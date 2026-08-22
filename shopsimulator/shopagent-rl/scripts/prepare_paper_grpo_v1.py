#!/usr/bin/env python3
"""Prepare the provenance-disjoint 800-row paper GRPO input."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.constraint_causal_pairs import build_v2_pairs  # noqa: E402
from experiment.counterfactual_pairs import (  # noqa: E402
    HELD_OUT_MECHANISMS,
    TRAIN_MECHANISMS,
    build_pairs,
    dump_jsonl,
)
from scripts.build_certified_grpo_data import build_rows  # noqa: E402
from scripts.freeze_final_atomic_test import _excluded_tasks, _round_robin_by_category  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(pair: dict) -> dict:
    relation = [
        *((pair.get("original") or {}).get("expected_action_intents") or []),
        *((pair.get("counterfactual") or {}).get("expected_action_intents") or []),
    ][:2]
    return {**pair, "split": "paper-grpo-v1", "expected_relation": relation}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path,
                    default=ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz")
    ap.add_argument("--exclude-list", type=Path,
                    default=ROOT / "experiments/splits/final_atomic_test_v1_exclusions.txt")
    ap.add_argument("--final-test", type=Path,
                    default=ROOT / "data/counterfactual/final_atomic_test_v1.jsonl")
    ap.add_argument("--system", type=Path, default=ROOT / "configs/teacher_gpt-5.6-terra.yaml")
    ap.add_argument("--tokenizer", type=Path, default=Path(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/"
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    ))
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--pairs-out", type=Path,
                    default=ROOT / "data/counterfactual/paper_grpo_pairs_v1.jsonl")
    ap.add_argument("--environment-out", type=Path,
                    default=ROOT / "data/paper_grpo_environment_tasks_v1.json")
    ap.add_argument("--parquet-out", type=Path,
                    default=ROOT / "data/grpo_certified_paper_v1_800_pairblocked.parquet")
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "data/grpo_certified_paper_v1_800_pairblocked.summary.json")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    with gzip.open(args.products, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    system_prompt = yaml.safe_load(args.system.read_text(encoding="utf-8"))["system_prompt"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def prompt_tokens(pair: dict) -> int:
        return max(
            len(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(pair[side]["observation"])},
            ], add_generation_prompt=True, tokenize=True))
            for side in ("original", "counterfactual")
        )
    exclusion_paths: list[Path] = []
    for line in args.exclude_list.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            path = Path(value)
            exclusion_paths.append(path if path.is_absolute() else ROOT / path)
    exclusion_paths.append(args.final_test)

    excluded: set[int] = set()
    for path in exclusion_paths:
        excluded.update(_excluded_tasks(path))
    excluded_asins = {
        str(products[i].get("asin", "")) for i in excluded if 0 <= i < len(products)
    }
    candidates = [
        i for i, product in enumerate(products)
        if i not in excluded and str(product.get("asin", "")) not in excluded_asins
    ]
    atomic = build_pairs(products, ({"task_id": i, "goal": {}} for i in candidates)).pairs
    v2 = build_v2_pairs(atomic).pairs
    by_type: dict[str, list[dict]] = defaultdict(list)
    for pair in (*atomic, *v2):
        by_type[pair["intervention_type"]].append(pair)

    requested = {
        "price_above_budget": 100,
        "option_swap": 60,
        "nuisance_display_note": 40,
    }
    # The held-out-mechanism claim rests on training never seeing them.  Assert
    # it here instead of relying on this dict staying correct by accident.
    leaked = sorted(set(requested) & HELD_OUT_MECHANISMS)
    if leaked:
        raise SystemExit(
            f"held-out mechanisms must not appear in training data: {leaked}. "
            "They are reserved for the frozen test set "
            "(scripts/freeze_final_atomic_test.py)."
        )
    unknown = sorted(set(requested) - TRAIN_MECHANISMS)
    if unknown:
        raise SystemExit(f"unknown training mechanisms: {unknown}")
    pair_records: list[dict] = []
    pair_tasks: set[int] = set()
    for offset, (intervention_type, count) in enumerate(requested.items()):
        available = [
            p for p in by_type[intervention_type]
            if p["task_id"] not in pair_tasks
            and len(system_prompt) + max(
                len(str(p[side]["observation"])) for side in ("original", "counterfactual")
            ) <= 3000
        ]
        available = [p for p in available if prompt_tokens(p) <= args.max_prompt_tokens]
        selected = _round_robin_by_category(available, count, args.seed + offset)
        if len(selected) != count:
            raise SystemExit(f"insufficient {intervention_type} pairs")
        pair_records.extend(seal(pair) for pair in selected)
        pair_tasks.update(pair["task_id"] for pair in selected)

    remaining = [task_id for task_id in candidates if task_id not in pair_tasks]
    rng = random.Random(args.seed + 100)
    rng.shuffle(remaining)
    environment_tasks = sorted(remaining[:400])
    if len(environment_tasks) != 400:
        raise SystemExit("insufficient environment tasks")

    pair_records.sort(key=lambda p: (p["intervention_type"], p["task_id"]))
    dump_jsonl(pair_records, args.pairs_out)
    args.environment_out.write_text(
        json.dumps(environment_tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = build_rows(
        environment_tasks,
        [p for p in pair_records if p["intervention_type"] == "price_above_budget"],
        [p for p in pair_records if p["intervention_type"] == "option_swap"],
        [p for p in pair_records if p["intervention_type"] == "nuisance_display_note"],
        system_prompt,
        environment_repeat=1,
        environment_tasks=-1,
        price_pairs=100,
        option_pairs=60,
        nuisance_pairs=40,
        option_intervention_type="option_swap",
        seed=args.seed,
        pair_block_size=8,
    )
    schema = pa.schema([
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("extra_info", pa.struct([("index", pa.int64())])),
        ("task_id", pa.int64()),
        ("data_source", pa.string()),
        ("sample_mode", pa.string()),
        ("pair_id", pa.string()),
        ("relation_id", pa.string()),
        ("intervention_type", pa.string()),
        ("side", pa.string()),
        ("expected_action_intents", pa.list_(pa.string())),
        ("expected_relation", pa.list_(pa.string())),
        ("allowed_actions", pa.list_(pa.string())),
    ])
    args.parquet_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.parquet_out)
    rendered_lengths = [
        len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True))
        for row in rows
    ]
    if max(rendered_lengths) > args.max_prompt_tokens:
        raise SystemExit(
            f"rendered prompt exceeds limit: {max(rendered_lengths)} > {args.max_prompt_tokens}"
        )

    final_records = [json.loads(line) for line in args.final_test.read_text(encoding="utf-8").splitlines()]
    final_tasks = {p["task_id"] for p in final_records}
    final_asins = {str(p["source"]["asin"]) for p in final_records}
    train_asins = {str(products[t].get("asin", "")) for t in pair_tasks | set(environment_tasks)}
    summary = {
        "schema_version": "shopsim-paper-grpo-v1",
        "seed": args.seed,
        "rows": len(rows),
        "environment_rows": sum(r["sample_mode"] == "environment" for r in rows),
        "counterfactual_rows": sum(r["sample_mode"] == "counterfactual" for r in rows),
        "pair_count": len(pair_records),
        "pairs_by_intervention": dict(Counter(p["intervention_type"] for p in pair_records)),
        "unique_pair_tasks": len(pair_tasks),
        "unique_environment_tasks": len(environment_tasks),
        "pair_environment_task_overlap": len(pair_tasks & set(environment_tasks)),
        "final_test_task_overlap": len((pair_tasks | set(environment_tasks)) & final_tasks),
        "final_test_product_overlap": len(train_asins & final_asins),
        "max_prompt_tokens": max(rendered_lengths),
        "prompt_token_limit": args.max_prompt_tokens,
        "excluded_tasks_union": len(excluded),
        "pairs": str(args.pairs_out.relative_to(ROOT)),
        "pairs_sha256": sha256(args.pairs_out),
        "environment_tasks": str(args.environment_out.relative_to(ROOT)),
        "environment_tasks_sha256": sha256(args.environment_out),
        "parquet": str(args.parquet_out.relative_to(ROOT)),
        "parquet_sha256": sha256(args.parquet_out),
        "final_test": str(args.final_test.relative_to(ROOT)),
        "final_test_sha256": sha256(args.final_test),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
