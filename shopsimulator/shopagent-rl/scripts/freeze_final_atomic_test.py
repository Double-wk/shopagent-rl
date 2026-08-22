#!/usr/bin/env python3
"""Freeze a provenance-disjoint atomic counterfactual test split.

The split is generated only from product/task records that are absent from all
listed training, development, and historical evaluation artifacts.  It uses
the programmatic atomic builders, so no teacher call or model output is needed.
Selected tasks are unique across intervention types; this makes the final test
task-disjoint even within the test file itself.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.constraint_causal_pairs import build_v2_pairs  # noqa: E402
from experiment.counterfactual_pairs import (  # noqa: E402
    HELD_OUT_MECHANISMS,
    TRAIN_MECHANISMS,
    build_pairs,
    dump_jsonl,
)


def _json_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _excluded_tasks(path: Path) -> set[int]:
    """Read task IDs, pair JSONL, SFT JSONL, or split-summary JSON.

    A missing artifact is a hard error: silently returning an empty set would
    under-exclude and quietly weaken the contamination guarantee that the frozen
    test set rests on.  Retire an artifact by commenting it out of the exclusion
    list with a reason, not by deleting the file.
    """
    if not path.exists():
        raise SystemExit(
            f"exclusion artifact missing: {path}\n"
            "Refusing to build a split that under-excludes. If this artifact is "
            "genuinely retired, comment it out of the exclusion list with a reason."
        )
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["task_id"])
        return {int(v.as_py()) for v in table.column("task_id") if v.as_py() is not None}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list) and all(isinstance(v, (int, float)) for v in payload):
        return {int(v) for v in payload}
    if isinstance(payload, dict) and isinstance(payload.get("task_ids"), list):
        return {int(v) for v in payload["task_ids"]}
    result: set[int] = set()
    for row in _json_records(path):
        task_id = row.get("task_id")
        if isinstance(task_id, int):
            result.add(task_id)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_robin_by_category(records: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str((record.get("product") or {}).get("category", ""))].append(record)
    for values in groups.values():
        rng.shuffle(values)
    categories = sorted(groups)
    selected: list[dict] = []
    while len(selected) < limit and categories:
        next_categories: list[str] = []
        for category in categories:
            values = groups[category]
            if values:
                selected.append(values.pop())
                if len(selected) >= limit:
                    break
            if values:
                next_categories.append(category)
        categories = next_categories
    return selected


def _seal_pair(pair: dict) -> dict:
    original = pair.get("original") or {}
    counterfactual = pair.get("counterfactual") or {}
    relation = [
        *(original.get("expected_action_intents") or []),
        *(counterfactual.get("expected_action_intents") or []),
    ][:2]
    return {
        **pair,
        "split": "final-atomic-test-v1",
        "expected_relation": [str(value) for value in relation],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path,
                    default=ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data/counterfactual/final_atomic_test_v1.jsonl")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="artifact containing task IDs; repeatable")
    ap.add_argument("--exclude-list", type=Path, default=None,
                    help="newline-delimited exclusion artifact paths, relative to repo root")
    ap.add_argument("--price", type=int, default=150)
    ap.add_argument("--option", type=int, default=75)
    ap.add_argument("--nuisance", type=int, default=75)
    # Held-out mechanisms.  Training data (scripts/prepare_paper_grpo_v1.py)
    # draws only price_above_budget / option_swap / nuisance_display_note, so
    # accuracy on these two measures generalization to an intervention
    # mechanism never seen during training -- not in-distribution recall.
    ap.add_argument("--option-unavailable", type=int, default=75,
                    help="held-out: unseen constraint type (availability)")
    ap.add_argument("--option-price-over-budget", type=int, default=75,
                    help="held-out: seen constraint (budget) on an unseen surface")
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    with gzip.open(args.products, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    exclusion_paths = list(args.exclude)
    if args.exclude_list:
        for line in args.exclude_list.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                path = Path(value)
                exclusion_paths.append(path if path.is_absolute() else ROOT / path)

    excluded: set[int] = set()
    exclusion_stats: dict[str, int] = {}
    for path in exclusion_paths:
        ids = _excluded_tasks(path)
        exclusion_stats[str(path)] = len(ids)
        excluded.update(ids)

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
    for pair in atomic:
        by_type[pair["intervention_type"]].append(pair)
    for pair in v2:
        by_type[pair["intervention_type"]].append(pair)

    requested = {
        "price_above_budget": args.price,
        "option_swap": args.option,
        "nuisance_display_note": args.nuisance,
        "option_unavailable": args.option_unavailable,
        "option_price_over_budget": args.option_price_over_budget,
    }
    requested = {name: n for name, n in requested.items() if n > 0}
    chosen: list[dict] = []
    chosen_tasks: set[int] = set()
    selected_by_type: Counter[str] = Counter()
    for intervention_type, limit in requested.items():
        available = [p for p in by_type[intervention_type] if p["task_id"] not in chosen_tasks]
        selected = _round_robin_by_category(available, limit, args.seed + len(chosen))
        if len(selected) != limit:
            raise SystemExit(
                f"insufficient disjoint {intervention_type} pairs: {len(selected)} < {limit}"
            )
        chosen.extend(selected)
        chosen_tasks.update(p["task_id"] for p in selected)
        selected_by_type[intervention_type] = len(selected)

    # Stable order makes the output/hash independent of dictionary iteration.
    chosen = [_seal_pair(pair) for pair in chosen]
    chosen.sort(key=lambda p: (str(p["intervention_type"]), int(p["task_id"]), str(p["pair_id"])))
    output = args.output
    dump_jsonl(chosen, output)
    summary_path = args.summary or output.with_suffix(".summary.json")
    categories = Counter(str((p.get("product") or {}).get("category", "")) for p in chosen)
    asins = {str((p.get("source") or {}).get("asin", "")) for p in chosen}
    summary = {
        "schema_version": "shopsim-final-atomic-test-v1",
        "split": "final-atomic-test-v1",
        "sealed": True,
        "seed": args.seed,
        "products": str(args.products),
        "products_sha256": _sha256(args.products),
        "excluded_artifacts": [str(p) for p in exclusion_paths],
        "excluded_task_counts": exclusion_stats,
        "excluded_tasks_union": len(excluded),
        "excluded_products_union": len(excluded_asins),
        "candidate_tasks": len(candidates),
        "requested_by_intervention": requested,
        "selected_by_intervention": dict(selected_by_type),
        # The generalization split, recorded so a reader can verify which
        # mechanisms the training data was allowed to contain.  Asserted at
        # build time in scripts/prepare_paper_grpo_v1.py.
        "train_mechanisms": sorted(TRAIN_MECHANISMS),
        "held_out_mechanisms": sorted(HELD_OUT_MECHANISMS),
        "verified_by_mechanism": {
            name: next(
                (str((p.get("intervention") or {}).get("verified_by", "reward_and_state"))
                 for p in chosen if p["intervention_type"] == name),
                "",
            )
            for name in sorted(requested)
        },
        "rows": len(chosen),
        "unique_tasks": len(chosen_tasks),
        "unique_products": len(asins),
        "category_count": len(categories),
        "category_histogram": dict(categories),
        "output": str(output),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    summary["output_sha256"] = _sha256(output)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
