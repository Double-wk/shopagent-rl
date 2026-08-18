#!/usr/bin/env python3
"""Build task-disjoint dev/test pairs with an explicit, visible price budget."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.counterfactual_pairs import build_pairs, dump_jsonl, load_products, validate_pair
from experiment.explicit_budget_pairs import make_budget_explicit, validate_explicit_budget_pair


def load_json(path: Path) -> list[int]:
    return [int(value) for value in json.loads(path.read_text(encoding="utf-8"))]


def jsonl_task_ids(path: Path) -> set[int]:
    result: set[int] = set()
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task_id = json.loads(line).get("task_id")
            if isinstance(task_id, int):
                result.add(task_id)
    return result


def parquet_task_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    import pyarrow.parquet as pq

    return {int(value) for value in pq.read_table(path, columns=["task_id"])["task_id"].to_pylist()}


def build_split(products: list[dict], task_ids: list[int], out: Path, split: str) -> dict:
    built = build_pairs(products, [{"task_id": task_id} for task_id in task_ids])
    base_invalid = [
        (pair.get("pair_id"), errors)
        for pair in built.pairs
        if (errors := validate_pair(pair))
    ]
    if base_invalid:
        raise SystemExit(f"invalid atomic {split} pairs: {base_invalid[:5]}")
    pairs = [make_budget_explicit(pair) for pair in built.pairs]
    invalid = []
    for pair in pairs:
        errors = validate_explicit_budget_pair(pair)
        if errors:
            invalid.append((pair.get("pair_id"), errors))
    if invalid:
        raise SystemExit(f"invalid {split} pairs: {invalid[:5]}")

    dump_jsonl(pairs, out)
    summary = {
        "split": split,
        "seeded_task_count": len(task_ids),
        "task_count_with_pairs": len({pair["task_id"] for pair in pairs}),
        "pair_count": len(pairs),
        "by_intervention_type": dict(Counter(pair["intervention_type"] for pair in pairs)),
        "by_budget_source": dict(Counter(
            (pair.get("goal") or {}).get("price_upper_source")
            for pair in pairs if pair["intervention_type"] == "price_above_budget"
        )),
        "task_ids": task_ids,
        "output": str(out),
        "builder_stats": built.stats,
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tasks-per-split", type=int, default=500)
    parser.add_argument("--train-start", type=int, default=1459)
    parser.add_argument("--train-stop", type=int, default=23421)
    parser.add_argument(
        "--products", type=Path,
        default=ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz",
    )
    parser.add_argument(
        "--dev-out", type=Path,
        default=ROOT / "data/counterfactual/explicit_budget_dev_v1.jsonl",
    )
    parser.add_argument(
        "--test-out", type=Path,
        default=ROOT / "data/counterfactual/explicit_budget_test_v1.jsonl",
    )
    args = parser.parse_args()

    excluded = set(load_json(ROOT / "data/sft_collect_targets_5000.json"))
    excluded.update(load_json(ROOT / "data/grpo_prompts_1000.json"))
    excluded.update(load_json(ROOT / "data/ablation_tasks_500.json"))
    excluded.update(jsonl_task_ids(ROOT / "data/sft_train_certified_corrective_mix.jsonl"))
    excluded.update(parquet_task_ids(ROOT / "data/grpo_certified_natural_train.parquet"))
    excluded.update(jsonl_task_ids(ROOT / "data/counterfactual/heldout_atomic_pairs_v2.jsonl"))

    candidates = [task_id for task_id in range(args.train_start, args.train_stop) if task_id not in excluded]
    needed = args.tasks_per_split * 2
    if len(candidates) < needed:
        raise SystemExit(f"only {len(candidates)} eligible tasks, need {needed}")
    random.Random(args.seed).shuffle(candidates)
    dev_ids = sorted(candidates[:args.tasks_per_split])
    test_ids = sorted(candidates[args.tasks_per_split:needed])
    if set(dev_ids) & set(test_ids):
        raise SystemExit("dev/test task overlap")

    products = load_products(args.products)
    dev = build_split(products, dev_ids, args.dev_out, "explicit-budget-dev-v1")
    test = build_split(products, test_ids, args.test_out, "explicit-budget-test-v1-sealed")
    report = {
        "seed": args.seed,
        "excluded_task_count": len(excluded),
        "eligible_task_count": len(candidates),
        "dev": {key: value for key, value in dev.items() if key != "task_ids"},
        "test": {key: value for key, value in test.items() if key != "task_ids"},
        "dev_test_overlap": 0,
        "test_policy": "sealed: do not evaluate until method and checkpoint selection are frozen",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
