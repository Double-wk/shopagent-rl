#!/usr/bin/env python3
"""Build a task-disjoint atomic counterfactual holdout from reserved task IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.counterfactual_pairs import (  # noqa: E402
    build_pairs,
    dump_jsonl,
    load_products,
    validate_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=Path, default=ROOT / "data/ablation_tasks_500.json")
    parser.add_argument("--train-task-ids", type=Path, default=ROOT / "data/grpo_prompts_1000.json")
    parser.add_argument("--eval-task-ids", type=Path, default=ROOT / "data/final200.json")
    parser.add_argument(
        "--products", type=Path,
        default=ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data/counterfactual/heldout_atomic_pairs_v2.jsonl",
    )
    args = parser.parse_args()

    heldout_ids = [int(value) for value in json.loads(args.task_ids.read_text())]
    train_ids = set(json.loads(args.train_task_ids.read_text()))
    eval_ids = set(json.loads(args.eval_task_ids.read_text()))
    overlap = set(heldout_ids) & (train_ids | eval_ids)
    if overlap:
        raise SystemExit(f"heldout task overlap: {sorted(overlap)[:10]}")

    result = build_pairs(load_products(args.products), [{"task_id": value} for value in heldout_ids])
    invalid = [(pair["pair_id"], validate_pair(pair)) for pair in result.pairs]
    invalid = [(pair_id, errors) for pair_id, errors in invalid if errors]
    if invalid:
        raise SystemExit(f"refusing to write invalid pairs: {invalid[:5]}")
    dump_jsonl(result.pairs, args.out)
    summary = {
        "schema_version": "shopsim-atomic-constraint-pairs-v1",
        "split": "heldout-v2",
        "task_ids": str(args.task_ids),
        "train_overlap": 0,
        "final200_overlap": 0,
        "output": str(args.out),
        "stats": result.stats,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
