#!/usr/bin/env python3
"""Generate the v1 atomic option/price paired evaluation set (CPU-only)."""
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
    load_json_records,
    load_products,
    validate_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build validated ShopSimulator commitment-point counterfactual pairs."
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=ROOT.parent / "ShopSimulator/shop_env/data/items_eval_train.json.gz",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=ROOT / "outputs/sft/v1/evaluation/eval_sft3793_f200_t10x512_0810_1612.jsonl",
        help="Eval JSONL/report supplying realized goals and price_upper values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/counterfactual/final200_atomic_pairs_v1.jsonl",
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    products = load_products(args.products)
    trajectories = load_json_records(args.trajectories)
    result = build_pairs(products, trajectories)

    invalid = [(pair["pair_id"], validate_pair(pair)) for pair in result.pairs]
    invalid = [(pair_id, errors) for pair_id, errors in invalid if errors]
    if invalid:
        preview = json.dumps(invalid[:5], ensure_ascii=False)
        raise SystemExit(f"refusing to write {len(invalid)} invalid pairs: {preview}")

    dump_jsonl(result.pairs, args.output)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = {
        "schema_version": "shopsim-atomic-constraint-pairs-v1",
        "products": str(args.products),
        "trajectories": str(args.trajectories),
        "output": str(args.output),
        "stats": result.stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

