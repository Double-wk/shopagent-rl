#!/usr/bin/env python3
"""Build v2 goal-side and nuisance controls from validated v1 atomic pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.constraint_causal_pairs import build_v2_pairs, validate_v2_pair  # noqa: E402
from experiment.counterfactual_pairs import dump_jsonl, load_json_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/counterfactual/final200_atomic_pairs_v1.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/counterfactual/final200_constraint_causal_v2.jsonl")
    args = parser.parse_args()
    result = build_v2_pairs(load_json_records(args.input))
    invalid = [(p["pair_id"], validate_v2_pair(p)) for p in result.pairs]
    invalid = [(pid, errors) for pid, errors in invalid if errors]
    if invalid:
        raise SystemExit(f"refusing to write invalid pairs: {invalid[:5]}")
    dump_jsonl(result.pairs, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "stats": result.stats},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
