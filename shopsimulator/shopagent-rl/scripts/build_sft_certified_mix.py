#!/usr/bin/env python3
"""Build the certified SFT mixture with a consistent trajectory horizon."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/sft_train.jsonl")
    parser.add_argument("--certified", type=Path, default=ROOT / "data/sft_certified_train.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/sft_train_certified_mix.jsonl")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    baseline = list(iter_jsonl(args.baseline))
    certified = list(iter_jsonl(args.certified))
    kept = [record for record in baseline if int(record.get("n_steps", 0)) <= args.max_turns]
    dropped = [record for record in baseline if int(record.get("n_steps", 0)) > args.max_turns]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in (*kept, *certified):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "out": str(args.out),
        "max_turns": args.max_turns,
        "baseline_total": len(baseline),
        "baseline_kept": len(kept),
        "baseline_dropped": len(dropped),
        "dropped_turns": dict(sorted(Counter(int(record["n_steps"]) for record in dropped).items())),
        "certified_records": len(certified),
        "mix_records": len(kept) + len(certified),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
