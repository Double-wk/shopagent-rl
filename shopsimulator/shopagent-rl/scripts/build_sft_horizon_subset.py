#!/usr/bin/env python3
"""Build a trajectory-length subset without modifying the SFT mother set.

The validated ``data/sft_train.jsonl`` remains the complete teacher-success
mother set. This script creates a separate analysis/training view for a fixed
interaction horizon, preserving record order and all original fields.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    """Prefer repository-relative paths in the audit summary."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "sft_train.jsonl",
        help="validated SFT mother set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "sft_train_horizon10.jsonl",
        help="separate subset JSONL",
    )
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="optional summary path; defaults to <output>.summary.json",
    )
    args = parser.parse_args()

    if args.min_steps < 0 or args.max_steps < args.min_steps:
        parser.error("require 0 <= min-steps <= max-steps")

    records = list(iter_jsonl(args.input))
    kept: list[dict[str, Any]] = []
    all_steps: Counter[int] = Counter()
    dropped_steps: Counter[int] = Counter()
    for record in records:
        steps = int(record.get("n_steps", 0))
        all_steps[steps] += 1
        if args.min_steps <= steps <= args.max_steps:
            kept.append(record)
        else:
            dropped_steps[steps] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = {
        "input": display_path(args.input),
        "output": display_path(args.output),
        "min_steps": args.min_steps,
        "max_steps": args.max_steps,
        "input_records": len(records),
        "output_records": len(kept),
        "dropped_records": len(records) - len(kept),
        "step_distribution": dict(sorted(all_steps.items())),
        "dropped_step_distribution": dict(sorted(dropped_steps.items())),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
