#!/usr/bin/env python3
"""Merge interrupted Final-N JSONL shards and recompute canonical metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.eval.metrics import aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    expected = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    records: dict[int, dict] = {}
    for input_name in args.inputs:
        with Path(input_name).open(encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = int(record["task_id"])
                if task_id in records:
                    raise ValueError(f"duplicate task_id: {task_id}")
                records[task_id] = record

    missing = [task_id for task_id in expected if task_id not in records]
    extra = sorted(set(records) - set(expected))
    if missing or extra:
        raise ValueError(f"task mismatch: missing={missing}, extra={extra}")

    ordered = [records[task_id] for task_id in expected]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as dst:
        for record in ordered:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    official_dir = out_path.parent / f"{out_path.stem}_tasks"
    official_dir.mkdir(parents=True, exist_ok=True)
    for record in ordered:
        official = {
            key: record.get(key, {} if key != "reward" else 0.0)
            for key in ("task_id", "reward", "reward_detail", "goal", "purchase", "conversation")
        }
        (official_dir / f"{record['task_id']}.json").write_text(
            json.dumps(official, ensure_ascii=False), encoding="utf-8"
        )

    sys.path.insert(0, str(ROOT.parent / "ShopSimulator"))
    from get_score import calculate_metrics  # noqa: E402

    official_metrics = calculate_metrics(
        [str(official_dir / f"{task_id}.json") for task_id in expected]
    )
    report = {
        "n_tasks": len(ordered),
        "aggregate_metrics": aggregate(ordered),
        "official_metrics": official_metrics,
        "inputs": args.inputs,
    }
    report_path = out_path.parent / f"{out_path.stem}_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"merged results -> {out_path}")
    print(f"official metrics -> {report_path}")


if __name__ == "__main__":
    main()
