"""Construct the zero-overlap Final-200 eval task split.

Final-200 = 200 task ids from [0, task_pool_size) that are NOT in the collected
training trajectories (data/trajectories_raw/). Deterministic (seeded) so the same
200 tasks evaluate Base / SFT / GRPO. Written to data/final200.json.

Run:
    python -m eval.build_final200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the zero-overlap Final-200 split.")
    ap.add_argument("--pool", type=int, default=1459, help="total task pool size")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", default=str(_ROOT / "data" / "trajectories_raw" / "gpt-5.6-terra"))
    ap.add_argument("--out", default=str(_ROOT / "data" / "final200.json"))
    args = ap.parse_args()

    raw = Path(args.raw)
    collected = set()
    if raw.exists():
        # Current collection writes JSONL (one record per trajectory); older runs
        # wrote per-task *.json. Read task_ids from whichever exists so the eval
        # split never overlaps ANY collected trajectory (train or eval-range).
        jsonl = raw / "trajectories_raw.jsonl"
        if jsonl.exists():
            import json as _json
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        collected.add(int(_json.loads(line).get("task_id")))
                    except (ValueError, _json.JSONDecodeError):
                        continue
        else:
            collected = {int(p.stem) for p in raw.glob("*.json")}
    available = [t for t in range(args.pool) if t not in collected]

    rng = random.Random(args.seed)
    k = min(args.n, len(available))
    if k < args.n:
        print(f"WARNING: only {len(available)} non-overlapping tasks available (< {args.n}); "
              f"collect more, or this overlaps training.")
    final = sorted(rng.sample(available, k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final), encoding="utf-8")
    print(f"Final-{len(final)} -> {out}  (excluded {len(collected)} collected task ids)")


if __name__ == "__main__":
    main()
