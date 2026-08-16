"""Sample GRPO *task_ids* — random, DISJOINT from the current SFT set + eval (seeded).

Mirrors sample_sft_targets.py. GRPO is self-supervised RL: the student samples its
own rollouts and the env reward scores them, so GRPO needs ONLY task_ids — NO teacher
trajectory collection (unlike SFT).

Disjointness (matches sample_sft_targets.py invariant): GRPO ids EXCLUDE
  * SFT   (data/sft_collect_targets_5000.json — the current SFT target set)
  * eval  (data/final200.json, task_id in [0,1459) anyway)
so SFT / GRPO / EVAL remain mutually exclusive.

Reproducible: same args + same SFT/eval files -> identical GRPO set (seed 42).

Output: data/grpo_prompts_<N>.json
Usage:  python scripts/sample_grpo_targets.py --num 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA = _ROOT / "data"
TRAIN_LO, TRAIN_HI = 1459, 23421   # train task_id range, exclusive upper (from configs)


def _load_ids(rel: str) -> set[int]:
    p = DATA / rel
    return set(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else set()


def _pct(a: list[int], q: float) -> int:
    if not a:
        return -1
    s = sorted(a)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return int(s[lo] + (s[hi] - s[lo]) * (k - lo))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample GRPO task_ids (random, disjoint from SFT+eval).")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft", default="sft_collect_targets_5000.json",
                    help="SFT target file to exclude (the current SFT set)")
    args = ap.parse_args()

    sft = _load_ids(args.sft)
    eval_ids = _load_ids("final200.json")
    exclude = sft | eval_ids

    pool = [t for t in range(TRAIN_LO, TRAIN_HI) if t not in exclude]
    if args.num > len(pool):
        raise SystemExit(f"--num {args.num} > available pool {len(pool)} (train range minus SFT+eval)")

    grpo = sorted(random.Random(args.seed).sample(pool, args.num))
    out_path = DATA / f"grpo_prompts_{args.num}.json"
    out_path.write_text(json.dumps(grpo, ensure_ascii=False), encoding="utf-8")

    g = set(grpo)
    print(f"=== GRPO targets (seed={args.seed}, N={args.num}) ===")
    print(f"train range        : [{TRAIN_LO}, {TRAIN_HI})  ({TRAIN_HI - TRAIN_LO} tasks)")
    print(f"excluded SFT       : {len(sft)}  ({args.sft})")
    print(f"excluded eval      : {len(eval_ids)}  (final200)")
    print(f"available pool     : {len(pool)}")
    print()
    print(f"sampled GRPO       : {len(grpo)}  range [{grpo[0]}..{grpo[-1]}]  "
          f"中位={_pct(grpo, .5)}  (均匀预期≈{(TRAIN_LO+TRAIN_HI)//2})")
    print(f"-> {out_path.relative_to(_ROOT)}")
    print()
    print(f"DISJOINTNESS (must all be 0):")
    print(f"  GRPO ∩ SFT  : {len(g & sft)}")
    print(f"  GRPO ∩ EVAL : {len(g & eval_ids)}")


if __name__ == "__main__":
    main()
