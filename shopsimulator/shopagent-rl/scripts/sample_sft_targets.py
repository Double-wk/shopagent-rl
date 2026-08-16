"""Sample SFT *teacher-collection targets* — random task_ids across the FULL train range.

WHY THIS EXISTS (fixes the SFT coverage bug, see README 🚧 TODO):
  The current SFT pool is biased — gpt-5.6-terra was collected as a *sequential prefix*
  (task_id 0–4824, median 2827) and deepseek is also front-loaded (median 2889). Neither
  is a random sample of the 21,962 train tasks. Validating them doesn't fix the bias.

  This script draws a CLEAN random sample of task_ids (seed 42) from the full train range,
  disjoint from GRPO prompts + eval, so teacher collection can be redone with proper
  coverage. Collection on these ids → validate → ~80% strict-pass → the new SFT set.

Disjointness (matches sample_budgets.py invariant): targets EXCLUDE
  * eval       (data/final200.json, task_id in [0,1459) anyway)
  * GRPO       (data/grpo_prompts_1000.json)
so SFT / GRPO / EVAL remain mutually exclusive.

Outputs:
  data/sft_collect_targets_<N>.json   (sorted list of task_ids to go collect)

Usage:
  python scripts/sample_sft_targets.py --num 5000 --seed 42
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


def _raw_ids(model: str) -> set[int]:
    """task_ids already collected (raw) for a given teacher model."""
    p = DATA / "trajectories_raw" / model / "trajectories_raw.jsonl"
    if not p.exists():
        return set()
    out: set[int] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line).get("task_id"))
            except json.JSONDecodeError:
                continue
    return out


def _pct(a: list[int], q: float) -> int:
    """Linear-interpolation percentile (no numpy needed)."""
    if not a:
        return -1
    s = sorted(a)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return int(s[lo] + (s[hi] - s[lo]) * (k - lo))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample SFT teacher-collection targets (random, full train range).")
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_ids = _load_ids("final200.json")
    grpo_ids = _load_ids("grpo_prompts_1000.json")
    exclude = eval_ids | grpo_ids

    pool = [t for t in range(TRAIN_LO, TRAIN_HI) if t not in exclude]
    if args.num > len(pool):
        raise SystemExit(f"--num {args.num} > available pool {len(pool)} (train range minus eval+grpo)")

    sample = sorted(random.Random(args.seed).sample(pool, args.num))

    # --- overlap with already-collected raw data (how many can we reuse vs must collect fresh) ---
    # Teacher = gpt-5.6-terra ONLY (deepseek is NOT used as the SFT source), so reuse counts
    # terra trajectories only — a task_id covered by deepseek but not terra still needs collecting.
    terra = _raw_ids("gpt-5.6-terra")
    existing = terra
    already = [t for t in sample if t in existing]
    need_new = [t for t in sample if t not in existing]

    out_path = DATA / f"sft_collect_targets_{args.num}.json"
    out_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    # --- report ---
    print(f"=== SFT collection targets (seed={args.seed}, N={args.num}) ===")
    print(f"train range        : [{TRAIN_LO}, {TRAIN_HI})  ({TRAIN_HI - TRAIN_LO} tasks)")
    print(f"excluded eval      : {len(eval_ids & set(range(TRAIN_LO, TRAIN_HI)))}  (final200)")
    print(f"excluded GRPO      : {len(grpo_ids)}  (grpo_prompts_1000)")
    print(f"available pool     : {len(pool)}")
    print()
    print(f"sampled targets    : {len(sample)}  -> {out_path.relative_to(_ROOT)}")
    print(f"  分布(verify random): min={sample[0]}  p25={_pct(sample, .25)}  "
          f"中位={_pct(sample, .5)}  p75={_pct(sample, .75)}  max={sample[-1]}")
    print(f"  真随机预期: 中位≈{(TRAIN_LO+TRAIN_HI)//2}  max≈{TRAIN_HI}  "
          f"({'✓ 随机OK' if _pct(sample, .5) > 9000 else '✗ 仍偏前段!'})")
    print()
    print(f"=== 跟现有 raw 数据的重叠（teacher = terra only）===")
    print(f"  已有 terra     : {len(terra)} raw  (teacher 唯一来源; deepseek 不计入)")
    print(f"  目标里 terra 已采过的 : {len(already)}  (可复用，不重采)")
    print(f"  >>> 需新采(terra)     : {len(need_new)}  (采集时间主要花在这部分)")
    print(f"  按 80% strict-pass 产率 → 预计可用 SFT ≈ {int(args.num * 0.8)}")


if __name__ == "__main__":
    main()
