"""Sample ABLATION *task_ids* — random, DISJOINT from SFT + GRPO + eval (seeded).

消融实验（ablation）用：held-out 诊断集，和全部训练集（SFT/GRPO）+ 评测集（eval）
完全互斥，保证消融结果不被训练数据污染。

级联互斥顺序（"一步一步"）：
  1. SFT      : 固定（data/sft_collect_targets_5000.json）
  2. GRPO     : train \\ (SFT ∪ eval)              -> grpo_prompts_<N>.json
  3. ABLATION : train \\ (SFT ∪ GRPO ∪ eval)       -> ablation_tasks_<N>.json   <-- 本脚本

三者都 seed 42、train 范围 [1459,23421) 内随机，互相零重叠。

Reproducible: same args + same SFT/GRPO/eval files -> identical ablation set.
Output: data/ablation_tasks_<N>.json
Usage:  python scripts/sample_ablation_targets.py --num 500 --seed 42
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
    ap = argparse.ArgumentParser(description="Sample ABLATION task_ids (random, disjoint from SFT+GRPO+eval).")
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft", default="sft_collect_targets_5000.json",
                    help="SFT target file to exclude")
    ap.add_argument("--grpo", default="grpo_prompts_1000.json",
                    help="GRPO target file to exclude (must already exist)")
    args = ap.parse_args()

    sft = _load_ids(args.sft)
    grpo = _load_ids(args.grpo)
    eval_ids = _load_ids("final200.json")
    exclude = sft | grpo | eval_ids

    pool = [t for t in range(TRAIN_LO, TRAIN_HI) if t not in exclude]
    if args.num > len(pool):
        raise SystemExit(f"--num {args.num} > available pool {len(pool)} (train range minus SFT+GRPO+eval)")

    abl = sorted(random.Random(args.seed).sample(pool, args.num))
    out_path = DATA / f"ablation_tasks_{args.num}.json"
    out_path.write_text(json.dumps(abl, ensure_ascii=False), encoding="utf-8")

    a = set(abl)
    print(f"=== ABLATION targets (seed={args.seed}, N={args.num}) ===")
    print(f"train range        : [{TRAIN_LO}, {TRAIN_HI})  ({TRAIN_HI - TRAIN_LO} tasks)")
    print(f"excluded SFT       : {len(sft)}  ({args.sft})")
    print(f"excluded GRPO      : {len(grpo)}  ({args.grpo})")
    print(f"excluded eval      : {len(eval_ids)}  (final200)")
    print(f"available pool     : {len(pool)}")
    print()
    print(f"sampled ABLATION   : {len(abl)}  range [{abl[0]}..{abl[-1]}]  "
          f"中位={_pct(abl, .5)}  (均匀预期≈{(TRAIN_LO+TRAIN_HI)//2})")
    print(f"-> {out_path.relative_to(_ROOT)}")
    print()
    print(f"DISJOINTNESS (must all be 0):")
    print(f"  ABLATION ∩ SFT  : {len(a & sft)}")
    print(f"  ABLATION ∩ GRPO : {len(a & grpo)}")
    print(f"  ABLATION ∩ EVAL : {len(a & eval_ids)}")
    print()
    used = len(sft) + len(grpo) + len(abl)
    print(f"train 区间占用汇总 : SFT {len(sft)} + GRPO {len(grpo)} + ABLATION {len(abl)} = {used}"
          f"  / {TRAIN_HI - TRAIN_LO}  (剩 {TRAIN_HI - TRAIN_LO - used} 未用)")


if __name__ == "__main__":
    main()
