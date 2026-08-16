"""Fix the experiment's data budgets by RANDOM SAMPLING (seeded, reproducible).

Why this exists (methodology for the resume experiment): full teacher collection
isn't on the critical path — SFT already has 2613 strict-pass trajectories, GRPO
is self-supervised (env reward, needs only task_ids), eval is a fixed 200 held-out
set. So instead of "however much we collected", commit to FIXED, SAMPLED budgets:
reproducible numbers that stand up in a resume ("trained on N SFT / M GRPO prompts
/ 200 held-out tasks").

This script produces three DISJOINT task sets:
  * SFT    : random --sft_size (default 2000) from the strict-pass pool
  * GRPO   : random --grpo_size (default 1000) task_ids from train range, EXCLUDING
             SFT + eval   (GRPO needs no teacher data — student samples its own
             rollouts; only task_ids + env reward matter)
  * EVAL   : unchanged (data/final200.json, task_id in [0,1459))

All sampling uses --seed (default 42). Idempotent: re-running with the same args
+ same source data yields identical sets.

Outputs:
  data/sft_train.jsonl            (the --sft_size sampled SFT records; training file)
  data/sft_train_full_<N>.jsonl   (snapshot of the full strict-pass pool, preserved)
  data/trajectories_sft/*.json    (synced per-task files for the sampled SFT set)
  data/grpo_prompts_<M>.json      (list of --grpo_size task_ids for GRPO)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SFT = _ROOT / "data" / "sft_train.jsonl"
SFT_DIR = _ROOT / "data" / "trajectories_sft"
EVAL = _ROOT / "data" / "final200.json"
GRPO_RANGE = (1459, 23421)   # train task_id range (exclusive upper), from configs


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample fixed SFT/GRPO data budgets.")
    ap.add_argument("--sft_size", type=int, default=2000)
    ap.add_argument("--grpo_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- SFT: random sample from the strict-pass pool ---
    # Pool source = the FULL strict-pass snapshot if it exists (so re-running with
    # a different --sft_size always samples from the complete pool, not from an
    # already-subsampled sft_train.jsonl). First run has no snapshot yet, so it
    # seeds the snapshot from sft_train.jsonl (which build_sft_data left full).
    snaps = sorted(glob.glob(str(_ROOT / "data" / "sft_train_full_*.jsonl")))
    pool_file = Path(snaps[-1]) if snaps else SFT
    pool = [json.loads(l) for l in pool_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_pool = len(pool)
    if args.sft_size > n_pool:
        raise SystemExit(f"--sft_size {args.sft_size} > pool {n_pool} (from {pool_file.name}); lower it or collect more")
    # (re)write the full-pool snapshot so the source of truth is preserved
    full_snap = _ROOT / "data" / f"sft_train_full_{n_pool}.jsonl"
    full_snap.write_text(pool_file.read_text(encoding="utf-8"), encoding="utf-8")
    sft_sel = random.Random(args.seed).sample(pool, args.sft_size)
    sft_ids = {r["task_id"] for r in sft_sel}

    # write combined + sync per-task files (clear stale first)
    for f in glob.glob(str(SFT_DIR / "*.json")):
        os.remove(f)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SFT, "w", encoding="utf-8") as f:
        for r in sft_sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(SFT_DIR / f"{r['task_id']}.json", "w", encoding="utf-8") as pf:
                json.dump(r, pf, ensure_ascii=False, indent=2)

    # --- GRPO: task_ids from train range, EXCLUDING sft + eval ---
    eval_ids = set(json.loads(EVAL.read_text(encoding="utf-8"))) if EVAL.exists() else set()
    exclude = sft_ids | eval_ids
    candidates = [t for t in range(GRPO_RANGE[0], GRPO_RANGE[1]) if t not in exclude]
    if args.grpo_size > len(candidates):
        raise SystemExit(f"--grpo_size {args.grpo_size} > available {len(candidates)}")
    grpo = sorted(random.Random(args.seed).sample(candidates, args.grpo_size))
    grpo_path = _ROOT / "data" / f"grpo_prompts_{args.grpo_size}.json"
    grpo_path.write_text(json.dumps(grpo, ensure_ascii=False), encoding="utf-8")

    # --- report + disjointness check ---
    grpo_ids = set(grpo)
    print(f"SFT  : {len(sft_sel):>5}  (sampled from {n_pool} strict-pass)  -> {SFT.relative_to(_ROOT)}")
    print(f"       full pool snapshot                         -> {full_snap.relative_to(_ROOT)}")
    print(f"GRPO : {len(grpo):>5}  task_ids (train range, excl. SFT+eval)  -> {grpo_path.relative_to(_ROOT)}")
    print(f"EVAL : {len(eval_ids):>5}  (unchanged)                         -> {EVAL.relative_to(_ROOT)}")
    print(f"\nDISJOINTNESS (must all be 0):")
    print(f"  SFT ∩ GRPO : {len(sft_ids & grpo_ids)}")
    print(f"  SFT ∩ EVAL : {len(sft_ids & eval_ids)}")
    print(f"  GRPO ∩ EVAL: {len(grpo_ids & eval_ids)}")
    print(f"  SFT in train range: {sum(1 for t in sft_ids if GRPO_RANGE[0] <= t < GRPO_RANGE[1])}/{len(sft_ids)}")


if __name__ == "__main__":
    main()
