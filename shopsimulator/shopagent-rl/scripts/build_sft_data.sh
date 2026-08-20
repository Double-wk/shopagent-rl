#!/usr/bin/env bash
# Build the SFT training set from raw teacher trajectories — TRAIN-RANGE ONLY.
#
# WHY THIS EXISTS (data-integrity guard for the resume experiment):
#   data/trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl is contaminated:
#   an earlier, range-unrestricted collection wrote 600 EVAL-range trajectories
#   (task_id in [0,1459)) as lines 1-600; the current --start 1459 collector
#   appended train-range (task_id in [1459,23421)) from line 601 onward.
#   teacher.validate.py has NO range guard (checks ok/strict/steps/legal only),
#   so feeding it the raw file leaks eval tasks into the SFT train set.
#
# This script:
#   1. Non-destructively filters the raw jsonl to train-range task_id in [1459,23421)
#      -> data/trajectories_raw/gpt-5.6-terra_trainonly/trajectories_raw.jsonl
#   2. Clears any stale per-task SFT files (validate.py does not clear them).
#   3. Runs teacher.validate on the filtered set -> data/trajectories_sft/ + sft_train.jsonl
#
# Idempotent + non-destructive: original raw file is never modified.
#
# Args:  $1 = MAX_KEEP (default 6800)
#        $2 = SAMPLE   (optional) randomly subsample this many train trajectories
#                      BEFORE validation (seeded). Use for the minimal first
#                      end-to-end run (e.g. SAMPLE=300) — small data, fast SFT.
set -euo pipefail

cd "$(dirname "$0")/.."   # scripts/.. -> shopagent-rl/
source /workspace/shopsimulator/shopagent-rl/scripts/paths.sh
PY="$SHOPAGENT_PY"
shopagent_require_py

RAW=data/trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl
TRAINONLY_DIR=data/trajectories_raw/gpt-5.6-terra_trainonly
TRAINONLY=$TRAINONLY_DIR/trajectories_raw.jsonl
MIN_TASK_ID=1459
MAX_TASK_ID=23421   # exclusive
MAX_KEEP="${1:-6800}"
SAMPLE="${2:-}"
SEED=42

echo "== 1. filter raw -> train-only (task_id in [$MIN_TASK_ID, $MAX_TASK_ID))${SAMPLE:+, random-sample=$SAMPLE} =="
mkdir -p "$TRAINONLY_DIR"
$PY - "$RAW" "$TRAINONLY" "$MIN_TASK_ID" "$MAX_TASK_ID" "$SAMPLE" "$SEED" <<'EOF'
import json, sys, random
raw, out = sys.argv[1], sys.argv[2]
lo, hi = int(sys.argv[3]), int(sys.argv[4])
sample = int(sys.argv[5]) if sys.argv[5] else None
seed = int(sys.argv[6])
kept = []
n_in=n_eval=n_bad=0
with open(raw, encoding="utf-8") as fi:
    for line in fi:
        line=line.strip()
        if not line: continue
        n_in+=1
        try: rec=json.loads(line)
        except json.JSONDecodeError: n_bad+=1; continue
        tid=rec.get("task_id",-1)
        if lo<=tid<hi:
            kept.append(line)
        else:
            n_eval+=1   # eval-range or out-of-range -> excluded
if sample and len(kept) > sample:
    kept = random.Random(seed).sample(kept, sample)
with open(out, "w", encoding="utf-8") as fo:
    for line in kept:
        fo.write(line + "\n")
print(f"  in={n_in}  kept_train={len(kept)}  excluded={n_eval}  bad_json={n_bad}"
      + (f"  sampled={sample}" if sample else ""))
EOF

echo "== 2. clear stale per-task SFT files =="
mkdir -p data/trajectories_sft
# remove old per-task jsons (keep the dir)
find data/trajectories_sft -maxdepth 1 -name '*.json' -delete
echo "  cleared."

echo "== 3. validate (strict-pass) -> data/trajectories_sft/ + sft_train.jsonl =="
$PY -m experiment.teacher.validate --raw "$TRAINONLY_DIR" --max_keep "$MAX_KEEP"

echo "done. SFT set -> data/sft_train.jsonl"
