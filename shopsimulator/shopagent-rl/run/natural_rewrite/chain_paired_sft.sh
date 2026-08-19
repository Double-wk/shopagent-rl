#!/usr/bin/env bash
# Chain: wait for train-side natural generation -> build paired SFT data ->
# concat with baseline trajectories -> launch paired-constraint SFT training.
# All steps detached-safe; every step logs to run/natural_rewrite/.
set -euo pipefail
cd /workspace/shopsimulator/shopagent-rl
PY=/overlay/miniconda3/envs/shopsim/bin/python
LOG=run/natural_rewrite

# 1. wait for the generation process (pid passed as $1) to finish
while kill -0 "$1" 2>/dev/null; do sleep 20; done
if ! rg -q '"candidates"' "$LOG/train_natural_v1.log"; then
  echo "generation did not complete cleanly; aborting" >&2
  exit 1
fi

# 2. build paired records (drop catalog-verbatim rewrites)
$PY scripts/build_paired_sft_data.py \
  --natural data/counterfactual/train_natural_option_swap_v1.jsonl \
  --v2 data/counterfactual/train_constraint_causal_v2.jsonl \
  --exclude-verbatim \
  --out data/sft_paired_train.jsonl

# 3. concat baseline + paired into the mixed training file
cat data/sft_train.jsonl data/sft_paired_train.jsonl > data/sft_train_paired_mix.jsonl
wc -l data/sft_train_paired_mix.jsonl

# 4. launch training (writes its own log; pid recorded)
setsid nohup $PY -m experiment.sft.train --config configs/sft_paired.yaml \
  > "$LOG/sft_paired_train.log" 2>&1 < /dev/null &
echo $! > "$LOG/sft_paired_train.pid"
echo "training launched: pid $(cat $LOG/sft_paired_train.pid)"
