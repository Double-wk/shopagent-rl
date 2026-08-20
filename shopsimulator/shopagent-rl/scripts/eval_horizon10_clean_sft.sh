#!/usr/bin/env bash
# Run the two formal evaluations for horizon10-clean-v1 serially on one GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

ADAPTER="${ADAPTER:-$PWD/outputs/sft/v6_horizon10_clean_from_base/model/training_output/lora_adapter}"
STAMP="${STAMP:-$(date +%m%d_%H%M)}"
CF_OUT="${CF_OUT:-$PWD/outputs/sft/v6_horizon10_clean_from_base/evaluation/counterfactual_heldout_v2_${STAMP}.jsonl}"
FINAL_OUT="${FINAL_OUT:-$PWD/outputs/sft/v6_horizon10_clean_from_base/evaluation/final200_t10x512_${STAMP}.jsonl}"

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "clean SFT adapter is not ready: $ADAPTER" >&2
  exit 1
fi
mkdir -p "$(dirname "$CF_OUT")" "$(dirname "$FINAL_OUT")"

echo "[$(date -Is)] starting clean SFT counterfactual eval"
bash scripts/run_counterfactual_eval.sh \
  --pairs data/counterfactual/heldout_atomic_pairs_v2.jsonl \
  --tag SFT_HORIZON10_CLEAN_V1_HELDOUT_V2 \
  --adapter "$ADAPTER" \
  --out "$CF_OUT" \
  --max_tokens 160 \
  --temperature 0 \
  --gpu_memory_utilization 0.35 \
  --wave 64

echo "[$(date -Is)] starting clean SFT Final-200 eval"
bash scripts/run_eval.sh \
  --tag SFT_HORIZON10_CLEAN_V1 \
  --adapter "$ADAPTER" \
  --out "$FINAL_OUT" \
  --max_turns 10 \
  --max_tokens 512 \
  --wave 16

echo "[$(date -Is)] clean SFT evaluations complete"
echo "counterfactual: $CF_OUT"
echo "final200: $FINAL_OUT"
