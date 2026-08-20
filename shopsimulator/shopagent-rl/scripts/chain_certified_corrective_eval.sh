#!/usr/bin/env bash
# Wait for corrective SFT, then enforce the heldout price gate before Final-200.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
source "$ROOT/scripts/paths.sh"
PY="$SHOPAGENT_PY"
shopagent_require_py
TRAIN_LOG="$ROOT/run/certified_corrective_sft.log"
ADAPTER="$ROOT/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter"
CF_OUT="$ROOT/outputs/counterfactual/cf_sft_v4_certified_corrective_heldout_v2.jsonl"
CF_METRICS="${CF_OUT%.jsonl}_metrics.json"
EV_DIR="$ROOT/outputs/sft/v4_certified_corrective/evaluation"

cd "$ROOT"
mkdir -p "$EV_DIR"

while ! rg -q 'SFT adapter saved|Traceback|OutOfMemoryError' "$TRAIN_LOG"; do
    sleep 60
done

if ! rg -q 'SFT adapter saved' "$TRAIN_LOG" || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "Corrective SFT did not produce an adapter; see $TRAIN_LOG" >&2
    exit 1
fi

echo "=== heldout-v2 natural counterfactual gate ==="
bash scripts/run_counterfactual_eval.sh \
    --pairs data/counterfactual/heldout_atomic_pairs_v2.jsonl \
    --tag SFT_V4_CERTIFIED_CORRECTIVE_HELDOUT_V2 \
    --adapter "$ADAPTER" \
    --out "$CF_OUT" \
    --max_tokens 160 \
    --temperature 0 \
    --gpu_memory_utilization 0.35 \
    --wave 64

PRICE_ACC="$($PY - "$CF_METRICS" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
print(metrics["by_intervention_type"]["price_above_budget"]["counterfactual_action_accuracy"])
PY
)"
echo "heldout price_above_budget accuracy: $PRICE_ACC"

if ! "$PY" - "$PRICE_ACC" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.3 else 1)
PY
then
    echo "PRICE GATE FAILED (< 0.3); Final-200 and GRPO remain blocked."
    exit 2
fi

STAMP="$(date +%m%d_%H%M)"
FINAL_OUT="$EV_DIR/final200_t10x512_${STAMP}.jsonl"
FINAL_METRICS="${FINAL_OUT%.jsonl}_official_metrics.json"
echo "=== price gate passed; Final-200 (10 turns x 512) ==="
bash scripts/run_eval.sh \
    --tag SFT_V4_CERTIFIED_CORRECTIVE \
    --adapter "$ADAPTER" \
    --out "$FINAL_OUT" \
    --max_turns 10 \
    --max_tokens 512 \
    --wave 16

STRICT_RATE="$($PY - "$FINAL_METRICS" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))["metrics"]
print(metrics["平均分_r_success"])
PY
)"
echo "Final-200 strict success: $STRICT_RATE"

if ! "$PY" - "$STRICT_RATE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.16 else 1)
PY
then
    echo "FINAL GATE FAILED (< 0.16); Certified GRPO remains blocked."
    exit 3
fi

echo "Both corrective SFT gates passed. Certified GRPO is eligible for manual launch."
