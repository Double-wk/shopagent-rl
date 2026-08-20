#!/usr/bin/env bash
# Paired-C1-hard step-200 double eval: Final-200 + 212-pair counterfactual probe.
# 基座是 SFT v2 Paired adapter 起训的 GRPO（paired_c1hard_200_direct, b4/n4/lr1e-5/200 steps）。
# Run ONLY after the paired_c1hard_200_direct GRPO run has finished (the two vLLM
# engines and the trainer are mutually exclusive on the single 48G card) and
# the pack_api env pool is serving on SHOP_ENV_BASE_URL.
#
# Produces:
#   outputs/grpo/paired_c1_hard/evaluation/eval_grpo_pairedc1hard_s200_f200_t10x512_<stamp>.jsonl
#   outputs/counterfactual/cf_pairedc1hard_s200.jsonl (+ _metrics.json)
set -euo pipefail

cd "$(dirname "$0")/.."      # -> shopagent-rl/

ADAPTER=/overlay/shopagent_rl_artifacts/grpo_runs/paired_c1hard_200_direct/export_step_200/lora_adapter
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter not found: $ADAPTER" >&2
    echo "       training not finished yet, or checkpoint layout changed" >&2
    exit 1
fi

STAMP="$(date +%m%d_%H%M)"
EV=outputs/grpo/paired_c1_hard/evaluation
mkdir -p "$EV"

echo "=== [1/2] Final-200 eval (10 turns x 512 tokens, same protocol as c1_hard/v2b) ==="
bash scripts/run_eval.sh \
    --tag GRPO_PAIRED_C1HARD \
    --adapter "$ADAPTER" \
    --out "$EV/eval_grpo_pairedc1hard_s200_f200_t10x512_${STAMP}.jsonl" \
    --max_turns 10 \
    --max_tokens 512 \
    --wave 16 \
    >"$EV/eval_grpo_pairedc1hard_s200_f200_t10x512_${STAMP}.log" 2>&1
echo "Final-200 done: $EV/eval_grpo_pairedc1hard_s200_f200_t10x512_${STAMP}.jsonl"

# The engine holds ~all of VRAM; let it go before the next one starts.
sleep 20

echo "=== [2/2] counterfactual probe (212 atomic pairs) ==="
bash scripts/run_counterfactual_eval.sh \
    --tag Paired_C1_hard \
    --adapter "$ADAPTER" \
    --out outputs/counterfactual/cf_pairedc1hard_s200.jsonl

echo "=== both evals complete ==="
