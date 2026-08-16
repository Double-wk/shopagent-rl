#!/usr/bin/env bash
# D1 + D2 diagnostics (docs/price-blindness-next.md §2). GPU stages run
# sequentially on the single card; D3 is CPU-only and runs inline first.
#
# D1: arithmetic-vs-shopping comparison probe, 4 checkpoints
# D2: cf-probe prompt variants salience / instruct on GRPO C1-hard
#
# Outputs:
#   outputs/price_blindness/d1_<tag>.jsonl (+ _metrics.json)
#   outputs/price_blindness/cf_c1hard_<variant>.jsonl (+ _metrics.json)
#   outputs/price_blindness/d3_hard_zero_audit.json
set -euo pipefail

cd "$(dirname "$0")/.."      # -> shopagent-rl/

source scripts/vllm_env_shopA.sh   # ONE-env: sets PY

OUT=outputs/price_blindness
mkdir -p "$OUT"

C1HARD=/overlay/shopagent_rl_grpo_outputs/grpo/c1_hard/full_200_b4_n8_env32/checkpoint_step_200/lora_adapter
V2B=/overlay/shopagent_rl_grpo_outputs/grpo/v2/export_step_200/lora_adapter
SFT=outputs/sft/v1/model/training_output/lora_adapter

echo "=== [D3] hard-zero trigger audit (CPU) ==="
"$PY" -m experiment.eval.audit_hard_zero_rate \
    outputs/grpo/c1_hard/logs/grpo_c1hard_200_b4n8_env32.log \
    > "$OUT/d3_hard_zero_audit.json"
"$PY" -c "import json;d=json.load(open('$OUT/d3_hard_zero_audit.json'));print({k:d[k] for k in ('rollouts','zero_reward','hard_zero_buys','hard_zero_rate_per_rollout','hard_zero_rate_per_buy')})"

echo "=== [D1] comparison probe: Base ==="
"$PY" -m experiment.eval.run_comparison_probe --tag Base \
    --out "$OUT/d1_base.jsonl" >"$OUT/d1_base.stdout.log" 2>&1
tail -3 "$OUT/d1_base.stdout.log"

for NAME in SFT V2B C1HARD; do
    case $NAME in
        SFT)    AD=$SFT ;;
        V2B)    AD=$V2B ;;
        C1HARD) AD=$C1HARD ;;
    esac
    echo "=== [D1] comparison probe: $NAME ==="
    "$PY" -m experiment.eval.run_comparison_probe --tag "$NAME" \
        --adapter "$AD" \
        --out "$OUT/d1_$(echo "$NAME" | tr 'A-Z' 'a-z').jsonl" \
        >"$OUT/d1_$(echo "$NAME" | tr 'A-Z' 'a-z').stdout.log" 2>&1
    tail -3 "$OUT/d1_$(echo "$NAME" | tr 'A-Z' 'a-z').stdout.log"
done

echo "=== [D2] cf-probe variants on C1-hard ==="
for VARIANT in salience instruct; do
    echo "--- variant: $VARIANT ---"
    bash scripts/run_counterfactual_eval.sh \
        --tag "C1_hard_$VARIANT" \
        --adapter "$C1HARD" \
        --variant "$VARIANT" \
        --out "$OUT/cf_c1hard_${VARIANT}.jsonl" \
        >"$OUT/cf_c1hard_${VARIANT}.stdout.log" 2>&1
    "$PY" -c "
import json; m=json.load(open('$OUT/cf_c1hard_${VARIANT}_metrics.json'))
p=m['by_intervention_type']['price_above_budget']
print('price cf acc:', p['counterfactual_action_accuracy'], '| commit err:', p['commit_persistence_error'])"
done

echo "=== D1-D3 complete: $OUT ==="
