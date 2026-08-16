#!/usr/bin/env bash
# Serial SFT -> GRPO Final-200 eval. Single 48G card: the two vLLM engines are
# mutually exclusive, so they MUST run one after the other, never concurrently.
#
# Horizon: 10 turns / 512 max_tokens. The two evaluated models use the same
# protocol; historical reports are archived and should not be compared directly.
# Re-run Base at 512 if a three-way comparison is needed.
set -uo pipefail

cd /workspace/shopsimulator/shopagent-rl

STAMP="$(date +%m%d_%H%M)"
mkdir -p outputs/sft/v1/evaluation outputs/sft/v1/logs \
         outputs/grpo/v1/evaluation outputs/grpo/v1/logs

run_one() {
    local tag="$1" adapter="$2" out="$3" log="$4"
    echo "=================================================================="
    echo "[$(date +%H:%M:%S)] START $tag  adapter=$adapter"
    echo "=================================================================="
    bash scripts/run_eval.sh \
        --tag "$tag" \
        --adapter "$adapter" \
        --out "$out" \
        --max_turns 10 \
        --max_tokens 512 \
        --wave 16 \
        >"$log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] $tag exit=$rc  lines=$(wc -l <"$out" 2>/dev/null || echo 0)"

    # The engine holds ~all of VRAM; make sure it is gone before the next tag.
    sleep 20
    echo "[$(date +%H:%M:%S)] VRAM after $tag: $(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'Used Memory \(B\): \K[0-9]+')"
    return $rc
}

run_one SFT  outputs/sft/v1/model/training_output/lora_adapter \
             outputs/sft/v1/evaluation/eval_sft3793_f200_t10x512_${STAMP}.jsonl \
             outputs/sft/v1/logs/eval_f200_t10x512_${STAMP}.log
sft_rc=$?

run_one GRPO outputs/grpo/v1/model/checkpoint_step_200/lora_adapter \
             outputs/grpo/v1/evaluation/eval_grpo_env16_s200_f200_t10x512_${STAMP}.jsonl \
             outputs/grpo/v1/logs/eval_f200_t10x512_${STAMP}.log
grpo_rc=$?

echo "=================================================================="
echo "[$(date +%H:%M:%S)] DONE  sft_rc=$sft_rc  grpo_rc=$grpo_rc  stamp=$STAMP"
for f in outputs/sft/v1/evaluation/eval_sft3793_f200_t10x512_${STAMP}_official_metrics.json \
         outputs/grpo/v1/evaluation/eval_grpo_env16_s200_f200_t10x512_${STAMP}_official_metrics.json; do
    echo "--- $f"; cat "$f" 2>/dev/null || echo "  (missing)"
done
