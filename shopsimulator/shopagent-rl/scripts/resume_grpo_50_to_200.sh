#!/usr/bin/env bash
# Wait for the active 40->50 recovery run, validate its final checkpoint, then
# continue to step 200 with the exact same GRPO hyperparameters.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
CURRENT_PID_FILE="$ROOT/run/grpo_env16_0812_rocmfix_resume40.pid"
CHECKPOINT=/overlay/shopagent_rl_artifacts/grpo_runs/global_step_50

if [ ! -s "$CURRENT_PID_FILE" ]; then
    echo "missing current PID file: $CURRENT_PID_FILE" >&2
    exit 1
fi
current_pid=$(sed -n '1p' "$CURRENT_PID_FILE")
echo "waiting for recovery run PID $current_pid to finish"
while kill -0 "$current_pid" 2>/dev/null; do
    sleep 5
done

# A successful FSDP checkpoint must contain all training state, not merely the
# Hydra outputs directory or a partially created global_step_50 directory.
required=(
    actor/model_world_size_1_rank_0.pt
    actor/optim_world_size_1_rank_0.pt
    actor/extra_state_world_size_1_rank_0.pt
    actor/fsdp_config.json
    actor/lora_train_meta.json
    data.pt
)
for rel in "${required[@]}"; do
    if [ ! -s "$CHECKPOINT/$rel" ]; then
        echo "refusing to resume: incomplete checkpoint, missing $CHECKPOINT/$rel" >&2
        exit 2
    fi
done

echo "validated complete checkpoint: $CHECKPOINT"
exec env \
    FOREGROUND=1 \
    RUN_NAME=grpo_env16_0812_rocmfix_resume50_to200 \
    TOTAL_STEPS=200 \
    TRAIN_BATCH=4 \
    ROLLOUT_N=4 \
    PPO_MINI_BATCH=4 \
    PPO_MICRO_BATCH=1 \
    LOGPROB_MICRO_BATCH=1 \
    GPU_MEM_UTIL=0.25 \
    MAX_PROMPT_LENGTH=512 \
    RESPONSE_LENGTH=8192 \
    MAX_MODEL_LEN=10240 \
    SHOPSIM_TURN_MAX_TOKENS=160 \
    SHOPSIM_OBS_MAX_CHARS=1800 \
    VERL_VLLM_WEIGHT_TRANSFER=auto \
    LR=1e-5 \
    SAVE_FREQ=10 \
    bash "$ROOT/scripts/run_grpo.sh" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$CHECKPOINT"
