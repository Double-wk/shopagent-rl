#!/usr/bin/env bash
# GRPO v2b: formal 200-step run after v2a validated the signal path.
# Four independent tasks per step, eight rollouts per task, and a 32-slot
# ShopSimulator pool.  Start from SFT v1, never from the degraded GRPO v1.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
VERSION_DIR="$ROOT/outputs/grpo/v2"
RUN_NAME=grpo_v2b_200_b4n8_env32
OVERLAY_DIR=/overlay/shopagent_rl_grpo_outputs/grpo/v2/full_200_b4_n8_env32

mkdir -p "$VERSION_DIR/config" "$VERSION_DIR/logs" "$VERSION_DIR/model" "$VERSION_DIR/evaluation" "$OVERLAY_DIR"
cp "$ROOT/configs/grpo.yaml" "$VERSION_DIR/config/grpo_v2b_200_b4n8_env32_base.yaml"

exec env \
  RUN_NAME="$RUN_NAME" \
  LOG_DIR="$VERSION_DIR/logs" \
  OUTPUT_DIR="$OVERLAY_DIR" \
  TOTAL_STEPS=200 \
  SAVE_FREQ=10 \
  TRAIN_BATCH=4 \
  ROLLOUT_N=8 \
  PPO_MINI_BATCH=4 \
  SHOP_ENV_MAX_NUM=32 \
  TEMPERATURE=0.85 \
  SHOPSIM_OBS_MAX_CHARS=1200 \
  SFT_ADAPTER="$ROOT/outputs/sft/v1/model/training_output/lora_adapter" \
  bash "$ROOT/scripts/run_grpo.sh"
