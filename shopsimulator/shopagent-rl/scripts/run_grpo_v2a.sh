#!/usr/bin/env bash
# GRPO v2a: 50-step signal-quality validation, starting from SFT v1.
#
# This is deliberately not a continuation of GRPO v1.  It keeps the env16
# rollout budget but changes group structure from 4 tasks x 4 samples to
# 2 tasks x 8 samples, so outcome-GRPO receives within-task variation.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
VERSION_DIR="$ROOT/outputs/grpo/v2"
OVERLAY_DIR=/overlay/shopagent_rl_grpo_outputs/grpo/v2/diagnostic_50

mkdir -p "$VERSION_DIR/config" "$VERSION_DIR/logs" "$VERSION_DIR/model" "$VERSION_DIR/evaluation" "$OVERLAY_DIR"
cp "$ROOT/configs/grpo.yaml" "$VERSION_DIR/config/grpo_base.yaml"

exec env \
  RUN_NAME=grpo_v2a_signal50 \
  LOG_DIR="$VERSION_DIR/logs" \
  OUTPUT_DIR="$OVERLAY_DIR" \
  TOTAL_STEPS=50 \
  SAVE_FREQ=10 \
  TRAIN_BATCH=2 \
  ROLLOUT_N=8 \
  PPO_MINI_BATCH=2 \
  SHOP_ENV_MAX_NUM=16 \
  TEMPERATURE=0.85 \
  SHOPSIM_OBS_MAX_CHARS=1400 \
  SFT_ADAPTER="$ROOT/outputs/sft/v1/model/training_output/lora_adapter" \
  bash "$ROOT/scripts/run_grpo.sh"
