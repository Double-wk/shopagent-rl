#!/usr/bin/env bash
# GRPO C1-hard: the v2b recipe bit-exact except the terminal reward gets a
# budget gate (SHOPSIM_REWARD_BUDGET_MODE=hard) -- a purchase that breaks the
# stated budget (r_price == 0) scores 0 instead of keeping up to 0.8 partial
# credit. Motivated by the counterfactual probe (outputs/counterfactual,
# 2026-08-14: commit_persistence_error SFT 0.085 -> v1 0.203 -> v2b 0.519).
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
VERSION_DIR="$ROOT/outputs/grpo/c1_hard"
RUN_NAME=grpo_c1hard_200_b4n8_env32
OVERLAY_DIR=/overlay/shopagent_rl_artifacts/grpo_runs/c1_hard/full_200_b4_n8_env32

mkdir -p "$VERSION_DIR/config" "$VERSION_DIR/logs" "$VERSION_DIR/model" "$VERSION_DIR/evaluation" "$OVERLAY_DIR"
cp "$ROOT/configs/grpo.yaml" "$VERSION_DIR/config/grpo_c1hard_200_b4n8_env32_base.yaml"

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
  SHOPSIM_REWARD_BUDGET_MODE=hard \
  SFT_ADAPTER="$ROOT/outputs/sft/v1/model/training_output/lora_adapter" \
  bash "$ROOT/scripts/run_grpo.sh"
