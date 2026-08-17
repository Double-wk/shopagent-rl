#!/usr/bin/env bash
# Mixed environment + atomic counterfactual GRPO, initialized from certified SFT.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/grpo_certified_natural_train.parquet}"
export SFT_ADAPTER="${SFT_ADAPTER:-$ROOT/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter}"
export OUTPUT_DIR="${OUTPUT_DIR:-/overlay/shopagent_rl_grpo_outputs/grpo/certified_single_seed}"
export RUN_NAME="${RUN_NAME:-certified_grpo}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export SHOPSIM_REWARD_BUDGET_MODE="${SHOPSIM_REWARD_BUDGET_MODE:-hard}"
export SHOPSIM_CERTIFIED_REWARD_WEIGHT="${SHOPSIM_CERTIFIED_REWARD_WEIGHT:-1.0}"
export SHOPSIM_CF_LENIENT_REWARD="${SHOPSIM_CF_LENIENT_REWARD:-0.5}"

exec bash "$ROOT/scripts/run_grpo.sh" "$@"
