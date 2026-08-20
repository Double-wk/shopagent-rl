#!/usr/bin/env bash
# Matched natural-format Certified GRPO initialized from corrective SFT v4.
#
# METHOD=independent: environment reward + per-side certified action reward.
# METHOD=paired:      identical setup plus the joint pair-relation bonus.
#
# The pair-blocked parquet and disabled data/batch shuffling keep both sides of
# each intervention pair in the same training batch. The method comparison must
# differ only in algorithm.paired_intervention.enabled.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
METHOD="${METHOD:-independent}"
case "$METHOD" in
  independent) PAIRED_OBJECTIVE=False ;;
  paired) PAIRED_OBJECTIVE=True ;;
  *) echo "METHOD must be independent or paired" >&2; exit 2 ;;
esac

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/grpo_certified_natural_800_pairblocked.parquet}"
export SFT_ADAPTER="${SFT_ADAPTER:-$ROOT/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/artifacts/grpo_runs/certified_natural_$METHOD}"
export RUN_NAME="${RUN_NAME:-certified_natural_grpo_$METHOD}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export TRAIN_BATCH="${TRAIN_BATCH:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-4}"
export LR="${LR:-1e-5}"
export DATA_SHUFFLE=False
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export SHOPSIM_REWARD_BUDGET_MODE="${SHOPSIM_REWARD_BUDGET_MODE:-hard}"
export SHOPSIM_CERTIFIED_REWARD_WEIGHT="${SHOPSIM_CERTIFIED_REWARD_WEIGHT:-1.0}"
export SHOPSIM_CF_LENIENT_REWARD="${SHOPSIM_CF_LENIENT_REWARD:-0.5}"

exec bash "$ROOT/scripts/run_grpo.sh" \
  trainer.balance_batch=False \
  algorithm.paired_intervention.enabled="$PAIRED_OBJECTIVE" \
  algorithm.paired_intervention.weight="${PAIRED_REWARD_WEIGHT:-1.0}" \
  "$@"
