#!/usr/bin/env bash
# Formal matched GRPO from the horizon10-clean-v1 SFT checkpoint.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
BASE_SNAPSHOT=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
METHOD="${METHOD:-independent}"
case "$METHOD" in
  independent) PAIRED_OBJECTIVE=False ;;
  paired) PAIRED_OBJECTIVE=True ;;
  *) echo "METHOD must be independent or paired" >&2; exit 2 ;;
esac

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/grpo_certified_natural_800_pairblocked.parquet}"
export SFT_ADAPTER="${SFT_ADAPTER:-$ROOT/outputs/sft/v6_horizon10_clean_from_base/model/training_output/lora_adapter}"
export MODEL_PATH="${MODEL_PATH:-$BASE_SNAPSHOT}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "pinned Qwen3-1.7B-Base snapshot is not available: $MODEL_PATH" >&2
    exit 2
fi
if [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
    echo "clean SFT adapter is not ready: $SFT_ADAPTER" >&2
    exit 2
fi
EXPECTED_DATA="$ROOT/data/grpo_certified_natural_800_pairblocked.parquet"
EXPECTED_ROWS=800
EXPECTED_SHA=9536e6605afb155db2335c432a8eb86cb1ed93657178c042181a5f5f2155b266
if [ "$TRAIN_FILES" = "$EXPECTED_DATA" ]; then
    rows=$(/overlay/miniconda3/envs/shopsim/bin/python -c \
        'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' \
        "$TRAIN_FILES")
    sha=$(sha256sum "$TRAIN_FILES" | cut -d' ' -f1)
    if [ "$rows" -ne "$EXPECTED_ROWS" ] || [ "$sha" != "$EXPECTED_SHA" ]; then
        echo "horizon10 GRPO data mismatch: rows=$rows sha256=$sha" >&2
        exit 2
    fi
fi
export OUTPUT_DIR="${OUTPUT_DIR:-/overlay/shopagent_rl_grpo_outputs/grpo/horizon10_clean_v1_$METHOD}"
export RUN_NAME="${RUN_NAME:-horizon10_clean_v1_grpo_$METHOD}"
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
