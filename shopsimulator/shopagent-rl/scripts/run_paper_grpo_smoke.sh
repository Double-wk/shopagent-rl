#!/usr/bin/env bash
# Matched 10-step paper-v1 smoke. Does not select or evaluate on the sealed test.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
# shellcheck source=scripts/paths.sh
source "$ROOT/scripts/paths.sh"
shopagent_require_py

METHOD="${METHOD:-independent}"
case "$METHOD" in
  independent) PAIRED_OBJECTIVE=False; RELATION_MODE=explicit_relation ;;
  explicit_relation) PAIRED_OBJECTIVE=True; RELATION_MODE=explicit_relation ;;
  residual) PAIRED_OBJECTIVE=True; RELATION_MODE=relational_residual ;;
  legacy_joint) PAIRED_OBJECTIVE=True; RELATION_MODE=joint_bonus ;;
  preference_margin) PAIRED_OBJECTIVE=True; RELATION_MODE=preference_margin ;;
  *) echo "METHOD must be independent, explicit_relation, residual, legacy_joint, or preference_margin" >&2; exit 2 ;;
esac

INIT="${INIT:-clean}"
case "$INIT" in
  clean) INIT_ADAPTER="$ROOT/outputs/sft/v6_horizon10_clean_from_base/model/training_output/lora_adapter" ;;
  certified) INIT_ADAPTER="$ROOT/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter" ;;
  *) echo "INIT must be clean or certified" >&2; exit 2 ;;
esac

DATA="$ROOT/data/grpo_certified_paper_v1_800_pairblocked.parquet"
EXPECTED_ROWS=800
EXPECTED_SHA=532f71e0c43ff0603beaf130fdeed48ddeb382e2604ce80eb37f753291e534a4
rows=$("$SHOPAGENT_PY" -c \
  'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' "$DATA")
sha=$(sha256sum "$DATA" | cut -d' ' -f1)
if [ "$rows" -ne "$EXPECTED_ROWS" ] || [ "$sha" != "$EXPECTED_SHA" ]; then
  echo "paper-v1 GRPO data mismatch: rows=$rows sha256=$sha" >&2
  exit 2
fi
if [ ! -f "$INIT_ADAPTER/adapter_model.safetensors" ]; then
  echo "missing initialization adapter: $INIT_ADAPTER" >&2
  exit 2
fi

BASE_SNAPSHOT=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
export MODEL_PATH="${MODEL_PATH:-$BASE_SNAPSHOT}"
export SFT_ADAPTER="${SFT_ADAPTER:-$INIT_ADAPTER}"
export TRAIN_FILES="${TRAIN_FILES:-$DATA}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SHOPAGENT_GRPO_ARTIFACT_ROOT/paper_v1_smoke_${INIT}_${METHOD}}"
export RUN_NAME="${RUN_NAME:-paper_v1_smoke_${INIT}_${METHOD}}"
export TOTAL_STEPS="${TOTAL_STEPS:-10}"
export TRAIN_BATCH="${TRAIN_BATCH:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-4}"
export DATA_SHUFFLE=False
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export SHOPSIM_REWARD_BUDGET_MODE="${SHOPSIM_REWARD_BUDGET_MODE:-hard}"
export SHOPSIM_CERTIFIED_REWARD_WEIGHT="${SHOPSIM_CERTIFIED_REWARD_WEIGHT:-1.0}"
export SHOPSIM_CF_LENIENT_REWARD="${SHOPSIM_CF_LENIENT_REWARD:-0.5}"

exec bash "$ROOT/scripts/run_grpo.sh" \
  trainer.balance_batch=False \
  algorithm.paired_intervention.enabled="$PAIRED_OBJECTIVE" \
  algorithm.paired_intervention.mode="$RELATION_MODE" \
  algorithm.paired_intervention.weight="${PAIRED_REWARD_WEIGHT:-1.0}" \
  "$@"
