#!/usr/bin/env bash
# Gate C: matched 200-step paper-v2 run, one arm per invocation.
#
# Differences from the 10-step smoke (scripts/run_paper_grpo_smoke.sh) that
# matter for a result, not just for length:
#
#   * paper-v2 data, whose training pairs are restricted to the three training
#     mechanisms so that option_unavailable / option_price_over_budget stay
#     held out (experiment/counterfactual_pairs.py: TRAIN_MECHANISMS).
#   * relation_coeff=0.002 rather than 1.0.  Measured on a 6-step probe: at
#     coeff=1.0 pair-step grad_norm was 90-243 against ~0.4 on environment
#     steps, so grad_clip=1.0 attenuated PPO's own gradient ~200x and pair steps
#     became pure relation descent.  At 0.002 pair steps sit at 0.70-1.20 vs
#     0.42-0.52, and margin/flip diagnostics are unchanged.
#   * resume_mode=disable.  veRL defaults to `auto`
#     (verl/trainer/config/ppo_trainer.yaml:162) and will silently continue from
#     a same-named output directory; that produced a bogus "step:11" baseline
#     once already.
#   * SEED is recorded in the run name but is NOT a rollout seed.  RolloutConfig
#     has no `seed` field, so vllm_async_server.py:271's
#     `self.config.get("seed") or 0` is always 0 and vLLM sampling is already
#     fixed for a single replica -- there is no rollout-seed knob to turn.  The
#     supported lever is `data.seed` (verl/utils/dataset/rl_dataset.py:154),
#     which reorders the training stream.  Gate C's single seed keeps
#     data.shuffle=False so the step-by-step comparison lines up with the
#     10-step smoke; the multi-seed phase should vary data.seed with
#     data.shuffle=True.  That is safe for pairing precisely because the
#     counterfactual side is synthesized from partner_state_text carried on the
#     row itself, so the two sides never need to co-occur in a batch.
#
# Does not select on or evaluate against the sealed test set.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
# shellcheck source=scripts/paths.sh
source "$ROOT/scripts/paths.sh"
shopagent_require_py

METHOD="${METHOD:-independent}"
case "$METHOD" in
  independent)       PAIRED_OBJECTIVE=False; RELATION_MODE=preference_margin ;;
  preference_margin) PAIRED_OBJECTIVE=True;  RELATION_MODE=preference_margin ;;
  *) echo "Gate C compares independent vs preference_margin only" >&2; exit 2 ;;
esac

INIT="${INIT:-clean}"
case "$INIT" in
  clean)     INIT_ADAPTER="$ROOT/outputs/sft/v6_horizon10_clean_from_base/model/training_output/lora_adapter" ;;
  certified) INIT_ADAPTER="$ROOT/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter" ;;
  *) echo "INIT must be clean or certified" >&2; exit 2 ;;
esac

DATA="$ROOT/data/grpo_certified_paper_v2_800_pairblocked.parquet"
EXPECTED_ROWS=800
EXPECTED_SHA=a1a0455f40ed4a4289e9c69b7c3cf764d91e98c0c9f46b1d7b616de9d53c502b
rows=$("$SHOPAGENT_PY" -c \
  'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' "$DATA")
sha=$(sha256sum "$DATA" | cut -d' ' -f1)
if [ "$rows" -ne "$EXPECTED_ROWS" ] || [ "$sha" != "$EXPECTED_SHA" ]; then
  echo "paper-v2 GRPO data mismatch: rows=$rows sha256=$sha" >&2
  exit 2
fi
if [ ! -f "$INIT_ADAPTER/adapter_model.safetensors" ]; then
  echo "missing initialization adapter: $INIT_ADAPTER" >&2
  exit 2
fi

SEED="${SEED:-0}"
RELATION_COEFF="${RELATION_COEFF:-0.002}"

BASE_SNAPSHOT=/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
export MODEL_PATH="${MODEL_PATH:-$BASE_SNAPSHOT}"
export SFT_ADAPTER="${SFT_ADAPTER:-$INIT_ADAPTER}"
export TRAIN_FILES="${TRAIN_FILES:-$DATA}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SHOPAGENT_GRPO_ARTIFACT_ROOT/paper_v2_gate_c_${INIT}_${METHOD}_s${SEED}}"
export RUN_NAME="${RUN_NAME:-paper_v2_gate_c_${INIT}_${METHOD}_s${SEED}}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export TRAIN_BATCH="${TRAIN_BATCH:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-4}"
export DATA_SHUFFLE=False
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export SHOPSIM_REWARD_BUDGET_MODE="${SHOPSIM_REWARD_BUDGET_MODE:-hard}"
export SHOPSIM_CERTIFIED_REWARD_WEIGHT="${SHOPSIM_CERTIFIED_REWARD_WEIGHT:-1.0}"
export SHOPSIM_CF_LENIENT_REWARD="${SHOPSIM_CF_LENIENT_REWARD:-0.5}"

echo "[gate_c] method=$METHOD init=$INIT seed=$SEED steps=$TOTAL_STEPS"
echo "[gate_c] relation_coeff=$RELATION_COEFF (applied only when paired=$PAIRED_OBJECTIVE)"
echo "[gate_c] data=$DATA sha256=$sha rows=$rows"
echo "[gate_c] output=$OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ] && compgen -G "$OUTPUT_DIR/global_step_*" >/dev/null; then
  echo "[gate_c] refusing to start: $OUTPUT_DIR already holds checkpoints." >&2
  echo "[gate_c] resume_mode is disabled, so training would restart from step 1" >&2
  echo "[gate_c] and overwrite them. Move the directory aside first." >&2
  exit 2
fi

exec bash "$ROOT/scripts/run_grpo.sh" \
  trainer.balance_batch=False \
  trainer.resume_mode=disable \
  data.seed="$SEED" \
  algorithm.paired_intervention.enabled="$PAIRED_OBJECTIVE" \
  algorithm.paired_intervention.mode="$RELATION_MODE" \
  algorithm.paired_intervention.weight=1.0 \
  algorithm.paired_intervention.relation_coeff="$RELATION_COEFF" \
  "$@"
