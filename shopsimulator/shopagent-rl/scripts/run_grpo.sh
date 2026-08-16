#!/usr/bin/env bash
# veRL multi-turn GRPO for the ShopSim agent — vLLM async backend, shop-A env.
# Continues from the SFT LoRA adapter (r=32/alpha=64/7-target-modules — the
# adapter's own config carries these; we only point at it).
#
# Single env: SFT + GRPO + eval all run in shop-A (transformers 4.57.6 after the
# 2026-08-08 downgrade). Sources the ROCm vLLM shims (amdsmi link + functorch
# shim) and runs from the verl repo root so the hydra searchpath in
# configs/grpo.yaml (file://verl/trainer/config) resolves.
#
# Usage:
#   bash scripts/run_grpo.sh                 # standard defaults: 50 steps
#   TOTAL_STEPS=50 N_GPUS=1 bash scripts/run_grpo.sh    # standard run
#   ROLLOUT_N=8 TRAIN_BATCH=8 bash ...                  # bigger GRPO groups
# Pre-req: pack_api ShopSim env must be serving on SHOP_ENV_BASE_URL (env16 pool).
set -euo pipefail

SHOP_A=/workspace/shopsimulator/shopagent-rl
VERL_ROOT="$SHOP_A"   # veRL 0.8.0 lives in shopagent-rl/verl (self-contained); cd here so hydra searchpath file://verl/trainer/config resolves
PY=/overlay/miniconda3/envs/shop-A/bin/python

# ---- persistent log + background detach (new session; survives terminal/SSH close) ----
# Log lives under run/ (NOT /tmp) so real runs persist. Override name: RUN_NAME=...
# Debug in foreground (log to terminal): FOREGROUND=1 bash run_grpo.sh
if [ -z "${RUN_NAME:-}" ]; then
    if [ "${TOTAL_STEPS:-50}" -le 50 ]; then RUN_NAME="grpo_smoke"
    else RUN_NAME="grpo_full_${TOTAL_STEPS}"; fi
fi
LOG_DIR="${LOG_DIR:-$SHOP_A/run}"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.log"; PIDF="$LOG_DIR/${RUN_NAME}.pid"
if [ "${FOREGROUND:-0}" != "1" ] && [ "${GRPO_DAEMONIZED:-0}" != "1" ]; then
    export GRPO_DAEMONIZED=1
    # A new session matters in managed terminals too: ordinary nohup can still be
    # reaped together with the caller's process group when that terminal closes.
    setsid nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"; disown
    echo "ShopSim GRPO detached — PID $(cat "$PIDF") (PPID=1)"
    echo "  log : tail -f $LOG"
    echo "  pid : $PIDF"
    echo "  stop: kill \$(cat $PIDF)"
    exit 0
fi

# 1) ROCm vLLM shims (amdsmi symlink + functorch shim), offline mode, spawn main-guard.
source "$SHOP_A/scripts/vllm_env_shopA.sh"

# 2) shopagent-rl must be importable from veRL's worker cwd (the verl repo root) so the
#    agent loop can `from shop_env... import ...`.
export PYTHONPATH="$SHOP_A:${PYTHONPATH:-}"

# 3) pack_api ShopSim env base URL (env16 pool the agent loop drives).
: "${SHOP_ENV_BASE_URL:=http://127.0.0.1:5000}"
export SHOP_ENV_BASE_URL

# 注：不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments —— vLLM memory pool 与之不兼容
# (multiproc_executor 断言 "not in conf"), 会直接让 vLLM 引擎初始化失败。靠 micro_batch 降配控显存。

cd "$VERL_ROOT"   # so hydra searchpath file://verl/trainer/config resolves

# ---- per-run knobs (override via env) ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
SFT_ADAPTER="${SFT_ADAPTER:-$SHOP_A/outputs/sft/v1/model/training_output/lora_adapter}"
TRAIN_FILES="${TRAIN_FILES:-$SHOP_A/data/grpo_train.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"        # standard run: 50 training steps
TRAIN_BATCH="${TRAIN_BATCH:-4}"         # distinct tasks per step; batch*rollout.n <= env16 pool
ROLLOUT_N="${ROLLOUT_N:-4}"             # GRPO group size -> batch*n rollouts/step (keep <= env16 pool)
SHOP_ENV_MAX_NUM="${SHOP_ENV_MAX_NUM:-16}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-$TRAIN_BATCH}" # task groups per actor update; must divide TRAIN_BATCH
LR="${LR:-1e-5}"
N_GPUS="${N_GPUS:-1}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
# vLLM reserves a KV-cache pool at startup. These defaults leave room for the
# LoRA actor and long multi-turn responses on the shared 48G card.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
# SFT assistant turns: P99=109 tokens, maximum=166.  160 retains 99.99% of
# them while acting only as a fallback; <|im_end|> is the normal turn boundary.
TURN_MAX_TOKENS="${SHOPSIM_TURN_MAX_TOKENS:-160}"
OBS_MAX_CHARS="${SHOPSIM_OBS_MAX_CHARS:-1800}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"          # vLLM loads the cached Base; sync only the LoRA adapter
OUTPUT_DIR="${OUTPUT_DIR:-/overlay/shopagent_rl_grpo_outputs/grpo}"
if (( TRAIN_BATCH % PPO_MINI_BATCH != 0 )); then
    echo "PPO_MINI_BATCH ($PPO_MINI_BATCH) must divide TRAIN_BATCH ($TRAIN_BATCH)" >&2
    exit 2
fi
if (( TRAIN_BATCH * ROLLOUT_N > SHOP_ENV_MAX_NUM )); then
    echo "TRAIN_BATCH*ROLLOUT_N ($((TRAIN_BATCH * ROLLOUT_N))) exceeds env pool ($SHOP_ENV_MAX_NUM)" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
export HYDRA_FULL_ERROR=1   # 训练期若再死,拿完整 stack
export SHOPSIM_TURN_MAX_TOKENS="$TURN_MAX_TOKENS"
export SHOPSIM_OBS_MAX_CHARS="$OBS_MAX_CHARS"
# On ROCm, auto selects POSIX shared memory for the per-step actor->vLLM weight
# transfer. Fresh HIP IPC buckets otherwise remain mapped until process exit.
: "${VERL_VLLM_WEIGHT_TRANSFER:=auto}"
export VERL_VLLM_WEIGHT_TRANSFER

# lora_rank>0 selects veRL's adapter path; keep it aligned with the SFT adapter.

echo "=== ShopSim GRPO (veRL vLLM async) ==="
echo "  model      : $MODEL_PATH"
echo "  sft adapter: $SFT_ADAPTER  (exists=$([ -e "$SFT_ADAPTER" ] && echo yes || echo NO))"
echo "  data       : $TRAIN_FILES  (exists=$([ -e "$TRAIN_FILES" ] && echo yes || echo NO))"
echo "  steps=$TOTAL_STEPS  train_batch=$TRAIN_BATCH  rollout.n=$ROLLOUT_N  (=$((TRAIN_BATCH*ROLLOUT_N)) rollouts/step)  lr=$LR  n_gpus=$N_GPUS"
echo "  lengths: prompt=$MAX_PROMPT_LENGTH turn=$TURN_MAX_TOKENS response=$RESPONSE_LENGTH model=$MAX_MODEL_LEN obs_chars=$OBS_MAX_CHARS"
echo "  sampling : temperature=$TEMPERATURE"
echo "  actor batches: ppo_mini=$PPO_MINI_BATCH ppo_micro=$PPO_MICRO_BATCH logprob_micro=$LOGPROB_MICRO_BATCH | vLLM gpu_mem_util=$GPU_MEM_UTIL"
echo "  vLLM load_format: $LOAD_FORMAT"
echo "  weight transfer : $VERL_VLLM_WEIGHT_TRANSFER"
echo "  output     : $OUTPUT_DIR"
echo "  env        : $SHOP_ENV_BASE_URL"

exec "$PY" -m verl.trainer.main_ppo \
    --config-path "$SHOP_A/configs" \
    --config-name grpo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TRAIN_FILES" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.load_format="$LOAD_FORMAT" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    "$@"
