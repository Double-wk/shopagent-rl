#!/usr/bin/env bash
# Continue the formal horizon10-clean-v1 experiment serially on one GPU.
set -uo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
# shellcheck source=scripts/paths.sh
source "$ROOT/scripts/paths.sh"
PY="$SHOPAGENT_PY"
RUN_DIR="$ROOT/run"
INDEPENDENT_ROOT="$SHOPAGENT_GRPO_ARTIFACT_ROOT/horizon10_clean_v1_independent"
PAIRED_ROOT="$SHOPAGENT_GRPO_ARTIFACT_ROOT/horizon10_clean_v1_paired"
INDEPENDENT_LOG="$RUN_DIR/horizon10_clean_v1_grpo_independent.log"
PAIRED_LOG="$RUN_DIR/horizon10_clean_v1_grpo_paired.log"
STATUS_LOG="$RUN_DIR/horizon10_clean_v1_night_chain.log"
INDEPENDENT_PID_FILE="$RUN_DIR/horizon10_clean_v1_grpo_independent.pid"
LOCK_FILE="$RUN_DIR/horizon10_clean_v1_night_chain.lock"

mkdir -p "$RUN_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date -Is)] another horizon10 night chain is already running" >&2
    exit 2
fi

log() {
    echo "[$(date -Is)] $*" | tee -a "$STATUS_LOG"
}

fail() {
    log "ERROR: $*"
    exit 1
}

wait_for_pid() {
    local pid="$1" label="$2"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "$label process $pid exited"
}

wait_for_gpu_idle() {
    local used threshold=$((4 * 1024 * 1024 * 1024)) attempts=0
    while (( attempts < 60 )); do
        used=$(rocm-smi --showmeminfo vram 2>/dev/null |
            awk '/VRAM Total Used Memory \(B\)/ {print $NF; exit}')
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < threshold )); then
            log "GPU is idle enough for the next phase (used_bytes=$used)"
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 30
    done
    return 1
}

require_checkpoint() {
    local root="$1" label="$2"
    local actor="$root/global_step_200/actor"
    [[ -f "$actor/model_world_size_1_rank_0.pt" ]] ||
        fail "$label did not produce global_step_200 actor checkpoint"
    [[ -f "$actor/lora_train_meta.json" ]] ||
        fail "$label global_step_200 checkpoint is incomplete"
}

export_adapter() {
    local root="$1" label="$2"
    local actor="$root/global_step_200/actor"
    local adapter="$root/export_step_200/lora_adapter"
    if [[ -f "$adapter/adapter_model.safetensors" ]]; then
        log "$label adapter already exported: $adapter"
        return 0
    fi
    log "exporting $label step-200 adapter"
    "$PY" "$ROOT/scripts/export_lora_adapter.py" --ckpt "$actor" --out "$adapter" \
        >>"$STATUS_LOG" 2>&1 || return 1
    log "$label adapter exported: $adapter"
}

run_evaluations() {
    local method="$1" root="$2"
    local adapter="$root/export_step_200/lora_adapter"
    local out_dir="$ROOT/outputs/grpo/horizon10_clean_v1/$method/evaluation"
    local cf_out="$out_dir/counterfactual_heldout_v2.jsonl"
    local final_out="$out_dir/final200_t10x512.jsonl"
    mkdir -p "$out_dir"

    log "starting $method heldout-v2 counterfactual evaluation"
    bash "$ROOT/scripts/run_counterfactual_eval.sh" \
        --pairs "$ROOT/data/counterfactual/heldout_atomic_pairs_v2.jsonl" \
        --tag "GRPO_HORIZON10_CLEAN_V1_${method^^}_HELDOUT_V2" \
        --adapter "$adapter" \
        --out "$cf_out" \
        --max_tokens 160 \
        --temperature 0 \
        --gpu_memory_utilization 0.35 \
        --wave 64 \
        >>"$STATUS_LOG" 2>&1 || return 1

    wait_for_gpu_idle || return 1
    log "starting $method Final-200 evaluation"
    bash "$ROOT/scripts/run_eval.sh" \
        --tag "GRPO_HORIZON10_CLEAN_V1_${method^^}" \
        --adapter "$adapter" \
        --out "$final_out" \
        --max_turns 10 \
        --max_tokens 512 \
        --wave 16 \
        >>"$STATUS_LOG" 2>&1 || return 1
    log "$method evaluations complete"
}

cd "$ROOT"
: >"$STATUS_LOG"
log "night chain started"

if [[ -f "$INDEPENDENT_PID_FILE" ]]; then
    independent_pid=$(<"$INDEPENDENT_PID_FILE")
elif [[ -n "${INDEPENDENT_PID:-}" ]]; then
    independent_pid="$INDEPENDENT_PID"
else
    fail "Independent PID file is missing"
fi
[[ "$independent_pid" =~ ^[0-9]+$ ]] || fail "invalid Independent PID: $independent_pid"

if kill -0 "$independent_pid" 2>/dev/null; then
    log "waiting for Independent training process $independent_pid"
    wait_for_pid "$independent_pid" "Independent training"
else
    log "Independent process $independent_pid has already exited; checking checkpoint"
fi

require_checkpoint "$INDEPENDENT_ROOT" "Independent training"
wait_for_gpu_idle || fail "GPU did not become idle after Independent training"
export_adapter "$INDEPENDENT_ROOT" "Independent" || fail "Independent adapter export failed"
run_evaluations independent "$INDEPENDENT_ROOT" ||
    fail "Independent evaluation failed; Paired training was not started"

wait_for_gpu_idle || fail "GPU did not become idle before Paired training"
log "starting formal Paired GRPO"
FOREGROUND=1 METHOD=paired \
    bash "$ROOT/scripts/run_horizon10_clean_grpo.sh" >"$PAIRED_LOG" 2>&1
paired_rc=$?
log "Paired GRPO exited with status $paired_rc"
(( paired_rc == 0 )) || fail "Paired GRPO failed; see $PAIRED_LOG"

require_checkpoint "$PAIRED_ROOT" "Paired training"
wait_for_gpu_idle || fail "GPU did not become idle after Paired training"
export_adapter "$PAIRED_ROOT" "Paired" || fail "Paired adapter export failed"
run_evaluations paired "$PAIRED_ROOT" || fail "Paired evaluation failed"

log "night chain complete: Independent and Paired training/evaluations finished"
touch "$RUN_DIR/horizon10_clean_v1_night_chain.complete"
