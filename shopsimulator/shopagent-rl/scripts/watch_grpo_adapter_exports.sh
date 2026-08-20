#!/usr/bin/env bash
# Persist selected LoRA adapters while a GRPO run keeps raw checkpoints on /overlay.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
# shellcheck source=scripts/paths.sh
source "$ROOT/scripts/paths.sh"
shopagent_require_py

RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the GRPO checkpoint directory}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT under tracked outputs/}
TRAIN_PID_FILE=${TRAIN_PID_FILE:?set TRAIN_PID_FILE}
MILESTONES=${MILESTONES:-"50 100 150 200"}
POLL_SECONDS=${POLL_SECONDS:-30}
LOG=${LOG:-$ROOT/run/grpo_adapter_export_watch.log}
LOCK=${LOCK:-$ROOT/run/grpo_adapter_export_watch.lock}

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "adapter export watcher is already running" >&2; exit 2; }

log() {
    echo "[$(date -Is)] $*" | tee -a "$LOG"
}

checkpoint_ready() {
    local actor=$1 size_a size_b
    [[ -f "$actor/model_world_size_1_rank_0.pt" ]] || return 1
    [[ -f "$actor/lora_train_meta.json" ]] || return 1
    [[ -f "$actor/extra_state_world_size_1_rank_0.pt" ]] || return 1
    size_a=$(stat -c %s "$actor/model_world_size_1_rank_0.pt")
    sleep 5
    size_b=$(stat -c %s "$actor/model_world_size_1_rank_0.pt")
    [[ "$size_a" = "$size_b" && "$size_a" -gt 1000000000 ]]
}

persist_step() {
    local step=$1
    local actor="$RUN_ROOT/global_step_$step/actor"
    local overlay_adapter="$RUN_ROOT/export_step_$step/lora_adapter"
    local persistent="$OUTPUT_ROOT/model/checkpoint_step_$step/lora_adapter"
    local marker="$persistent/.export_complete"

    [[ ! -f "$marker" ]] || return 0
    checkpoint_ready "$actor" || return 1

    log "exporting step $step LoRA from $actor"
    mkdir -p "$overlay_adapter" "$persistent"
    "$SHOPAGENT_PY" "$ROOT/scripts/export_lora_adapter.py" \
        --ckpt "$actor" --out "$overlay_adapter" >>"$LOG" 2>&1

    cp "$overlay_adapter/adapter_config.json" "$persistent/adapter_config.json"
    sha256sum "$overlay_adapter/adapter_model.safetensors" \
        >"$persistent/adapter_model.safetensors.sha256"
    gzip -c "$overlay_adapter/adapter_model.safetensors" | \
        split -b 48m -d -a 2 - "$persistent/adapter_model.safetensors.gz.part"
    printf '%s\n' "$actor" >"$persistent/source_checkpoint.txt"
    touch "$marker"
    log "step $step LoRA persisted under $persistent"
}

train_pid=$(<"$TRAIN_PID_FILE")
[[ "$train_pid" =~ ^[0-9]+$ ]] || { echo "invalid training PID: $train_pid" >&2; exit 2; }
log "watching PID $train_pid; milestones: $MILESTONES"

while :; do
    pending=0
    for step in $MILESTONES; do
        marker="$OUTPUT_ROOT/model/checkpoint_step_$step/lora_adapter/.export_complete"
        if [[ ! -f "$marker" ]]; then
            pending=1
            persist_step "$step" || true
        fi
    done

    if ! kill -0 "$train_pid" 2>/dev/null; then
        log "training PID $train_pid exited; final export scan complete"
        break
    fi
    (( pending == 1 )) || { log "all requested adapters persisted"; break; }
    sleep "$POLL_SECONDS"
done
