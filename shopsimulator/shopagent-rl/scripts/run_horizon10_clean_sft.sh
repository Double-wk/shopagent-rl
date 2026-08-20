#!/usr/bin/env bash
# Formal horizon10-clean-v1 SFT, initialized directly from Qwen3-1.7B-Base.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
source "$ROOT/scripts/paths.sh"
PY="$SHOPAGENT_PY"
shopagent_require_py
CONFIG="$ROOT/configs/sft_horizon10_clean_v1.yaml"
DATA="$ROOT/data/sft_train_horizon10.jsonl"
OUTPUT="$ROOT/outputs/sft/v6_horizon10_clean_from_base/model/training_output"
ADAPTER="$OUTPUT/lora_adapter/adapter_model.safetensors"
EXPECTED_ROWS=3624
EXPECTED_SHA=2d72041a9a3202550d1282bee85cfeb8b4f977059726950583118688dc1ee964
RUN_NAME="${RUN_NAME:-sft_horizon10_clean_v1}"
LOG_DIR="${LOG_DIR:-$ROOT/run}"
LOG="$LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"

rows=$(wc -l < "$DATA")
sha=$(sha256sum "$DATA" | cut -d' ' -f1)
if [ "$rows" -ne "$EXPECTED_ROWS" ] || [ "$sha" != "$EXPECTED_SHA" ]; then
    echo "horizon10 SFT data mismatch: rows=$rows sha256=$sha" >&2
    exit 2
fi
if [ -e "$ADAPTER" ] && [ -z "${RESUME_FROM_CHECKPOINT:-}" ]; then
    echo "refusing to overwrite completed clean SFT adapter: $ADAPTER" >&2
    exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT"
if [ "${FOREGROUND:-0}" != "1" ] && [ "${HORIZON10_CLEAN_SFT_DAEMONIZED:-0}" != "1" ]; then
    export HORIZON10_CLEAN_SFT_DAEMONIZED=1
    setsid nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    disown
    echo "horizon10-clean-v1 SFT detached: PID $(cat "$PID_FILE")"
    echo "  log: $LOG"
    exit 0
fi

cd "$ROOT"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cmd=("$PY" -m experiment.sft.train --config "$CONFIG")
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
exec "${cmd[@]}" "$@"
