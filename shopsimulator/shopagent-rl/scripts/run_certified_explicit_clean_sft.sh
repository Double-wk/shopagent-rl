#!/usr/bin/env bash
# Train the clean explicit-budget SFT baseline independently of v3/v4 adapters.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
PY=/overlay/miniconda3/envs/shop-A/bin/python
RUN_NAME="${RUN_NAME:-certified_explicit_clean_sft}"
LOG_DIR="${LOG_DIR:-$ROOT/run}"
LOG="$LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"
mkdir -p "$LOG_DIR"

if [ "${FOREGROUND:-0}" != "1" ] && [ "${CERTIFIED_EXPLICIT_CLEAN_DAEMONIZED:-0}" != "1" ]; then
    export CERTIFIED_EXPLICIT_CLEAN_DAEMONIZED=1
    setsid nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    disown
    echo "Certified explicit clean SFT detached: PID $(cat "$PID_FILE")"
    echo "  log: $LOG"
    exit 0
fi

cd "$ROOT"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec "$PY" -m experiment.sft.train --config configs/sft_certified_explicit_clean.yaml "$@"
