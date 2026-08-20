#!/usr/bin/env bash
# Continue v3 from natural-format paired price examples to remove the summary shortcut.
set -euo pipefail

ROOT=/workspace/shopsimulator/shopagent-rl
source "$ROOT/scripts/paths.sh"
PY="$SHOPAGENT_PY"
shopagent_require_py
RUN_NAME="${RUN_NAME:-certified_corrective_sft}"
LOG_DIR="${LOG_DIR:-$ROOT/run}"
LOG="$LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"
mkdir -p "$LOG_DIR"

if [ "${FOREGROUND:-0}" != "1" ] && [ "${CERTIFIED_CORRECTIVE_DAEMONIZED:-0}" != "1" ]; then
    export CERTIFIED_CORRECTIVE_DAEMONIZED=1
    setsid nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    disown
    echo "Certified corrective SFT detached: PID $(cat "$PID_FILE")"
    echo "  log: $LOG"
    exit 0
fi

cd "$ROOT"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec "$PY" -m experiment.sft.train --config configs/sft_certified_corrective.yaml "$@"
