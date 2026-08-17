#!/usr/bin/env bash
# Self-healing wrapper around the official single_eval agent.py.
#
# Why: vLLM 0.16 on this ROCm box intermittently answers "model does not
# exist" (404) for its own static served model while the engine core is
# briefly stalled (see agent.py backoff note). agent.py retries + its resume
# scan make reruns cheap: finished tasks are skipped via output-dir scan, so
# we simply relaunch until nothing is pending.
#
# Usage:
#   setsid nohup bash scripts/run_single_eval_resilient.sh \
#     <yaml> <max_workers> <rounds> > <log> 2>&1 &
set -u

YAML_PATH=${1:?usage: run_single_eval_resilient.sh <yaml> <max_workers> <rounds>}
MAX_WORKERS=${2:-4}
ROUNDS=${3:-20}
PY=/overlay/miniconda3/envs/shop-A/bin/python
cd /workspace/shopsimulator/ShopSimulator/single_eval

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

for round in $(seq 1 "$ROUNDS"); do
    echo "[wrapper] === round $round/$ROUNDS $(date '+%F %T') ==="
    "$PY" agent.py --yaml_name "$YAML_PATH" --multithread --max_workers "$MAX_WORKERS"
    # rerun the same finished-task scan agent.py uses
    pending=$("$PY" - "$YAML_PATH" <<'PYEOF'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
ac = cfg["agent_config"]
out = os.path.join(ac["output_path"], ac["model_name"])
os.makedirs(out, exist_ok=True)
done = set()
for f in os.listdir(out):
    if f.endswith(".json"):
        try: done.add(int(f.rsplit(".", 1)[0]))
        except ValueError: pass
all_t = range(ac.get("task_nums", 0))
print(len([i for i in all_t if i not in done]))
PYEOF
)
    echo "[wrapper] pending after round $round: $pending"
    [ "$pending" -eq 0 ] && { echo "[wrapper] DONE, all tasks finished"; exit 0; }
    sleep 10
done
echo "[wrapper] exhausted $ROUNDS rounds, $pending still pending"
exit 1
