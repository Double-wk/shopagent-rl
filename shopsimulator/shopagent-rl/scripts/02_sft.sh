#!/usr/bin/env bash
# Phase 2: validate the 5,000 collected trajectories (-> 3,793 strict) and
# train the Qwen3-1.7B-Base LoRA adapter.
# Uses the validated data/sft_train.jsonl already produced by the data pipeline.
set -euo pipefail
PY=/overlay/miniconda3/envs/shopsim/bin/python
cd /workspace/shopsimulator/shopagent-rl
"$PY" -m experiment.teacher.validate --max_keep 3793
exec "$PY" -m experiment.sft.train --config configs/sft.yaml
