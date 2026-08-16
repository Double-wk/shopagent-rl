#!/usr/bin/env bash
# Phase 1: collect teacher trajectories (GPT-5.6-SOL via mcgrox OpenAI-compatible API).
# Requires the ShopSimulator service running on :5000 (run 00_start_env.sh first).
# Collects 10,000 trajectories from official train data (random sample of 21,962 tasks).
set -euo pipefail
PY=/overlay/miniconda3/envs/shop-A/bin/python
cd /workspace/shopsimulator/shop_A
exec "$PY" -m teacher.collect --config configs/teacher_gpt-5.6-sol.yaml --num 10000 --workers 8
