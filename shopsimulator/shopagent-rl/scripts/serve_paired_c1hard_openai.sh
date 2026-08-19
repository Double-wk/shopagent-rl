#!/usr/bin/env bash
# OpenAI-compatible vLLM server for the Paired-C1-hard s200 policy.
#
# Purpose: let the OFFICIAL ShopSimulator single_eval / multi_eval harnesses
# (OpenAI-client based) drive our local policy:
#   base_url   = http://127.0.0.1:8000/v1
#   model_name = shopagent-paired-c1hard
#
# v2: serves the MERGED model (scripts/merge_paired_c1hard.py output), not
# base+LoRA. vLLM 0.16's dynamic LoRA name resolution intermittently 404s a
# correctly-named adapter on this ROCm build; a plain model has no LoRA route.
#
# Detached launch (memory rule: long tasks run detached, one engine max):
#   setsid nohup bash scripts/serve_paired_c1hard_openai.sh \
#     > run/serve_paired_c1hard.log 2>&1 &
#   pgrep -f 'entrypoints[.]openai' > run/serve_paired_c1hard.pid
set -euo pipefail

cd "$(dirname "$0")/.."      # -> shopagent-rl/
source scripts/vllm_env_shopA.sh   # ONE-env: shopsim python + ROCm shims (sets PY)

MODEL=/overlay/qwen3_1.7b_paired_c1hard_merged
if [ ! -f "$MODEL/model.safetensors" ]; then
    echo "ERROR: merged model not found: $MODEL" >&2
    echo "       run scripts/merge_paired_c1hard.py first" >&2
    exit 1
fi

exec "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name shopagent-paired-c1hard \
    --chat-template configs/qwen3_base_chat_template.jinja \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --enforce-eager \
    --trust-remote-code \
    --port 8000
