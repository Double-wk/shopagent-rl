#!/usr/bin/env bash
# Phase-1 counterfactual pair eval (see docs/counterfactual-eval.md) for one
# checkpoint. Pure vLLM inference on the stored pair states — no pack_api/env.
#
# Usage:
#   bash scripts/run_counterfactual_eval.sh --tag Base \
#       --out outputs/counterfactual/cf_base_v1.jsonl
#   bash scripts/run_counterfactual_eval.sh --tag SFT \
#       --adapter outputs/sft/v1/model/training_output/lora_adapter \
#       --out outputs/counterfactual/cf_sft_v1.jsonl
set -euo pipefail

cd "$(dirname "$0")/.."      # -> shopagent-rl/

source scripts/vllm_env_shopA.sh   # ONE-env: shopsim python + ROCm shims (sets PY)

exec "$PY" -m experiment.eval.run_counterfactual "$@"
