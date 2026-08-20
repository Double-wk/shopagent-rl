#!/usr/bin/env bash
# Run a Final-200 eval (Base / SFT / GRPO) with the local vLLM policy.
# Sources the shopsim ROCm vLLM launcher (amdsmi/functorch/shims/spawn/offline env;
# ONE-env design — eval runs in the same shopsim env as SFT/GRPO, PY is set by the script),
# then runs the round-stepped wave runner.
#
# Usage:
#   bash scripts/run_eval.sh --tag Base --out outputs/eval_base.jsonl
#   bash scripts/run_eval.sh --tag SFT --out outputs/eval_sft.jsonl \
#       --adapter outputs/sft/v1/model/training_output/lora_adapter
#   bash scripts/run_eval.sh --tag GRPO --out outputs/eval_grpo.jsonl \
#       --adapter /workspace/artifacts/grpo_runs/<run>/global_step_50/actor
set -euo pipefail

# shopagent-rl root regardless of where this is invoked from
cd "$(dirname "$0")/.."      # -> shopagent-rl/

source scripts/vllm_env_shopA.sh   # ONE-env: shopsim python + ROCm shims (sets PY)

exec "$PY" -m experiment.eval.run_final200 "$@"
