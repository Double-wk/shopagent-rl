#!/usr/bin/env bash
# Run the phase-1 counterfactual pair eval for all four checkpoints SEQUENTIALLY
# (each spawns its own vLLM engine; they cannot co-reside on the single GPU).
set -euo pipefail
cd "$(dirname "$0")/.."      # -> shopagent-rl/

OUT=outputs/counterfactual
mkdir -p "$OUT"

run() {  # run <tag> <outfile> [adapter]
  local tag="$1" out="$2" adapter="${3:-}"
  if [ -f "${out%.jsonl}_metrics.json" ]; then
    echo "[skip] $tag already done"
    return
  fi
  echo "=== counterfactual eval: $tag ==="
  if [ -n "$adapter" ]; then
    bash scripts/run_counterfactual_eval.sh --tag "$tag" --out "$out" --adapter "$adapter"
  else
    bash scripts/run_counterfactual_eval.sh --tag "$tag" --out "$out"
  fi
}

run Base       "$OUT/cf_base_v1.jsonl"
run SFT        "$OUT/cf_sft_v1.jsonl"          outputs/sft/v1/model/training_output/lora_adapter
run GRPO_v1    "$OUT/cf_grpo_v1_s200.jsonl"    outputs/grpo/v1/model/checkpoint_step_200/lora_adapter
run GRPO_v2b   "$OUT/cf_grpo_v2b_s200.jsonl"   /workspace/artifacts/grpo_runs/v2/export_step_200/lora_adapter

echo "=== all counterfactual evals complete ==="
