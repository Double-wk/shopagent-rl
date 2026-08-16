#!/usr/bin/env bash
# Relaunch of the formal paired-constraint SFT after the 2026-08-16 04:32 run
# wedged in allocator retry (CPU spin, GPU idle, VRAM 51.1/48, no log for 1h).
# Only change vs the wedged run: PYTORCH_ALLOC_CONF=expandable_segments:True
# (pure SFT, no vLLM — the GRPO caveat about expandable_segments does not apply).
set -euo pipefail
cd /workspace/shopsimulator/shopagent-rl
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo $$ > run/natural_rewrite/sft_paired_formal2.pid
exec /overlay/miniconda3/envs/shop-A/bin/python -m experiment.sft.train \
  --config configs/sft_paired.yaml
