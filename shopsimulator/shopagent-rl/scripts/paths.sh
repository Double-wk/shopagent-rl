#!/usr/bin/env bash
# Single source of truth for machine paths. Source this, never hardcode.
#
# Why this file exists: until 2026-08-19 the GRPO trainer wrote to
# /overlay/shopagent_rl_grpo_outputs, which lived on the container's ephemeral
# disk. An instance restart wiped it and took the horizon10-clean-v1 Independent
# adapter and every intermediate checkpoint with it. The SFT outputs, already
# under /workspace, survived untouched. Nothing that a rerun would need may live
# outside /workspace again.
#
# Two tiers, both persistent:
#   SHOPAGENT_OUTPUT_ROOT     reproducibility-critical, committed to git
#                             (exported LoRA adapters as .gz split volumes,
#                             eval reports, metrics, provenance)
#   SHOPAGENT_ARTIFACT_ROOT   bulk resumable trainer state, git-ignored
#                             (raw FSDP checkpoints, optimizer shards, ray temp)
#                             Safe to prune; only needed to resume a live run.
#
# Every value is overridable so the workspace can move without touching scripts.

SHOPAGENT_ROOT="${SHOPAGENT_ROOT:-/workspace/shopsimulator/shopagent-rl}"
SHOPAGENT_PY="${SHOPAGENT_PY:-/workspace/miniconda3/envs/shopsim/bin/python}"
SHOPAGENT_ARTIFACT_ROOT="${SHOPAGENT_ARTIFACT_ROOT:-/workspace/artifacts}"
SHOPAGENT_OUTPUT_ROOT="${SHOPAGENT_OUTPUT_ROOT:-$SHOPAGENT_ROOT/outputs}"
SHOPAGENT_GRPO_ARTIFACT_ROOT="${SHOPAGENT_GRPO_ARTIFACT_ROOT:-$SHOPAGENT_ARTIFACT_ROOT/grpo_runs}"

export SHOPAGENT_ROOT SHOPAGENT_PY SHOPAGENT_ARTIFACT_ROOT \
       SHOPAGENT_OUTPUT_ROOT SHOPAGENT_GRPO_ARTIFACT_ROOT

# Fail early and loudly rather than 40 minutes into a rollout.
shopagent_require_py() {
    if [ ! -x "$SHOPAGENT_PY" ]; then
        echo "shopsim interpreter not found: $SHOPAGENT_PY" >&2
        echo "rebuild the env or export SHOPAGENT_PY=/path/to/python" >&2
        return 2
    fi
}
