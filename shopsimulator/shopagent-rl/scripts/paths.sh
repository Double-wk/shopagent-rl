#!/usr/bin/env bash
# Single source of truth for machine paths. Source this, never hardcode.
#
# The split is by REPRODUCIBILITY, not by size:
#
#   SHOPAGENT_OUTPUT_ROOT     /workspace/.../outputs — persistent, git-tracked.
#                             Exported LoRA adapters (70MB, as .gz split
#                             volumes), eval reports, metrics, provenance.
#                             Everything the paper needs to be reproducible.
#
#   SHOPAGENT_ARTIFACT_ROOT   /overlay — the container's LARGE but EPHEMERAL
#                             disk (~3T free, wiped on instance restart).
#                             Raw FSDP checkpoints, optimizer shards, ray temp:
#                             big, and losing them costs a rerun, not a result.
#
# WHY THE DISTINCTION MATTERS: before 2026-08-19 the exported adapter was written
# under /overlay too. An instance restart wiped the disk and the horizon10-clean-v1
# Independent adapter went with it — 4 hours of training reduced to a single-seed
# evaluation report that can never be extended. The raw checkpoint is the ONLY
# source for that adapter, so:
#
#   >>> Export to $SHOPAGENT_OUTPUT_ROOT as soon as a checkpoint is written. <<<
#   >>> Do not wait until the run ends.  scripts/export_lora_adapter.py       <<<
#
# Every value is overridable so the workspace can move without touching scripts.

SHOPAGENT_ROOT="${SHOPAGENT_ROOT:-/workspace/shopsimulator/shopagent-rl}"
# The conda env physically lives on the persistent volume; a ROCm/vLLM rebuild
# costs hours, so it is not treated as throwaway bulk despite its size.
SHOPAGENT_PY="${SHOPAGENT_PY:-/workspace/miniconda3/envs/shopsim/bin/python}"
SHOPAGENT_ARTIFACT_ROOT="${SHOPAGENT_ARTIFACT_ROOT:-/overlay/shopagent_rl_artifacts}"
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

# /overlay is recreated empty after every instance restart.
shopagent_prepare_artifact_root() {
    mkdir -p "$SHOPAGENT_GRPO_ARTIFACT_ROOT" || {
        echo "cannot create artifact root: $SHOPAGENT_GRPO_ARTIFACT_ROOT" >&2
        echo "export SHOPAGENT_ARTIFACT_ROOT to a writable large disk" >&2
        return 2
    }
    echo "[paths] artifacts -> $SHOPAGENT_GRPO_ARTIFACT_ROOT (EPHEMERAL: export adapters to outputs/ promptly)"
}
