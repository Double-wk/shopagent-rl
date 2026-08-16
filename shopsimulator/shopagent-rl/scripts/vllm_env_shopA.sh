#!/usr/bin/env bash
# shop-A ships the canonical vendored veRL 0.8.0 as an editable package.
# Install it once after a fresh clone: $PY -m pip install --no-deps -e .
# shop-A already ships the verified vLLM 0.16.0 code at commit 89a77b108.
# This script only wires the two ROCm shims and the offline HF cache.
#
# Source before any vLLM/GRPO/eval script:
#   source scripts/vllm_env_shopA.sh
#   $PY my_script.py
# IMPORTANT: scripts using LLM() MUST guard execution in `if __name__ == "__main__":`
# (V1 engine forces `spawn`; top-level LLM() recurses in the worker).
# DO NOT `pip install` vLLM deps without --no-deps — it clobbers ROCm torch.
set -u

PY=/overlay/miniconda3/envs/shop-A/bin/python

# 1. amdsmi (system binding) -> PYTHONPATH so vLLM's rocm platform plugin finds it.
mkdir -p /tmp/amdsmi_link
ln -sfn /usr/local/lib/python3.12/dist-packages/amdsmi /tmp/amdsmi_link/amdsmi

# 2. Restore the checked-in functorch compatibility shim after every reboot.
SHIM_SRC=/workspace/scripts/functorch_shim/functorch
SHIM_DST=/tmp/functorch_shim/functorch
[ -f "$SHIM_SRC/compile.py" ] || { echo "ERROR: missing $SHIM_SRC"; return 1; }
mkdir -p "$SHIM_DST"
cp "$SHIM_SRC/__init__.py" "$SHIM_SRC/compile.py" "$SHIM_DST/"

# PYTHONPATH is retained for the ShopSim project modules and ROCm shims. veRL
# itself resolves through the editable `shop-a-verl` installation.
export PYTHONPATH=/workspace/shopsimulator/shopagent-rl:/tmp/amdsmi_link:/tmp/functorch_shim:${PYTHONPATH:-}
# HF_HUB_CACHE is the hub subdirectory, not its parent cache directory.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/root/.cache/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "shop-A vLLM env ready: PY=$PY  PYTHONPATH=$PYTHONPATH"
echo "  remember: guard your LLM() call in  if __name__ == '__main__':"
