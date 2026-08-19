#!/usr/bin/env bash
# vLLM on ROCm, shopsim env — the single ShopSimulator environment.
# shopsim ships the full lock plus the vLLM 0.16.0 build at commit 89a77b108.
# veRL 0.8.0 is the vendored source under shopsimulator/shopagent-rl, installed once as
# the editable `shop-a-verl` package (see shop_A/pyproject.toml); `import verl`
# resolves to that source in bare Python, exactly like shopsim and opd-rocm.
#
# Source before any vLLM / GPU script:
#   source /workspace/scripts/vllm_env_rocm_base.sh
#   $PY my_script.py
# IMPORTANT: scripts using LLM() MUST guard execution in `if __name__ == "__main__":`
# (V1 engine forces `spawn`; top-level LLM() recurses in the worker).
# DO NOT `pip install` vLLM deps without --no-deps — it clobbers ROCm torch.
set -u

PY=/overlay/miniconda3/envs/shopsim/bin/python

# 1. amdsmi (system binding) -> PYTHONPATH so vLLM's rocm platform plugin finds it.
#    Without this the engine dies in DeviceConfig: "Device string must not be empty".
mkdir -p /tmp/amdsmi_link
ln -sfn /usr/local/lib/python3.12/dist-packages/amdsmi /tmp/amdsmi_link/amdsmi

# 2. Restore the checked-in functorch compatibility shim after every reboot.
SHIM_SRC=/workspace/scripts/functorch_shim/functorch
SHIM_DST=/tmp/functorch_shim/functorch
[ -f "$SHIM_SRC/compile.py" ] || { echo "ERROR: missing $SHIM_SRC"; return 1; }
mkdir -p "$SHIM_DST"
cp "$SHIM_SRC/__init__.py" "$SHIM_SRC/compile.py" "$SHIM_DST/"

# shop_A stays first on PYTHONPATH for the ShopSim project modules and ROCm
# shims (amdsmi/functorch); veRL itself resolves through the editable install.
export PYTHONPATH=/workspace/shopsimulator/shopagent-rl:/tmp/amdsmi_link:/tmp/functorch_shim:${PYTHONPATH:-}
# HF_HUB_CACHE is the hub subdirectory, not its parent cache directory.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/root/.cache/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "shopsim vLLM env ready: PY=$PY  PYTHONPATH=$PYTHONPATH"
echo "  remember: guard your LLM() call in  if __name__ == '__main__':"
