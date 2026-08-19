#!/usr/bin/env bash
set -euo pipefail

# Build the pinned ROCm vLLM checkout into a named conda environment.
CONDA_ROOT="${CONDA_ROOT:-/overlay/miniconda3}"
CONDA_ENV="${1:-shopsim}"
VLLM_SRC="${VLLM_SRC:-/overlay/vllm-rocm-src}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.1}"
# 89a77b108 is the verified gfx1100/ROCm 7.2.1 build (v0.16.0 tag).
EXPECTED_VLLM_COMMIT="${VLLM_COMMIT:-89a77b108}"

test -f "$CONDA_ROOT/etc/profile.d/conda.sh" || {
    echo "conda not found under $CONDA_ROOT" >&2
    exit 1
}
test -d "$VLLM_SRC/.git" || {
    echo "vLLM checkout not found at $VLLM_SRC" >&2
    exit 1
}
test -x "$ROCM_PATH/bin/rocminfo" || {
    echo "ROCm not found under $ROCM_PATH" >&2
    exit 1
}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== Building vLLM for ROCm ==="
echo "env=$CONDA_ENV source=$VLLM_SRC rocm=$ROCM_PATH"
python --version
ACTUAL_VLLM_COMMIT="$(git -C "$VLLM_SRC" rev-parse HEAD)"
case "$ACTUAL_VLLM_COMMIT" in
    "$EXPECTED_VLLM_COMMIT"*) ;;
    *)
        echo "expected vLLM $EXPECTED_VLLM_COMMIT, found $ACTUAL_VLLM_COMMIT" >&2
        exit 1
        ;;
esac
echo "vLLM commit=$ACTUAL_VLLM_COMMIT"

python -m pip install \
    'cmake>=3.26.1' ninja 'packaging>=24.2' \
    'setuptools>=77.0.3,<80.0.0' 'setuptools-scm>=8' wheel \
    'jinja2>=3.1.6' regex build
rm -rf "$VLLM_SRC/build"

export ROCM_PATH CUDA_HOME="$ROCM_PATH"
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
python -m pip install -e "$VLLM_SRC" --no-build-isolation --no-deps

python -m pip show vllm | sed -n '1,8p'
