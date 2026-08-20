# Offline Environment Artifacts

These files allow the verified ShopSimulator GPU environment to be restored
without downloading or rebuilding vLLM on an identical machine.

## Compatibility

The prebuilt vLLM wheel is only for this stack:

- Linux x86_64
- CPython 3.12
- AMD gfx1100
- ROCm 7.2.1
- PyTorch 2.9.1 ROCm build `gitff65f5bc`
- Triton 3.5.1 ROCm build `gita272dfa8`

Do not install it on another GPU architecture or with a different PyTorch/ROCm
ABI. Use the source archive and `scripts/build_vllm_rocm.sh` instead.

## Direct Install

```bash
PY=/workspace/miniconda3/envs/shopsim/bin/python

"$PY" -m pip install --no-deps \
  /workspace/shopsimulator/envs_config/vllm_wheels/triton_kernels-1.0.0-py3-none-any.whl \
  /workspace/shopsimulator/envs_config/vllm_wheels/vllm-0.16.0+rocm721-cp312-cp312-linux_x86_64.whl
```

The vLLM wheel is based on upstream commit
`89a77b10846fd96273cce78d86d2556ea582d26e` and includes
`shopagent-rl/patches/vllm-0.16.0-rocm-sleep-release.patch`.

## Checksums

```text
357b97f878b2ff0007e7191fc114f93492dc7e7065c0cdcc6fac52d7c032c3de  vllm-0.16.0-89a77b108.tar.gz
a589cc6642e8d14b396b6ff25c403c35ac22e130aa38946a7130cba2125af670  vllm_wheels/triton_kernels-1.0.0-py3-none-any.whl
95fa5f134e975af97ae7e80880674566c99a6e56ce72d0dbd9553aa879586bc9  vllm_wheels/vllm-0.16.0+rocm721-cp312-cp312-linux_x86_64.whl
```

## Validation

The wheel was validated on 2026-08-20:

- `vllm._C`, `vllm._rocm_C`, and `vllm._moe_C` imported successfully.
- Qwen3-1.7B generated successfully with `RESULT=PASS`.
- Three sleep/wake cycles completed with `SLEEP_WAKE_RESULT=PASS`.
- Sleeping VRAM residual was stable at approximately 3.71 GiB.
