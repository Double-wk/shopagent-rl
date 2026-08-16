#!/usr/bin/env python3
"""Verify that vLLM sleep returns physical VRAM on the pinned ROCm stack."""

import torch
from vllm import LLM, SamplingParams


GIB = 1024**3


def memory_gib() -> tuple[float, float, float]:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB, free / GIB, total / GIB


def main() -> None:
    baseline = memory_gib()
    print(f"baseline used={baseline[0]:.3f} free={baseline[1]:.3f} total={baseline[2]:.3f}")

    llm = LLM(
        model="Qwen/Qwen3-1.7B-Base",
        dtype="bfloat16",
        gpu_memory_utilization=0.25,
        max_model_len=10240,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        enable_sleep_mode=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    awake = memory_gib()
    print(f"awake0 used={awake[0]:.3f} free={awake[1]:.3f}")

    sampling = SamplingParams(temperature=0.7, max_tokens=64)
    first_asleep_used = None
    for cycle in range(1, 4):
        output = llm.generate(["Compute 17 + 28. Answer:"], sampling, use_tqdm=False)
        assert output[0].outputs[0].token_ids, "engine produced no tokens"

        before_sleep = memory_gib()
        llm.sleep(level=1)
        asleep = memory_gib()
        freed = asleep[1] - before_sleep[1]
        residual = asleep[0] - baseline[0]
        print(
            f"cycle={cycle} before_sleep_used={before_sleep[0]:.3f} "
            f"asleep_used={asleep[0]:.3f} freed={freed:.3f} residual={residual:.3f}"
        )
        minimum_freed = 3.0 if cycle == 1 else 2.5
        assert freed > minimum_freed, f"sleep freed only {freed:.3f} GiB"
        # vLLM V1 keeps some non-sleep-pool buffers resident.  The upstream
        # test allows 7 GiB; use a tighter 5 GiB ceiling here and, more
        # importantly, reject growth across repeated cycles.
        assert residual < 5.0, f"sleep residual is {residual:.3f} GiB"
        if first_asleep_used is None:
            first_asleep_used = asleep[0]
        else:
            drift = asleep[0] - first_asleep_used
            assert drift < 0.25, f"sleep residual drifted by {drift:.3f} GiB"

        llm.wake_up()
        awake = memory_gib()
        print(f"cycle={cycle} awake_used={awake[0]:.3f}")

    print("SLEEP_WAKE_RESULT=PASS")


if __name__ == "__main__":
    main()
