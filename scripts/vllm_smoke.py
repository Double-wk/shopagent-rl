#!/usr/bin/env python3
"""Real vLLM engine smoke test for the local ROCm environment."""
import os


def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-1.7B-Base")
    llm = LLM(
        model=model,
        dtype="float16",
        gpu_memory_utilization=0.60,
        max_model_len=1024,
        enforce_eager=True,
    )
    # ROCm runtime convention: use a small positive temperature for greedy.
    outputs = llm.generate(
        ["Compute 17 + 28. Answer:"],
        SamplingParams(temperature=0.01, max_tokens=48),
    )
    result = outputs[0].outputs[0]
    print(f"PROMPT: {outputs[0].prompt}")
    print(f"OUTPUT: {result.text!r}")
    print(f"NTOKENS: {len(result.token_ids)}")
    assert result.token_ids, "engine produced no tokens"
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
