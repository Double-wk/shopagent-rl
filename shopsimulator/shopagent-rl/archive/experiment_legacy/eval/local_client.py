"""Local vLLM inference client with the teacher-client chat interface.

Drops into eval/run_eval.py as a drop-in replacement for OpenAITeacherClient:
exposes `.chat(messages, system, max_tokens, temperature) -> str` so the
collect.py::run_one multi-turn rollout body is reused unchanged. ONE LLM engine
is shared across all eval tasks (created once in __main__, called from worker
threads — vLLM.generate is concurrency-safe and batches internally).

Eval three checkpoints from one client shape:
  * Base : model=<base path>, adapter=None
  * SFT  : model=<base path>, adapter=<SFT lora_adapter>
  * GRPO : model=<base path>, adapter=<GRPO lora_adapter>
(vLLM native LoRA: enable_lora + LoRARequest — no merge step. If ROCm rejects
native LoRA at runtime, fall back to merging the adapter first and pointing
model= at the merged dir with adapter=None.)

ROCm: import only under the sourced vllm_env_shopA.sh shims; LLM() must be
constructed under `if __name__ == "__main__":` (V1 engine forces spawn).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class VLLMClient:
    """vLLM-backed chat client (teacher-client compatible interface)."""

    def __init__(
        self,
        model: str,
        adapter: Optional[str] = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 8192,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.adapter = adapter
        enable_lora = bool(adapter)
        self.lora_request = (
            LoRARequest("adapter", lora_int_id=1, lora_local_path=adapter)
            if enable_lora else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        # Pad token: Qwen3 base may lack one; chat_template uses generation prompt,
        # but vLLM needs a pad id for batched left-pad. Fall back to eos if absent.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=model,
            dtype=dtype,
            enable_lora=enable_lora,
            max_loras=1 if enable_lora else 0,
            max_lora_rank=64,                  # match SFT adapter r=32 (rank+slack)
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
        )
        cfg = "base+LoRA" if enable_lora else "base"
        print(f"[VLLMClient] loaded {cfg}: model={model} adapter={adapter}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant message content (teacher-client compatible)."""
        from vllm import SamplingParams

        full: List[Dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        # apply_chat_template -> the same ChatML framing SFT trained on (tokenize=False
        # lets vLLM tokenize, consistent with its BOS handling).
        prompt = self.tokenizer.apply_chat_template(
            full, add_generation_prompt=True, tokenize=False
        )
        out = self.llm.generate(
            [prompt],
            SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0),
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        return out[0].outputs[0].text
