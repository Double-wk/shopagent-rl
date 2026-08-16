"""Local vLLM policy engine for ShopSimulator eval (Base / SFT / GRPO).

A single shared engine with BATCHED multi-turn chat. Optional LoRA adapter
(SFT/GRPO eval); Base eval passes adapter_path=None.

CONSTRUCT INSIDE `if __name__ == "__main__":` (the caller main() does this) —
vLLM's V1 EngineCore subprocess forces `spawn`; building LLM() at import time
recurses forever in the spawned worker. See runs/vllm_env.sh.

SFT/INFERENCE ALIGNMENT (the token-boundary risk flagged in the plan):
  * Uses the tokenizer's OWN chat_template — the same one sft/train.py applies via
    apply_chat_template — so eval prompts are byte-identical to training prompts.
  * vLLM's llm.chat() applies that template with add_generation_prompt=True.
  * Generation stops at <|im_end|> (id 151645, the Qwen3 ChatML turn terminator),
    NOT at eos=<|endoftext|> (151643): the template terminates assistant turns with
    <|im_end|>; stopping only at eos lets the model ramble past the turn boundary
    into a hallucinated next user/assistant turn.

Run via scripts/run_eval.sh (sources the local ROCm/vLLM environment).
"""
from __future__ import annotations

from typing import List, Optional


class LocalVLLM:
    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        # 0.80 not 0.90: this is a shared GPU; vLLM's init-time check
        # (free mem >= util*total) tripped at 0.9 with ~5 GiB of background use.
        # A 1.7B bf16 model (~3.4 GiB weights) needs a fraction of VRAM, and
        # 0.80*48 GiB still leaves a massive KV cache for batched multi-turn.
        gpu_memory_utilization: float = 0.80,
        # The native Qwen3 context limit is 32768.  Keep eval within the
        # verified GRPO context budget; 10 turns × 512 tokens fits comfortably.
        max_model_len: int = 10240,
        dtype: str = "bfloat16",
        # enforce_eager=True: skip torch.compile/inductor + CUDA-graph capture.
        # On ROCm the compiled path is flaky — it pulls names from the legacy
        # `functorch.compile` (make_boxed_func, nop, ...) one at a time and the
        # functorch shim is incomplete. Eager matches the verified-working config
        # (see memory vllm-rocm-usable-with-setup) and STILL keeps vLLM's paged-
        # attention batching win (the 233 tok/s batch number was eager). If we
        # later want compiled mode for GRPO throughput, complete the shim first.
        enforce_eager: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams  # noqa: F401  (import-time check)
        from vllm.lora.request import LoRARequest  # noqa: F401

        self.adapter_path = adapter_path
        self.base_model = base_model
        self.max_model_len = max_model_len

        # Resolve stop token ids from the tokenizer (robust to id drift across
        # Qwen builds): <|im_end|> is the real turn terminator; include eos too.
        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.stop_token_ids: List[int] = []
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self.stop_token_ids.append(i)

        kwargs = dict(
            model=base_model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        if adapter_path:
            # LoRA: enable at the engine, then attach per-request.
            kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=64, max_cpu_loras=4)
        self.llm = LLM(**kwargs)
        self.lora_request = None
        if adapter_path:
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        print(
            f"[LocalVLLM] loaded {base_model}"
            f"{' + LoRA ' + adapter_path if adapter_path else ' (base, no adapter)'}"
            f"  stop_token_ids={self.stop_token_ids}"
        )

    def generate_batch(
        self,
        conversations: List[List[dict]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> List[str]:
        """Batched chat completion. Each conversation is a list of {role, content}
        (system + user + assistant turns). Returns one assistant text per input,
        in order."""
        from vllm import SamplingParams

        # Truncate each conversation to fit max_model_len. ShopSimulator runs up
        # to 30 turns and a long horizon accumulates ~65K tokens — past the
        # model's 32K native context. vLLM rejects the WHOLE batch if even one
        # conv is over-length (VLLMValidationError), and run_final200 then
        # soft-fails EVERY active task in the wave to reward=0. So we trim
        # over-length convs first: keep system + first user (instruction) + the
        # most-recent turn-pairs, drop the middle. No-op for short convs. Native
        # 32K is preserved (no YaRN extrapolation) so Base/SFT/GRPO all see the
        # same truncation = fair, and stay within verified context quality.
        input_limit = self.max_model_len - max_tokens - 512
        conversations = [self._truncate(c, input_limit) for c in conversations]

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=self.stop_token_ids or None,
        )
        outs = self.llm.chat(
            conversations, sp, lora_request=self.lora_request, use_tqdm=False
        )
        results: List[str] = []
        for o in outs:
            results.append(o.outputs[0].text if o.outputs else "")
        return results

    def _ntok(self, messages: List[dict]) -> int:
        """Approx token count: sum of per-message content tokens (ignores the
        small chat-template overhead per turn, which the 512-token slack in
        generate_batch comfortably covers). Order-independent, so it works on
        arbitrary sub-sequences (e.g. a single assistant/user pair)."""
        return sum(len(self.tok.encode(m.get("content", ""))) for m in messages)

    def _truncate(self, conv: List[dict], limit: int) -> List[dict]:
        """Keep system msgs + the first non-system turn (the instruction) + the
        most-recent (assistant,user) turn-pairs; drop middle pairs until under
        `limit` tokens. Returns conv unchanged if already short. Removes whole
        turn-pairs so chat alternation (user/assistant) is preserved."""
        if len(conv) <= 2:
            return conv
        try:
            if self._ntok(conv) <= limit:
                return conv
        except Exception:  # noqa: BLE001
            return conv
        sys_msgs = [m for m in conv if m.get("role") == "system"]
        nonsys = [m for m in conv if m.get("role") != "system"]
        if len(nonsys) <= 1:
            return conv
        head = [nonsys[0]]                       # instruction — always kept
        tail = nonsys[1:]                        # [asst, user, asst, user, ...]
        pairs = [(tail[i], tail[i + 1]) for i in range(0, len(tail) - 1, 2)]
        kept: List[tuple] = []
        cur = self._ntok(sys_msgs + head)
        for p in reversed(pairs):
            try:
                nt = self._ntok(list(p))
            except Exception:  # noqa: BLE001
                break
            if cur + nt > limit:
                break
            kept.append(p)
            cur += nt
        kept.reverse()
        result = head + [m for p in kept for m in p]
        return sys_msgs + result
