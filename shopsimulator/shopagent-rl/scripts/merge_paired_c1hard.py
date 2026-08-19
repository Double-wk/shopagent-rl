"""Merge the Paired-C1-hard s200 LoRA adapter into the Qwen3-1.7B base weights.

Motivation: vLLM 0.16's dynamic LoRA name resolution intermittently 404s a
correctly-named adapter (observed on this ROCm build: identical requests flap
between 200 and 404 across minutes). Serving a merged plain model removes the
LoRA runtime path entirely.

Runs on CPU (torch.device('cpu'), bf16) so it can run while the vLLM server
holds the GPU. Output is a self-contained model dir: merged weights + tokenizer
files + generation_config with eos=[<|im_end|>, <|endoftext|>].

Usage:
  /overlay/miniconda3/envs/shopsim/bin/python scripts/merge_paired_c1hard.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

BASE = "/overlay/qwen3_1.7b_base_imend"   # symlinked snapshot + fixed generation_config
ADAPTER = "/overlay/shopagent_rl_grpo_outputs/grpo/paired_c1hard_200_direct/export_step_200/lora_adapter"
OUT = "/overlay/qwen3_1.7b_paired_c1hard_merged"


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (Path(OUT) / "model.safetensors").exists():
        print(f"[merge] {OUT} already exists, skipping (delete it to redo)")
        return

    print(f"[merge] base={BASE} adapter={ADAPTER}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    merged = model.merge_and_unload()

    Path(OUT).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT, safe_serialization=True)

    # tokenizer + template: copy from the base snapshot set
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.save_pretrained(OUT)

    # turn terminator as eos, so OpenAI-compat generation stops at <|im_end|>
    gen_cfg = Path(OUT) / "generation_config.json"
    shutil.copyfile(Path(BASE) / "generation_config.json", gen_cfg)

    print(f"[merge] wrote {OUT}")
    print(f"[merge] generation_config: {gen_cfg.read_text().strip()}")


if __name__ == "__main__":
    main()
