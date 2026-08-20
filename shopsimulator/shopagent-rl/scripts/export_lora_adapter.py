"""Export a PEFT LoRA adapter from a veRL/FSDP actor checkpoint.

The FSDP checkpoint (model_world_size_1_rank_0.pt) stores the wrapped PEFT model:
base weights as `...base_layer.weight` and LoRA mats as `...lora_A.default.weight`.
vLLM's LoRARequest expects a plain PEFT adapter dir (adapter_config.json +
adapter_model.safetensors with `...lora_A.weight` keys, no `.default`), matching
the layout verl saves for the SFT adapter and the GRPO v2b export.

Usage:
  python scripts/export_lora_adapter.py \
      --ckpt /workspace/artifacts/grpo_runs/<run>/global_step_200/actor \
      --out  outputs/grpo/<run>/model/checkpoint_step_200/lora_adapter

The raw FSDP checkpoint under artifacts/ is pruned (trainer.max_actor_ckpt_to_keep)
and never committed; the 70MB adapter this writes under outputs/ is the artifact
that reproduces the run, so export before the checkpoint rotates out.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="veRL actor checkpoint dir")
    ap.add_argument("--out", required=True, help="destination lora_adapter dir")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = ckpt / "lora_train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    r = meta.get("r", 32)
    lora_alpha = meta.get("lora_alpha", 2 * r)

    sd = torch.load(ckpt / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and not any("lora" in k for k in sd):
        sd = sd["model"]

    adapter = {}
    for key, value in sd.items():
        if ".lora_" not in key or "base_layer" in key:
            continue
        # `...lora_A.default.weight` -> `...lora_A.weight` (PEFT export format)
        adapter[key.replace(".default.", ".")] = value.contiguous().to(torch.float32)

    if not adapter:
        raise SystemExit(f"no LoRA tensors found in {ckpt}")

    target_modules = sorted(
        {k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for k in adapter}
    )
    config = {
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "auto_mapping": None,
        "peft_version": "0.20.0",
        "base_model_name_or_path": None,
        "revision": None,
        "inference_mode": True,
        "r": r,
        "target_modules": target_modules,
        "exclude_modules": None,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "use_rslora": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "megatron_config": None,
        "loftq_config": {},
        "use_dora": False,
        "lora_bias": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=4))
    save_file(adapter, out / "adapter_model.safetensors")
    print(f"exported {len(adapter)} LoRA tensors (r={r}, alpha={lora_alpha}) -> {out}")


if __name__ == "__main__":
    main()
