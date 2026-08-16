"""LoRA SFT of Qwen3-1.7B-Base on validated teacher trajectories.

Dataset: data/sft_train.jsonl (from teacher.validate), one record per trajectory:
  {"messages": [{role, content}, ...], "task_id":..., "reward":...}
Loss is masked to ASSISTANT tokens only (prompt-prefix masking via the tokenizer's
chat template). 8K context, bf16, gradient checkpointing, LoRA adapter saved to
output_dir/lora_adapter.

Run (shopsimulator env):
    python -m experiment.sft.train --config configs/sft.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenize_fn(tokenizer, max_len: int):
    """Return a map fn that adds input_ids + assistant-only labels."""

    def _chat_ids(messages: List[Dict[str, str]], add_gen: bool, cap: int = None) -> List[int]:
        # transformers 5.x: apply_chat_template(tokenize=True) returns a
        # BatchEncoding (dict-like of Encoding objs), NOT a flat int list — which
        # silently corrupts input_ids/labels. return_dict=False restores a list;
        # coerce to List[int] to be version-independent.
        kw = dict(tokenize=True, add_generation_prompt=add_gen, return_dict=False)
        if cap is not None:
            kw.update(truncation=True, max_length=cap)
        out = tokenizer.apply_chat_template(messages, **kw)
        if out and isinstance(out[0], (list, tuple)):  # batched-shape -> single conv
            out = out[0]
        return [int(x) for x in out]

    def fn(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example["messages"]
        full = _chat_ids(messages, add_gen=False, cap=max_len)
        labels = [-100] * len(full)
        prefix: List[Dict[str, str]] = []
        for m in messages:
            prefix.append(m)
            if m["role"] == "assistant":
                pre = _chat_ids(prefix[:-1], add_gen=True)
                whole = _chat_ids(prefix, add_gen=False)
                start = len(pre)
                end = min(len(whole), len(full))
                for j in range(start, end):
                    labels[j] = full[j]
                if len(whole) >= len(full):   # reached the truncation boundary
                    break
        return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}

    return fn


class PadCollator:
    """Pad input_ids + labels to the per-batch max length.

    Replaces DataCollatorForSeq2Seq: in transformers 5.x that collator calls
    tokenizer.pad(), whose strict input-type check (tokenization_utils_base.py
    ~2707) raises `type ... unknown: <class 'dict'>` on the shape it passes.
    Padding by hand avoids tokenizer.pad() entirely and is version-independent.
    """

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad = pad_token_id
        self.lpad = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            padlen = maxlen - len(ids)
            input_ids.append(ids + [self.pad] * padlen)
            attn.append([1] * len(ids) + [0] * padlen)
            labels.append(lab + [self.lpad] * padlen)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT of Qwen3-1.7B-Base.")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "sft.yaml"))
    ap.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Resume Trainer state (optimizer/scheduler/RNG) from a checkpoint directory.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]

    tok = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = load_jsonl(data_cfg["train_file"])
    ds = Dataset.from_list(records)
    ds = ds.map(build_tokenize_fn(tok, train_cfg["max_seq_length"]),
                remove_columns=ds.column_names, num_proc=4)
    print(f"sft examples: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model.config.use_cache = False
    # NOTE: do NOT call model.gradient_checkpointing_enable() here. The Trainer
    # enables it from TrainingArguments(gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False}). Calling it here
    # first with default kwargs (use_reentrant=True) double-enables and can leave
    # the model on the memory-hungry path, negating activation savings (a 1.7B
    # model peaked at ~42 GiB at batch=4 with the double-enable; it should be
    # ~12-15 GiB once checkpointing actually takes effect).

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=model_cfg["lora_r"],
        lora_alpha=model_cfg["lora_alpha"], lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["lora_target_modules"], bias="none",
    )
    model = get_peft_model(model, lora)
    # LoRA + gradient checkpointing: the frozen base needs input embeddings to
    # require grad, or checkpointed activations get no gradient (and the memory
    # savings silently fail to apply). Standard PEFT fix.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    collator = PadCollator(tok.pad_token_id, label_pad_token_id=-100)
    # Gradient checkpointing is config-driven (train.gradient_checkpointing):
    # ON saves activation memory (needed at per_device>=2 on 8K seqs) but ~2x
    # slower (recomputes the forward in backward). At per_device=1 the
    # activations fit UN-checkpointed (~25 GiB), so we turn it OFF for speed and
    # recover the recompute cost. The effective batch (per_device*grad_accum) is
    # unchanged either way, so alignment to the paper's batch=32 holds regardless.
    use_gc = train_cfg.get("gradient_checkpointing", True)
    ta_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=True,
        gradient_checkpointing=use_gc,
        report_to="none",
        dataloader_num_workers=2,
    )
    if train_cfg["save_strategy"] == "steps":
        # checkpoint every N steps: crash-recovery (resume) + mid-run eval
        ta_kwargs["save_steps"] = train_cfg.get("save_steps", 500)
    if use_gc:
        # use_reentrant=False is the mode that actually saves memory with
        # PEFT/LoRA and needs input embeddings to require grad
        # (model.enable_input_require_grads above).
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    targs = TrainingArguments(**ta_kwargs)
    # transformers 5.x renamed Trainer's `tokenizer=` kwarg to `processing_class=`
    # (passing `tokenizer=` now raises TypeError). The semantics are identical.
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator, processing_class=tok)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    out = str(Path(cfg["output_dir"]) / "lora_adapter")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("SFT adapter saved to", out)


if __name__ == "__main__":
    main()
