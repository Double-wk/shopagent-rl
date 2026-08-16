"""Final-200 eval runner for the ShopSim agent (Base / SFT / GRPO).

Runs a local vLLM model (optionally + LoRA adapter) as a shopping agent over the
final200 held-out task split, recording per-task terminal reward + strict_success,
then aggregates via eval/metrics.py. The SAME runner + task list + greedy decoding
evaluates all three checkpoints so the Base->SFT->GRPO comparison is apples-to-apples.

The multi-turn body mirrors teacher/collect.py::run_one exactly (reset ->
client.chat -> env.step -> format_observation), but with ONE shared VLLMClient
(created in __main__) instead of a per-task teacher.

Run (after sourcing the ROCm vLLM env):
    python -m experiment.eval.run_eval --model Qwen/Qwen3-1.7B-Base --tag base
    python -m experiment.eval.run_eval --model Qwen/Qwen3-1.7B-Base \
        --adapter experiment/outputs/sft_new3793/lora_adapter --tag sft

Pre-req: pack_api ShopSim env serving on SHOP_ENV_BASE_URL (the 40-env pool).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

SHOP_A = Path("/workspace/shopsimulator/shop_A")
sys.path.insert(0, str(SHOP_A))

from shop_env.client import ShopEnvClient            # noqa: E402
from shop_env.wrapper import ShopSimEnv              # noqa: E402
from shop_env.obs_format import format_observation   # noqa: E402
from shop_env import reward as R                     # noqa: E402
from eval.metrics import aggregate, print_report     # noqa: E402


def _load_sys_prompt() -> str:
    # Same PROMPT_TEMPLATE_zh SFT trained on and GRPO rolls out under -> identical framing.
    cfg = yaml.safe_load((SHOP_A / "configs" / "teacher_gpt-5.6-terra.yaml").read_text())
    return cfg["system_prompt"]


def eval_one(
    task_id: int,
    client,
    base_url: str,
    sys_prompt: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    if_persona: bool,
    reset_retries: int = 4,
    reset_backoff: float = 6.0,
) -> Dict[str, Any]:
    """Run one task to terminal (or cap) against the shared client. Per-task env."""
    env = ShopSimEnv(ShopEnvClient(base_url), max_steps=max_turns, if_persona=if_persona)
    rec: Dict[str, Any] = {
        "task_id": task_id, "ok": False, "n_steps": 0, "illegal_steps": 0,
        "reward": 0.0, "reward_detail": {}, "reached_cap": False, "strict_success": False,
    }
    try:
        obs = None
        for attempt in range(reset_retries):
            try:
                obs, _ = env.reset(task_id)
                break
            except RuntimeError as e:
                if "Unable to get available environment" not in str(e):
                    raise
                if attempt == reset_retries - 1:
                    raise
                time.sleep(reset_backoff * (attempt + 1))
            except requests.exceptions.RequestException:
                if attempt == reset_retries - 1:
                    raise
                time.sleep(reset_backoff * (attempt + 1))

        conv: List[Dict[str, str]] = [{"role": "user", "content": format_observation(obs)}]
        done = False
        for _ in range(env.max_steps):
            resp = client.chat(conv, system=sys_prompt, max_tokens=max_tokens, temperature=temperature)
            obs, _r, step_done, info = env.step(resp)
            conv.append({"role": "assistant", "content": resp})
            rec["n_steps"] += 1
            if not info.get("legal", True):
                rec["illegal_steps"] += 1
            if step_done:
                done = True
                rec["reward"] = float(info.get("reward", _r))
                rec["reward_detail"] = info.get("reward_detail", {}) or {}
                break
            conv.append({"role": "user", "content": format_observation(obs)})

        rec["reached_cap"] = not done
        rec["strict_success"] = R.strict_success(rec["reward_detail"], rec["reward"])
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        env.release()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval a model on the final200 ShopSim split.")
    ap.add_argument("--model", required=True, help="base model path/id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (SFT/GRPO); omit for Base")
    ap.add_argument("--tasks", default=str(SHOP_A / "data" / "final200.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0=greedy (reproducible eval)")
    ap.add_argument("--tag", required=True, help="label for the report row (base/sft/grpo)")
    ap.add_argument("--out", default=str(SHOP_A / "experiment" / "outputs" / "eval"))
    args = ap.parse_args()

    # VLLMClient import is heavy (vllm); do it here, under __main__ (spawn guard).
    from experiment.eval.local_client import VLLMClient

    base_url = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")
    sys_prompt = _load_sys_prompt()
    task_ids: List[int] = json.loads(Path(args.tasks).read_text())

    print(f"[eval:{args.tag}] model={args.model} adapter={args.adapter} "
          f"tasks={len(task_ids)} workers={args.workers} temp={args.temperature}")

    client = VLLMClient(model=args.model, adapter=args.adapter)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(eval_one, t, client, base_url, sys_prompt,
                      args.max_turns, args.max_tokens, args.temperature, False): t
            for t in task_ids
        }
        n_done = 0
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[task {tid}] crashed: {e!r}")
                continue
            results.append(rec)
            n_done += 1
            if n_done % 20 == 0:
                m = aggregate(results)
                print(f"  ...{n_done}/{len(task_ids)} strict_success={m['strict_success_rate']:.3f}")

    m = print_report(args.tag, results)
    m["elapsed_s"] = round(time.time() - t0, 1)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.tag}.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "model": args.model, "adapter": args.adapter, "metrics": m},
                  f, ensure_ascii=False, indent=2)
    print(f"wrote {out_jsonl}  +  {out_dir}/{args.tag}_summary.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
