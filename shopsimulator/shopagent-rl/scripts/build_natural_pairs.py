#!/usr/bin/env python
"""Generate verified natural-language constraint-swap pairs with a teacher.

Reads a v2 pairs JSONL, takes every ``option_goal_swap_structured`` pair, asks
the configured teacher to rewrite the instruction so the wanted option flips,
and keeps only rewrites that pass the programmatic gate in
``experiment.constraint_causal_natural.verify_rewrite``.  Rejected rewrites are
written out with their failure reasons for auditing — they never enter the
output pairs.

Example:
  python scripts/build_natural_pairs.py --pairs data/counterfactual/final200_constraint_causal_v2.jsonl \
    --config configs/teacher_gpt-5.6-terra.yaml \
    --endpoint-substring gptgod.online \
    --out data/counterfactual/natural_option_swap_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.constraint_causal_natural import (  # noqa: E402
    REWRITE_SYSTEM_PROMPT,
    build_natural_pair,
    build_rewrite_prompt,
    instruction_of,
    iter_structured_candidates,
    verify_rewrite,
)
from experiment.teacher.client_openai import OpenAITeacherClient  # noqa: E402


def rewrite_one(client: OpenAITeacherClient, pair: dict, attempts: int) -> dict:
    """Ask the teacher for a rewrite, retrying with failure feedback."""
    instruction = instruction_of(pair)
    intervention = pair["intervention"]
    option_a, option_b = intervention["from"], intervention["to"]
    feedback = ""
    last = None
    for attempt in range(1, attempts + 1):
        prompt = build_rewrite_prompt(instruction, option_a, option_b)
        if feedback:
            prompt += f"\n\n上一次改写被程序否决，原因：{feedback}。请严格遵守上述要求重新改写。"
        raw = client.chat([{"role": "user", "content": prompt}],
                          system=REWRITE_SYSTEM_PROMPT, max_tokens=512, temperature=0.0)
        check = verify_rewrite(instruction, raw, option_a, option_b)
        check.detail["attempt"] = attempt
        last = {"pair_id": pair["pair_id"], "rewrite": raw, "check": check}
        if check.accepted:
            return {"pair": pair, "raw": raw, "check": check}
        feedback = "；".join(check.reasons)
    return {"pair": pair, **last}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="v2 pairs JSONL")
    ap.add_argument("--config", required=True, help="teacher YAML config")
    ap.add_argument("--endpoint-substring", default="",
                    help="only use endpoints whose base_url contains this (empty = all)")
    ap.add_argument("--out", required=True, help="output natural pairs JSONL")
    ap.add_argument("--rejected-out", default="", help="optional rejected rewrites JSONL")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    args = ap.parse_args()

    v2_pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    candidates = iter_structured_candidates(v2_pairs)
    if args.limit:
        candidates = candidates[:args.limit]

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    teacher_cfg = dict(cfg["teacher"])
    if args.endpoint_substring:
        matching = [e for e in teacher_cfg.get("endpoints", [])
                    if args.endpoint_substring in e.get("base_url", "")]
        if len(matching) != 1:
            raise SystemExit(f"expected exactly one matching endpoint, got {len(matching)}")
        teacher_cfg["endpoints"] = matching
    client = OpenAITeacherClient(
        endpoints=teacher_cfg["endpoints"], model=teacher_cfg.get("model"),
        timeout=teacher_cfg.get("timeout", 120), max_retries=teacher_cfg.get("max_retries", 5),
    )
    model = teacher_cfg.get("model", "")

    accepted, rejected = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rewrite_one, client, p, args.attempts): p["pair_id"]
                   for p in candidates}
        for future in as_completed(futures):
            pair_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # teacher/network failure on one pair
                rejected.append({"pair_id": pair_id, "error": repr(exc)[:200]})
                continue
            if "check" in result and result["check"].accepted:
                accepted.append(build_natural_pair(
                    result["pair"], result["raw"], result["check"], model))
            else:
                rejected.append({
                    "pair_id": pair_id,
                    "rewrite": result.get("raw", ""),
                    "reasons": result["check"].reasons if "check" in result else ["no_result"],
                    "detail": result["check"].detail if "check" in result else {},
                })
            done = len(accepted) + len(rejected)
            print(f"[{done}/{len(candidates)}] {pair_id} "
                  f"{'accepted' if 'check' in result and result['check'].accepted else 'rejected'}",
                  flush=True)

    accepted.sort(key=lambda p: p["task_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in accepted:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.rejected_out:
        rp = Path(args.rejected_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "candidates": len(candidates), "accepted": len(accepted), "rejected": len(rejected),
        "accept_rate": round(len(accepted) / max(1, len(candidates)), 3),
        "out": str(out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
