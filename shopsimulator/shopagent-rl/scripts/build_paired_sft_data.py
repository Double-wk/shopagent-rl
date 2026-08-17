#!/usr/bin/env python
"""Build paired-constraint SFT records from natural counterfactual pairs.

For every verified natural goal-swap pair we emit three single-turn examples
that share the official system prompt and the pair's rendered observation:

* original   -> Thought + ``click[buy now]``        (satisfied  -> COMMIT)
* counterfactual -> Thought + ``click[<new option>]`` (violated -> targeted recovery)
* nuisance   -> Thought + ``click[buy now]``        (irrelevant change -> COMMIT)

Validated atomic price pairs can additionally emit an over-budget recovery
example whose Thought contains the executable numeric comparison and whose
action leaves the item page instead of buying.

The assistant targets are templates: only the ACTION is certified by the
programmatic gate; the Thought text is boilerplate consistent with the
trajectory format.  Output records are drop-in compatible with
``experiment.sft.train`` ({"messages", "task_id", ...}).

Example:
  python scripts/build_paired_sft_data.py \
    --natural data/counterfactual/train_natural_option_swap_v1.jsonl \
    --v2 data/counterfactual/train_constraint_causal_v2.jsonl \
    --system-from data/sft_train.jsonl \
    --out data/sft_paired_train.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = None  # loaded from the baseline SFT file to stay byte-identical


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_system_prompt(path: str) -> str:
    rec = json.loads(next(l for l in open(path, encoding="utf-8") if l.strip()))
    for m in rec["messages"]:
        if m["role"] == "system":
            return m["content"]
    raise SystemExit(f"no system message found in {path}")


def assistant(thought: str, action: str) -> str:
    return f"Thought: {thought}  \nAction: {action}"


def _number(value: object) -> str:
    number = float(value)
    return f"{number:g}"


def emit(pair: dict, side: str) -> dict | None:
    itype = pair.get("intervention_type")
    obs = (pair.get(side) or {}).get("observation")
    if not obs:
        return None
    if side == "counterfactual" and itype == "option_goal_swap_natural":
        new_option = pair["intervention"]["to"]
        action = f"click[{new_option}]"
        thought = f"用户需求的目标规格已改为{new_option}，当前已选规格不再满足要求，先改选目标规格。"
    else:
        # original side of any pair, and nuisance counterfactual side: commit.
        action = "click[buy now]"
        if side == "original":
            thought = "商品、规格与预算都已核对并满足需求，直接提交购买。"
        else:
            thought = "页面展示信息的变化与任务需求无关，已选规格和价格仍然满足，提交购买。"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs},
            {"role": "assistant", "content": assistant(thought, action)},
        ],
        # keep the baseline record schema so the files can be concatenated
        # into one Dataset without column mismatch
        "n_steps": 1,
        "reward": 1.0,
        "task_id": pair["task_id"],
        "pair_id": pair["pair_id"],
        "intervention_type": itype,
        "side": side,
    }


def emit_price(pair: dict, side: str = "counterfactual") -> dict | None:
    """Build a certified price-comparison target from an atomic price pair."""
    if pair.get("intervention_type") != "price_above_budget":
        return None
    state = pair.get(side) or {}
    obs = state.get("observation")
    if not obs:
        return None

    price = _number(state["current_price"])
    budget = _number(pair["goal"]["price_upper"])
    # Upstream realized goals may randomize price_upper independently of the
    # colloquial budget in instruction_text. Keep the numeric target observable
    # in this structured training control rather than leaking it only in Thought.
    # Natural heldout evaluation intentionally does not add this summary.
    obs = f"{obs}\n任务约束摘要: 预算上限={budget}元。"
    if side == "counterfactual":
        action = state.get("allowed_actions", ["click[back to search]"])[0]
        thought = (
            f"当前价格{price}元 > 预算上限{budget}元，已经超出预算，"
            "不能购买，返回搜索更便宜的替代商品。"
        )
    else:
        action = "click[buy now]"
        thought = (
            f"当前价格{price}元 <= 预算上限{budget}元，商品和规格均已满足要求，"
            "可以提交购买。"
        )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs},
            {"role": "assistant", "content": assistant(thought, action)},
        ],
        "n_steps": 1,
        "reward": 1.0,
        "task_id": pair["task_id"],
        "pair_id": pair["pair_id"],
        "intervention_type": pair["intervention_type"],
        "side": side,
    }


def main() -> None:
    global SYSTEM_PROMPT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--natural", required=True, help="natural pairs JSONL (v3-natural)")
    ap.add_argument("--v2", default="", help="v2 pairs JSONL for nuisance controls")
    ap.add_argument("--atomic", default="", help="atomic pairs JSONL for price examples")
    ap.add_argument("--system-from", default=str(ROOT / "data" / "sft_train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-verbatim", action="store_true",
                    help="skip pairs whose rewrite pasted the catalog option string")
    ap.add_argument("--limit", type=int, default=0, help="cap natural pairs (0 = all)")
    ap.add_argument("--price-limit", type=int, default=0,
                    help="cap price pairs (0 = all); E1 uses 1000-2000")
    ap.add_argument("--include-price-original", action="store_true",
                    help="also emit the within-budget side for each price pair")
    ap.add_argument("--mix-out", default="",
                    help="optional baseline+method JSONL output")
    args = ap.parse_args()

    SYSTEM_PROMPT = load_system_prompt(args.system_from)
    natural = load_jsonl(args.natural)
    if args.exclude_verbatim:
        natural = [p for p in natural
                   if not p["intervention"]["verification_detail"].get("catalog_verbatim")]
    if args.limit:
        natural = natural[:args.limit]

    records: list[dict] = []
    for p in natural:
        for side in ("original", "counterfactual"):
            rec = emit(p, side)
            if rec:
                records.append(rec)

    price_pairs = []
    if args.atomic:
        price_pairs = [
            pair for pair in load_jsonl(args.atomic)
            if pair.get("intervention_type") == "price_above_budget"
        ]
        if args.price_limit:
            price_pairs = price_pairs[:args.price_limit]
        for pair in price_pairs:
            sides = ("original", "counterfactual") if args.include_price_original \
                else ("counterfactual",)
            for side in sides:
                rec = emit_price(pair, side)
                if rec:
                    records.append(rec)

    nuisance_by_task = {}
    if args.v2:
        for p in load_jsonl(args.v2):
            if p.get("intervention_type") == "nuisance_display_note":
                nuisance_by_task.setdefault(p["task_id"], p)
    covered = 0
    for p in natural:
        n = nuisance_by_task.get(p["task_id"])
        if n:
            covered += 1
            rec = emit(n, "counterfactual")
            if rec:
                rec["pair_id"] = f"{p['task_id']}:nuisance_display_note"
                records.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "natural_pairs": len(natural), "nuisance_matched": covered,
        "price_pairs": len(price_pairs),
        "records": len(records),
        "by_side": dict(Counter(r["side"] for r in records)),
        "by_intervention_type": dict(Counter(r["intervention_type"] for r in records)),
        "out": str(out),
    }
    if args.mix_out:
        mix_out = Path(args.mix_out)
        mix_out.parent.mkdir(parents=True, exist_ok=True)
        with mix_out.open("w", encoding="utf-8") as f:
            for source in (Path(args.system_from), out):
                with source.open(encoding="utf-8") as src:
                    for line in src:
                        if line.strip():
                            f.write(line)
        summary["mix_out"] = str(mix_out)
        summary["mix_records"] = sum(1 for line in mix_out.open(encoding="utf-8") if line.strip())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
