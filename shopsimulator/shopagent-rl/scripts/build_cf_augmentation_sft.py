#!/usr/bin/env python
"""Build the counterfactual-data-augmentation baseline from GRPO training pairs.

This is the baseline that answers "isn't the gain just from having the pairs?".
It consumes the *same* pairs the paired-objective arm trains on and turns each
side into one supervised single-turn example.  A run that SFTs on this output and
then trains with the independent objective therefore differs from the paired arm
in exactly one respect: whether the pair enters training as two imitation targets
or as an interventional preference margin between the two sides.

Why not scripts/build_paired_sft_data.py: its ``emit`` special-cases only
``option_goal_swap_natural`` on the counterfactual side and commits on every
other mechanism, which is wrong for ``option_swap`` (whose counterfactual must
select the target option, not buy).  Here the action is read from the pair's own
``allowed_actions``, which the pair builder already certified, so the target is
correct for any mechanism without per-mechanism branching.

Thought text is boilerplate keyed to the expected intent; only the ACTION is
certified, exactly as in the other paired-SFT builders.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.counterfactual_pairs import HELD_OUT_MECHANISMS  # noqa: E402

# One boilerplate rationale per canonical intent.  Price gets a numeric variant
# below because the comparison is the skill being taught.
THOUGHTS = {
    "COMMIT": "商品、规格与预算都已核对并满足需求，直接提交购买。",
    "SELECT_TARGET_OPTION": "当前已选规格不是需求的目标规格，先改选目标规格再提交。",
    "SEARCH_ALTERNATIVE": "当前商品无法同时满足需求与约束，返回搜索替代商品。",
}
NUISANCE_THOUGHT = "页面展示信息的变化与任务需求无关，已选规格和价格仍然满足，提交购买。"


def _number(value: object) -> str:
    return f"{float(value):g}"


def load_system_prompt(path: Path) -> str:
    record = json.loads(next(line for line in path.open(encoding="utf-8") if line.strip()))
    for message in record["messages"]:
        if message["role"] == "system":
            return message["content"]
    raise SystemExit(f"no system message found in {path}")


def thought_for(pair: dict, side: str, intent: str) -> str:
    itype = pair.get("intervention_type")
    if itype == "nuisance_display_note" and side == "counterfactual":
        return NUISANCE_THOUGHT
    if itype == "price_above_budget":
        # Teach the executable comparison, not a keyword.
        price = _number((pair.get(side) or {})["current_price"])
        budget = _number(pair["goal"]["price_upper"])
        if intent == "SEARCH_ALTERNATIVE":
            return (f"当前价格{price}元 > 预算上限{budget}元，已经超出预算，"
                    "不能购买，返回搜索更便宜的替代商品。")
        return (f"当前价格{price}元 <= 预算上限{budget}元，商品和规格均已满足要求，"
                "可以提交购买。")
    return THOUGHTS.get(intent, THOUGHTS["COMMIT"])


def emit(pair: dict, side: str, system_prompt: str) -> dict | None:
    state = pair.get(side) or {}
    observation = state.get("observation")
    actions = state.get("allowed_actions") or []
    intents = state.get("expected_action_intents") or []
    if not observation or not actions or not intents:
        return None
    intent = str(intents[0])
    action = str(actions[0])
    thought = thought_for(pair, side, intent)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
            {"role": "assistant", "content": f"Thought: {thought}  \nAction: {action}"},
        ],
        # Keep the baseline record schema so this file can be concatenated with
        # the trajectory SFT set into one Dataset without a column mismatch.
        "n_steps": 1,
        "reward": 1.0,
        "task_id": pair["task_id"],
        "pair_id": pair["pair_id"],
        "intervention_type": pair["intervention_type"],
        "side": side,
        "expected_action_intent": intent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path,
                    default=ROOT / "data/counterfactual/paper_grpo_pairs_v2.jsonl",
                    help="the same pairs the paired GRPO arm trains on")
    ap.add_argument("--system-from", type=Path, default=ROOT / "data/sft_train.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/sft_cf_augmentation_v2.jsonl")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="optional trajectory SFT set to concatenate for a mixture")
    ap.add_argument("--mix-out", type=Path, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    system_prompt = load_system_prompt(args.system_from)
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines()
             if line.strip()]

    # The augmentation baseline must be as blind to the held-out mechanisms as
    # the paired arm is, or it is not a controlled comparison.
    leaked = sorted({p["intervention_type"] for p in pairs} & HELD_OUT_MECHANISMS)
    if leaked:
        raise SystemExit(
            f"refusing to build: input pairs contain held-out mechanisms {leaked}. "
            "The baseline would then see states the paired arm never sees."
        )

    records: list[dict] = []
    skipped: Counter[str] = Counter()
    for pair in pairs:
        for side in ("original", "counterfactual"):
            record = emit(pair, side, system_prompt)
            if record is None:
                skipped[f"{pair['intervention_type']}:{side}"] += 1
            else:
                records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "shopsim-cf-augmentation-v2",
        "pairs_in": len(pairs),
        "records_out": len(records),
        "by_intervention": dict(Counter(r["intervention_type"] for r in records)),
        "by_intent": dict(Counter(r["expected_action_intent"] for r in records)),
        "by_side": dict(Counter(r["side"] for r in records)),
        "unique_tasks": len({r["task_id"] for r in records}),
        "skipped": dict(skipped),
        "held_out_mechanisms_excluded": sorted(HELD_OUT_MECHANISMS),
        "pairs_path": str(args.pairs),
        "out": str(args.out),
    }

    if args.baseline and args.mix_out:
        baseline_lines = [line for line in args.baseline.read_text(encoding="utf-8").splitlines()
                          if line.strip()]
        args.mix_out.parent.mkdir(parents=True, exist_ok=True)
        with args.mix_out.open("w", encoding="utf-8") as handle:
            for line in baseline_lines:
                handle.write(line + "\n")
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary["baseline_rows"] = len(baseline_lines)
        summary["mix_rows"] = len(baseline_lines) + len(records)
        summary["mix_out"] = str(args.mix_out)

    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
