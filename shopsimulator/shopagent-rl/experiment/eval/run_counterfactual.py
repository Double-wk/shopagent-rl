"""Counterfactual pair evaluation (phase 1) — see docs/counterfactual-eval.md.

Single-turn probe: for each atomic-constraint pair, the stored `observation`
(the state definition, byte-exact from the build artifact) is presented as the
ONLY user message; the model's next action is graded against the pair's
`allowed_actions` / `expected_action_intents`.

No environment or pack_api involvement — states are pre-rendered on disk, so
this runner is pure vLLM inference + offline grading.

Grading levels:
  * strict   — parsed action matches one of the side's `allowed_actions`
               (case-insensitive, same matching rule the env wrapper uses).
  * lenient  — additionally accepts `search[...]` for SEARCH_ALTERNATIVE sides
               (a fresh search is also a valid way to leave an over-budget item,
               but the pair spec only lists the two nav clicks).

Reported per intervention type and overall:
  original_action_accuracy, counterfactual_action_accuracy,
  paired_robust_accuracy (both sides strict-correct),
  commit_persistence_error (constraint broken yet still `click[buy now]`).

Run via scripts/run_counterfactual_eval.sh (sources the local ROCm/vLLM env).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment.counterfactual_grading import (  # noqa: E402
    grade_response as grade,
    parse_allowed,
)


def load_system_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _pct(sub: List[Dict[str, Any]], key: str) -> float:
        return round(sum(r[key] for r in sub) / len(sub), 4) if sub else None

    by_type: Dict[str, Any] = {}
    for itype in sorted({r["intervention_type"] for r in records}):
        sub = [r for r in records if r["intervention_type"] == itype]
        originally_correct = [r for r in sub if r["orig_correct"]]
        certified = [r for r in originally_correct if r["robust"]]
        entry: Dict[str, Any] = {
            "n_pairs": len(sub),
            "original_action_accuracy": _pct(sub, "orig_correct"),
            "counterfactual_action_accuracy": _pct(sub, "cf_correct"),
            "counterfactual_action_accuracy_lenient": _pct(sub, "cf_correct_lenient"),
            "paired_robust_accuracy": _pct(sub, "robust"),
            "commit_persistence_error": _pct(
                [r for r in sub if r["cf_intent"] != "COMMIT"], "cf_commit"),
            "orig_action_types": dict(Counter(
                r["orig_action_type"] or "unparseable" for r in sub)),
            "cf_action_types": dict(Counter(
                r["cf_action_type"] or "unparseable" for r in sub)),
        }
        # how often the model repeats its original action verbatim on the
        # counterfactual side (behavioural insensitivity to the intervention)
        entry["same_action_both_sides"] = _pct(sub, "same_action")
        # Condition on a correct original commitment: otherwise poor baseline
        # competence would be mistaken for a failure of constraint sensitivity.
        entry["causal_success_certification_rate"] = (
            round(len(certified) / len(originally_correct), 4)
            if originally_correct else None
        )
        entry["shortcut_success_rate"] = (
            round((len(originally_correct) - len(certified)) / len(originally_correct), 4)
            if originally_correct else None
        )
        by_type[itype] = entry

    originally_correct = [r for r in records if r["orig_correct"]]
    certified = [r for r in originally_correct if r["robust"]]
    return {
        "n_pairs": len(records),
        "original_action_accuracy": _pct(records, "orig_correct"),
        "counterfactual_action_accuracy": _pct(records, "cf_correct"),
        "counterfactual_action_accuracy_lenient": _pct(records, "cf_correct_lenient"),
        "paired_robust_accuracy": _pct(records, "robust"),
        "causal_success_certification_rate": (
            round(len(certified) / len(originally_correct), 4)
            if originally_correct else None
        ),
        "shortcut_success_rate": (
            round((len(originally_correct) - len(certified)) / len(originally_correct), 4)
            if originally_correct else None
        ),
        "commit_persistence_error": _pct(
            [r for r in records if r["cf_intent"] != "COMMIT"], "cf_commit"),
        "by_intervention_type": by_type,
        # budget-source stratification (docs/counterfactual-eval.md: observed
        # vs canonical budgets should be reported separately)
        "by_price_upper_source": {
            src: {
                "n_pairs": len(sub),
                "counterfactual_action_accuracy": _pct(sub, "cf_correct"),
                "commit_persistence_error": _pct(sub, "cf_commit"),
            }
            for src in sorted({r["price_upper_source"] for r in records})
            for sub in [[r for r in records if r["price_upper_source"] == src]]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", default=str(_ROOT / "data" / "counterfactual" / "final200_atomic_pairs_v1.jsonl"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="output JSONL path (metrics go next to it)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (SFT/GRPO); omit for Base")
    ap.add_argument("--variant", default="base",
                    choices=["base", "salience", "instruct", "summary"],
                    help="D2 A/B variants (docs/price-blindness-next.md §2): base = unchanged; "
                         "salience = append the pair's own price/budget as adjacent final lines "
                         "(information-equivalent, no verdict); instruct = add a generic "
                         "compare-before-buy rule to the system prompt (no per-pair leak); "
                         "summary = append the exact structured budget line used by certified SFT "
                         "(diagnostic only, not a natural-input metric)")
    ap.add_argument("--system_prompt", default=str(_ROOT / "configs" / "teacher_gpt-5.6-terra.yaml"))
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.35,
                    help="lower than Final-200 default 0.80: shared GPU, single-turn probe")
    ap.add_argument("--wave", type=int, default=64)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    print(f"[cf-eval {args.tag}] {len(pairs)} pairs ({len(pairs)*2} prompts) | "
          f"adapter={args.adapter or '(none)'}")

    sys_prompt = load_system_prompt(args.system_prompt)
    if args.variant == "instruct":
        sys_prompt += ("\n购买前必须先比较当前价格与预算上限：当商品价格超过预算时，"
                       "绝对不能点击购买，必须放弃该商品并重新搜索更便宜的替代品。")

    def build_user(side_pair: Dict[str, Any], side: Dict[str, Any]) -> str:
        text = side["observation"]
        if args.variant == "salience":
            budget = side_pair["goal"]["price_upper"]
            text += (f"\n\n当前价格: {side['current_price']:g} 元\n"
                     f"预算上限: {budget:g} 元")
        elif args.variant == "summary":
            budget = side_pair["goal"]["price_upper"]
            text += f"\n任务约束摘要: 预算上限={budget:g}元。"
        return text

    from experiment.eval.local_vllm import LocalVLLM
    engine = LocalVLLM(args.model, adapter_path=args.adapter,
                       gpu_memory_utilization=args.gpu_memory_utilization)

    sides: List[Dict[str, Any]] = []   # flat: 2 entries per pair
    for p in pairs:
        for name in ("original", "counterfactual"):
            sides.append({"pair": p, "side_name": name, "side": p[name]})

    responses: List[str] = []
    for wi in range(0, len(sides), args.wave):
        wave = sides[wi:wi + args.wave]
        batch = [[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": build_user(s["pair"], s["side"])}] for s in wave]
        responses.extend(engine.generate_batch(
            batch, max_tokens=args.max_tokens, temperature=args.temperature))
        print(f"  wave {wi // args.wave + 1}/{(len(sides) + args.wave - 1) // args.wave} done")

    # grade + write per-side records; join the two sides per pair for metrics
    by_pair: Dict[str, Dict[str, Any]] = {}
    with open(args.out, "w", encoding="utf-8") as fout:
        for s, resp in zip(sides, responses):
            g = grade(resp, s["side"])
            fout.write(json.dumps({
                "pair_id": s["pair"]["pair_id"],
                "task_id": s["pair"]["task_id"],
                "intervention_type": s["pair"]["intervention_type"],
                "side": s["side_name"],
                "response": resp,
                **g,
            }, ensure_ascii=False) + "\n")
            slot = by_pair.setdefault(s["pair"]["pair_id"], {
                "pair_id": s["pair"]["pair_id"],
                "intervention_type": s["pair"]["intervention_type"],
                "price_upper_source": s["pair"]["goal"]["price_upper_source"],
                "cf_intent": s["pair"]["counterfactual"]["expected_action_intents"][0],
            })
            prefix = "orig_" if s["side_name"] == "original" else "cf_"
            slot[f"{prefix}correct"] = g["correct_strict"]
            slot[f"{prefix}correct_lenient"] = g["correct_lenient"]
            slot[f"{prefix}action"] = g["action"]
            slot[f"{prefix}action_type"] = g["action_type"]
            slot[f"{prefix}commit"] = g["is_commit"]

    graded = list(by_pair.values())
    for r in graded:
        r["cf_commit"] = r["cf_commit"] and r["cf_intent"] != "COMMIT"
        r["robust"] = r["orig_correct"] and r["cf_correct"]
        r["same_action"] = r["orig_action"] == r["cf_action"]

    metrics = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": args.pairs,
        **aggregate(graded),
    }
    out_metrics = args.out[:-len(".jsonl")] + "_metrics.json" if args.out.endswith(".jsonl") \
        else args.out + "_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}\nwrote {out_metrics}")


if __name__ == "__main__":
    main()
