"""D1 comparison-capability probe — see docs/price-blindness-next.md §2.

Isolates the numeric-comparison step from the shopping task. For every
price_above_budget pair (and its within-budget original side), the SAME
(price, budget) numbers are asked in two forms:

  arithmetic:  "1575 是否不超过 1300？只需回答 是 或 否 。"
  shopping:    "商品价格是1575元，我的预算是1300元，可以购买吗？只需回答 可以 或 不可以 。"

Expected answer: within budget (original side) -> 是/可以;
over budget (counterfactual side) -> 否/不可以.

If a checkpoint answers the arithmetic form correctly but fails the shopping
form, the failure is contextual prior suppression (E2); if both fail, the
1.7B lacks the comparison itself (E1/E3).

Run via scripts/run_comparison_probe.sh (one checkpoint per invocation).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiment.eval.run_counterfactual import load_system_prompt  # noqa: E402

_NEG_POS = {
    # negative markers must be checked before their positive substrings
    "否": ["不可以", "不能买", "不是", "否", "超过", "不行", "买不了"],
    "是": ["可以", "能买", "是", "符合", "能购买"],
}


def grade_answer(text: str) -> str:
    """Return the earliest-occurring verdict marker's polarity, or 'unparseable'."""
    best, best_pos = None, None
    for polarity, markers in _NEG_POS.items():
        for m in markers:
            pos = text.find(m)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best, best_pos = polarity, pos
    return best or "unparseable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", default=str(_ROOT / "data" / "counterfactual" / "final200_atomic_pairs_v1.jsonl"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--system_prompt", default=str(_ROOT / "configs" / "teacher_gpt-5.6-terra.yaml"))
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    ap.add_argument("--wave", type=int, default=128)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    pairs = [p for p in pairs if p["intervention_type"] == "price_above_budget"]

    sys_prompt = load_system_prompt(args.system_prompt)

    questions = []   # (meta, user_text)
    for p in pairs:
        budget = p["goal"]["price_upper"]
        for side_name in ("original", "counterfactual"):
            price = p[side_name]["current_price"]
            within = price <= budget
            meta = {"pair_id": p["pair_id"], "side": side_name, "price": price,
                    "budget": budget, "within": within}
            questions.append((
                {**meta, "form": "arithmetic"},
                f"{price:g} 是否不超过 {budget:g}？只需回答 是 或 否 。",
            ))
            questions.append((
                {**meta, "form": "shopping"},
                f"商品价格是{price:g}元，我的预算是{budget:g}元，可以购买吗？只需回答 可以 或 不可以 。",
            ))

    print(f"[D1 {args.tag}] {len(pairs)} pairs x 2 sides x 2 forms = {len(questions)} questions | "
          f"adapter={args.adapter or '(none)'}")

    from experiment.eval.local_vllm import LocalVLLM
    engine = LocalVLLM(args.model, adapter_path=args.adapter,
                       gpu_memory_utilization=args.gpu_memory_utilization)

    responses = []
    for wi in range(0, len(questions), args.wave):
        wave = questions[wi:wi + args.wave]
        batch = [[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": q[1]}] for q in wave]
        responses.extend(engine.generate_batch(
            batch, max_tokens=args.max_tokens, temperature=args.temperature))

    records = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for (meta, _q), resp in zip(questions, responses):
            verdict = grade_answer(resp)
            correct = (verdict == ("是" if meta["within"] else "否"))
            rec = {**meta, "response": resp, "verdict": verdict, "correct": correct}
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def acc(sub):
        return round(sum(r["correct"] for r in sub) / len(sub), 4) if sub else None

    summary = {"tag": args.tag, "adapter": args.adapter, "n_questions": len(records)}
    for form in ("arithmetic", "shopping"):
        for side in ("original", "counterfactual"):
            sub = [r for r in records if r["form"] == form and r["side"] == side]
            summary[f"{form}/{side}"] = {"n": len(sub), "acc": acc(sub),
                                         "unparseable": sum(r["verdict"] == "unparseable" for r in sub)}
    metrics_path = str(Path(args.out).with_name(Path(args.out).stem + "_metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as fm:
        json.dump(summary, fm, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
