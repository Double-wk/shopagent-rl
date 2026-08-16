#!/usr/bin/env python3
"""
Failure-mode decomposition of ShopSimulator eval trajectories.

Reads outputs/eval_*_full_report.json, parses the per-turn action sequence from
each conversation, classifies each trajectory into a failure mode, and decomposes
where reward is lost. Purpose: decide whether the RL bottleneck is
  (a) structural truncation / degeneration  (hit cap before buying, or rotted)
  (b) purchase quality                       (bought the wrong thing)
which forks the research direction (signal/compression vs credit-assignment).

This is the data-first diagnostic that should run BEFORE committing to any
top-down framing -- see the RBC / Budget-Aware premise checks for why.

Read-only. Prints one summary block per eval file.
"""
import json
import re
import glob
import os
from collections import Counter, defaultdict

OUT = "/workspace/shopsimulator/shopagent-rl/outputs"

# matches "Action: search[keywords]" / "Action: click[value]" (upstream PROMPT_TEMPLATE_zh)
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[([^\]]*)\]", re.I)
BUY_PAT = re.compile(r"buy\s*now", re.I)
SUCCESS_REWARD = 0.9  # loose proxy for a good purchase (reward is the shaped 4-comp sum)


def load(path):
    with open(path) as f:
        txt = f.read().strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]


def parse_actions(conversation):
    """One entry per assistant turn. ('none', '', False) = malformed/garbage turn."""
    acts = []
    for msg in conversation or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        m = ACTION_RE.search(content)
        if not m:
            acts.append(("none", "", False))
            continue
        name, arg = m.group(1).lower(), m.group(2)
        acts.append((name, arg, name == "click" and bool(BUY_PAT.search(arg))))
    return acts


def classify(rec):
    n_steps = rec.get("n_steps", 0) or 0
    illegal = rec.get("illegal_steps", 0) or 0
    purchased = bool(rec.get("purchased", False))
    reached_cap = bool(rec.get("reached_cap", False))
    reward = float(rec.get("reward", 0.0) or 0.0)
    illegal_ratio = illegal / n_steps if n_steps else 0

    if purchased and reward >= SUCCESS_REWARD:
        return "success"
    if purchased and reward < SUCCESS_REWARD:
        return "wrong_purchase"
    if reached_cap and not purchased:
        return "truncated_at_cap"
    if illegal_ratio >= 0.5 and not purchased:
        return "degenerate_illegal"
    if not purchased and not reached_cap:
        return "incomplete_nocap"
    return "other"


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*_full_report.json")))
    if not files:
        print("no *_full_report.json under", OUT)
        return
    for path in files:
        name = os.path.basename(path)
        try:
            recs = load(path)
        except Exception as e:  # noqa
            print(f"{name}: LOAD ERROR {e}")
            continue
        official = {}
        if isinstance(recs, dict):
            official = recs.get("official_metrics") or recs.get("aggregate_metrics") or {}
            recs = (
                recs.get("tasks")
                or next((v for v in recs.values() if isinstance(v, list) and v), [])
            )
        if not isinstance(recs, list) or not recs:
            print(f"{name}: no task list found, skipping (type={type(recs).__name__})")
            continue

        n = len(recs)
        modes = Counter()
        act_counts = Counter()
        buy_attempts = 0
        buy_success = 0
        cap_count = 0
        purchased_count = 0
        rewards = []
        lost = Counter()  # component -> #purchased trajectories where component == 0
        illegal_steps_total = 0
        n_steps_total = 0

        for rec in recs:
            conv = rec.get("conversation") or rec.get("messages") or []
            for a in parse_actions(conv):
                act_counts[a[0]] += 1
            modes[classify(rec)] += 1
            rewards.append(float(rec.get("reward", 0.0) or 0.0))
            if rec.get("purchased"):
                purchased_count += 1
                rd = rec.get("reward_detail") or {}
                for comp in ("r_type", "r_att", "r_option", "r_price"):
                    if float(rd.get(comp, 0.0) or 0.0) == 0.0:
                        lost[comp] += 1
            if rec.get("reached_cap"):
                cap_count += 1
            if any(a[2] for a in parse_actions(conv)):
                buy_attempts += 1
                if float(rec.get("reward", 0.0) or 0.0) >= SUCCESS_REWARD:
                    buy_success += 1
            illegal_steps_total += rec.get("illegal_steps", 0) or 0
            n_steps_total += rec.get("n_steps", 0) or 0

        mean_r = sum(rewards) / n if n else 0.0
        print("=" * 72)
        print(f"{name}   (n={n})")
        if official:
            print("  official_metrics:", json.dumps(official, ensure_ascii=False))
        print(f"  mean_reward={mean_r:.3f}  purchased={purchased_count}  reached_cap={cap_count}  "
              f"illegal_ratio={illegal_steps_total/max(n_steps_total,1):.2f}")
        print(f"  action-type counts: {dict(act_counts)}")
        print(f"  buy attempts={buy_attempts}  buy->success(reward>={SUCCESS_REWARD})={buy_success}")
        print("  failure modes:")
        for m, c in modes.most_common():
            print(f"    {m:20s} {c:4d}  ({100*c/n:5.1f}%)")
        print(f"  among purchased (n={purchased_count}), lost-reward components (=0):")
        for comp in ("r_type", "r_att", "r_option", "r_price"):
            print(f"    {comp:9s} lost in {lost[comp]:4d}")
        print()


if __name__ == "__main__":
    main()
