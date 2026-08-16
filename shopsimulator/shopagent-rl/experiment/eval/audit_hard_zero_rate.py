"""D3 hard-zero trigger audit — see docs/price-blindness-next.md §2.

Parses a c1-hard GRPO training log and estimates how often the budget_mode=hard
terminal zeroing actually fired on policy.

Identification: in hard mode a real Buy whose weighted sum would be >= 0.2*r_type
(> 0) can only score exactly 0.0000 through the hard zero (over-budget purchase,
r_price = 0 -> reward 0). Cap/aborted rollouts also print reward=0.0000 but have
no `done=True` buy turn. So for every `summary ... reward=0.0000` line we look
back for a `turn ... done=True` line with the same task_id.

Caveat: ray collapses identical lines with a "[repeated N across cluster]"
suffix — multiplicity is multiplied in where present, but collapsed interleaving
can still attribute a buy to the wrong task_id twin. Treat the result as an
order-of-magnitude estimate, not an exact count.

Usage:
  python -m experiment.eval.audit_hard_zero_rate <training.log>
"""
from __future__ import annotations

import json
import re
import sys

_TURN = re.compile(r"task_id=(\d+) turn=\d+ .*done=True")
_SUMMARY = re.compile(
    r"task_id=(\d+) summary turns=(\d+) actions=([\w,]*) legal=\d+ "
    r"response_tokens=\d+ reward=([\d.]+)")
_REPEAT = re.compile(r"\[repeated (\d+)x across cluster\]")


def main() -> None:
    log_path = sys.argv[1]
    # per task_id: did we see a done=True turn since the last summary?
    bought = {}
    stats = {"rollouts": 0, "zero_reward": 0, "hard_zero_buys": 0, "buys_total": 0}
    by_decile = [{"n": 0, "zero": 0, "hard": 0} for _ in range(10)]

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # first pass: total rollout count for decile bucketing
    total = 0
    for line in lines:
        if _SUMMARY.search(line):
            rm = _REPEAT.search(line)
            total += int(rm.group(1)) if rm else 1

    seen = 0
    for line in lines:
        mult = 1
        if rm := _REPEAT.search(line):
            mult = int(rm.group(1))
        if tm := _TURN.search(line):
            bought[tm.group(1)] = True
            stats["buys_total"] += mult
        elif sm := _SUMMARY.search(line):
            tid, reward = sm.group(1), float(sm.group(4))
            dec = min(9, int(10 * seen / max(1, total)))
            seen += mult
            stats["rollouts"] += mult
            by_decile[dec]["n"] += mult
            if reward == 0.0:
                stats["zero_reward"] += mult
                by_decile[dec]["zero"] += mult
                if bought.get(tid):
                    stats["hard_zero_buys"] += mult
                    by_decile[dec]["hard"] += mult
            bought.pop(tid, None)

    out = {
        "log": log_path,
        "note": "hard_zero_buys is an upper-bound-ish estimate (ray line dedup + "
                "task_id reuse across steps); buy attribution may double-count",
        **stats,
        "hard_zero_rate_per_rollout": round(stats["hard_zero_buys"] / max(1, stats["rollouts"]), 5),
        "hard_zero_rate_per_buy": round(stats["hard_zero_buys"] / max(1, stats["buys_total"]), 5),
        "by_decile": [
            {"decile": i, "n": d["n"], "zero_reward": d["zero"], "hard_zero_buys": d["hard"]}
            for i, d in enumerate(by_decile)
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
