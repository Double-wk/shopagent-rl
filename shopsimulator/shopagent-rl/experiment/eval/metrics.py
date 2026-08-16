"""Aggregate eval metrics over a list of run results.

Each run result (from an agent runner) is a dict with at least:
  {task_id, done, reward, reward_detail, n_steps, illegal_steps}
Reports the strict-success rate (the project's headline metric, matching the
ShopSimulator paper) plus per-dimension segmented reward means, avg steps, and
action-legality, for direct Base / SFT / GRPO comparison.

The headline metric is strict success rate; report only measured results.
"""
from __future__ import annotations

from typing import Any, Dict, List

from shop_env import reward as R


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    n_success = sum(
        R.strict_success(r.get("reward_detail", {}), r.get("reward", 0.0)) for r in results
    )
    comps = [R.components(r.get("reward_detail", {})) for r in results]
    mean = lambda k: sum(c[k] for c in comps) / n
    steps = [r.get("n_steps", 0) for r in results]
    illegal = [r.get("illegal_steps", 0) for r in results]
    return {
        "n": n,
        "strict_success_rate": n_success / n,
        "n_success": n_success,
        "r_type": mean("r_type"),       # 品类
        "r_att": mean("r_att"),         # 属性
        "r_option": mean("r_option"),   # 规格
        "r_price": mean("r_price"),     # 价格
        "avg_steps": sum(steps) / n,
        "avg_illegal_steps": sum(illegal) / n,
    }


def print_report(tag: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = aggregate(results)
    print(f"[{tag}] n={m['n']} strict_success={m.get('strict_success_rate', 0):.3f} "
          f"(r_type={m.get('r_type',0):.2f} r_att={m.get('r_att',0):.2f} "
          f"r_option={m.get('r_option',0):.2f} r_price={m.get('r_price',0):.2f}) "
          f"avg_steps={m.get('avg_steps',0):.1f}")
    return m
