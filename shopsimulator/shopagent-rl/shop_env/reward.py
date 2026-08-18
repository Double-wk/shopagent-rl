"""Segmented reward derived from the env's reward_detail.

reward_detail is populated ONLY at terminal steps (env.done == True, i.e. the
agent clicked Buy Now). Keys come from web_agent_site/engine/goal.py
`get_reward(..., verbose=True)`:

    query_match, category_match, title_score,   # -> 品类 / type
    num_attr_matches, r_att,                    # -> 属性 / attribute
    num_option_matches, r_option,               # -> 规格 / spec (options)
    r_price                                     # -> 价格 / price
    r_type                                      # 0.5 | 1.0 multiplier

Mapping to the project's "按品类、属性、规格、价格分段 Reward":
    品类 -> r_type (and query_match / category_match / title_score)
    属性 -> r_att
    规格 -> r_option
    价格 -> r_price
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def components(detail: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return the four normalized segmented reward components in [0, 1]."""
    detail = detail or {}
    return {
        "r_type": _f(detail.get("r_type", 0.0)),      # 品类  (0.5 or 1.0)
        "r_att": _f(detail.get("r_att", 0.0)),        # 属性
        "r_option": _f(detail.get("r_option", 0.0)),  # 规格
        "r_price": _f(detail.get("r_price", 0.0)),    # 价格
    }


def strict_success(detail: Optional[Dict[str, Any]], total_reward: Optional[float] = None, eps: float = 1e-6) -> bool:
    """Strict success: every dimension fully satisfied.

    Equivalent to the env's total_reward == 1.0 (all attributes + all options
    matched, price within budget, type multiplier == 1.0).
    """
    if total_reward is not None and _f(total_reward) >= 1.0 - eps:
        return True
    c = components(detail)
    return c["r_type"] >= 1.0 and c["r_att"] >= 1.0 and c["r_option"] >= 1.0 and c["r_price"] >= 1.0


# Default per-dimension weights for the shaped (segmented) terminal reward used
# by GRPO. Tunable; attribute/spec weighted higher since they're the long-horizon
# discriminative skill the project targets.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "r_type": 0.20,
    "r_att": 0.30,
    "r_option": 0.30,
    "r_price": 0.20,
}


def shaped(
    detail: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    legal_bonus: float = 0.0,
    illegal: bool = False,
    budget_mode: str = "none",
    budget_penalty: float = 0.5,
) -> float:
    """Weighted segmented terminal reward for GRPO credit assignment.

    Args:
        detail: reward_detail dict from a terminal step (may be empty/non-terminal).
        weights: per-component weights (sum need not be 1). Defaults to DEFAULT_WEIGHTS.
        legal_bonus: small bonus added when the final action was legal.
        illegal: if True (the terminal/purchase action was illegal), return 0.
        budget_mode: reward variant. ``strict`` returns the original
            ShopSimulator multiplicative reward ``r_type * r_att *
            r_option * r_price``; ``hard``/``pen`` are C1 fixes for
            price-blind committing, motivated by the
            counterfactual probe (outputs/counterfactual, 2026-08-14:
            commit_persistence_error SFT 0.085 -> GRPO v1 0.203 -> v2b 0.519
            while price cf accuracy stays ~0 -- the weighted sum lets an
            over-budget buy keep up to 0.8 partial credit). "none" = vanilla
            weighted sum. "hard" zeroes the whole terminal reward on a
            budget-breaking buy (r_price == 0), scoring it like not buying at
            all. "pen" keeps the sum but subtracts budget_penalty.
        budget_penalty: penalty magnitude for budget_mode="pen".
    """
    if illegal:
        return 0.0
    w = weights or DEFAULT_WEIGHTS
    c = components(detail)
    if budget_mode == "strict":
        # ShopSimulator's original bottleneck reward.  Keep the empty-detail
        # case at zero (non-terminal/capped rollouts have no purchase credit).
        return (
            c["r_type"] * c["r_att"] * c["r_option"] * c["r_price"]
            if detail else 0.0
        ) + legal_bonus
    score = sum(w.get(k, 0.0) * v for k, v in c.items())
    if budget_mode != "none" and detail and c["r_price"] <= 0.0:
        # Only a real terminal buy populates reward_detail (wrapper returns {}
        # on step-cap / non-terminal), and r_price == 0 there means the
        # purchased item exceeded the budget. The `detail and` guard keeps a
        # no-buy rollout from tripping the gate (pen mode must not fine it).
        if budget_mode == "hard":
            return 0.0
        if budget_mode == "pen":
            score -= budget_penalty
    return score + legal_bonus
