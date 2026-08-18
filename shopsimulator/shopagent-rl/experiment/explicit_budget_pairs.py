"""Make programmatic price constraints explicit in atomic pair observations."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "shopsim-explicit-budget-pairs-v1"
BUDGET_PREFIX = "用户最终确认：总预算上限为"
BUDGET_SUFFIX = "元，并以此为准。"


def _number(value: Any) -> str:
    return f"{float(value):g}"


def _with_explicit_budget(observation: str, budget: Any) -> str:
    head, sep, tail = observation.partition("\n")
    if not sep or not head.startswith("Instruction: "):
        raise ValueError("observation is missing its Instruction line")
    clarification = f"{BUDGET_PREFIX}{_number(budget)}{BUDGET_SUFFIX}"
    return f"{head} {clarification}\n{tail}"


def make_budget_explicit(pair: dict[str, Any]) -> dict[str, Any]:
    """Expose the verifier budget identically on both sides of a price pair."""
    result = deepcopy(pair)
    result["schema_version"] = SCHEMA_VERSION
    if result.get("intervention_type") != "price_above_budget":
        return result

    goal = result.get("goal") or {}
    budget = goal.get("price_upper")
    if budget is None:
        raise ValueError(f"price pair {result.get('pair_id')} has no budget")
    derivation_source = goal.get("price_upper_source", "unknown")
    goal["price_upper_derivation_source"] = derivation_source
    goal["price_upper_source"] = "programmatic_explicit_budget"
    goal["constraint_visibility"] = "explicit_final_clarification"
    result["goal"] = goal

    for side_name in ("original", "counterfactual"):
        side = result.get(side_name) or {}
        side["observation"] = _with_explicit_budget(str(side.get("observation") or ""), budget)
        result[side_name] = side
    return result


def validate_explicit_budget_pair(pair: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if pair.get("schema_version") != SCHEMA_VERSION:
        errors.append("unexpected schema_version")
    if pair.get("intervention_type") != "price_above_budget":
        return errors

    goal = pair.get("goal") or {}
    budget = goal.get("price_upper")
    expected = f"{BUDGET_PREFIX}{_number(budget)}{BUDGET_SUFFIX}"
    if goal.get("price_upper_source") != "programmatic_explicit_budget":
        errors.append("budget is not marked explicit")
    for side_name in ("original", "counterfactual"):
        observation = str((pair.get(side_name) or {}).get("observation") or "")
        if observation.count(expected) != 1:
            errors.append(f"{side_name} does not expose the exact budget once")
    original = pair.get("original") or {}
    counterfactual = pair.get("counterfactual") or {}
    if float(original.get("current_price", float("inf"))) > float(budget):
        errors.append("original price exceeds explicit budget")
    if float(counterfactual.get("current_price", float("-inf"))) <= float(budget):
        errors.append("counterfactual price does not exceed explicit budget")
    return errors
