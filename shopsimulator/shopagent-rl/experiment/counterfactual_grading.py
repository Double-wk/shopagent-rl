"""Shared action grading for counterfactual evaluation and training."""
from __future__ import annotations

import re
from typing import Any

from shop_env.wrapper import parse_model_action


_ACTION_RE = re.compile(r"^(search|click)\s*\[(.*?)\]$", re.IGNORECASE)


def parse_allowed(action_str: str) -> tuple[str, str] | None:
    match = _ACTION_RE.match(str(action_str).strip())
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def predicted_intent(action: Any) -> str | None:
    """Map a parsed action to a coarse, intervention-level intent.

    This deliberately ignores the clicked option value. Exact option/value
    correctness remains the job of ``correct_strict``; the coarse intent is
    used only to verify the relation between the two sides of a pair.
    """
    if action is None:
        return None
    action_type = str(getattr(action, "type", "")).lower()
    value = str(getattr(action, "value", "")).strip().lower()
    if action_type == "click" and value == "buy now":
        return "COMMIT"
    if action_type == "search":
        return "SEARCH_ALTERNATIVE"
    if action_type == "click" and value in {"back to search", "< prev"}:
        return "SEARCH_ALTERNATIVE"
    if action_type == "click":
        return "SELECT_TARGET_OPTION"
    return None


def grade_response(response: str, side: dict[str, Any]) -> dict[str, Any]:
    """Grade one response against a validated counterfactual side."""
    action = parse_model_action(response)
    allowed = [parse_allowed(value) for value in side.get("allowed_actions", [])]
    allowed = [value for value in allowed if value is not None]
    intents = [str(value) for value in side.get("expected_action_intents", [])]

    result: dict[str, Any] = {
        "action": f"{action.type}[{action.value}]" if action else None,
        "action_type": action.type if action else None,
        "is_commit": bool(
            action
            and action.type == "click"
            and action.value.strip().lower() == "buy now"
        ),
        "intents": intents,
        "predicted_intent": predicted_intent(action),
    }
    if action is None:
        result.update(correct_strict=False, correct_lenient=False, side_correct=False, unparseable=True)
        return result

    result["unparseable"] = False
    result["correct_strict"] = any(
        action.type == action_type
        and action.value.strip().lower() == action_value.lower()
        for action_type, action_value in allowed
    )
    correct_lenient = result["correct_strict"]
    if not correct_lenient and "SEARCH_ALTERNATIVE" in intents and action.type == "search":
        correct_lenient = True
    result["correct_lenient"] = correct_lenient
    result["side_correct"] = bool(result["correct_strict"])
    return result
