"""Derive v2 constraint-causal pairs from validated atomic commitment states.

The v1 builder changes product state (wrong option / over-budget price).  This
module adds two controls without inventing product facts:

* ``option_goal_swap`` changes only the user-requested option while keeping the
  selected product option and price fixed; and
* ``nuisance_display_note`` changes presentation-only text, for which the
  intended action must remain ``COMMIT``.

Exact, one-occurrence option mentions produce the natural-language subset.  A
separately labelled structured-summary subset covers semantically expressed
options without pretending that it is natural-language generalisation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA_VERSION_V2 = "shopsim-constraint-causal-pairs-v2"


@dataclass(frozen=True)
class BuildResult:
    pairs: list[dict[str, Any]]
    stats: dict[str, int]


def _instruction_line(observation: str) -> tuple[str, str]:
    head, sep, tail = observation.partition("\n")
    if not sep or not head.startswith("Instruction: "):
        return "", observation
    return head[len("Instruction: "):], tail


def _with_instruction(observation: str, instruction: str) -> str:
    _old, tail = _instruction_line(observation)
    return f"Instruction: {instruction}\n{tail}"


def _with_constraint_summary(observation: str, option: str, budget: Any) -> str:
    """Append the same verifier-derived constraint summary to both worlds."""
    return observation + f"\n任务约束摘要: 目标规格={option}；预算上限={budget}元。"


def build_v2_pairs(v1_pairs: Iterable[dict[str, Any]]) -> BuildResult:
    """Create bidirectional-goal and nuisance controls from v1 option pairs."""
    pairs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_tasks: set[int] = set()

    for source in v1_pairs:
        stats["records_seen"] += 1
        if source.get("intervention_type") != "option_swap":
            continue
        original = source.get("original") or {}
        broken = source.get("counterfactual") or {}
        goal = source.get("goal") or {}
        target = (goal.get("goal_options") or [None])[0]
        alternative = (broken.get("selected_options") or {}).get(
            (source.get("product") or {}).get("option_group")
        )
        observation = str(original.get("observation") or "")
        instruction, _tail = _instruction_line(observation)
        if not isinstance(target, str) or not isinstance(alternative, str) or not instruction:
            stats["rejected_missing_option_or_instruction"] += 1
            continue
        # Replace exactly once: this is the validity gate for a true user-side
        # intervention, rather than an approximate string-generation heuristic.
        natural_rewrite = target in instruction and instruction.count(target) == 1 and alternative not in instruction
        if natural_rewrite:
            pair_original = original
            swapped_observation = _with_instruction(
                observation, instruction.replace(target, alternative, 1)
            )
            intervention_type = "option_goal_swap"
            checks = {
                "same_product_state": True,
                "same_selected_options": True,
                "same_price": original.get("current_price") == broken.get("current_price"),
                "single_exact_instruction_rewrite": True,
            }
            stats["pairs_option_goal_swap"] += 1
        else:
            # Coverage fallback for the common case where the user describes a
            # specification semantically rather than copying its catalog value.
            # It is explicitly separated in metadata and reports; it is useful
            # for controlled training/diagnosis, not evidence of natural-text
            # generalisation.
            pair_original = {
                **original,
                "observation": _with_constraint_summary(
                    observation, target, goal.get("price_upper")
                ),
            }
            swapped_observation = _with_constraint_summary(
                observation, alternative, goal.get("price_upper")
            )
            intervention_type = "option_goal_swap_structured"
            checks = {
                "same_product_state": True,
                "same_selected_options": True,
                "same_price": original.get("current_price") == broken.get("current_price"),
                "same_constraint_summary_format": True,
                "only_target_option_changes": True,
            }
            stats["pairs_option_goal_swap_structured"] += 1
        shared = {
            "schema_version": SCHEMA_VERSION_V2,
            "task_id": source["task_id"],
            "source_pair_id": source["pair_id"],
            "source": source.get("source", {}),
            "goal": source["goal"],
            "product": source.get("product", {}),
            "original": pair_original,
        }
        pairs.append({
            **shared,
            "pair_id": f"{source['task_id']}:{intervention_type}",
            "intervention_type": intervention_type,
            "intervention": {
                "field": "instruction.goal_option",
                "from": target,
                "to": alternative,
                "validity_checks": checks,
            },
            "counterfactual": {
                **pair_original,
                "observation": swapped_observation,
                "expected_action_intents": ["SELECT_TARGET_OPTION"],
                "allowed_actions": [f"click[{alternative}]"],
            },
        })

        # One nuisance control per task avoids making presentation variants
        # dominate the benchmark.  It preserves all executable state and only
        # appends a clearly non-task display note.
        task_id = source["task_id"]
        if task_id not in seen_tasks:
            seen_tasks.add(task_id)
            nuisance = {
                **shared,
                "pair_id": f"{task_id}:nuisance_display_note",
                "intervention_type": "nuisance_display_note",
                "intervention": {
                    "field": "display_note",
                    "from": "页面主题=蓝色",
                    "to": "页面主题=红色",
                    "validity_checks": {
                        "same_product_state": True,
                        "same_instruction": True,
                        "same_price": True,
                        "irrelevant_to_goal": True,
                    },
                },
                "counterfactual": {
                    **pair_original,
                    "observation": pair_original["observation"] + "\n展示备注: 页面主题色为红色。",
                },
            }
            pairs.append(nuisance)
            stats["pairs_nuisance_display_note"] += 1

    stats["pairs_total"] = len(pairs)
    stats["tasks_with_pairs"] = len({p["task_id"] for p in pairs})
    return BuildResult(pairs=pairs, stats=dict(sorted(stats.items())))


def validate_v2_pair(pair: dict[str, Any]) -> list[str]:
    """Validate v2-specific directional and invariance controls."""
    errors: list[str] = []
    if pair.get("schema_version") != SCHEMA_VERSION_V2:
        return ["unexpected schema_version"]
    original, cf = pair.get("original") or {}, pair.get("counterfactual") or {}
    checks = (pair.get("intervention") or {}).get("validity_checks") or {}
    if not checks or not all(value is True for value in checks.values()):
        errors.append("intervention validity check failed")
    itype = pair.get("intervention_type")
    if itype in {"option_goal_swap", "option_goal_swap_structured"}:
        if original.get("selected_options") != cf.get("selected_options"):
            errors.append("goal swap changed product state")
        if original.get("current_price") != cf.get("current_price"):
            errors.append("goal swap changed price")
        if cf.get("expected_action_intents") != ["SELECT_TARGET_OPTION"]:
            errors.append("goal swap missing recovery intent")
    elif itype == "nuisance_display_note":
        if original.get("expected_action_intents") != cf.get("expected_action_intents"):
            errors.append("nuisance changed intended action")
        if original.get("allowed_actions") != cf.get("allowed_actions"):
            errors.append("nuisance changed allowed actions")
    else:
        errors.append("unknown intervention_type")
    if original.get("observation") == cf.get("observation"):
        errors.append("observations are identical")
    return errors
