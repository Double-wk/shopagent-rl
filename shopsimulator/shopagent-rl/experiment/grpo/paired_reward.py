"""Pair-level certified reward coupling for intervention GRPO."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch


def _matched_pairs(
    relation_ids: Sequence[Any],
    sides: Sequence[Any],
    rollout_ids: Sequence[Any] | None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Index original/counterfactual rows by relation and rollout id."""
    grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    side_occurrence: dict[tuple[str, str], int] = defaultdict(int)
    for index, (relation_id, side) in enumerate(zip(relation_ids, sides, strict=True)):
        relation_id = str(relation_id or "")
        side = str(side or "")
        if not relation_id or side not in {"original", "counterfactual"}:
            continue
        if rollout_ids is None:
            key = (relation_id, side)
            rollout_id = str(side_occurrence[key])
            side_occurrence[key] += 1
        else:
            rollout_id = str(rollout_ids[index])
        grouped[relation_id][side][rollout_id] = index
    return grouped


def add_joint_certified_bonus(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    relation_ids: Sequence[Any],
    sides: Sequence[Any],
    rollout_ids: Sequence[Any] | None = None,
    *,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Add a bonus when matched original/CF rollouts are both certified.

    Each pair's original and counterfactual rollouts are matched by rollout ID
    (or occurrence order). The joint score is the minimum of the two scalar
    rewards, so partial/lenient credit cannot exceed the weaker side.
    """
    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("reward and response-mask tensors must have the same 2-D shape")
    batch_size = token_level_rewards.shape[0]
    if len(relation_ids) != batch_size or len(sides) != batch_size:
        raise ValueError("pair metadata length does not match reward batch")
    if rollout_ids is not None and len(rollout_ids) != batch_size:
        raise ValueError("rollout metadata length does not match reward batch")
    if weight < 0:
        raise ValueError("paired reward weight must be non-negative")

    scores = token_level_rewards.sum(dim=-1)
    grouped = _matched_pairs(relation_ids, sides, rollout_ids)

    result = token_level_rewards.clone()
    matched_rollouts = 0
    joint_successes = 0
    bonus_total = 0.0
    complete_relations = 0
    for by_side in grouped.values():
        original = by_side.get("original", {})
        counterfactual = by_side.get("counterfactual", {})
        common = sorted(set(original) & set(counterfactual))
        if not common:
            continue
        complete_relations += 1
        for rollout_id in common:
            original_index = original[rollout_id]
            counterfactual_index = counterfactual[rollout_id]
            joint = torch.minimum(scores[original_index], scores[counterfactual_index]).clamp_min(0)
            bonus = joint * weight
            matched_rollouts += 1
            joint_successes += int(float(joint) > 0)
            bonus_total += float(bonus)
            for row_index in (original_index, counterfactual_index):
                valid = torch.nonzero(response_mask[row_index] > 0, as_tuple=False).flatten()
                if len(valid):
                    result[row_index, int(valid[-1])] += bonus

    stats: dict[str, float | int] = {
        "complete_relations": complete_relations,
        "matched_rollouts": matched_rollouts,
        "joint_successes": joint_successes,
        "joint_success_rate": joint_successes / matched_rollouts if matched_rollouts else 0.0,
        "mean_joint_bonus": bonus_total / matched_rollouts if matched_rollouts else 0.0,
    }
    return result, stats


def add_relational_advantage(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_rewards: torch.Tensor,
    relation_ids: Sequence[Any],
    sides: Sequence[Any],
    rollout_ids: Sequence[Any] | None = None,
    *,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Couple both sides of an intervention pair with a signed utility.

    Unlike ``add_joint_certified_bonus``, this is applied after ordinary GRPO
    advantages are computed. A failed relation receives a negative signal, so
    a policy that emits the same wrong action for every rollout is not silently
    treated as neutral. The utility is deliberately based on certified side
    rewards for this first implementation; intent-level relation certificates
    can be added without changing the trainer interface.
    """
    if advantages.shape != response_mask.shape or token_level_rewards.shape != advantages.shape:
        raise ValueError("advantages, rewards, and response mask must have the same shape")
    if len(relation_ids) != len(advantages) or len(sides) != len(advantages):
        raise ValueError("pair metadata length does not match advantage batch")
    if rollout_ids is not None and len(rollout_ids) != len(advantages):
        raise ValueError("rollout metadata length does not match advantage batch")
    if weight < 0:
        raise ValueError("relational advantage weight must be non-negative")

    grouped = _matched_pairs(relation_ids, sides, rollout_ids)
    pair_rows: list[tuple[int, int, float]] = []
    for by_side in grouped.values():
        original = by_side.get("original", {})
        counterfactual = by_side.get("counterfactual", {})
        for rollout_id in sorted(set(original) & set(counterfactual)):
            oi, ci = original[rollout_id], counterfactual[rollout_id]
            original_ok = float(token_level_rewards[oi].sum()) > 0.0
            counterfactual_ok = float(token_level_rewards[ci].sum()) > 0.0
            if original_ok and counterfactual_ok:
                utility = 1.0
            elif original_ok and not counterfactual_ok:
                utility = -1.0
            elif counterfactual_ok:
                utility = 0.0
            else:
                utility = -0.5
            pair_rows.append((oi, ci, utility))

    result = advantages.clone()
    if not pair_rows:
        return result, {
            "complete_relations": 0,
            "matched_rollouts": 0,
            "positive_relations": 0,
            "negative_relations": 0,
            "mean_relation_utility": 0.0,
        }

    utilities = torch.tensor([row[2] for row in pair_rows], dtype=advantages.dtype, device=advantages.device)
    # Preserve a non-zero signed signal when a batch contains only failures or
    # only successes; standardize only when there is actual variation.
    if len(utilities) > 1 and float(utilities.std(unbiased=False)) > 1e-6:
        utilities = (utilities - utilities.mean()) / (utilities.std(unbiased=False) + 1e-6)

    for (oi, ci, _), utility in zip(pair_rows, utilities, strict=True):
        for row_index in (oi, ci):
            result[row_index] += utility * weight * response_mask[row_index]

    return result, {
        "complete_relations": len({str(r or "") for r in relation_ids if str(r or "")}),
        "matched_rollouts": len(pair_rows),
        "positive_relations": sum(float(row[2]) > 0 for row in pair_rows),
        "negative_relations": sum(float(row[2]) < 0 for row in pair_rows),
        "mean_relation_utility": float(torch.tensor([row[2] for row in pair_rows]).mean()),
    }


def add_relational_residual_advantage(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_rewards: torch.Tensor,
    relation_ids: Sequence[Any],
    sides: Sequence[Any],
    rollout_ids: Sequence[Any] | None = None,
    *,
    weight: float = 1.0,
    success_bonus: float = 0.25,
    failure_penalty: float = 1.0,
    joint_failure_penalty: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Add an asymmetric residual that repairs only the failing pair side.

    Ordinary GRPO retains the per-side task signal. This residual adds a small
    shared preservation signal when the certified relation is satisfied. If
    exactly one side fails, only that side is penalized; the correct side is
    left unchanged. This avoids the v1 prototype's failure mode of suppressing
    a correct original action because its counterfactual partner was wrong.
    """
    if advantages.shape != response_mask.shape or token_level_rewards.shape != advantages.shape:
        raise ValueError("advantages, rewards, and response mask must have the same shape")
    if len(relation_ids) != len(advantages) or len(sides) != len(advantages):
        raise ValueError("pair metadata length does not match advantage batch")
    if rollout_ids is not None and len(rollout_ids) != len(advantages):
        raise ValueError("rollout metadata length does not match advantage batch")
    if min(weight, success_bonus, failure_penalty, joint_failure_penalty) < 0:
        raise ValueError("relational residual weights must be non-negative")

    grouped = _matched_pairs(relation_ids, sides, rollout_ids)
    result = advantages.clone()
    matched_rollouts = 0
    joint_successes = 0
    original_failures = 0
    counterfactual_failures = 0
    joint_failures = 0
    residual_total = 0.0

    for by_side in grouped.values():
        original = by_side.get("original", {})
        counterfactual = by_side.get("counterfactual", {})
        for rollout_id in sorted(set(original) & set(counterfactual)):
            oi, ci = original[rollout_id], counterfactual[rollout_id]
            original_ok = float(token_level_rewards[oi].sum()) > 0.0
            counterfactual_ok = float(token_level_rewards[ci].sum()) > 0.0
            matched_rollouts += 1

            if original_ok and counterfactual_ok:
                original_residual = counterfactual_residual = success_bonus
                joint_successes += 1
            elif original_ok:
                original_residual, counterfactual_residual = 0.0, -failure_penalty
                counterfactual_failures += 1
            elif counterfactual_ok:
                original_residual, counterfactual_residual = -failure_penalty, 0.0
                original_failures += 1
            else:
                original_residual = counterfactual_residual = -joint_failure_penalty
                joint_failures += 1

            for row_index, residual in (
                (oi, original_residual),
                (ci, counterfactual_residual),
            ):
                result[row_index] += residual * weight * response_mask[row_index]
                residual_total += residual * weight

    return result, {
        "complete_relations": sum(
            bool(by_side.get("original")) and bool(by_side.get("counterfactual"))
            for by_side in grouped.values()
        ),
        "matched_rollouts": matched_rollouts,
        "joint_successes": joint_successes,
        "original_failures": original_failures,
        "counterfactual_failures": counterfactual_failures,
        "joint_failures": joint_failures,
        "mean_side_residual": residual_total / (2 * matched_rollouts) if matched_rollouts else 0.0,
    }
