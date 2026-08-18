"""Pair-level certified reward coupling for intervention GRPO."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch


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
    grouped: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(list)
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
        grouped[relation_id][side].append((rollout_id, index))

    result = token_level_rewards.clone()
    matched_rollouts = 0
    joint_successes = 0
    bonus_total = 0.0
    complete_relations = 0
    for by_side in grouped.values():
        original = dict(by_side.get("original", []))
        counterfactual = dict(by_side.get("counterfactual", []))
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
