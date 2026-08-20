"""Paired-intervention reward coupling on batches that carry no pair.

`data/grpo_certified_natural_800_pairblocked.parquet` interleaves 4-row
counterfactual blocks with 4-row environment blocks, so with TRAIN_BATCH=4 a
paired GRPO run alternates between steps that contain two complete pairs and
steps that contain none. The 2026-08-19 paired run crashed at its first
environment block, so the environment path is covered here explicitly.
"""
import unittest

import torch

from experiment.grpo.paired_reward import add_joint_certified_bonus


def _rewards(scores: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """One reward tensor per row with the scalar score on the last valid token."""
    rewards = torch.zeros(len(scores), 4)
    mask = torch.ones(len(scores), 4)
    for row, score in enumerate(scores):
        rewards[row, -1] = score
    return rewards, mask


class EnvironmentBlockTests(unittest.TestCase):
    def test_environment_only_block_adds_no_bonus(self) -> None:
        """Empty pair metadata must be inert, not fatal."""
        rewards, mask = _rewards([0.55, 0.0, 1.0, 0.0])

        result, stats = add_joint_certified_bonus(
            rewards, mask, [""] * 4, [""] * 4, ["0", "0", "0", "0"]
        )

        torch.testing.assert_close(result, rewards)
        self.assertEqual(stats["complete_relations"], 0)
        self.assertEqual(stats["matched_rollouts"], 0)
        self.assertEqual(stats["mean_joint_bonus"], 0.0)

    def test_mixed_block_only_credits_the_complete_pair(self) -> None:
        """Environment rows in the same batch must not absorb pair bonus."""
        rewards, mask = _rewards([1.0, 0.5, 0.55, 0.0])
        relation_ids = ["18789:price_above_budget", "18789:price_above_budget", "", ""]
        sides = ["original", "counterfactual", "", ""]

        result, stats = add_joint_certified_bonus(
            rewards, mask, relation_ids, sides, ["r0", "r0", "r0", "r0"]
        )

        self.assertEqual(stats["complete_relations"], 1)
        self.assertEqual(stats["matched_rollouts"], 1)
        # joint = min(1.0, 0.5) = 0.5, added to both sides of the pair only.
        self.assertAlmostEqual(float(result[0, -1]), 1.5)
        self.assertAlmostEqual(float(result[1, -1]), 1.0)
        torch.testing.assert_close(result[2:], rewards[2:])

    def test_unmatched_side_is_not_credited(self) -> None:
        """A pair whose CF side never came back must not get a one-sided bonus."""
        rewards, mask = _rewards([1.0, 0.0])
        result, stats = add_joint_certified_bonus(
            rewards, mask, ["p:price_above_budget", ""], ["original", ""], ["r0", "r0"]
        )

        torch.testing.assert_close(result, rewards)
        self.assertEqual(stats["complete_relations"], 0)


if __name__ == "__main__":
    unittest.main()
