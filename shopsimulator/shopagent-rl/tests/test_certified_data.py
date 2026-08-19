from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

import scripts.build_paired_sft_data as sft_builder
from scripts.build_certified_grpo_data import build_rows
from experiment.explicit_budget_pairs import (
    BUDGET_PREFIX,
    make_budget_explicit,
    validate_explicit_budget_pair,
)
from experiment.grpo.paired_reward import add_joint_certified_bonus


SYSTEM = "system"


def price_pair() -> dict:
    return {
        "pair_id": "7:price_above_budget",
        "task_id": 7,
        "intervention_type": "price_above_budget",
        "goal": {"price_upper": 100},
        "original": {
            "observation": "Instruction: test\n当前价格: 90",
            "current_price": 90,
            "expected_action_intents": ["COMMIT"],
            "allowed_actions": ["click[buy now]"],
        },
        "counterfactual": {
            "observation": "Instruction: test\n当前价格: 105",
            "current_price": 105,
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]", "click[< prev]"],
        },
    }


class CertifiedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        sft_builder.SYSTEM_PROMPT = SYSTEM

    def test_price_sft_contains_true_comparison_and_recovery(self) -> None:
        record = sft_builder.emit_price(price_pair(), "counterfactual")
        target = record["messages"][-1]["content"]
        prompt = record["messages"][-2]["content"]
        self.assertNotIn("任务约束摘要", prompt)
        self.assertIn("105元 > 预算上限100元", target)
        self.assertIn("Action: click[back to search]", target)

        original = sft_builder.emit_price(price_pair(), "original")
        original_target = original["messages"][-1]["content"]
        self.assertIn("90元 <= 预算上限100元", original_target)
        self.assertIn("Action: click[buy now]", original_target)

    def test_certified_grpo_rows_keep_both_sides(self) -> None:
        rows = build_rows(
            [1], [price_pair()], [], [], SYSTEM,
            environment_repeat=1,
            price_pairs=1,
            option_pairs=0,
            nuisance_pairs=0,
            seed=1,
        )
        self.assertEqual(len(rows), 3)
        cf_rows = [row for row in rows if row["sample_mode"] == "counterfactual"]
        self.assertEqual({row["side"] for row in cf_rows}, {"original", "counterfactual"})
        self.assertTrue(all("任务约束摘要" not in row["prompt"][-1]["content"] for row in cf_rows))
        self.assertEqual(len({row["extra_info"]["index"] for row in rows}), 3)

    def test_pair_blocking_keeps_complete_relations_in_each_batch(self) -> None:
        rows = build_rows(
            [1, 2, 3, 4], [price_pair(), {**price_pair(), "pair_id": "8:price_above_budget", "task_id": 8}],
            [], [], SYSTEM,
            environment_repeat=1,
            price_pairs=2,
            option_pairs=0,
            nuisance_pairs=0,
            seed=1,
            pair_block_size=4,
        )
        self.assertEqual(len(rows), 8)
        for start in range(0, len(rows), 4):
            block = rows[start:start + 4]
            relation_ids = {row["relation_id"] for row in block if row["relation_id"]}
            if not relation_ids:
                self.assertTrue(all(row["sample_mode"] == "environment" for row in block))
                continue
            self.assertEqual(len(relation_ids), 2)
            for relation_id in relation_ids:
                sides = {row["side"] for row in block if row["relation_id"] == relation_id}
                self.assertEqual(sides, {"original", "counterfactual"})

    def test_environment_task_limit_is_deterministic(self) -> None:
        kwargs = dict(
            environment_repeat=1,
            environment_tasks=4,
            price_pairs=0,
            option_pairs=0,
            nuisance_pairs=0,
            seed=7,
        )
        first = build_rows(list(range(20)), [], [], [], SYSTEM, **kwargs)
        second = build_rows(list(range(20)), [], [], [], SYSTEM, **kwargs)
        self.assertEqual([row["task_id"] for row in first], [row["task_id"] for row in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({row["task_id"] for row in first}), 4)

    def test_prompt_char_limit_replaces_overlong_pairs_before_sampling(self) -> None:
        long_pair = price_pair()
        long_pair["pair_id"] = "9:price_above_budget"
        long_pair["task_id"] = 9
        long_pair["original"]["observation"] = "x" * 100
        long_pair["counterfactual"]["observation"] = "y" * 100
        rows = build_rows(
            [], [long_pair, price_pair()], [], [], SYSTEM,
            environment_repeat=0,
            price_pairs=1,
            option_pairs=0,
            nuisance_pairs=0,
            seed=1,
            max_counterfactual_prompt_chars=80,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["pair_id"] for row in rows}, {"7:price_above_budget"})

    def test_explicit_budget_is_identical_and_visible_on_both_sides(self) -> None:
        pair = make_budget_explicit(price_pair())
        self.assertEqual(pair["goal"]["price_upper_source"], "programmatic_explicit_budget")
        self.assertEqual(pair["goal"]["price_upper_derivation_source"], "unknown")
        for side in ("original", "counterfactual"):
            observation = pair[side]["observation"]
            self.assertIn(f"{BUDGET_PREFIX}100元", observation)
        self.assertEqual(validate_explicit_budget_pair(pair), [])

    def test_excluded_task_ids_accept_lists_and_split_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            list_path = root / "list.json"
            summary_path = root / "summary.json"
            list_path.write_text(json.dumps([1, 2]), encoding="utf-8")
            summary_path.write_text(json.dumps({"task_ids": [2, 3]}), encoding="utf-8")
            excluded = sft_builder.load_excluded_task_ids(
                [str(list_path), str(summary_path)]
            )
        self.assertEqual(excluded, {1, 2, 3})

    def test_joint_certified_bonus_couples_matching_rollouts(self) -> None:
        rewards = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.7],
        ])
        mask = torch.ones_like(rewards)
        updated, stats = add_joint_certified_bonus(
            rewards,
            mask,
            ["p", "p", "p", "p", ""],
            ["original", "original", "counterfactual", "counterfactual", ""],
            [0, 1, 0, 1, 0],
            weight=1.0,
        )
        self.assertEqual(updated.sum(dim=-1).tolist()[:4], [2.0, 1.0, 2.0, 0.0])
        self.assertAlmostEqual(float(updated.sum(dim=-1)[4]), 0.7, places=5)
        self.assertEqual(stats["complete_relations"], 1)
        self.assertEqual(stats["matched_rollouts"], 2)
        self.assertEqual(stats["joint_successes"], 1)


if __name__ == "__main__":
    unittest.main()
