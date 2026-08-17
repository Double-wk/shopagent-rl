from __future__ import annotations

import unittest

import scripts.build_paired_sft_data as sft_builder
from scripts.build_certified_grpo_data import build_rows


SYSTEM = "system"


def price_pair() -> dict:
    return {
        "pair_id": "7:price_above_budget",
        "task_id": 7,
        "intervention_type": "price_above_budget",
        "goal": {"price_upper": 100},
        "original": {
            "observation": "当前价格: 90",
            "current_price": 90,
            "expected_action_intents": ["COMMIT"],
            "allowed_actions": ["click[buy now]"],
        },
        "counterfactual": {
            "observation": "当前价格: 105",
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


if __name__ == "__main__":
    unittest.main()
