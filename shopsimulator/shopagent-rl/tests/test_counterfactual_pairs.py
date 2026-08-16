from __future__ import annotations

import unittest

from experiment.counterfactual_pairs import (
    build_pairs,
    canonical_price_upper,
    normalize_option,
    validate_pair,
)


class CounterfactualPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.products = [
            {
                "asin": "p0",
                "tag": "eval",
                "title": "测试商品",
                "category": "测试类目",
                "pricing": [49],
                "customization_options": {
                    "尺码": [
                        {"value": "M/标准", "price": 49, "is_available": True},
                        {"value": "L | 标准", "price": 49, "is_available": True},
                        {"value": "XL", "price": 59, "is_available": True},
                    ]
                },
            }
        ]
        self.records = [
            {
                "task_id": 0,
                "goal": {
                    "instruction_text": "要 M 标准尺码，预算 50 元。",
                    "goal_options": ["M | 标准"],
                    "price_upper": 50,
                },
            }
        ]

    def test_normalizes_engine_punctuation(self) -> None:
        self.assertEqual(normalize_option("M/标准"), normalize_option("M | 标准"))

    def test_builds_atomic_option_and_price_pairs(self) -> None:
        result = build_pairs(self.products, self.records)
        self.assertEqual(result.stats["pairs_total"], 2)
        by_type = {pair["intervention_type"]: pair for pair in result.pairs}

        option_pair = by_type["option_swap"]
        self.assertEqual(option_pair["original"]["current_price"], 49)
        self.assertEqual(option_pair["counterfactual"]["current_price"], 49)
        self.assertEqual(
            option_pair["counterfactual"]["allowed_actions"], ["click[M/标准]"]
        )
        self.assertEqual(validate_pair(option_pair), [])

        price_pair = by_type["price_above_budget"]
        self.assertGreater(price_pair["counterfactual"]["current_price"], 50)
        self.assertEqual(
            price_pair["original"]["selected_options"],
            price_pair["counterfactual"]["selected_options"],
        )
        self.assertEqual(validate_pair(price_pair), [])

    def test_rejects_original_that_is_already_over_budget(self) -> None:
        self.records[0]["goal"]["price_upper"] = 40
        result = build_pairs(self.products, self.records)
        self.assertEqual(result.pairs, [])
        self.assertEqual(result.stats["rejected_original_over_budget"], 1)

    def test_recovers_missing_goal_from_raw_product(self) -> None:
        self.products[0]["instructions"] = [
            {
                "instruction": "要 M 标准尺码。",
                "instruction_options": ["M | 标准"],
            }
        ]
        result = build_pairs(self.products, [{"task_id": 0, "goal": None}])
        self.assertEqual(result.stats["pairs_total"], 2)
        self.assertEqual(result.stats["goals_with_canonical_budget"], 1)
        self.assertEqual(
            result.pairs[0]["goal"]["price_upper_source"], "canonical_from_target_price"
        )

    def test_canonical_budget_is_strictly_above_price(self) -> None:
        self.assertEqual(canonical_price_upper(49), 50)
        self.assertEqual(canonical_price_upper(50), 60)


if __name__ == "__main__":
    unittest.main()
