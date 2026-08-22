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
        self.assertEqual(result.stats["pairs_total"], 4)
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
        self.assertEqual(result.stats["pairs_total"], 4)
        self.assertEqual(result.stats["goals_with_canonical_budget"], 1)
        self.assertEqual(
            result.pairs[0]["goal"]["price_upper_source"], "canonical_from_target_price"
        )

    def test_held_out_pairs_share_a_select_target_frame(self) -> None:
        result = build_pairs(self.products, self.records)
        by_type = {pair["intervention_type"]: pair for pair in result.pairs}
        for name in ("option_unavailable", "option_price_over_budget"):
            pair = by_type[name]
            original = pair["original"]
            # The cheap sibling is selected, so the goal option is not yet won.
            self.assertEqual(original["expected_action_intents"], ["SELECT_TARGET_OPTION"])
            self.assertEqual(original["allowed_actions"], ["click[M/标准]"])
            self.assertEqual(
                pair["counterfactual"]["expected_action_intents"], ["SEARCH_ALTERNATIVE"]
            )
            # Atomic: selection and the displayed scalar price never move.
            self.assertEqual(
                original["selected_options"], pair["counterfactual"]["selected_options"]
            )
            self.assertEqual(
                original["current_price"], pair["counterfactual"]["current_price"]
            )
            self.assertEqual(validate_pair(pair), [])

    def test_option_unavailable_removes_only_the_goal_option_from_clickables(self) -> None:
        result = build_pairs(self.products, self.records)
        pair = {p["intervention_type"]: p for p in result.pairs}["option_unavailable"]
        before = pair["original"]["observation"]
        after = pair["counterfactual"]["observation"]
        self.assertIn('"M/标准"', before)
        self.assertNotIn('"M/标准"', after.split("可点击的按钮: ")[1])
        self.assertIn("缺货", after)
        # The sibling stays clickable: the intervention is not a blanket block.
        self.assertIn('"L | 标准"', after.split("可点击的按钮: ")[1])
        self.assertEqual(pair["intervention"]["verified_by"], "action_legality")

    def test_option_price_over_budget_hides_violation_from_the_scalar(self) -> None:
        result = build_pairs(self.products, self.records)
        pair = {p["intervention_type"]: p for p in result.pairs}["option_price_over_budget"]
        after = pair["counterfactual"]["observation"]
        # 当前价格 still shows the affordable sibling, so a price-blind policy
        # that only reads the scalar cannot detect the violation.
        self.assertIn("当前价格: 49", after)
        self.assertGreater(pair["intervention"]["to"], 50)
        self.assertIn(f"M/标准(￥{pair['intervention']['to']:g})", after)
        # The goal option stays clickable -- this is a budget, not availability.
        self.assertIn('"M/标准"', after.split("可点击的按钮: ")[1])
        self.assertEqual(pair["intervention"]["verified_by"], "structured_option_price")

    def test_held_out_pairs_need_an_affordable_sibling(self) -> None:
        self.products[0]["customization_options"]["尺码"] = [
            {"value": "M/标准", "price": 49, "is_available": True},
        ]
        result = build_pairs(self.products, self.records)
        types = {pair["intervention_type"] for pair in result.pairs}
        self.assertEqual(types, {"price_above_budget"})
        self.assertEqual(result.stats["skipped_no_affordable_sibling"], 1)

    def test_canonical_budget_is_strictly_above_price(self) -> None:
        self.assertEqual(canonical_price_upper(49), 50)
        self.assertEqual(canonical_price_upper(50), 60)


if __name__ == "__main__":
    unittest.main()
