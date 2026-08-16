from __future__ import annotations

import unittest

from experiment.constraint_causal_pairs import build_v2_pairs, validate_v2_pair


class ConstraintCausalPairTests(unittest.TestCase):
    def test_builds_goal_swap_and_nuisance_control(self) -> None:
        v1 = [{
            "pair_id": "1:option_swap", "task_id": 1,
            "intervention_type": "option_swap", "source": {},
            "goal": {"goal_options": ["M"], "price_upper": 50, "price_upper_source": "observed_eval_goal"},
            "product": {"option_group": "尺码"},
            "original": {
                "selected_options": {"尺码": "M"}, "current_price": 49,
                "expected_action_intents": ["COMMIT"], "allowed_actions": ["click[buy now]"],
                "observation": "Instruction: 需要 M 尺码。\n商品: 测试\n当前价格: 49",
            },
            "counterfactual": {"selected_options": {"尺码": "L"}, "current_price": 49},
        }]
        result = build_v2_pairs(v1)
        self.assertEqual(result.stats["pairs_total"], 2)
        by_type = {p["intervention_type"]: p for p in result.pairs}
        goal_swap = by_type["option_goal_swap"]
        self.assertIn("需要 L 尺码", goal_swap["counterfactual"]["observation"])
        self.assertEqual(goal_swap["counterfactual"]["allowed_actions"], ["click[L]"])
        self.assertEqual(validate_v2_pair(goal_swap), [])
        nuisance = by_type["nuisance_display_note"]
        self.assertEqual(nuisance["original"]["allowed_actions"], nuisance["counterfactual"]["allowed_actions"])
        self.assertEqual(validate_v2_pair(nuisance), [])


if __name__ == "__main__":
    unittest.main()
