from __future__ import annotations

import unittest

from experiment.counterfactual_grading import grade_response


class CounterfactualGradingTests(unittest.TestCase):
    def test_strict_price_recovery(self) -> None:
        side = {
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]", "click[< prev]"],
        }
        grade = grade_response("Thought: 超预算。\nAction: click[back to search]", side)
        self.assertTrue(grade["correct_strict"])
        self.assertFalse(grade["is_commit"])

    def test_search_is_lenient_price_recovery(self) -> None:
        side = {
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]"],
        }
        grade = grade_response("Action: search[更便宜的商品]", side)
        self.assertFalse(grade["correct_strict"])
        self.assertTrue(grade["correct_lenient"])

    def test_commit_fails_over_budget_side(self) -> None:
        side = {
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]"],
        }
        grade = grade_response("Action: click[buy now]", side)
        self.assertFalse(grade["correct_lenient"])
        self.assertTrue(grade["is_commit"])


if __name__ == "__main__":
    unittest.main()
