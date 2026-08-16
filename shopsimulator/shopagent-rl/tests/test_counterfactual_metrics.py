from __future__ import annotations

import unittest

from experiment.eval.run_counterfactual import aggregate


class CounterfactualMetricTests(unittest.TestCase):
    def test_certification_conditions_on_original_success(self) -> None:
        records = [
            {"intervention_type": "price_above_budget", "price_upper_source": "observed_eval_goal", "orig_correct": True, "cf_correct": True, "cf_correct_lenient": True, "robust": True, "cf_intent": "SEARCH_ALTERNATIVE", "cf_commit": False, "orig_action_type": "click", "cf_action_type": "search", "same_action": False},
            {"intervention_type": "price_above_budget", "price_upper_source": "observed_eval_goal", "orig_correct": True, "cf_correct": False, "cf_correct_lenient": False, "robust": False, "cf_intent": "SEARCH_ALTERNATIVE", "cf_commit": True, "orig_action_type": "click", "cf_action_type": "click", "same_action": True},
            {"intervention_type": "price_above_budget", "price_upper_source": "observed_eval_goal", "orig_correct": False, "cf_correct": False, "cf_correct_lenient": False, "robust": False, "cf_intent": "SEARCH_ALTERNATIVE", "cf_commit": False, "orig_action_type": "search", "cf_action_type": "search", "same_action": True},
        ]
        metrics = aggregate(records)
        self.assertEqual(metrics["causal_success_certification_rate"], 0.5)
        self.assertEqual(metrics["shortcut_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
