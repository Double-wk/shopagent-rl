from __future__ import annotations

import unittest

from experiment.constraint_causal_natural import (
    REWRITE_SYSTEM_PROMPT,
    alnum_tokens,
    build_natural_pair,
    build_rewrite_prompt,
    instruction_of,
    iter_structured_candidates,
    verify_rewrite,
)


class TokenTests(unittest.TestCase):
    def test_alnum_tokens_mixed(self) -> None:
        self.assertEqual(alnum_tokens("预算100元，85ml就够了"), ["100", "85ml"])
        self.assertEqual(alnum_tokens("50g*3(整盒)"), ["50g*3"])


class VerifyRewriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = "最近手脚发麻，想买一款泰国按摩药油，预算100元左右，85ml就够了。"
        self.option_a = "玉菩药堂驱风油85ml"
        self.option_b = "玉菩药堂驱风油膏50g*3(整盒)"

    def test_accepts_faithful_rewrite(self) -> None:
        rewrite = "最近手脚发麻，想买一款泰国按摩药油，预算100元左右，要50g*3整盒装的。"
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertTrue(check.accepted, check.reasons)

    def test_rejects_budget_drift(self) -> None:
        rewrite = "最近手脚发麻，想买一款泰国按摩药油，预算200元左右，要50g*3整盒装的。"
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertFalse(check.accepted)
        self.assertTrue(any(r.startswith("lost_unrelated_tokens") for r in check.reasons))

    def test_rejects_residual_old_option(self) -> None:
        rewrite = "最近手脚发麻，想买一款泰国按摩药油，预算100元左右，85ml的就行。"
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertFalse(check.accepted)
        self.assertTrue(any(r.startswith("old_option_tokens_remain") for r in check.reasons))

    def test_rejects_missing_new_option(self) -> None:
        rewrite = "最近手脚发麻，想买一款泰国按摩药油，预算100元左右，要大容量的。"
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertFalse(check.accepted)
        self.assertTrue(any(r.startswith("new_option_tokens_missing") for r in check.reasons))

    def test_rejects_topic_drift(self) -> None:
        rewrite = "想给家里买一台空气炸锅，预算100元左右，要50g*3整盒装的。"
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertFalse(check.accepted)
        self.assertTrue(any(r.startswith("bigram_jaccard_low") for r in check.reasons))

    def test_rejects_bloat(self) -> None:
        rewrite = ("最近手脚发麻，想买一款泰国按摩药油，预算100元左右，要50g*3整盒装的。"
                   + "另外顺便再详细描述很多与任务无关的内容" * 10)
        check = verify_rewrite(self.original, rewrite, self.option_a, self.option_b)
        self.assertFalse(check.accepted)
        self.assertTrue(any(r.startswith("length_ratio_out_of_bounds") for r in check.reasons))

    def test_rejects_empty(self) -> None:
        check = verify_rewrite(self.original, "  ", self.option_a, self.option_b)
        self.assertFalse(check.accepted)


class BuildNaturalPairTests(unittest.TestCase):
    def _structured_pair(self) -> dict:
        return {
            "schema_version": "shopsim-constraint-causal-pairs-v2",
            "pair_id": "16:option_goal_swap_structured", "task_id": 16,
            "source": {"tag": "eval"}, "product": {"option_group": "颜色分类"},
            "goal": {"goal_options": ["玉菩药堂驱风油85ml"], "price_upper": 180},
            "original": {
                "selected_options": {"颜色分类": "玉菩药堂驱风油85ml"}, "current_price": 126,
                "expected_action_intents": ["COMMIT"], "allowed_actions": ["click[buy now]"],
                "observation": ("Instruction: 想买按摩药油，预算100元，85ml就够了。\n"
                                "商品: 测试\n任务约束摘要: 目标规格=玉菩药堂驱风油85ml；预算上限=180元。"),
            },
            "counterfactual": {"current_price": 126},
            "intervention_type": "option_goal_swap_structured",
            "intervention": {"field": "instruction.goal_option", "from": "玉菩药堂驱风油85ml",
                             "to": "玉菩药堂驱风油膏50g*3(整盒)", "validity_checks": {}},
        }

    def test_builds_pair_and_strips_summary(self) -> None:
        pair = self._structured_pair()
        rewrite = "想买按摩药油，预算100元，要50g*3的。"
        check = verify_rewrite(instruction_of(pair), rewrite, "玉菩药堂驱风油85ml", "玉菩药堂驱风油膏50g*3(整盒)")
        self.assertTrue(check.accepted, check.reasons)
        natural = build_natural_pair(pair, rewrite, check, model="test-teacher")
        self.assertEqual(natural["intervention_type"], "option_goal_swap_natural")
        self.assertNotIn("任务约束摘要", natural["original"]["observation"])
        self.assertNotIn("任务约束摘要", natural["counterfactual"]["observation"])
        self.assertIn(rewrite, natural["counterfactual"]["observation"])
        self.assertNotIn("85ml", natural["counterfactual"]["observation"])
        self.assertEqual(natural["counterfactual"]["allowed_actions"], ["click[玉菩药堂驱风油膏50g*3(整盒)]"])
        self.assertEqual(natural["counterfactual"]["expected_action_intents"], ["SELECT_TARGET_OPTION"])
        self.assertEqual(natural["goal"]["goal_options"], ["玉菩药堂驱风油膏50g*3(整盒)"])
        self.assertTrue(natural["intervention"]["validity_checks"]["programmatic_rewrite_verification"])

    def test_iter_structured_candidates_filters(self) -> None:
        pair = self._structured_pair()
        pairs = [pair, {**pair, "intervention_type": "nuisance_display_note"}]
        self.assertEqual(len(iter_structured_candidates(pairs)), 1)


class PromptTests(unittest.TestCase):
    def test_prompt_mentions_both_options(self) -> None:
        prompt = build_rewrite_prompt("原文", "规格A", "规格B")
        self.assertIn("规格A", prompt)
        self.assertIn("规格B", prompt)
        self.assertIn("预算", prompt)
        self.assertTrue(REWRITE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
