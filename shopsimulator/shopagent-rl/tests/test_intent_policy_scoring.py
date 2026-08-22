"""Unit tests for canonical-intent scoring on the restricted action set.

The parser and aggregation are exercised with a deterministic stub model so the
logic is covered on CPU without loading 1.7B weights. Real-model numerics are
verified separately as part of the Gate A gradient acceptance.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from experiment.grpo.intent_policy_scoring import (
    intent_log_probs,
    intent_of_action,
    parse_legal_actions,
    score_actions,
)
from experiment.grpo.preference_margin import CANONICAL_INTENTS


def _state(buttons: list[str], price: float | None = None) -> str:
    """Render a state the way shop_env/obs_format.py does."""
    lines = [
        "商品页面 [SEP] 详情",
        "搜索功能是否可用: False",
        f"可点击的按钮: {json.dumps(buttons, ensure_ascii=False)}",
    ]
    if price is not None:
        lines.append(f"当前价格: {price}")
    return "\n\n".join(lines)


class StubTokenizer:
    """Character-level tokenizer; ids are codepoints offset past the pad id."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = True):
        return type("Enc", (), {"input_ids": [ord(c) % 97 + 2 for c in text]})()


class StubModel:
    """Deterministic logits: favors whichever token id appears in `boost`."""

    def __init__(self, vocab: int = 128, boost: set[int] | None = None):
        self.vocab = vocab
        self.boost = boost or set()
        self.weight = torch.zeros(vocab, requires_grad=True)

    def __call__(self, input_ids, attention_mask=None):
        n, t = input_ids.shape
        logits = torch.zeros(n, t, self.vocab)
        for tid in self.boost:
            logits[:, :, tid] = 3.0
        # Tie to a parameter so gradient-flow assertions are meaningful.
        logits = logits + self.weight.view(1, 1, -1)
        return type("Out", (), {"logits": logits})()


class TestParseLegalActions:
    def test_price_case_exposes_navigation(self):
        actions = parse_legal_actions(_state(["buy now", "back to search", "< prev"], 99.0))
        assert actions == ["click[buy now]", "click[back to search]", "click[< prev]"]

    def test_option_case_exposes_variants(self):
        actions = parse_legal_actions(_state(["buy now", "红色", "XL"]))
        assert "click[红色]" in actions and "click[XL]" in actions

    def test_order_is_preserved_for_determinism(self):
        buttons = ["buy now", "reviews", "红色"]
        assert parse_legal_actions(_state(buttons)) == parse_legal_actions(_state(buttons))

    def test_malformed_state_yields_empty(self):
        assert parse_legal_actions("没有按钮这一行") == []
        assert parse_legal_actions("可点击的按钮: [不是合法json\n") == []

    def test_non_string_buttons_are_dropped(self):
        assert parse_legal_actions('可点击的按钮: ["buy now", 7, null]\n') == ["click[buy now]"]


class TestIntentOfAction:
    @pytest.mark.parametrize("button,expected", [
        ("buy now", "COMMIT"),
        ("BUY NOW", "COMMIT"),
        ("back to search", "SEARCH_ALTERNATIVE"),
        ("< prev", "SEARCH_ALTERNATIVE"),
        ("红色", "SELECT_TARGET_OPTION"),
        ("description", None),
        ("reviews", None),
    ])
    def test_mapping(self, button, expected):
        assert intent_of_action(f"click[{button}]") == expected

    def test_lenient_whitespace(self):
        assert intent_of_action("  click[ buy now ]  ") == "COMMIT"

    def test_malformed_action_returns_none(self):
        assert intent_of_action("search buy now") is None
        assert intent_of_action("click[]") is None


class TestScoreActions:
    def test_empty_actions_returns_empty(self):
        out = score_actions(StubModel(), StubTokenizer(), "state", [])
        assert out.shape == (0,)

    def test_one_score_per_action_and_finite(self):
        out = score_actions(StubModel(), StubTokenizer(), _state(["buy now", "红色"]),
                            ["click[buy now]", "click[红色]"])
        assert out.shape == (2,)
        assert torch.isfinite(out).all()

    def test_padding_does_not_leak_across_rows(self):
        """A long candidate must not change a short candidate's score."""
        state = _state(["buy now", "红色"])
        short_alone = score_actions(StubModel(), StubTokenizer(), state, ["click[buy now]"])
        with_long = score_actions(StubModel(), StubTokenizer(), state,
                                  ["click[buy now]", "click[" + "长" * 40 + "]"])
        assert torch.allclose(short_alone[0], with_long[0], atol=1e-5)


class TestIntentLogProbs:
    def test_shape_matches_canonical_intents(self):
        lp, info = intent_log_probs(StubModel(), StubTokenizer(),
                                    _state(["buy now", "back to search", "红色"]))
        assert lp.shape == (len(CANONICAL_INTENTS),)
        assert info["scorable"] and info["n_legal"] == 3

    def test_distribution_is_normalized_over_legal_set(self):
        """All buttons map to some intent here, so probabilities must sum to 1."""
        lp, _ = intent_log_probs(StubModel(), StubTokenizer(),
                                 _state(["buy now", "back to search", "红色"]))
        assert lp.exp().sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_inert_buttons_leave_mass_outside_intents(self):
        lp, info = intent_log_probs(StubModel(), StubTokenizer(),
                                    _state(["buy now", "description", "reviews"]))
        assert info["n_mapped"] == 1
        assert lp.exp().sum().item() < 1.0

    def test_absent_intent_is_neg_inf_not_zero(self):
        """No navigation button => SEARCH_ALTERNATIVE is unavailable, not unlikely."""
        lp, _ = intent_log_probs(StubModel(), StubTokenizer(), _state(["buy now", "红色"]))
        idx = CANONICAL_INTENTS.index("SEARCH_ALTERNATIVE")
        assert lp[idx].item() == -math.inf
        assert torch.isfinite(lp[CANONICAL_INTENTS.index("COMMIT")])

    def test_multiple_variants_aggregate_by_logsumexp(self):
        """A second same-intent action must take mass from the other intents.

        Asserted via COMMIT rather than SELECT_TARGET_OPTION: raw sequence
        log-probs are length-biased, so the shorter label already holds
        essentially all the mass and its normalized log-prob rounds to 0.0 in
        float32. Mass conservation states the same invariant without saturating.
        """
        one, _ = intent_log_probs(StubModel(), StubTokenizer(), _state(["buy now", "红色"]))
        two, _ = intent_log_probs(StubModel(), StubTokenizer(), _state(["buy now", "红色", "蓝色"]))
        commit = CANONICAL_INTENTS.index("COMMIT")
        select = CANONICAL_INTENTS.index("SELECT_TARGET_OPTION")
        assert two[commit].item() < one[commit].item()
        assert two[select].item() >= one[select].item()

    def test_malformed_state_is_reported_not_crashed(self):
        lp, info = intent_log_probs(StubModel(), StubTokenizer(), "缺少按钮行")
        assert not info["scorable"]
        assert torch.isinf(lp).all()

    def test_gradient_reaches_parameters(self):
        model = StubModel()
        lp, _ = intent_log_probs(model, StubTokenizer(), _state(["buy now", "红色"]))
        finite = lp[torch.isfinite(lp)]
        grad = torch.autograd.grad(finite.sum(), model.weight, allow_unused=True)[0]
        assert grad is not None and grad.abs().sum() > 0
