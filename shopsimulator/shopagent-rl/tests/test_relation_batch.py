"""Unit tests for pair grouping and relation-loss assembly."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from experiment.grpo.preference_margin import CANONICAL_INTENTS, INTENT_TO_INDEX
from experiment.grpo.relation_batch import (
    anchor_loss,
    compute_batch_relation_loss,
    group_pairs,
)
from tests.test_intent_policy_scoring import StubModel, StubTokenizer, _state


def _row(pair_id, side, relation, expected, buttons=("buy now", "back to search", "红色"),
         intervention="price_above_budget"):
    return {
        "pair_id": pair_id,
        "side": side,
        "expected_relation": relation,
        "expected_action_intents": expected,
        "intervention_type": intervention,
        "state_text": _state(list(buttons)),
    }


def _flip_pair(pair_id="p1"):
    rel = ["COMMIT", "SEARCH_ALTERNATIVE"]
    return [
        _row(pair_id, "original", rel, ["COMMIT"]),
        _row(pair_id, "counterfactual", rel, ["SEARCH_ALTERNATIVE"]),
    ]


def _preserve_pair(pair_id="p2"):
    rel = ["COMMIT", "COMMIT"]
    return [
        _row(pair_id, "original", rel, ["COMMIT"], intervention="nuisance_display_note"),
        _row(pair_id, "counterfactual", rel, ["COMMIT"], intervention="nuisance_display_note"),
    ]


class TestGroupPairs:
    def test_complete_flip_pair(self):
        pairs, c = group_pairs(_flip_pair())
        assert c["pairs"] == 1 and c["decision_changing"] == 1
        assert pairs[0]["intent_original"] == "COMMIT"
        assert pairs[0]["intent_cf"] == "SEARCH_ALTERNATIVE"
        assert pairs[0]["is_decision_changing"] is True

    def test_preserve_pair_derived_from_equal_intents(self):
        pairs, c = group_pairs(_preserve_pair())
        assert c["decision_preserving"] == 1
        assert pairs[0]["is_decision_changing"] is False

    def test_half_pair_is_dropped_not_treated_as_singleton(self):
        pairs, c = group_pairs(_flip_pair()[:1])
        assert pairs == [] and c["dropped_incomplete"] == 1

    def test_environment_rows_without_pair_id_are_dropped(self):
        rows = _flip_pair() + [_row("", "", [], [])]
        pairs, c = group_pairs(rows)
        assert c["pairs"] == 1 and c["dropped_no_pair_id"] == 1

    def test_unknown_side_is_dropped(self):
        rows = _flip_pair() + [_row("p9", "sideways", ["COMMIT", "COMMIT"], ["COMMIT"])]
        _, c = group_pairs(rows)
        assert c["dropped_bad_side"] == 1

    def test_malformed_relation_is_dropped(self):
        rows = _flip_pair("bad")
        for r in rows:
            r["expected_relation"] = ["COMMIT", "NOT_AN_INTENT"]
        pairs, c = group_pairs(rows)
        assert pairs == [] and c["dropped_bad_relation"] == 2

    def test_numpy_and_json_metadata_are_accepted(self):
        """Parquet hands back numpy arrays; a JSON string must work too."""
        rows = _flip_pair()
        rows[0]["expected_relation"] = np.array(["COMMIT", "SEARCH_ALTERNATIVE"])
        rows[0]["pair_id"] = np.str_("p1")
        rows[1]["expected_action_intents"] = json.dumps(["SEARCH_ALTERNATIVE"])
        pairs, c = group_pairs(rows)
        assert c["pairs"] == 1 and pairs[0]["intent_cf"] == "SEARCH_ALTERNATIVE"

    def test_original_row_with_partner_state_completes_its_own_pair(self):
        """The whole point of the trainer's partner attachment.

        With `ppo_micro_batch_size_per_gpu=1` the counterfactual row is never in
        the same micro-batch, so the pair must be completable from one row.
        """
        row = _flip_pair()[0]
        row["partner_state_text"] = _state(["back to search", "buy now"])
        row["partner_expected_action_intents"] = ["SEARCH_ALTERNATIVE"]
        pairs, c = group_pairs([row])
        assert c["pairs"] == 1 and c["pairs_from_partner_state"] == 1
        assert c["dropped_incomplete"] == 0
        assert pairs[0]["is_decision_changing"] is True
        cf = pairs[0]["counterfactual"]
        assert cf["state_text"] == row["partner_state_text"]
        assert cf["expected_action_intents"] == ["SEARCH_ALTERNATIVE"]

    def test_empty_partner_state_does_not_fabricate_a_pair(self):
        row = _flip_pair()[0]
        row["partner_state_text"] = ""
        pairs, c = group_pairs([row])
        assert pairs == [] and c["dropped_incomplete"] == 1
        assert c["pairs_from_partner_state"] == 0

    def test_counterfactual_row_carrying_partner_state_is_ignored(self):
        """Only `original` rows complete pairs, so a pair is counted once."""
        row = _flip_pair()[1]
        row["partner_state_text"] = _state(["buy now"])
        pairs, c = group_pairs([row])
        assert pairs == [] and c["dropped_incomplete"] == 1

    def test_real_counterfactual_row_wins_over_carried_state(self):
        rows = _flip_pair()
        rows[0]["partner_state_text"] = _state(["carried"])
        pairs, c = group_pairs(rows)
        assert c["pairs"] == 1 and c["pairs_from_partner_state"] == 0
        assert pairs[0]["counterfactual"] is rows[1]

    def test_partner_state_pair_is_not_double_counted(self):
        """All n rollouts of a side share one pair_id, hence one pair."""
        rows = []
        for _ in range(4):
            row = dict(_flip_pair()[0])
            row["partner_state_text"] = _state(["back to search"])
            row["partner_expected_action_intents"] = ["SEARCH_ALTERNATIVE"]
            rows.append(row)
        _, c = group_pairs(rows)
        assert c["rows"] == 4 and c["pairs"] == 1

    def test_two_pairs_are_kept_separate(self):
        pairs, c = group_pairs(_flip_pair("a") + _preserve_pair("b"))
        assert c["pairs"] == 2
        assert {p["pair_id"] for p in pairs} == {"a", "b"}


class TestAnchorLoss:
    def test_penalizes_low_probability_target(self):
        lp = torch.log(torch.tensor([0.7, 0.2, 0.1]))
        hi = anchor_loss(lp, [CANONICAL_INTENTS[0]])
        lo = anchor_loss(lp, [CANONICAL_INTENTS[2]])
        assert lo > hi
        assert hi.item() == pytest.approx(-math.log(0.7), abs=1e-6)

    def test_multiple_targets_use_combined_mass(self):
        lp = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        both = anchor_loss(lp, CANONICAL_INTENTS[:2])
        assert both.item() == pytest.approx(-math.log(0.8), abs=1e-6)

    def test_unavailable_target_yields_zero_with_grad_path(self):
        base = torch.zeros(len(CANONICAL_INTENTS), requires_grad=True)
        lp = base + torch.tensor([0.0, -math.inf, -math.inf])
        out = anchor_loss(lp, [CANONICAL_INTENTS[1]])
        assert out.item() == 0.0 and out.requires_grad

    def test_empty_expectation_is_zero(self):
        lp = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        assert anchor_loss(lp, []).item() == 0.0


class TestComputeBatchRelationLoss:
    def test_no_pairs_returns_exact_zero(self):
        loss, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), [])
        assert loss.item() == 0.0 and stats["pairs_used"] == 0

    def test_flip_pair_produces_positive_loss_and_stats(self):
        loss, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), _flip_pair())
        assert stats["pairs_used"] == 1
        assert loss.item() > 0
        assert "margin/price_above_budget" in stats
        assert stats["preserve_loss_mean"] == 0.0

    def test_preserve_pair_routes_to_preserve_term(self):
        loss, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), _preserve_pair())
        assert stats["pairs_used"] == 1
        assert stats["flip_loss_mean"] == 0.0
        assert loss.item() >= 0

    def test_identical_states_give_zero_margin(self):
        """A flip pair whose two sides are byte-identical must have M=0."""
        rows = _flip_pair()
        rows[1]["state_text"] = rows[0]["state_text"]
        _, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), rows)
        assert stats["margin_mean"] == pytest.approx(0.0, abs=1e-5)

    def test_preserve_pairs_excluded_from_margin_mean(self):
        """M is structurally 0 when intent_original == intent_cf.

        Averaging those zeros in would drag margin_mean toward 0 and hide real
        movement on the decision-changing pairs.
        """
        flip_only, s_flip = compute_batch_relation_loss(
            StubModel(), StubTokenizer(), _flip_pair("a"))
        mixed, s_mixed = compute_batch_relation_loss(
            StubModel(), StubTokenizer(), _flip_pair("a") + _preserve_pair("b"))
        assert s_mixed["pairs_used"] == 2
        assert s_mixed["margin_mean"] == pytest.approx(s_flip["margin_mean"], abs=1e-6)

    def test_preserve_only_batch_reports_zero_margin(self):
        _, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), _preserve_pair())
        assert stats["pairs_used"] == 1
        assert stats["margin_mean"] == 0.0
        assert stats["flip_rate"] == 0.0

    def test_unscorable_pair_is_counted_and_skipped(self):
        rows = _flip_pair()
        rows[0]["state_text"] = "没有按钮"
        loss, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), rows)
        assert stats["dropped_unscorable"] == 1 and stats["pairs_used"] == 0
        assert loss.item() == 0.0

    def test_pair_missing_compared_intent_is_skipped(self):
        """SEARCH_ALTERNATIVE not clickable => margin would be infinite."""
        rows = _flip_pair()
        for r in rows:
            r["state_text"] = _state(["buy now", "红色"])
        loss, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), rows)
        assert stats["dropped_unscorable"] == 1 and loss.item() == 0.0

    def test_weights_scale_the_loss(self):
        base, _ = compute_batch_relation_loss(StubModel(), StubTokenizer(), _flip_pair(),
                                              anchor_weight=0.0)
        doubled, _ = compute_batch_relation_loss(StubModel(), StubTokenizer(), _flip_pair(),
                                                 flip_weight=2.0, anchor_weight=0.0)
        assert doubled.item() == pytest.approx(2 * base.item(), rel=1e-5)

    def test_anchor_weight_zero_removes_anchor_contribution(self):
        with_anchor, s1 = compute_batch_relation_loss(StubModel(), StubTokenizer(), _flip_pair())
        without, _ = compute_batch_relation_loss(StubModel(), StubTokenizer(), _flip_pair(),
                                                 anchor_weight=0.0)
        assert with_anchor.item() > without.item()
        assert s1["anchor_loss_mean"] > 0

    def test_mixed_batch_averages_over_both_kinds(self):
        rows = _flip_pair("a") + _preserve_pair("b")
        _, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), rows)
        assert stats["pairs_used"] == 2
        assert stats["flip_loss_mean"] > 0 and stats["preserve_loss_mean"] >= 0

    def test_gradient_reaches_parameters(self):
        model = StubModel()
        loss, stats = compute_batch_relation_loss(model, StubTokenizer(), _flip_pair())
        assert stats["pairs_used"] == 1
        grad = torch.autograd.grad(loss, model.weight, allow_unused=True)[0]
        assert grad is not None and grad.abs().sum() > 0

    def test_flip_rate_counts_only_decision_changing_pairs(self):
        rows = _flip_pair("a") + _preserve_pair("b")
        _, stats = compute_batch_relation_loss(StubModel(), StubTokenizer(), rows)
        assert 0.0 <= stats["flip_rate"] <= 1.0
        # One flip pair: the rate can only be 0 or 1, never a preserve-diluted 0.5.
        assert stats["flip_rate"] in (0.0, 1.0)
