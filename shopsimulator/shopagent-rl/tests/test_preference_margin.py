"""Tests for preference margin optimization logic."""
import math

import pytest
import torch

from experiment.grpo.preference_margin import (
    CANONICAL_INTENTS,
    INTENT_TO_INDEX,
    compute_preference_margin,
    compute_relation_losses,
    flip_loss,
    intent_from_action,
    preserve_loss,
)


class TestIntentMapping:
    """Test intent mapping from actions."""

    def test_commit_from_buy_now(self):
        """Click 'buy now' maps to COMMIT."""
        class Action:
            type = "click"
            value = "buy now"

        assert intent_from_action(Action()) == "COMMIT"

    def test_search_alternative_from_search(self):
        """Search action maps to SEARCH_ALTERNATIVE."""
        class Action:
            type = "search"
            value = "laptop"

        assert intent_from_action(Action()) == "SEARCH_ALTERNATIVE"

    def test_search_alternative_from_back(self):
        """Click 'back to search' maps to SEARCH_ALTERNATIVE."""
        class Action:
            type = "click"
            value = "back to search"

        assert intent_from_action(Action()) == "SEARCH_ALTERNATIVE"

    def test_select_option_from_click(self):
        """Generic click maps to SELECT_TARGET_OPTION."""
        class Action:
            type = "click"
            value = "option 3"

        assert intent_from_action(Action()) == "SELECT_TARGET_OPTION"

    def test_none_from_none(self):
        """None action returns None."""
        assert intent_from_action(None) is None


class TestPreferenceMargin:
    """Test preference margin computation."""

    def test_margin_single_pair(self):
        """Compute margin for a single pair."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])  # COMMIT dominant
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])  # SEARCH dominant

        margin = compute_preference_margin(
            log_p_original, log_p_cf, "COMMIT", "SEARCH_ALTERNATIVE"
        )

        # pref_original = log_p_commit - log_p_search = -0.1 - (-2.0) = 1.9
        # pref_cf = log_p_commit - log_p_search = -2.0 - (-0.1) = -1.9
        # margin = 1.9 - (-1.9) = 3.8
        expected = 1.9 - (-1.9)
        assert torch.isclose(margin, torch.tensor(expected))

    def test_margin_batch(self):
        """Compute margin for a batch of pairs."""
        log_p_original = torch.tensor([
            [-0.1, -2.0, -1.5],  # Pair 1: COMMIT dominant
            [-2.0, -0.1, -1.5],  # Pair 2: SEARCH dominant
        ])
        log_p_cf = torch.tensor([
            [-2.0, -0.1, -1.5],  # Pair 1: SEARCH dominant
            [-0.1, -2.0, -1.5],  # Pair 2: COMMIT dominant
        ])

        margin = compute_preference_margin(
            log_p_original, log_p_cf, "COMMIT", "SEARCH_ALTERNATIVE"
        )

        assert margin.shape == (2,)
        # Pair 1: margin should be positive (flipped correctly)
        assert margin[0] > 0
        # Pair 2: margin should be negative (flipped incorrectly)
        assert margin[1] < 0

    def test_margin_invalid_intent(self):
        """Unknown intent raises error."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])

        with pytest.raises(ValueError, match="Unknown intents"):
            compute_preference_margin(
                log_p_original, log_p_cf, "UNKNOWN", "COMMIT"
            )


class TestFlipLoss:
    """Test flip loss for decision-changing interventions."""

    def test_flip_loss_positive_margin(self):
        """Positive margin (correct flip) should have low loss."""
        margin = torch.tensor([3.8])  # Strong positive margin
        loss = flip_loss(margin)

        # Loss should be small when margin > 0
        assert loss < 0.5

    def test_flip_loss_negative_margin(self):
        """Negative margin (wrong flip) should have high loss."""
        margin = torch.tensor([-3.8])  # Strong negative margin
        loss = flip_loss(margin)

        # Loss should be large when margin < 0
        assert loss > 2.0

    def test_flip_loss_with_threshold(self):
        """Threshold shifts the decision boundary."""
        margin = torch.tensor([0.5])
        loss_no_threshold = flip_loss(margin, margin_threshold=0.0)
        loss_with_threshold = flip_loss(margin, margin_threshold=1.0)

        # With higher threshold, same margin should give higher loss
        assert loss_with_threshold > loss_no_threshold

    def test_flip_loss_temperature(self):
        """Temperature softens by compressing the loss toward the tau->inf limit.

        loss = softplus(-(margin - m) / tau), so raising tau pulls the scaled
        argument toward 0 and the loss toward softplus(0) = ln 2, whatever the
        sign of the margin. It does NOT monotonically lower the loss: for an
        already-satisfied (positive) margin a higher temperature gives a *larger*
        loss, because the reward for being correct is being blunted.
        """
        ln2 = math.log(2.0)

        # Satisfied margin: sharper (low tau) => closer to 0, softer => toward ln 2.
        satisfied = torch.tensor([1.0])
        assert flip_loss(satisfied, temperature=0.5) < flip_loss(satisfied, temperature=2.0)
        assert flip_loss(satisfied, temperature=2.0) < ln2

        # Violated margin: sharper => larger penalty, softer => decays toward ln 2.
        violated = torch.tensor([-1.0])
        assert flip_loss(violated, temperature=0.5) > flip_loss(violated, temperature=2.0)
        assert flip_loss(violated, temperature=2.0) > ln2

        # The shared limit: large tau drives either sign to ln 2.
        for m in (satisfied, violated):
            assert torch.isclose(flip_loss(m, temperature=1e4), torch.tensor(ln2), atol=1e-3)


class TestPreserveLoss:
    """Test preserve loss for decision-preserving interventions."""

    def test_preserve_loss_identical_distributions(self):
        """Identical distributions should have zero loss."""
        log_p = torch.tensor([-0.1, -2.0, -1.5])
        loss = preserve_loss(log_p, log_p)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_preserve_loss_different_distributions(self):
        """Different distributions should have positive loss."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])

        loss = preserve_loss(log_p_original, log_p_cf)
        assert loss > 0

    def test_preserve_loss_batch(self):
        """Preserve loss works with batches."""
        log_p_original = torch.randn(4, 3)
        log_p_cf = torch.randn(4, 3)

        loss = preserve_loss(log_p_original, log_p_cf)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0


class TestRelationLosses:
    """Test combined relation loss computation."""

    def test_decision_changing_uses_flip_loss(self):
        """Decision-changing intervention should use flip loss."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])

        result = compute_relation_losses(
            log_p_original,
            log_p_cf,
            "COMMIT",
            "SEARCH_ALTERNATIVE",
            is_decision_changing=True,
        )

        assert result["flip_loss"] > 0
        assert torch.isclose(result["preserve_loss"], torch.tensor(0.0))
        assert result["loss"] == result["flip_loss"]

    def test_decision_preserving_uses_preserve_loss(self):
        """Decision-preserving intervention should use preserve loss."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])

        result = compute_relation_losses(
            log_p_original,
            log_p_cf,
            "COMMIT",
            "COMMIT",  # Same intent = decision-preserving
            is_decision_changing=False,
        )

        assert torch.isclose(result["flip_loss"], torch.tensor(0.0))
        assert result["preserve_loss"] > 0
        assert result["loss"] == result["preserve_loss"]

    def test_weights_are_applied(self):
        """Loss weights should be applied correctly."""
        log_p_original = torch.tensor([-0.1, -2.0, -1.5])
        log_p_cf = torch.tensor([-2.0, -0.1, -1.5])

        result = compute_relation_losses(
            log_p_original,
            log_p_cf,
            "COMMIT",
            "SEARCH_ALTERNATIVE",
            is_decision_changing=True,
            flip_weight=2.0,
        )

        assert result["loss"] == 2.0 * result["flip_loss"]


def test_canonical_intents_constant():
    """Verify canonical intents are properly defined."""
    assert "COMMIT" in CANONICAL_INTENTS
    assert "SEARCH_ALTERNATIVE" in CANONICAL_INTENTS
    assert "SELECT_TARGET_OPTION" in CANONICAL_INTENTS

    # All intents should have indices
    assert len(INTENT_TO_INDEX) == len(CANONICAL_INTENTS)
    assert set(INTENT_TO_INDEX.keys()) == set(CANONICAL_INTENTS)
