"""Tests for RayPPOTrainer._maybe_attach_partner_states.

This method is the reason PPO's micro-batch size can stay at 1 under the
preference-margin objective. The relation loss needs both sides of a pair, but
with `ppo_micro_batch_size_per_gpu=1` the two rows never share a micro-batch, and
forcing them together (`force_group_size = 2 * rollout.n`) would multiply PPO's
own memory by 8 and change gradient-accumulation granularity for the paired arm
only. Instead the trainer -- which sees the whole batch -- copies the partner's
state onto each `original` row.
"""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(enabled=True, mode="preference_margin"):
    stub = object.__new__(RayPPOTrainer)
    stub.config = OmegaConf.create(
        {"algorithm": {"paired_intervention": {"enabled": enabled, "mode": mode}}}
    )
    return stub


def _batch(rows):
    n = len(rows)
    non_tensor = {
        key: np.array([r.get(key) for r in rows], dtype=object)
        for key in ("pair_id", "side", "state_text", "expected_action_intents")
    }
    return DataProto.from_dict(
        tensors={"input_ids": torch.zeros(n, 2, dtype=torch.long)},
        non_tensors=non_tensor,
    )


def _row(pair_id, side, state, expected):
    return {"pair_id": pair_id, "side": side, "state_text": state,
            "expected_action_intents": expected}


def _interleaved(rollout_n=4):
    """Mimic `batch.repeat(interleave=True)`: n rollouts per row, sides apart."""
    rows = []
    for side, state, exp in (
        ("original", "STATE_O", ["COMMIT"]),
        ("counterfactual", "STATE_C", ["SEARCH_ALTERNATIVE"]),
    ):
        rows.extend(_row("p1", side, state, exp) for _ in range(rollout_n))
    return rows


class TestGating:
    def test_disabled_attaches_nothing(self):
        batch = _batch(_interleaved())
        _trainer(enabled=False)._maybe_attach_partner_states(batch)
        assert "partner_state_text" not in batch.non_tensor_batch

    def test_other_modes_attach_nothing(self):
        for mode in ("joint_bonus", "relational_residual", "explicit_relation"):
            batch = _batch(_interleaved())
            _trainer(mode=mode)._maybe_attach_partner_states(batch)
            assert "partner_state_text" not in batch.non_tensor_batch

    def test_missing_metadata_is_a_no_op_not_a_crash(self):
        """Environment-only batches lack the columns; must not raise."""
        batch = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(2, 2, dtype=torch.long)}
        )
        _trainer()._maybe_attach_partner_states(batch)
        assert "partner_state_text" not in batch.non_tensor_batch


class TestAttachment:
    def test_original_rows_get_the_counterfactual_state(self):
        batch = _batch(_interleaved())
        _trainer()._maybe_attach_partner_states(batch)
        partner = batch.non_tensor_batch["partner_state_text"]
        sides = batch.non_tensor_batch["side"]
        for side, value in zip(sides, partner):
            assert value == ("STATE_C" if side == "original" else "")

    def test_counterfactual_rows_get_nothing(self):
        """Otherwise each pair would be counted twice per batch."""
        batch = _batch(_interleaved())
        _trainer()._maybe_attach_partner_states(batch)
        for side, value in zip(batch.non_tensor_batch["side"],
                               batch.non_tensor_batch["partner_state_text"]):
            if side == "counterfactual":
                assert value == ""

    def test_partner_anchor_targets_are_attached(self):
        batch = _batch(_interleaved())
        _trainer()._maybe_attach_partner_states(batch)
        exp = batch.non_tensor_batch["partner_expected_action_intents"]
        for side, value in zip(batch.non_tensor_batch["side"], exp):
            if side == "original":
                assert list(value) == ["SEARCH_ALTERNATIVE"]
            else:
                assert list(value) == []

    def test_columns_are_row_aligned(self):
        """The actor drops a partner column whose length does not match."""
        rows = _interleaved()
        batch = _batch(rows)
        _trainer()._maybe_attach_partner_states(batch)
        assert len(batch.non_tensor_batch["partner_state_text"]) == len(rows)

    def test_half_pair_gets_no_partner(self):
        batch = _batch([_row("p1", "original", "STATE_O", ["COMMIT"])])
        _trainer()._maybe_attach_partner_states(batch)
        assert list(batch.non_tensor_batch["partner_state_text"]) == [""]

    def test_environment_rows_in_mixed_batch_get_no_partner(self):
        rows = _interleaved(2) + [_row("", "", "", []) for _ in range(2)]
        batch = _batch(rows)
        _trainer()._maybe_attach_partner_states(batch)
        partner = list(batch.non_tensor_batch["partner_state_text"])
        assert partner[-2:] == ["", ""]
        assert partner[0] == "STATE_C"

    def test_multiple_pairs_do_not_cross_contaminate(self):
        rows = [
            _row("p1", "original", "O1", ["COMMIT"]),
            _row("p1", "counterfactual", "C1", ["SEARCH_ALTERNATIVE"]),
            _row("p2", "original", "O2", ["COMMIT"]),
            _row("p2", "counterfactual", "C2", ["COMMIT"]),
        ]
        batch = _batch(rows)
        _trainer()._maybe_attach_partner_states(batch)
        assert list(batch.non_tensor_batch["partner_state_text"]) == ["C1", "", "C2", ""]

    def test_survives_micro_batch_split(self):
        """A single-row chunk must still carry a complete pair.

        This is the property the whole design exists for: `chunk` is how the
        actor's micro-batches are formed, and after it the counterfactual row is
        gone from the chunk holding the original.
        """
        batch = _batch(_interleaved(4))
        _trainer()._maybe_attach_partner_states(batch)
        chunks = batch.chunk(8)
        first = chunks[0].non_tensor_batch
        assert first["side"][0] == "original"
        assert first["partner_state_text"][0] == "STATE_C"

    def test_grouping_consumes_the_attached_columns(self):
        """End-to-end: trainer attaches, group_pairs forms a pair from one row."""
        from experiment.grpo.relation_batch import group_pairs

        batch = _batch(_interleaved(2))
        _trainer()._maybe_attach_partner_states(batch)
        ntb = batch.non_tensor_batch
        row = {
            "pair_id": ntb["pair_id"][0],
            "side": ntb["side"][0],
            "state_text": ntb["state_text"][0],
            "expected_relation": ["COMMIT", "SEARCH_ALTERNATIVE"],
            "expected_action_intents": ntb["expected_action_intents"][0],
            "intervention_type": "price_above_budget",
            "partner_state_text": ntb["partner_state_text"][0],
            "partner_expected_action_intents": ntb["partner_expected_action_intents"][0],
        }
        pairs, counters = group_pairs([row])
        assert counters["pairs"] == 1 and counters["pairs_from_partner_state"] == 1
        assert pairs[0]["counterfactual"]["state_text"] == "STATE_C"
