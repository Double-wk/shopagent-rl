"""Tests for the actor-side relation loss hook and its config gating."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from experiment.grpo.relation_loss_hook import extract_pair_rows, make_relation_loss_fn
from tests.test_intent_policy_scoring import StubModel, StubTokenizer, _state

_FIELDS = ("pair_id", "side", "state_text", "expected_relation",
           "expected_action_intents", "intervention_type")


def _batch(rows: list[dict], extra_tensor: bool = True) -> TensorDict:
    """Build a TensorDict shaped like the actor micro-batch."""
    n = len(rows)
    td = TensorDict({}, batch_size=[n])
    if extra_tensor:
        td["response_mask"] = torch.ones(n, 4, dtype=torch.long)
    for field in _FIELDS:
        td[field] = NonTensorStack.from_list([NonTensorData(r.get(field)) for r in rows])
    return td


def _pair_rows(buttons=("buy now", "back to search", "红色")):
    rel = ["COMMIT", "SEARCH_ALTERNATIVE"]
    state = _state(list(buttons))
    return [
        {"pair_id": "p1", "side": "original", "state_text": state,
         "expected_relation": rel, "expected_action_intents": ["COMMIT"],
         "intervention_type": "price_above_budget"},
        {"pair_id": "p1", "side": "counterfactual", "state_text": state,
         "expected_relation": rel, "expected_action_intents": ["SEARCH_ALTERNATIVE"],
         "intervention_type": "price_above_budget"},
    ]


def _env_rows(n=2):
    return [{"pair_id": "", "side": "", "state_text": "",
             "expected_relation": [], "expected_action_intents": [],
             "intervention_type": ""} for _ in range(n)]


def _base_loss(model_output=None, data=None, dp_group=None):
    return torch.tensor(2.0, requires_grad=True), {"actor/pg_loss": 2.0}


class TestExtractPairRows:
    def test_reads_pair_rows_from_micro_batch(self):
        rows = extract_pair_rows(_batch(_pair_rows()))
        assert len(rows) == 2
        assert rows[0]["side"] == "original" and rows[1]["side"] == "counterfactual"
        assert "可点击的按钮" in rows[0]["state_text"]

    def test_environment_rows_are_skipped(self):
        assert extract_pair_rows(_batch(_env_rows())) == []

    def test_mixed_batch_keeps_only_pair_rows(self):
        rows = extract_pair_rows(_batch(_env_rows(2) + _pair_rows()))
        assert len(rows) == 2 and all(r["pair_id"] == "p1" for r in rows)

    def test_batch_without_metadata_is_empty_not_error(self):
        td = TensorDict({"response_mask": torch.ones(2, 3)}, batch_size=[2])
        assert extract_pair_rows(td) == []

    def test_partial_metadata_is_rejected(self):
        """A batch missing one required column must not be half-interpreted."""
        td = _batch(_pair_rows())
        del td["state_text"]
        assert extract_pair_rows(td) == []

    def test_partner_columns_are_carried_when_present(self):
        rows = _pair_rows()
        td = _batch(rows)
        td["partner_state_text"] = NonTensorStack.from_list(
            [NonTensorData("far side"), NonTensorData("")]
        )
        td["partner_expected_action_intents"] = NonTensorStack.from_list(
            [NonTensorData(["SEARCH_ALTERNATIVE"]), NonTensorData([])]
        )
        out = extract_pair_rows(td)
        assert out[0]["partner_state_text"] == "far side"
        assert out[0]["partner_expected_action_intents"] == ["SEARCH_ALTERNATIVE"]

    def test_partner_columns_are_optional(self):
        """Absent partner columns degrade to the co-located path, not an error."""
        out = extract_pair_rows(_batch(_pair_rows()))
        assert len(out) == 2
        assert "partner_state_text" not in out[0]


class _Wrapped:
    """Module wrapper mimicking FSDP's attribute nesting."""

    def __init__(self, inner):
        self._fsdp_wrapped_module = inner

    def parameters(self):
        return self._fsdp_wrapped_module.parameters()

    def __call__(self, **kwargs):
        return self._fsdp_wrapped_module(**kwargs)


class TestMakeRelationLossFn:
    def _model(self):
        m = StubModel()
        m.gradient_checkpointing = True
        m.training = True
        return m

    def test_adds_to_base_loss_when_pairs_present(self):
        model = self._model()
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        loss, metrics = fn(model_output=None, data=_batch(_pair_rows()))
        assert loss.item() > 2.0
        assert metrics["relation/pairs_used"] == 1.0
        assert "relation/margin_mean" in metrics

    def test_passthrough_when_no_pairs(self):
        model = self._model()
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        loss, metrics = fn(model_output=None, data=_batch(_env_rows()))
        assert loss.item() == 2.0
        assert not any(k.startswith("relation/") for k in metrics)

    def test_base_metrics_are_preserved(self):
        model = self._model()
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        _, metrics = fn(model_output=None, data=_batch(_pair_rows()))
        assert metrics["actor/pg_loss"] == 2.0

    def test_relation_coeff_scales_added_term(self):
        model = self._model()
        one = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        two = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer(),
                                    relation_coeff=2.0)
        a, _ = one(model_output=None, data=_batch(_pair_rows()))
        b, _ = two(model_output=None, data=_batch(_pair_rows()))
        assert (b.item() - 2.0) == 2 * (a.item() - 2.0)

    def test_gradient_flows_into_module(self):
        model = self._model()
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        loss, _ = fn(model_output=None, data=_batch(_pair_rows()))
        grad = torch.autograd.grad(loss, model.weight, allow_unused=True)[0]
        assert grad is not None and grad.abs().sum() > 0

    def test_missing_checkpointing_fails_loudly(self):
        model = StubModel()
        model.gradient_checkpointing = False
        model.training = True
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        try:
            fn(model_output=None, data=_batch(_pair_rows()))
        except RuntimeError as exc:
            assert "gradient checkpointing" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    def test_checkpointing_check_sees_through_fsdp_wrapper(self):
        model = self._model()
        fn = make_relation_loss_fn(_base_loss, lambda: _Wrapped(model), StubTokenizer())
        loss, metrics = fn(model_output=None, data=_batch(_pair_rows()))
        assert metrics["relation/pairs_used"] == 1.0 and loss.item() > 2.0

    def test_override_allows_running_without_checkpointing(self):
        model = StubModel()
        model.gradient_checkpointing = False
        model.training = True
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer(),
                                   require_gradient_checkpointing=False)
        loss, _ = fn(model_output=None, data=_batch(_pair_rows()))
        assert loss.item() > 2.0

    def test_absent_module_is_a_passthrough(self):
        fn = make_relation_loss_fn(_base_loss, lambda: None, StubTokenizer())
        loss, _ = fn(model_output=None, data=_batch(_pair_rows()))
        assert loss.item() == 2.0


class TestConfigGating:
    """The wrapper must attach for preference_margin and for nothing else."""

    def _worker_stub(self, enabled, mode):
        from verl.workers.engine_workers import ActorRolloutRefWorker

        cfg = OmegaConf.create({
            "algorithm": {"paired_intervention": {"enabled": enabled, "mode": mode}},
            "rollout": {"n": 4},
        })
        stub = object.__new__(ActorRolloutRefWorker)
        stub.config = cfg
        return stub

    def test_paired_config_none_when_disabled(self):
        assert self._worker_stub(False, "preference_margin")._paired_config() is None

    def test_paired_config_present_when_enabled(self):
        cfg = self._worker_stub(True, "preference_margin")._paired_config()
        assert cfg is not None and cfg["mode"] == "preference_margin"

    def test_other_modes_do_not_wrap(self):
        for mode in ("joint_bonus", "relational_residual", "explicit_relation"):
            stub = self._worker_stub(True, mode)
            sentinel = object()
            assert stub._maybe_wrap_relation_loss(sentinel, None) is sentinel

    def test_disabled_does_not_wrap(self):
        stub = self._worker_stub(False, "preference_margin")
        sentinel = object()
        assert stub._maybe_wrap_relation_loss(sentinel, None) is sentinel

    def test_micro_batching_is_left_untouched(self):
        """The worker must not reach into PPO's micro-batch split.

        An earlier version set `force_group_size = 2 * rollout.n` to co-locate
        pair sides. That multiplies PPO's micro-batch by 8 on a
        `ppo_micro_batch_size_per_gpu=1` chosen to fit 48 GiB, and changes
        gradient-accumulation granularity for the paired arm only -- which would
        break the "baseline unchanged" comparison. Pairs are completed by the
        trainer attaching `partner_state_text` instead.
        """
        stub = self._worker_stub(True, "preference_margin")
        assert not hasattr(stub, "_maybe_set_relation_group_size")
        assert getattr(stub, "_relation_force_group_size", None) is None
