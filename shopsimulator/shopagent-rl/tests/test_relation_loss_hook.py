"""Tests for the actor-side relation loss hook and its config gating."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

import verl.utils.tensordict_utils as tu
from experiment.grpo.relation_loss_hook import (
    _has_checkpointing_flag,
    _has_transformers_checkpointing,
    _has_verl_offload_checkpointing,
    _module_supports_checkpointing,
    extract_pair_rows,
    make_relation_loss_fn,
)
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


def _partner_rows(pair_id="p1", rollout_n=4):
    """Rows as they look after the trainer attaches partner state."""
    rel = ["COMMIT", "SEARCH_ALTERNATIVE"]
    state_o = _state(["buy now", "back to search"])
    state_c = _state(["back to search", "buy now"])
    rows = []
    for side in ("original", "counterfactual"):
        for _ in range(rollout_n):
            rows.append({
                "pair_id": pair_id, "side": side,
                "state_text": state_o if side == "original" else state_c,
                "expected_relation": rel,
                "expected_action_intents": (
                    ["COMMIT"] if side == "original" else ["SEARCH_ALTERNATIVE"]),
                "intervention_type": "price_above_budget",
                "partner_state_text": state_c if side == "original" else "",
                "partner_expected_action_intents": (
                    ["SEARCH_ALTERNATIVE"] if side == "original" else []),
            })
    return rows


def _partner_batch(rows, mini_batch_pair_rows):
    n = len(rows)
    td = TensorDict({}, batch_size=[n])
    td["response_mask"] = torch.ones(n, 4, dtype=torch.long)
    fields = _FIELDS + ("partner_state_text", "partner_expected_action_intents")
    for field in fields:
        td[field] = NonTensorStack.from_list([NonTensorData(r.get(field)) for r in rows])
    tu.assign_non_tensor(td, relation_pair_rows=mini_batch_pair_rows)
    return td


class TestAccumulationScaling:
    """The engine sums micro-batch losses with no 1/N.

    Without rescaling, the relation term's weight would be the number of
    pair-carrying micro-batches -- 8 at rollout.n=4 -- so `relation_coeff` would
    silently change meaning whenever rollout.n or ppo_micro_batch_size changed.
    """

    def _fn(self, coeff=1.0):
        model = StubModel()
        return model, make_relation_loss_fn(
            lambda model_output=None, data=None, dp_group=None: (
                torch.zeros((), requires_grad=True), {}),
            lambda: model, StubTokenizer(), relation_coeff=coeff,
            require_gradient_checkpointing=False)

    def _summed_over_singles(self, fn, rows, pair_rows):
        total = torch.zeros(())
        for row in rows:
            loss, _ = fn(None, _partner_batch([row], pair_rows))
            total = total + loss.detach()
        return float(total)

    @pytest.mark.parametrize("rollout_n", [1, 2, 4])
    def test_sum_over_micro_batches_equals_single_micro_batch(self, rollout_n):
        _, fn = self._fn()
        rows = _partner_rows(rollout_n=rollout_n)
        pair_rows = sum(1 for r in rows if r["partner_state_text"])
        whole, _ = fn(None, _partner_batch(rows, pair_rows))
        summed = self._summed_over_singles(fn, rows, pair_rows)
        assert float(whole.detach()) == pytest.approx(summed, abs=1e-4)

    def test_value_is_invariant_to_rollout_n(self):
        """The whole point: relation_coeff must not move with rollout.n."""
        _, fn = self._fn()
        values = []
        for rollout_n in (1, 2, 4, 8):
            rows = _partner_rows(rollout_n=rollout_n)
            pair_rows = sum(1 for r in rows if r["partner_state_text"])
            values.append(self._summed_over_singles(fn, rows, pair_rows))
        assert max(values) - min(values) < 1e-4

    def test_value_is_invariant_to_pair_count(self):
        _, fn = self._fn()
        one = _partner_rows("p1")
        two = _partner_rows("p1") + _partner_rows("p2")
        a = self._summed_over_singles(fn, one, sum(1 for r in one if r["partner_state_text"]))
        b = self._summed_over_singles(fn, two, sum(1 for r in two if r["partner_state_text"]))
        assert a == pytest.approx(b, abs=1e-4)

    def test_scale_is_reported(self):
        _, fn = self._fn()
        rows = _partner_rows(rollout_n=4)
        _, metrics = fn(None, _partner_batch(rows[:1], 4))
        assert metrics["relation/accumulation_scale"] == pytest.approx(0.25)

    def test_absent_denominator_does_not_rescale(self):
        """A caller without the trainer hook keeps the unscaled mean."""
        model = StubModel()
        fn = make_relation_loss_fn(
            lambda model_output=None, data=None, dp_group=None: (
                torch.zeros((), requires_grad=True), {}),
            lambda: model, StubTokenizer(), require_gradient_checkpointing=False)
        _, metrics = fn(None, _batch(_pair_rows()))
        assert metrics["relation/accumulation_scale"] == pytest.approx(1.0)

    def test_coeff_still_scales_linearly(self):
        rows = _partner_rows(rollout_n=4)
        pair_rows = sum(1 for r in rows if r["partner_state_text"])
        _, fn1 = self._fn(coeff=1.0)
        _, fn3 = self._fn(coeff=3.0)
        a = self._summed_over_singles(fn1, rows, pair_rows)
        b = self._summed_over_singles(fn3, rows, pair_rows)
        assert b == pytest.approx(3.0 * a, rel=1e-4)


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


class TestCheckpointingDetection:
    """`gradient_checkpointing` is never set on the top-level module.

    `gradient_checkpointing_enable()` sets it on the submodules owning the
    checkpointed blocks -- for `Qwen3ForCausalLM`, the inner `Qwen3Model` and its
    decoder layers. Verified on the real Qwen3-1.7B-Base: after enabling,
    `getattr(model, "gradient_checkpointing")` is *absent* while
    `model.model.gradient_checkpointing` is True. A top-level attribute check
    therefore rejects a correctly configured model, which is what took down the
    first Gate B paired run. These tests use the same nesting so a stub cannot
    hide the regression.
    """

    def test_flag_on_submodule_is_detected(self):
        inner = torch.nn.Linear(2, 2)
        inner.gradient_checkpointing = True
        outer = torch.nn.Sequential(inner)
        assert _has_checkpointing_flag(outer) is True

    def test_no_flag_anywhere_is_false(self):
        assert _has_checkpointing_flag(torch.nn.Sequential(torch.nn.Linear(2, 2))) is False

    def test_is_gradient_checkpointing_property_wins(self):
        """transformers exposes the recursive test directly; prefer it."""
        class HasProperty:
            is_gradient_checkpointing = True
        assert _has_checkpointing_flag(HasProperty()) is True

    def test_property_false_is_respected_over_submodule_walk(self):
        class SaysNo:
            is_gradient_checkpointing = False
            gradient_checkpointing = True
        assert _has_checkpointing_flag(SaysNo()) is False

    def test_plain_object_without_modules_is_false(self):
        assert _has_checkpointing_flag(object()) is False

    def test_submodule_flag_plus_eval_mode_still_fails_the_guard(self):
        """Checkpointing is gated on `self.training`, so eval() means no saving."""
        inner = torch.nn.Linear(2, 2)
        inner.gradient_checkpointing = True
        outer = torch.nn.Sequential(inner)
        outer.eval()
        assert _module_supports_checkpointing(outer) is False
        outer.train()
        assert _module_supports_checkpointing(outer) is True

    def test_nested_flag_passes_the_full_guard_through_fsdp(self):
        inner = torch.nn.Linear(2, 2)
        inner.gradient_checkpointing = True
        outer = torch.nn.Sequential(inner)
        outer.train()
        assert _module_supports_checkpointing(_Wrapped(outer)) is True


class TestVerlOffloadCheckpointing:
    """`enable_activation_offload=True` replaces transformers' checkpointing.

    veRL disables the transformers implementation (the two are incompatible) and
    wraps each layer's `forward` to route through `torch.utils.checkpoint` itself.
    The transformers flag is then False on a model that *is* checkpointed --
    which is what failed the second Gate B paired run. Built with the real
    `ActivationHandler` so the detection is tested against the actual wrapper, not
    a hand-made lookalike.
    """

    def _wrapped(self, enable_ckpt=True, n_layers=4):
        from verl.utils.activation_offload import (
            ActivationHandler,
            FSDPParameterFilter,
            get_activation_offload_context,
        )
        layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(n_layers)])
        model = torch.nn.Sequential(layers)
        ctx, sync = get_activation_offload_context(n_layers - 1, n_layers, FSDPParameterFilter())
        handler = ActivationHandler(ctx, sync, FSDPParameterFilter(), enable_ckpt=enable_ckpt)
        for layer in layers:
            handler.wrap_module_forward_method(layer)
        return model

    def test_offload_checkpointing_is_detected(self):
        assert _has_verl_offload_checkpointing(self._wrapped()) is True

    def test_transformers_flag_stays_false(self):
        """The exact trap: a checkpointed model reporting False on the old check."""
        model = self._wrapped()
        assert _has_transformers_checkpointing(model) is False
        assert _has_checkpointing_flag(model) is True

    def test_offload_without_checkpointing_is_not_accepted(self):
        assert _has_verl_offload_checkpointing(self._wrapped(enable_ckpt=False)) is False

    def test_unwrapped_model_is_not_accepted(self):
        assert _has_verl_offload_checkpointing(torch.nn.Sequential(torch.nn.Linear(4, 4))) is False

    def test_full_guard_respects_train_mode(self):
        """veRL's handler also short-circuits on `not module.training`."""
        model = self._wrapped()
        model.eval()
        assert _module_supports_checkpointing(model) is False
        model.train()
        assert _module_supports_checkpointing(model) is True

    def test_plain_object_is_not_accepted(self):
        assert _has_verl_offload_checkpointing(object()) is False

    def test_relation_loss_runs_under_offload_checkpointing(self):
        """End to end: the guard must let the paired run proceed.

        `StubModel` is not an `nn.Module`, so the wrapped layers are exposed via
        an explicit `modules()` -- which is the only thing the detector walks.
        """
        model = StubModel()
        wrapped = self._wrapped()
        model.modules = wrapped.modules
        model.training = True
        fn = make_relation_loss_fn(_base_loss, lambda: model, StubTokenizer())
        loss, metrics = fn(model_output=None, data=_batch(_pair_rows()))
        assert metrics["relation/pairs_used"] == 1.0 and loss.item() > 2.0


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
