"""Attach the paired relation loss to veRL's actor loss without touching PPO.

Why a wrapper instead of editing ``ppo_loss``:

``ppo_loss`` receives only ``(model_output, data, dp_group)``. The relation loss
needs the *module* — it re-scores the full legal action set at each paired state,
which is extra forward passes that the rollout's single forward cannot provide.
Wrapping keeps two properties that matter:

* When ``preference_margin`` is off, ``loss_fn`` is the unmodified
  ``partial(ppo_loss, config=...)``. The Independent baseline is therefore
  byte-for-byte unchanged, which Gate B depends on.
* The PPO term is computed by the original function on the original data, so the
  relation loss can only *add* a term, never perturb the existing one.

The wrapper reads ``pair_id`` / ``side`` / ``state_text`` / ``expected_relation``
/ ``expected_action_intents`` off the micro-batch. ``ppo_loss`` internally does
``data.select(...)``, which returns a new TensorDict, so those columns are still
present on the object handed to the wrapper.

Note on cost: scoring ~20-30 candidates over a ~1.2k-token prompt with grad
enabled needs gradient checkpointing to fit on one 48 GiB device (12.6 GiB with,
>48 GiB without). ``require_gradient_checkpointing`` defaults to True and fails
loudly rather than OOM-ing several minutes into a step.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

import verl.utils.tensordict_utils as tu

_ROW_FIELDS = (
    "pair_id",
    "side",
    "state_text",
    "expected_relation",
    "expected_action_intents",
    "intervention_type",
)


def extract_pair_rows(data: Any) -> list[dict[str, Any]]:
    """Pull the per-row relation metadata out of a micro-batch.

    Returns [] when the batch carries no relation metadata at all, so a mixed run
    (environment-only micro-batches) is a no-op rather than an error.
    """
    if "pair_id" not in data.keys():
        return []

    columns: dict[str, list[Any]] = {}
    for field in _ROW_FIELDS:
        if field not in data.keys():
            return []
        value = tu.get(data, field)
        columns[field] = list(value) if not isinstance(value, list) else value

    size = len(columns["pair_id"])
    if any(len(col) != size for col in columns.values()):
        return []

    rows = []
    for i in range(size):
        # An empty pair_id marks an environment row: no pair, no relation signal.
        if not str(columns["pair_id"][i] or ""):
            continue
        rows.append({field: columns[field][i] for field in _ROW_FIELDS})
    return rows


def _module_supports_checkpointing(module: Any) -> bool:
    inner = getattr(module, "_fsdp_wrapped_module", module)
    return bool(getattr(inner, "gradient_checkpointing", False)) and bool(
        getattr(inner, "training", False)
    )


def make_relation_loss_fn(
    base_loss_fn: Callable,
    module_getter: Callable[[], Any],
    tokenizer: Any,
    *,
    flip_weight: float = 1.0,
    preserve_weight: float = 1.0,
    anchor_weight: float = 1.0,
    margin_threshold: float = 0.0,
    temperature: float = 1.0,
    relation_coeff: float = 1.0,
    require_gradient_checkpointing: bool = True,
) -> Callable:
    """Wrap ``base_loss_fn`` so the relation loss is added to its scalar loss.

    ``module_getter`` is a callable rather than a module because the engine builds
    (and may re-wrap) the module after the loss function is installed.
    """
    from experiment.grpo.relation_batch import compute_batch_relation_loss

    def relation_loss_fn(model_output, data, dp_group=None):
        loss, metrics = base_loss_fn(model_output=model_output, data=data, dp_group=dp_group)

        rows = extract_pair_rows(data)
        if not rows:
            return loss, metrics

        module = module_getter()
        if module is None:
            return loss, metrics

        if require_gradient_checkpointing and not _module_supports_checkpointing(module):
            raise RuntimeError(
                "preference_margin relation loss needs gradient checkpointing on a "
                "training module: scoring the legal action set with grad enabled "
                "needs >48 GiB without it. Enable "
                "actor_rollout_ref.model.enable_gradient_checkpointing, or set "
                "paired.require_gradient_checkpointing=false to accept the risk."
            )

        relation_loss, stats = compute_batch_relation_loss(
            module,
            tokenizer,
            rows,
            flip_weight=flip_weight,
            preserve_weight=preserve_weight,
            anchor_weight=anchor_weight,
            margin_threshold=margin_threshold,
            temperature=temperature,
            device=loss.device,
        )

        if stats.get("pairs_used", 0) > 0:
            loss = loss + relation_coeff * relation_loss.to(loss.dtype)

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                metrics[f"relation/{key}"] = float(value)
        return loss, metrics

    return relation_loss_fn


__all__ = ["extract_pair_rows", "make_relation_loss_fn"]
