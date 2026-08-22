"""Assemble the paired relation loss from a micro-batch of CF rows.

The relation loss is defined on *pairs*, but the trainer hands us a flat
micro-batch. This module does three things and nothing else:

  1. group rows into (original, counterfactual) pairs by ``pair_id``
  2. score each side's canonical-intent distribution under the current policy
  3. combine them into flip / preserve / anchor terms

Design notes that are easy to get wrong:

* A pair contributes only when *both* sides are reachable from the same
  micro-batch. A half-pair carries no relational signal, so it is counted in
  ``dropped_incomplete`` and skipped rather than silently treated as a singleton.
  "Reachable" is deliberately weaker than "both rows present": an ``original``
  row carrying ``partner_state_text`` completes its own pair. That is what the
  trainer attaches (``_maybe_attach_partner_states``), and it is why PPO's
  micro-batch size can stay at 1 -- forcing the two rows into one micro-batch
  would multiply PPO's own memory by ``2 * rollout.n``. Since the relation loss
  reads only the state (the row's prompt), which is identical across all ``n``
  rollouts of a side, carrying it is equivalent to co-locating the rows.
* ``expected_relation`` is the pair-level ``[z_original, z_counterfactual]``.
  Decision-changing is exactly ``z_o != z_c``; there is no separate label to
  trust, and deriving it removes a class of metadata inconsistency.
* The anchor term is what keeps the margin honest. A margin is a *difference* of
  differences, so a policy can satisfy it by making the wrong intent less wrong
  on one side. Anchoring each side to its own ``expected_action_intents``
  removes that degenerate solution.
* Rows whose state has no rendered button list (``sample_mode='environment'``)
  are unscorable by construction and are excluded before pairing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch

from experiment.grpo.intent_policy_scoring import intent_log_probs
from experiment.grpo.preference_margin import (
    CANONICAL_INTENTS,
    INTENT_TO_INDEX,
    compute_relation_losses,
)

ORIGINAL = "original"
COUNTERFACTUAL = "counterfactual"


def _as_str(value: Any) -> str:
    """Normalize numpy scalars / bytes / None into a plain str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _as_list(value: Any) -> list[str]:
    """Normalize numpy arrays, lists, JSON strings and scalars into list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json
            try:
                parsed = json.loads(text)
            except ValueError:
                return []
            return [_as_str(v) for v in parsed]
        return [text] if text else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_str(v) for v in value]
    return [_as_str(value)]


def group_pairs(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group flat rows into complete pairs.

    Each row needs ``pair_id``, ``side``, ``state_text`` and ``expected_relation``.
    An ``original`` row may additionally carry ``partner_state_text`` (plus
    ``partner_expected_action_intents``), in which case it completes its own pair
    without the counterfactual row being in this micro-batch.

    Returns (pairs, counters). Every returned pair has both sides and a
    two-element expected relation, so downstream code needs no further guards.
    """
    counters = {"rows": len(rows), "dropped_no_pair_id": 0, "dropped_bad_side": 0,
                "dropped_incomplete": 0, "dropped_bad_relation": 0,
                "pairs_from_partner_state": 0}
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        pair_id = _as_str(row.get("pair_id"))
        if not pair_id:
            counters["dropped_no_pair_id"] += 1
            continue
        side = _as_str(row.get("side"))
        if side not in (ORIGINAL, COUNTERFACTUAL):
            counters["dropped_bad_side"] += 1
            continue
        # Last write wins: a repeated (pair_id, side) is a sampling artifact, and
        # taking one of them keeps the pair usable instead of dropping it.
        by_pair[pair_id][side] = row

        partner_state = _as_str(row.get("partner_state_text"))
        if side == ORIGINAL and partner_state and COUNTERFACTUAL not in by_pair[pair_id]:
            # Synthesize the far side from what this row carries. Only the fields
            # the loss actually reads are needed: the state to score, and the
            # anchor targets for that state. A real counterfactual row, if one
            # also lands in this micro-batch, is preferred -- hence the guard and
            # the ordering (a real row seen later overwrites this).
            by_pair[pair_id][COUNTERFACTUAL] = {
                "pair_id": pair_id,
                "side": COUNTERFACTUAL,
                "state_text": partner_state,
                "expected_action_intents": row.get("partner_expected_action_intents"),
                "_synthesized": True,
            }

    pairs: list[dict[str, Any]] = []
    for pair_id, sides in by_pair.items():
        if ORIGINAL not in sides or COUNTERFACTUAL not in sides:
            counters["dropped_incomplete"] += len(sides)
            continue
        relation = _as_list(sides[ORIGINAL].get("expected_relation"))
        if len(relation) != 2 or any(z not in INTENT_TO_INDEX for z in relation):
            counters["dropped_bad_relation"] += 2
            continue
        pairs.append({
            "pair_id": pair_id,
            ORIGINAL: sides[ORIGINAL],
            COUNTERFACTUAL: sides[COUNTERFACTUAL],
            "intent_original": relation[0],
            "intent_cf": relation[1],
            "is_decision_changing": relation[0] != relation[1],
            "intervention_type": _as_str(sides[ORIGINAL].get("intervention_type")),
        })

    counters["pairs"] = len(pairs)
    # Counted after assembly, not when synthesizing: a real counterfactual row
    # arriving later overwrites the synthesized side, and this stat should say how
    # many pairs actually *relied* on the carried state.
    counters["pairs_from_partner_state"] = sum(
        1 for p in pairs if p[COUNTERFACTUAL].get("_synthesized")
    )
    counters["decision_changing"] = sum(p["is_decision_changing"] for p in pairs)
    counters["decision_preserving"] = counters["pairs"] - counters["decision_changing"]
    return pairs, counters


def anchor_loss(log_probs: torch.Tensor, expected: Iterable[str]) -> torch.Tensor:
    """Per-side correctness anchor: -log P(expected intent | state).

    With several acceptable intents, the target is their combined mass, so the
    policy is free to choose among them. Returns a zero that still carries grad
    history when the expectation is missing or unavailable in this state, so the
    caller can sum unconditionally.
    """
    idx = [INTENT_TO_INDEX[z] for z in expected if z in INTENT_TO_INDEX]
    idx = [i for i in idx if torch.isfinite(log_probs[i])]
    if not idx:
        # Zero, but still attached to the graph so callers can sum
        # unconditionally. Only finite entries may be touched: -inf * 0 is NaN,
        # and unavailable intents are exactly the -inf ones.
        finite = log_probs[torch.isfinite(log_probs)]
        if finite.numel() == 0:
            return torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
        return finite.sum() * 0.0
    return -torch.logsumexp(log_probs[idx], dim=0)


def compute_batch_relation_loss(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    flip_weight: float = 1.0,
    preserve_weight: float = 1.0,
    anchor_weight: float = 1.0,
    margin_threshold: float = 0.0,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean relation loss over the complete pairs in ``rows``.

    Returns (loss, stats). The loss is a scalar tensor carrying grad history; it
    is exactly zero (and ``stats["pairs_used"] == 0``) when no complete,
    scorable pair is present, which lets the caller add it unconditionally.
    """
    pairs, stats = group_pairs(rows)

    totals: list[torch.Tensor] = []
    flips: list[float] = []
    preserves: list[float] = []
    anchors: list[torch.Tensor] = []
    margins: list[float] = []
    # Kept apart from `margins`: flip_rate is only defined on decision-changing
    # pairs, and the two lists interleave in iteration order.
    flip_margins: list[float] = []
    per_type: dict[str, list[float]] = defaultdict(list)
    unscorable = 0

    for pair in pairs:
        lp_o, info_o = intent_log_probs(model, tokenizer, pair[ORIGINAL]["state_text"], device=device)
        lp_c, info_c = intent_log_probs(model, tokenizer, pair[COUNTERFACTUAL]["state_text"], device=device)
        if not (info_o["scorable"] and info_c["scorable"]):
            unscorable += 1
            continue

        # The margin needs both compared intents available on both sides;
        # otherwise it is +-inf and would poison the batch mean.
        needed = (INTENT_TO_INDEX[pair["intent_original"]], INTENT_TO_INDEX[pair["intent_cf"]])
        if not all(torch.isfinite(lp[i]) for lp in (lp_o, lp_c) for i in needed):
            unscorable += 1
            continue

        result = compute_relation_losses(
            log_probs_original=lp_o,
            log_probs_cf=lp_c,
            intent_original=pair["intent_original"],
            intent_cf=pair["intent_cf"],
            is_decision_changing=pair["is_decision_changing"],
            flip_weight=flip_weight,
            preserve_weight=preserve_weight,
            margin_threshold=margin_threshold,
            temperature=temperature,
        )

        a_loss = (anchor_loss(lp_o, _as_list(pair[ORIGINAL].get("expected_action_intents")))
                  + anchor_loss(lp_c, _as_list(pair[COUNTERFACTUAL].get("expected_action_intents"))))
        totals.append(result["loss"] + anchor_weight * a_loss)
        anchors.append(a_loss.detach())
        if pair["is_decision_changing"]:
            # Only decision-changing pairs have a meaningful margin. For a
            # preserving pair intent_original == intent_cf, so
            # M = (a - a) - (b - b) = 0 for *any* distributions: averaging those
            # structural zeros in would silently drag margin_mean toward 0 and
            # make it look like the flip term is not moving.
            margins.append(float(result["margin"]))
            per_type[pair["intervention_type"]].append(float(result["margin"]))
            flips.append(float(result["flip_loss"].detach()))
            flip_margins.append(float(result["margin"]))
        else:
            preserves.append(float(result["preserve_loss"].detach()))

    stats["dropped_unscorable"] = unscorable
    stats["pairs_used"] = len(totals)

    if not totals:
        # Keep the dtype/device contract without fabricating a grad path.
        zero = torch.zeros((), device=device) if device is not None else torch.zeros(())
        stats.update({"relation_loss": 0.0, "margin_mean": 0.0, "flip_rate": 0.0})
        return zero, stats

    loss = torch.stack(totals).mean()
    stats.update({
        "relation_loss": float(loss.detach()),
        # Averaged over decision-changing pairs only; 0.0 means "none in batch".
        "margin_mean": sum(margins) / len(margins) if margins else 0.0,
        # Fraction of pairs already on the right side of the threshold. For
        # preserving pairs the target is |M| small, so they are not counted here.
        "flip_rate": (sum(m > margin_threshold for m in flip_margins) / len(flip_margins)
                      if flip_margins else 0.0),
        "flip_loss_mean": sum(flips) / len(flips) if flips else 0.0,
        "preserve_loss_mean": sum(preserves) / len(preserves) if preserves else 0.0,
        "anchor_loss_mean": float(torch.stack(anchors).mean()),
    })
    for name, values in per_type.items():
        stats[f"margin/{name}"] = sum(values) / len(values)
    return loss, stats


__all__ = [
    "CANONICAL_INTENTS",
    "anchor_loss",
    "compute_batch_relation_loss",
    "group_pairs",
]
