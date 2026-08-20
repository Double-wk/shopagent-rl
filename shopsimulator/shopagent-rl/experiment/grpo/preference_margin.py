"""Policy-level preference margin optimization for intervention pairs.

Core idea: Instead of only rewarding "both sides correct", directly optimize
whether the policy's relative preference between two competing intents flips
in the correct direction after a decision-changing intervention.

For a price-CF pair where the original should COMMIT and the CF should SEARCH:
- Original world: π(COMMIT|x) > π(SEARCH|x)
- Counterfactual world: π(SEARCH|x') > π(COMMIT|x')

The Interventional Preference Margin:
    M_θ(x,x') = log π(z_o|x)/π(z_c|x) - log π(z_o|x')/π(z_c|x')

For decision-changing: encourage M_θ > m via flip loss
For decision-preserving: encourage preference stability via JS divergence
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# Canonical intent set for policy-level optimization
CANONICAL_INTENTS = [
    "COMMIT",
    "SEARCH_ALTERNATIVE",
    "SELECT_TARGET_OPTION",
    # Future extensions can include ASK, NAVIGATE, etc.
]

INTENT_TO_INDEX = {intent: idx for idx, intent in enumerate(CANONICAL_INTENTS)}


def intent_from_action(action: Any) -> str | None:
    """Map a parsed action to canonical intent.

    This is the same mapping used in counterfactual grading, duplicated here
    to avoid import cycles during rollout scoring.
    """
    if action is None:
        return None
    action_type = str(getattr(action, "type", "")).lower()
    value = str(getattr(action, "value", "")).strip().lower()
    if action_type == "click" and value == "buy now":
        return "COMMIT"
    if action_type == "search":
        return "SEARCH_ALTERNATIVE"
    if action_type == "click" and value in {"back to search", "< prev"}:
        return "SEARCH_ALTERNATIVE"
    if action_type == "click":
        return "SELECT_TARGET_OPTION"
    return None


def compute_preference_margin(
    log_probs_original: torch.Tensor,
    log_probs_cf: torch.Tensor,
    intent_original: str,
    intent_cf: str,
) -> torch.Tensor:
    """Compute interventional preference margin for a single pair.

    Args:
        log_probs_original: Log probabilities for canonical intents in original world
                            Shape: (num_intents,) or (batch, num_intents)
        log_probs_cf: Log probabilities for canonical intents in counterfactual world
                       Shape: (num_intents,) or (batch, num_intents)
        intent_original: The correct intent for the original world (e.g., "COMMIT")
        intent_cf: The correct intent for the counterfactual world (e.g., "SEARCH_ALTERNATIVE")

    Returns:
        Preference margin M_θ(x,x'). Positive means preference flipped correctly.
        Shape: () or (batch,)
    """
    if intent_original not in INTENT_TO_INDEX or intent_cf not in INTENT_TO_INDEX:
        raise ValueError(f"Unknown intents: {intent_original}, {intent_cf}")

    idx_o = INTENT_TO_INDEX[intent_original]
    idx_c = INTENT_TO_INDEX[intent_cf]

    # log π(z_o|x) - log π(z_c|x) for original world
    if log_probs_original.ndim == 1:
        pref_original = log_probs_original[idx_o] - log_probs_original[idx_c]
    else:
        pref_original = log_probs_original[:, idx_o] - log_probs_original[:, idx_c]

    # log π(z_o|x') - log π(z_c|x') for counterfactual world
    if log_probs_cf.ndim == 1:
        pref_cf = log_probs_cf[idx_o] - log_probs_cf[idx_c]
    else:
        pref_cf = log_probs_cf[:, idx_o] - log_probs_cf[:, idx_c]

    # M_θ = pref_original - pref_cf
    margin = pref_original - pref_cf
    return margin


def flip_loss(margin: torch.Tensor, margin_threshold: float = 0.0, temperature: float = 1.0) -> torch.Tensor:
    """Compute flip loss for decision-changing interventions.

    We want M_θ > m, so we use logistic loss: -log σ((M_θ - m) / τ)

    Args:
        margin: Preference margin M_θ(x,x')
        margin_threshold: The minimum margin m (default 0)
        temperature: Temperature τ for softening (default 1.0)

    Returns:
        Scalar loss value
    """
    scaled = (margin - margin_threshold) / temperature
    # -log σ(x) = log(1 + exp(-x)) = softplus(-x)
    return F.softplus(-scaled).mean()


def preserve_loss(
    log_probs_original: torch.Tensor,
    log_probs_cf: torch.Tensor,
) -> torch.Tensor:
    """Compute preserve loss for decision-preserving interventions.

    Use JS divergence between intent distributions to encourage stability.

    Args:
        log_probs_original: Log probs for canonical intents in original world
        log_probs_cf: Log probs for canonical intents in counterfactual world

    Returns:
        Scalar JS divergence loss
    """
    p = F.softmax(log_probs_original, dim=-1)
    q = F.softmax(log_probs_cf, dim=-1)

    # KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    kl_pq = F.kl_div(q, p, reduction="batchmean")
    kl_qp = F.kl_div(p, q, reduction="batchmean")

    # JS(P, Q) = 0.5 * (KL(P || M) + KL(Q || M)) where M = (P + Q) / 2
    m = (p + q) / 2
    kl_pm = F.kl_div(m, p, reduction="batchmean")
    kl_qm = F.kl_div(m, q, reduction="batchmean")
    js = 0.5 * (kl_pm + kl_qm)

    return js.mean()


def compute_relation_losses(
    log_probs_original: torch.Tensor,
    log_probs_cf: torch.Tensor,
    intent_original: str,
    intent_cf: str,
    is_decision_changing: bool,
    flip_weight: float = 1.0,
    preserve_weight: float = 1.0,
    margin_threshold: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute relation-level losses for a paired intervention.

    Args:
        log_probs_original: Log probs for canonical intents in original world
        log_probs_cf: Log probs for canonical intents in counterfactual world
        intent_original: The correct intent for original world
        intent_cf: The correct intent for counterfactual world
        is_decision_changing: True for decision-changing, False for decision-preserving
        flip_weight: Weight for flip loss (only used if decision-changing)
        preserve_weight: Weight for preserve loss (only used if decision-preserving)
        margin_threshold: Minimum margin threshold m
        temperature: Temperature τ for flip loss

    Returns:
        Dict with 'loss', 'flip_loss' (or 0), 'preserve_loss' (or 0), and 'margin'
    """
    margin = compute_preference_margin(log_probs_original, log_probs_cf, intent_original, intent_cf)

    if is_decision_changing:
        f_loss = flip_loss(margin, margin_threshold=margin_threshold, temperature=temperature)
        p_loss = torch.tensor(0.0, device=margin.device)
        total = flip_weight * f_loss
    else:
        f_loss = torch.tensor(0.0, device=margin.device)
        p_loss = preserve_loss(log_probs_original, log_probs_cf)
        total = preserve_weight * p_loss

    return {
        "loss": total,
        "flip_loss": f_loss,
        "preserve_loss": p_loss,
        "margin": margin.detach(),
    }
