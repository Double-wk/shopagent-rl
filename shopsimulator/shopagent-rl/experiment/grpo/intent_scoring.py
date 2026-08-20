"""Intent-level log probability computation for preference margin.

This module provides utilities for computing log probabilities of canonical intents
given a model and tokenizer. This is used to implement preference margin optimization.

The key idea: for each state (prompt), we compute the model's log probabilities
for a small set of canonical intent phrases like "COMMIT", "SEARCH_ALTERNATIVE", etc.
These log probabilities are then used to compute preference margins between
original and counterfactual states.
"""
from __future__ import annotations

from typing import Any

import torch


# Canonical intent phrases that we'll score
# These are simplified representations of the actual actions
INTENT_PHRASES = {
    "COMMIT": "Action: click[buy now]",
    "SEARCH_ALTERNATIVE": "Action: search[",
    "SELECT_TARGET_OPTION": "Action: click[",
}


def compute_intent_log_probs_from_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, float]:
    """Compute log probabilities for canonical intents given a prompt.

    This is a placeholder implementation. The full version would:
    1. Encode the prompt
    2. For each intent phrase, encode prompt + phrase
    3. Compute log probabilities for the phrase tokens
    4. Return the sum/mean log prob for each intent

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt (state observation)
        device: Device to run computation on

    Returns:
        Dict mapping intent names to log probabilities
    """
    # Placeholder: return uniform distribution
    # Full implementation requires actual model forward pass
    num_intents = len(INTENT_PHRASES)
    uniform_log_prob = -torch.log(torch.tensor(num_intents, dtype=torch.float32))
    return {intent: float(uniform_log_prob) for intent in INTENT_PHRASES}


def extract_intent_log_probs_from_rollout(
    response_text: str,
    log_probs: torch.Tensor,
    response_tokens: list[int],
    tokenizer: Any,
) -> dict[str, float]:
    """Extract intent log probabilities from existing rollout data.

    This attempts to find canonical intent phrases in the response text
    and extract their corresponding log probabilities.

    Args:
        response_text: The model's generated response
        log_probs: Token-level log probabilities from the rollout
        response_tokens: Token IDs for the response
        tokenizer: The tokenizer

    Returns:
        Dict mapping intent names to log probabilities found in response
    """
    # Placeholder: find intent phrases and extract their log probs
    # Full implementation would:
    # 1. Tokenize each intent phrase
    # 2. Find occurrences in response_tokens
    # 3. Extract corresponding log_probs
    return {}


def get_simple_intent_prior(
    intent: str,
    predicted_intent: str | None = None,
) -> float:
    """Get a simple prior/logit for an intent based on prediction.

    This is a very simplified approximation used for initial testing.
    The full implementation should use actual model log probabilities.

    Args:
        intent: The canonical intent to score
        predicted_intent: The intent predicted from the response (if available)

    Returns:
        A logit value representing the model's preference for this intent
    """
    if predicted_intent == intent:
        return 2.0  # High logit for correct intent
    elif predicted_intent:
        return -2.0  # Low logit for other intents
    else:
        return 0.0  # Neutral for unknown


def compute_intent_logits_from_predictions(
    predicted_intents: list[str | None],
    canonical_intents: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute intent logits from rollouts based on predicted intents.

    This is a simplified proxy for actual model log probabilities.
    It converts predicted intents into logits that can be used
    for computing preference margins during development/testing.

    Args:
        predicted_intents: List of predicted intents from rollouts
        canonical_intents: List of canonical intents to compute logits for

    Returns:
        Dict mapping canonical intents to logits (one per rollout)
    """
    if canonical_intents is None:
        from experiment.grpo.preference_margin import CANONICAL_INTENTS
        canonical_intents = CANONICAL_INTENTS

    # For each rollout, create a logit vector
    # High logit for the predicted intent, low for others
    logits = []
    for pred in predicted_intents:
        if pred is None:
            # Uniform distribution for unknown predictions
            logits.append(torch.zeros(len(canonical_intents)))
        else:
            vec = []
            for intent in canonical_intents:
                if intent == pred:
                    vec.append(3.0)  # High logit for predicted intent
                else:
                    vec.append(-3.0)  # Low logit for others
            logits.append(torch.tensor(vec))

    # Stack into (num_rollouts, num_intents) tensor
    stacked = torch.stack(logits) if logits else torch.zeros(0, len(canonical_intents))

    return {intent: stacked[:, i] for i, intent in enumerate(canonical_intents)}


def convert_logits_to_log_probs(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert logits to log probabilities for each intent.

    Args:
        logits: Dict mapping intent names to logit tensors

    Returns:
        Dict mapping intent names to log prob tensors
    """
    log_probs = {}
    first_tensor = next(iter(logits.values())) if logits else None

    if first_tensor is None or first_tensor.ndim == 0:
        return {k: torch.log_softmax(v, dim=-1) for k, v in logits.items()}

    # For batched tensors, apply log_softmax along the last dim
    # But we need to reconstruct the full tensor first
    intent_names = list(logits.keys())
    stacked = torch.stack([logits[name] for name in intent_names], dim=-1)

    # Apply log_softmax to get probabilities over intents
    log_probs_stacked = torch.log_softmax(stacked, dim=-1)

    # Split back into dict
    for i, name in enumerate(intent_names):
        log_probs[name] = log_probs_stacked[..., i]

    return log_probs
