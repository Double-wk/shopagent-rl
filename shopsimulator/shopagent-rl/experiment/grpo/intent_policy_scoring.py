"""Canonical-intent policy scoring over the restricted legal action set.

WHY NOT FIXED PHRASES: scoring a fixed phrase per intent does not give a
comparable margin. The previous basis used "Action: click[" for
SELECT_TARGET_OPTION and "Action: click[buy now]" for COMMIT — the former is a
strict *prefix* of the latter, so P(SELECT) >= P(COMMIT) holds by construction
and their log-odds carries no information. That pair is exactly the option-swap
margin. Unequal token lengths also bias sum-log-prob toward shorter phrases.

INSTEAD: at each decision state, take the actions the environment actually
accepts, score each one, softmax-normalize over that set, then group candidates
by intent with logsumexp. The result is a normalized distribution over mutually
exclusive events, so the preference margin is well defined.

The legal set comes from the observation's rendered clickable list, not from the
row's ``allowed_actions`` column — that column holds the *expected correct*
action (1-2 entries) and cannot serve as a normalizer.
"""
from __future__ import annotations

import json
import re
from typing import Any

import torch

from experiment.grpo.preference_margin import CANONICAL_INTENTS

# Matches the line emitted by shop_env/obs_format.py:26.
_CLICKABLE_RE = re.compile(r"可点击的按钮:\s*(\[.*?\])\s*(?:\n|$)", re.S)

# Buttons that navigate away from the current item rather than selecting within
# it. Kept lowercase; comparison is done on the casefolded button text.
_SEARCH_BUTTONS = frozenset({"back to search", "< prev", "next >"})

# Buttons that are page furniture, not decisions. They are still legal clicks,
# so they stay in the normalizer, but they map to no canonical intent.
_INERT_BUTTONS = frozenset({"description", "features", "reviews"})


def parse_legal_actions(state_text: str) -> list[str]:
    """Extract the environment's legal action strings from a rendered state.

    Returns action strings in ``click[<button>]`` form, in the order the
    environment listed them so scoring is deterministic. Empty list if the
    clickable line is absent or malformed — callers must treat that as
    unscorable rather than as "no actions available".
    """
    match = _CLICKABLE_RE.search(state_text)
    if match is None:
        return []
    try:
        buttons = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(buttons, list):
        return []
    return [f"click[{button}]" for button in buttons if isinstance(button, str)]


def intent_of_action(action: str) -> str | None:
    """Map a ``click[<button>]`` action string to its canonical intent.

    Mirrors preference_margin.intent_from_action, but works on the raw action
    string rather than a parsed action object, because the legal set is read
    back out of the rendered observation.
    """
    match = re.fullmatch(r"click\[(.*)\]", action.strip(), re.S)
    if match is None:
        return None
    button = match.group(1).strip().casefold()
    if button == "buy now":
        return "COMMIT"
    if button in _SEARCH_BUTTONS:
        return "SEARCH_ALTERNATIVE"
    if button in _INERT_BUTTONS:
        return None
    if not button:
        return None
    return "SELECT_TARGET_OPTION"


def score_actions(
    model: Any,
    tokenizer: Any,
    state_text: str,
    actions: list[str],
    device: torch.device | None = None,
    prefix: str = "Action: ",
) -> torch.Tensor:
    """Sum log P(action tokens | state) for each action, keeping the graph.

    One batched forward over ``state + prefix + action`` per candidate. A
    shared-prefix KV-cache pass is ~12x cheaper in tokens, but the cache cannot
    be duplicated per candidate while it carries grad (deepcopy rejects non-leaf
    tensors), and expanding it in place depends on transformers' private cache
    layout. Both agree to 5.7e-06, so correctness and portability win here; the
    scoring forward is small beside rollout generation.

    Returns a (len(actions),) float32 tensor. Gradients flow to model params.
    """
    if not actions:
        return torch.zeros(0, dtype=torch.float32)

    state_ids = tokenizer(state_text, add_special_tokens=False).input_ids
    cont_ids = [
        tokenizer(prefix + action, add_special_tokens=False).input_ids
        for action in actions
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    widths = [len(state_ids) + len(cont) for cont in cont_ids]
    total = max(widths)
    n = len(actions)

    input_ids = torch.full((n, total), pad_id, dtype=torch.long)
    attn = torch.zeros((n, total), dtype=torch.long)
    # Marks, per row, which positions are continuation tokens to be scored.
    score_mask = torch.zeros((n, total), dtype=torch.bool)
    for i, cont in enumerate(cont_ids):
        row = state_ids + cont
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        attn[i, : len(row)] = 1
        score_mask[i, len(state_ids) : len(row)] = True

    if device is None:
        # Infer from the model so callers holding a sharded/offloaded module do
        # not have to track placement themselves. Stubs without parameters()
        # simply stay on CPU.
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = None
    if device is not None:
        input_ids, attn, score_mask = input_ids.to(device), attn.to(device), score_mask.to(device)

    # Every row shares `state_ids`, so the first scored position is the same for
    # all of them and every scored position lies in the last (max_cont + 1)
    # columns. Restricting the LM head to that window keeps the logits tensor at
    # a few dozen MiB instead of the ~10 GiB a full (n, seq_len, 150k) block
    # would need -- which does not survive a backward pass on one GPU.
    keep = total - len(state_ids) + 1
    try:
        output = model(input_ids=input_ids, attention_mask=attn,
                       logits_to_keep=keep, return_dict=True)
    except TypeError:
        # Models/stubs without those arguments still work, just less frugally.
        output = model(input_ids=input_ids, attention_mask=attn)

    # Predict position t from the distribution at t-1.
    shifted_mask = score_mask[:, 1:]
    targets = input_ids[:, 1:]
    rows, cols = shifted_mask.nonzero(as_tuple=True)

    # `use_fused_kernels=True` replaces the model's forward with one that fuses the
    # LM-head projection into the log-prob computation and returns
    # CausalLMOutputForPPO(log_probs=..., entropy=...) -- there is no `.logits` to
    # read (verl/models/transformers/dense_common.py). Its `log_probs[t]` is
    # log P(input_ids[t+1] | prefix), which is exactly the per-token quantity the
    # gather below would produce, so use it directly. Verified against the plain
    # path on the real model and a real pair state: max abs diff 2.4e-07.
    #
    # It is not the cheaper path here, though: that forward ignores
    # `logits_to_keep` and computes log-probs for every position, measured 10.12
    # GiB against 4.96 GiB for the windowed logits path. Both fit; taking this
    # branch is about correctness under the configured model, not frugality.
    fused_log_probs = getattr(output, "log_probs", None)
    if fused_log_probs is not None:
        if rows.numel() == 0:
            return torch.zeros(n, device=fused_log_probs.device, dtype=torch.float32)
        tok_lp = fused_log_probs[rows, cols].float()
        out = torch.zeros(n, device=tok_lp.device, dtype=tok_lp.dtype)
        return out.index_add(0, rows, tok_lp)

    logits = output.logits
    offset = total - logits.shape[1]
    # Gather only the scored positions before the softmax. Running log_softmax
    # over the whole (n, seq_len, vocab) block would allocate several GiB for a
    # 150k vocab while all but a few dozen positions are masked out anyway.
    if rows.numel() == 0:
        return torch.zeros(n, device=logits.device, dtype=torch.float32)
    # `cols` indexes the shifted (t-1) axis of the full sequence; shift it into
    # the kept window. offset is 0 when the whole sequence came back.
    sel_logits = logits[rows, cols - offset].float()
    sel_lp = torch.log_softmax(sel_logits, dim=-1)
    tok_lp = sel_lp.gather(-1, targets[rows, cols].unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(n, device=tok_lp.device, dtype=tok_lp.dtype)
    return out.index_add(0, rows, tok_lp)


def intent_log_probs(
    model: Any,
    tokenizer: Any,
    state_text: str,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalized log P(intent | state) over the legal action set.

    Normalizing across the legal set — including inert buttons like
    "description" — makes the intents mutually exclusive and sum to <= 1, so the
    preference margin compares genuine alternatives. An intent with no legal
    action gets -inf; callers must drop those rather than treat them as
    low-probability, since -inf is "unavailable here", not "dispreferred".

    Returns (tensor of shape (len(CANONICAL_INTENTS),), info dict). The tensor is
    all -inf when the state is unparseable, with info["scorable"] False.
    """
    actions = parse_legal_actions(state_text)
    info: dict[str, Any] = {"n_legal": len(actions), "scorable": bool(actions)}
    neg_inf = float("-inf")

    if not actions:
        # Keep the info contract identical in both branches so callers can read
        # n_mapped without checking scorable first.
        info["n_mapped"] = 0
        return torch.full((len(CANONICAL_INTENTS),), neg_inf), info

    scores = score_actions(model, tokenizer, state_text, actions, device=device)
    # softmax over the legal set: the normalizer is every action the env accepts.
    normalized = torch.log_softmax(scores, dim=-1)

    intents = [intent_of_action(action) for action in actions]
    info["n_mapped"] = sum(intent is not None for intent in intents)

    out = []
    for intent in CANONICAL_INTENTS:
        idx = [i for i, mapped in enumerate(intents) if mapped == intent]
        if not idx:
            out.append(torch.tensor(neg_inf, device=normalized.device))
            continue
        # logsumexp: several buttons can realize one intent (many option clicks).
        out.append(torch.logsumexp(normalized[idx], dim=0))
    return torch.stack(out), info
