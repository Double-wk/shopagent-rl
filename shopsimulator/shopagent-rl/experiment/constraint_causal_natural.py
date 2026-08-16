"""Teacher-assisted natural-language rewrites of single-constraint goal changes.

The v2 builder could only produce exact-substring rewrites (3 pairs): users
usually describe specifications semantically, not in catalog spelling.  This
module turns each structured ``option_goal_swap_structured`` pair into a
natural pair by asking a teacher to rewrite the instruction so that the user
now wants option ``B`` instead of option ``A``.

The teacher is never trusted.  Every rewrite must pass purely programmatic
checks (:func:`verify_rewrite`) before it becomes a pair:

* every number/alnum token of the original instruction that does not belong to
  option ``A`` must survive unchanged (budgets, quantities, unrelated specs);
* no token unique to ``A`` may remain, and at least one token unique to ``B``
  must appear (the constraint really flipped);
* character-bigram overlap outside alnum tokens stays high (the rewrite did
  not drift to another product, category, or persona);
* length ratio stays bounded.

Output pairs reuse the v2 schema with ``intervention_type =
option_goal_swap_natural`` and carry the full verification transcript.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

SCHEMA_VERSION_NATURAL = "shopsim-constraint-causal-pairs-v3-natural"

_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[.*×x][A-Za-z0-9]+)*")
_WHITESPACE = re.compile(r"\s+")
# Digits attached to currency context: everything number-like matters, the
# verifier keeps this conservative on purpose (a false reject only costs a
# pair; a false accept corrupts the benchmark).
_PUNCT = re.compile(r"[，。！？；：、,.!?;:（）()\"'“”‘’\[\]【】<>《》…-]")


def alnum_tokens(text: str) -> list[str]:
    """Case-folded alphanumeric spec tokens ('85ml', '50g*3' → '50g*3')."""
    return [m.group(0).lower() for m in _ALNUM.finditer(text)]


def _chinese_stream(text: str) -> str:
    """Text with alnum tokens, whitespace and punctuation removed."""
    stripped = _ALNUM.sub(" ", text)
    stripped = _PUNCT.sub(" ", stripped)
    return _WHITESPACE.sub("", stripped)


def _bigrams(text: str) -> set[str]:
    s = _chinese_stream(text)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


REWRITE_SYSTEM_PROMPT = "你是中文购物需求改写器。只输出改写后的需求文本本身，不要解释、不要引号、不要前后缀。"


def build_rewrite_prompt(instruction: str, option_a: str, option_b: str) -> str:
    return (
        "原始用户需求：\n"
        f"{instruction}\n\n"
        "这个需求里用户想要的商品规格要换：\n"
        f"原来想要的规格：{option_a}\n"
        f"现在想要的规格：{option_b}\n\n"
        "请改写这条需求，要求：\n"
        "1. 只改动与规格相关的表述，让需求自然地表达用户想要新规格；\n"
        "2. 预算、品牌、品类、用途、语气、其他数量一律保持原样；\n"
        "3. 用用户的口吻描述新规格（如容量、数量、颜色、材质），"
        "不要把商品目录里的规格原文整串粘贴进需求；\n"
        "4. 长度与原文相近，不要添加原文没有的信息；\n"
        "5. 直接输出改写后的需求文本。"
    )


@dataclass
class RewriteCheck:
    """Outcome of one programmatic verification (all fields are evidence)."""
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def verify_rewrite(original: str, rewrite: str, option_a: str, option_b: str,
                   *, min_bigram_jaccard: float = 0.5,
                   length_ratio_bounds: tuple[float, float] = (0.5, 2.0)) -> RewriteCheck:
    """Programmatic gate: only the option constraint may change."""
    reasons: list[str] = []
    if not rewrite.strip():
        return RewriteCheck(False, ["empty_rewrite"])

    tok_a = set(alnum_tokens(option_a))
    tok_b = set(alnum_tokens(option_b))
    if not tok_b - tok_a and not _chinese_stream(option_b):
        return RewriteCheck(False, ["options_not_verifiable"])

    orig_tokens = alnum_tokens(original)
    new_tokens = set(alnum_tokens(rewrite))
    a_unique = tok_a - tok_b
    b_unique = tok_b - tok_a

    # 1. Numbers/specs unrelated to option A must survive verbatim.
    preserved = [t for t in orig_tokens if t.lower() not in a_unique]
    lost = [t for t in preserved if t.lower() not in new_tokens]
    if lost:
        reasons.append(f"lost_unrelated_tokens:{lost}")

    # 2. The constraint must actually flip.
    residual = [t for t in a_unique if t in new_tokens]
    if residual:
        reasons.append(f"old_option_tokens_remain:{residual}")
    if b_unique and not (b_unique & new_tokens):
        reasons.append("new_option_tokens_missing")

    # 3. Guard against drift to another product/persona.
    j = _bigram_jaccard(original, rewrite)
    if j < min_bigram_jaccard:
        reasons.append(f"bigram_jaccard_low:{j:.3f}")

    # 4. Length guard.
    ratio = len(_chinese_stream(rewrite)) / max(1, len(_chinese_stream(original)))
    lo, hi = length_ratio_bounds
    if not lo <= ratio <= hi:
        reasons.append(f"length_ratio_out_of_bounds:{ratio:.2f}")

    detail = {
        "a_unique": sorted(a_unique), "b_unique": sorted(b_unique),
        "bigram_jaccard": round(j, 4), "length_ratio": round(ratio, 3),
        "lost_unrelated_tokens": lost,
        # Audit flag (not a rejection): the rewrite pasted the catalog option
        # string verbatim.  Semantically faithful but unnatural; report the
        # natural/verbatim split instead of silently mixing them.
        "catalog_verbatim": option_b.strip() in rewrite,
    }
    return RewriteCheck(not reasons, reasons, detail)


def _bigram_jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def build_natural_pair(pair: dict[str, Any], rewrite: str,
                       check: RewriteCheck, model: str) -> dict[str, Any]:
    """Assemble a v3 natural pair from a verified rewrite of a v2 structured pair."""
    intervention = pair["intervention"]
    original = dict(pair["original"])
    # Strip the v2 structured-summary appendix if present: the natural pair
    # must present a plain instruction, both worlds.
    obs = original.get("observation") or ""
    obs = re.sub(r"\n任务约束摘要: [^\n]+$", "", obs)
    original["observation"] = obs

    instruction, _tail = obs.partition("\n")[0], ""
    instruction = instruction[len("Instruction: "):] if instruction.startswith("Instruction: ") else instruction
    new_instruction = rewrite.strip()
    new_observation = re.sub(
        r"^(Instruction: )[^\n]*", lambda m: m.group(1) + new_instruction, obs, count=1
    )
    alternative = intervention["to"]
    cf = {
        **original,
        "observation": new_observation,
        "expected_action_intents": ["SELECT_TARGET_OPTION"],
        "allowed_actions": [f"click[{alternative}]"],
    }
    return {
        "schema_version": SCHEMA_VERSION_NATURAL,
        "task_id": pair["task_id"],
        "source_pair_id": pair["pair_id"],
        "source": pair.get("source", {}),
        "goal": {**pair.get("goal", {}),
                 "instruction_text": new_instruction,
                 "goal_options": [alternative]},
        "product": pair.get("product", {}),
        "original": original,
        "counterfactual": cf,
        "intervention_type": "option_goal_swap_natural",
        "intervention": {
            "field": "instruction.goal_option",
            "from": intervention["from"],
            "to": alternative,
            "teacher_model": model,
            "teacher_rewrite": new_instruction,
            "verification_reasons": check.reasons,
            "verification_detail": check.detail,
            "validity_checks": {
                "same_product_state": True,
                "same_selected_options": True,
                "same_price": original.get("current_price") == pair["counterfactual"].get("current_price"),
                "programmatic_rewrite_verification": check.accepted,
            },
        },
        "pair_id": f"{pair['task_id']}:option_goal_swap_natural",
    }


def iter_structured_candidates(v2_pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The v2 structured goal-swap pairs eligible for natural rewriting."""
    return [p for p in v2_pairs if p.get("intervention_type") == "option_goal_swap_structured"]


def instruction_of(pair: dict[str, Any]) -> str:
    obs = (pair.get("original") or {}).get("observation") or ""
    head = obs.split("\n", 1)[0]
    return head[len("Instruction: "):] if head.startswith("Instruction: ") else head
