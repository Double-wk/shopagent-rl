"""Build validated, atomic pre-purchase counterfactuals from ShopSimulator data.

Every mechanism is backed by executable, structured state.  They are split into
a *training* family and a *held-out* family so that generalization to an unseen
intervention mechanism can be measured rather than assumed.

Training mechanisms (the goal option is already selected; the question is
whether to commit):

* ``option_swap``: keep product, instruction, price and available actions fixed,
  but change the selected option from the goal option to a same-price alternative.
* ``price_above_budget``: keep everything else fixed and move the selected
  product price just above the realized goal budget.

Held-out mechanisms.  Both share one frame -- an affordable sibling option is
selected and the goal option is still reachable, so the pre-intervention
decision is ``SELECT_TARGET_OPTION`` -- and differ only in how the intervention
blocks the goal option:

* ``option_unavailable``: the goal option goes out of stock.  An availability
  constraint, a *constraint type* absent from training.
* ``option_price_over_budget``: the goal option's own listed price breaks the
  budget while the displayed ``当前价格`` is unchanged.  A budget constraint on a
  *new surface*: the violation must be attributed to one option instead of read
  off a scalar.

Verification differs per mechanism and is recorded in ``intervention.verified_by``.
``option_price_over_budget`` is backed by the structured per-option price the
engine itself keys on (``option_to_price``).  ``option_unavailable`` is verified
against *action legality* -- the goal option leaves ``可点击的按钮`` -- because the
upstream engine exposes ``is_available`` without enforcing it in reward.

Free-text product attributes remain excluded: they are untyped, and the reward
matches them fuzzily against title, bullets and description, so "provably
unsatisfiable" cannot be established from the data.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "shopsim-atomic-constraint-pairs-v1"
NAV_ACTIONS = ["back to search", "< prev", "description", "features", "reviews", "buy now"]

# The generalization split.  Training data may only contain TRAIN_MECHANISMS;
# HELD_OUT_MECHANISMS exist solely in the frozen test set, so accuracy on them
# measures transfer to an unseen intervention mechanism.  ``nuisance_display_note``
# is built downstream (scripts/build_constraint_causal_v2.py) from v1 pairs and
# is a decision-preserving control, so it belongs to the training family.
TRAIN_MECHANISMS = frozenset({"price_above_budget", "option_swap", "nuisance_display_note"})
HELD_OUT_MECHANISMS = frozenset({"option_unavailable", "option_price_over_budget"})


def normalize_option(value: Any) -> str:
    """Normalize the punctuation variants used by raw data and the web engine."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return "".join(char for char in text if char.isalnum())


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list/report or JSONL trajectory file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("tasks", "records", "trajectories"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"cannot find trajectory records in {path}")


def load_products(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    if not isinstance(products, list):
        raise ValueError(f"expected a product list in {path}")
    return products


def _price(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def canonical_price_upper(price: float) -> float:
    """Choose a deterministic budget from the support used by upstream goal.py.

    Upstream samples two multiples of ten strictly around/above the generated
    product price and uses the larger one.  Eval trajectories that terminate
    before purchase do not persist that sampled goal.  For those records we use
    the smallest unambiguous upper bound above the target option price and mark
    its provenance as canonical rather than observed.
    """
    base = math.ceil(price / 10.0) * 10.0
    return base + 10.0 if base <= price else base


def _goal_option_locations(
    product: dict[str, Any], goal_options: Iterable[Any]
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    targets = {normalize_option(value) for value in goal_options}
    locations = []
    for group, entries in (product.get("customization_options") or {}).items():
        for entry in entries or []:
            if normalize_option(entry.get("value", "")) in targets:
                locations.append((str(group), entry, entries))
    return locations


def _render_observation(
    *,
    instruction: str,
    product: dict[str, Any],
    selected: dict[str, str],
    current_price: float,
    price_overrides: dict[str, float] | None = None,
    unavailable: Iterable[str] = (),
) -> str:
    """Render a compact commitment-point state without fabricating page content.

    ``price_overrides`` and ``unavailable`` are keyed by normalized option value
    and express structured interventions on the option list itself.  Per-option
    prices are shown whenever the product carries them, so the presence of a
    price annotation is a property of the product and never a marker of which
    mechanism produced the state.  An unavailable option is still listed -- the
    page shows it as out of stock -- but leaves ``可点击的按钮``, which is what
    makes clicking it an illegal action.
    """
    overrides = {normalize_option(k): v for k, v in (price_overrides or {}).items()}
    blocked = {normalize_option(value) for value in unavailable}
    lines = [
        f"Instruction: {instruction}",
        f"商品: {product.get('title', '')}",
    ]
    clickables = list(NAV_ACTIONS)
    for group, entries in (product.get("customization_options") or {}).items():
        rendered = []
        for entry in entries or []:
            value = str(entry.get("value", ""))
            key = normalize_option(value)
            price = overrides.get(key, _price(entry.get("price")))
            label = value if price is None else f"{value}(￥{price:g})"
            if key in blocked:
                rendered.append(f"{label}[缺货]")
            else:
                rendered.append(label)
                clickables.append(value)
        lines.append(f"{group}: " + " | ".join(rendered))
    selected_text = "; ".join(f"{key}={value}" for key, value in selected.items())
    lines.extend(
        [
            f"已选规格: {selected_text}",
            f"当前价格: {current_price:g}",
            "可点击的按钮: " + json.dumps(clickables, ensure_ascii=False),
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class BuildResult:
    pairs: list[dict[str, Any]]
    stats: dict[str, int]


def build_pairs(
    products: list[dict[str, Any]], trajectory_records: Iterable[dict[str, Any]]
) -> BuildResult:
    """Create paired states and reject confounded or unverifiable interventions."""
    pairs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for record in trajectory_records:
        stats["records_seen"] += 1
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or not (0 <= task_id < len(products)):
            stats["rejected_bad_task_id"] += 1
            continue
        product = products[task_id]
        observed_goal = record.get("goal") or {}
        raw_instruction = (product.get("instructions") or [{}])[0]
        instruction = str(
            observed_goal.get("instruction_text") or raw_instruction.get("instruction") or ""
        )
        goal_options = list(
            observed_goal.get("goal_options")
            or raw_instruction.get("instruction_options")
            or raw_instruction.get("options")
            or []
        )
        price_upper = _price(observed_goal.get("price_upper"))
        budget_source = "observed_eval_goal" if price_upper is not None else "canonical_from_target_price"
        if not instruction or not goal_options:
            stats["rejected_incomplete_goal"] += 1
            continue

        locations = _goal_option_locations(product, goal_options)
        if len(locations) != len(goal_options):
            stats["rejected_unmapped_goal_option"] += 1
            continue

        # Version 1 requires one target option. Multi-group interventions would
        # change more than one decision variable and are therefore not atomic.
        if len(locations) != 1:
            stats["rejected_multi_target"] += 1
            continue
        group, target, entries = locations[0]
        target_value = str(target.get("value", ""))
        target_price = _price(target.get("price"))
        if target_price is None:
            pricing = product.get("pricing") or []
            target_price = _price(pricing[0]) if pricing else None
        if target_price is None:
            stats["rejected_missing_target_price"] += 1
            continue
        if price_upper is None:
            price_upper = canonical_price_upper(target_price)
            stats["goals_with_canonical_budget"] += 1
        else:
            stats["goals_with_observed_budget"] += 1
        if price_upper <= 0 or target_price > price_upper:
            stats["rejected_original_over_budget"] += 1
            continue

        selected_target = {group: target_value}
        base = {
            "selected_options": selected_target,
            "current_price": target_price,
            "expected_action_intents": ["COMMIT"],
            "allowed_actions": ["click[buy now]"],
        }
        base["observation"] = _render_observation(
            instruction=instruction,
            product=product,
            selected=selected_target,
            current_price=target_price,
        )
        shared = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "source": {
                "asin": str(product.get("asin", "")),
                "tag": product.get("tag"),
            },
            "goal": {
                "instruction_text": instruction,
                "goal_options": goal_options,
                "price_upper": price_upper,
                "price_upper_source": budget_source,
            },
            "product": {
                "title": product.get("title", ""),
                "category": product.get("category", ""),
                "option_group": group,
            },
            "original": base,
        }

        same_price_alternatives = sorted(
            (
                entry
                for entry in entries
                if normalize_option(entry.get("value", "")) != normalize_option(target_value)
                and entry.get("is_available", True)
                and _price(entry.get("price")) is not None
                and math.isclose(float(entry["price"]), target_price, abs_tol=1e-9)
            ),
            key=lambda entry: normalize_option(entry.get("value", "")),
        )
        if same_price_alternatives:
            alternative = same_price_alternatives[0]
            alternative_value = str(alternative.get("value", ""))
            cf_selected = {group: alternative_value}
            counterfactual = {
                "selected_options": cf_selected,
                "current_price": target_price,
                "expected_action_intents": ["SELECT_TARGET_OPTION"],
                "allowed_actions": [f"click[{target_value}]"],
            }
            counterfactual["observation"] = _render_observation(
                instruction=instruction,
                product=product,
                selected=cf_selected,
                current_price=target_price,
            )
            pairs.append(
                {
                    **shared,
                    "pair_id": f"{task_id}:option_swap",
                    "intervention_type": "option_swap",
                    "intervention": {
                        "field": f"selected_options.{group}",
                        "from": target_value,
                        "to": alternative_value,
                        "validity_checks": {
                            "same_product": True,
                            "same_instruction": True,
                            "same_price": True,
                            "alternative_is_available": True,
                            "target_action_is_clickable": True,
                        },
                    },
                    "counterfactual": counterfactual,
                }
            )
            stats["pairs_option_swap"] += 1
        else:
            stats["skipped_no_same_price_alternative"] += 1

        delta = max(1.0, round(price_upper * 0.05, 2))
        over_budget_price = round(price_upper + delta, 2)
        counterfactual = {
            "selected_options": selected_target,
            "current_price": over_budget_price,
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]", "click[< prev]"],
        }
        counterfactual["observation"] = _render_observation(
            instruction=instruction,
            product=product,
            selected=selected_target,
            current_price=over_budget_price,
        )
        pairs.append(
            {
                **shared,
                "pair_id": f"{task_id}:price_above_budget",
                "intervention_type": "price_above_budget",
                "intervention": {
                    "field": "current_price",
                    "from": target_price,
                    "to": over_budget_price,
                    "validity_checks": {
                        "same_product": True,
                        "same_instruction": True,
                        "same_selected_option": True,
                        "original_within_budget": target_price <= price_upper,
                        "counterfactual_over_budget": over_budget_price > price_upper,
                    },
                },
                "counterfactual": counterfactual,
            }
        )
        stats["pairs_price_above_budget"] += 1

        # ---- Held-out mechanisms -------------------------------------------
        # Both need a *sibling* option that is affordable, so that the frame is
        # "a cheap sibling is selected, the goal option is still reachable" and
        # the pre-intervention decision is SELECT_TARGET_OPTION rather than
        # COMMIT.  That keeps each intervention atomic: only the goal option's
        # availability (M3) or its own listed price (M4) changes.
        target_entry_price = _price(target.get("price"))
        siblings = sorted(
            (
                entry
                for entry in entries
                if normalize_option(entry.get("value", "")) != normalize_option(target_value)
                and entry.get("is_available", True)
                and _price(entry.get("price")) is not None
                and float(entry["price"]) <= price_upper
            ),
            key=lambda entry: (_price(entry.get("price")), normalize_option(entry.get("value", ""))),
        )
        if not siblings or target_entry_price is None:
            if not siblings:
                stats["skipped_no_affordable_sibling"] += 1
            else:
                stats["skipped_target_option_unpriced"] += 1
            continue

        sibling = siblings[0]
        sibling_value = str(sibling.get("value", ""))
        sibling_price = float(sibling["price"])
        sibling_selected = {group: sibling_value}
        # The goal option is not selected yet, so the goal option reward is not
        # yet earned: the constraint-faithful action is to select it.
        held_out_base = {
            "selected_options": sibling_selected,
            "current_price": sibling_price,
            "expected_action_intents": ["SELECT_TARGET_OPTION"],
            "allowed_actions": [f"click[{target_value}]"],
        }
        held_out_base["observation"] = _render_observation(
            instruction=instruction,
            product=product,
            selected=sibling_selected,
            current_price=sibling_price,
        )
        held_out_shared = {**shared, "original": held_out_base}
        held_out_shared["product"] = {**shared["product"], "sibling_option": sibling_value}

        # M3 option_unavailable: the goal option goes out of stock.  Verified by
        # action legality -- it is no longer in 可点击的按钮 -- rather than by
        # reward, since the engine does not enforce is_available.
        counterfactual = {
            "selected_options": sibling_selected,
            "current_price": sibling_price,
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]", "click[< prev]"],
        }
        counterfactual["observation"] = _render_observation(
            instruction=instruction,
            product=product,
            selected=sibling_selected,
            current_price=sibling_price,
            unavailable=[target_value],
        )
        pairs.append(
            {
                **held_out_shared,
                "pair_id": f"{task_id}:option_unavailable",
                "intervention_type": "option_unavailable",
                "intervention": {
                    "field": f"customization_options.{group}.{target_value}.is_available",
                    "from": True,
                    "to": False,
                    "verified_by": "action_legality",
                    "validity_checks": {
                        "same_product": True,
                        "same_instruction": True,
                        "same_selected_option": True,
                        "same_current_price": True,
                        "target_clickable_before": True,
                        "target_not_clickable_after": True,
                    },
                },
                "counterfactual": counterfactual,
            }
        )
        stats["pairs_option_unavailable"] += 1

        # M4 option_price_over_budget: the goal option's own listed price breaks
        # the budget.  The displayed 当前价格 still shows the affordable sibling,
        # so the violation cannot be read off the scalar -- it has to be
        # attributed to one option.  Verified against the structured per-option
        # price the page itself resolves and displays
        # (web_agent_text_env.py:525-529, option_to_price).  Note this is a
        # display-and-data guarantee, not a reward one: engine `done()` scores
        # with the base ASIN price (web_agent_text_env.py:590), so the option
        # premium never reaches r_price.  That upstream gap is precisely why the
        # budget constraint is under-supervised by reward alone.
        blown_price = round(price_upper + max(1.0, round(price_upper * 0.05, 2)), 2)
        counterfactual = {
            "selected_options": sibling_selected,
            "current_price": sibling_price,
            "expected_action_intents": ["SEARCH_ALTERNATIVE"],
            "allowed_actions": ["click[back to search]", "click[< prev]"],
        }
        counterfactual["observation"] = _render_observation(
            instruction=instruction,
            product=product,
            selected=sibling_selected,
            current_price=sibling_price,
            price_overrides={target_value: blown_price},
        )
        pairs.append(
            {
                **held_out_shared,
                "pair_id": f"{task_id}:option_price_over_budget",
                "intervention_type": "option_price_over_budget",
                "intervention": {
                    "field": f"customization_options.{group}.{target_value}.price",
                    "from": target_entry_price,
                    "to": blown_price,
                    "verified_by": "structured_option_price",
                    "validity_checks": {
                        "same_product": True,
                        "same_instruction": True,
                        "same_selected_option": True,
                        "same_current_price": True,
                        "target_option_affordable_before": target_entry_price <= price_upper,
                        "target_option_over_budget_after": blown_price > price_upper,
                        "sibling_still_affordable": sibling_price <= price_upper,
                    },
                },
                "counterfactual": counterfactual,
            }
        )
        stats["pairs_option_price_over_budget"] += 1

    stats["pairs_total"] = len(pairs)
    stats["tasks_with_pairs"] = len({pair["task_id"] for pair in pairs})
    return BuildResult(pairs=pairs, stats=dict(sorted(stats.items())))


def validate_pair(pair: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the pair is usable."""
    errors = []
    original = pair.get("original") or {}
    counterfactual = pair.get("counterfactual") or {}
    intervention = pair.get("intervention") or {}
    if pair.get("schema_version") != SCHEMA_VERSION:
        errors.append("unexpected schema_version")
    if not pair.get("pair_id"):
        errors.append("missing pair_id")
    if original.get("observation") == counterfactual.get("observation"):
        errors.append("observations are identical")
    checks = intervention.get("validity_checks") or {}
    if not checks or not all(value is True for value in checks.values()):
        errors.append("intervention validity check failed")
    if not original.get("allowed_actions") or not counterfactual.get("allowed_actions"):
        errors.append("missing allowed actions")
    return errors


def dump_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
