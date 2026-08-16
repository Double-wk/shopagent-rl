"""Build validated, atomic pre-purchase counterfactuals from ShopSimulator data.

The first version deliberately supports only interventions backed by executable,
structured state in ShopSimulator:

* ``option_swap``: keep product, instruction, price and available actions fixed,
  but change the selected option from the goal option to a same-price alternative.
* ``price_above_budget``: keep everything else fixed and move the selected
  product price just above the realized goal budget.

Free-text attributes and ``is_available`` are intentionally excluded.  Product
attributes are not typed, while the upstream engine currently exposes option
values without enforcing the raw ``is_available`` flag.
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
) -> str:
    """Render a compact commitment-point state without fabricating page content."""
    lines = [
        f"Instruction: {instruction}",
        f"商品: {product.get('title', '')}",
    ]
    clickables = list(NAV_ACTIONS)
    for group, entries in (product.get("customization_options") or {}).items():
        values = [str(entry.get("value", "")) for entry in entries or []]
        lines.append(f"{group}: " + " | ".join(values))
        clickables.extend(values)
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
