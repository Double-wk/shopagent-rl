#!/usr/bin/env python3
"""Build a mixed environment/counterfactual parquet for Certified GRPO."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.explicit_budget_pairs import make_budget_explicit



def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _expected_relation(pair: dict) -> list[str]:
    """Return the ordered intent relation encoded by a certified pair."""
    original = pair.get("original") or {}
    counterfactual = pair.get("counterfactual") or {}
    values = [
        *(original.get("expected_action_intents") or []),
        *(counterfactual.get("expected_action_intents") or []),
    ]
    return [str(value) for value in values[:2]]


def load_excluded_task_ids(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("task_ids", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise SystemExit(f"excluded task IDs must be a list in {path}")
        excluded.update(int(value) for value in values)
    return excluded


def select_pairs(
    records: list[dict],
    intervention_type: str,
    limit: int,
    rng: random.Random,
    excluded_tasks: set[int],
    system_prompt: str = "",
    max_prompt_chars: int = -1,
) -> list[dict]:
    selected = [
        record for record in records
        if (record.get("intervention_type") == intervention_type
            and record.get("task_id") not in excluded_tasks)
        and (
            max_prompt_chars < 0
            or len(system_prompt) + max(
                len(str(record[side]["observation"]))
                for side in ("original", "counterfactual")
            ) <= max_prompt_chars
        )
    ]
    rng.shuffle(selected)
    return selected if limit < 0 else selected[:limit]


def environment_row(task_id: int, system_prompt: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": system_prompt}],
        "task_id": int(task_id),
        "data_source": "shopsim",
        "sample_mode": "environment",
        "pair_id": "",
        "relation_id": "",
        "intervention_type": "",
        "side": "",
        "expected_action_intents": [],
        "expected_relation": [],
        "allowed_actions": [],
    }


def counterfactual_row(pair: dict, side: str, system_prompt: str) -> dict:
    state = pair[side]
    observation = state["observation"]
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ],
        "task_id": int(pair["task_id"]),
        "data_source": "shopsim_counterfactual",
        "sample_mode": "counterfactual",
        "pair_id": str(pair["pair_id"]),
        "relation_id": str(pair["pair_id"]),
        "intervention_type": str(pair["intervention_type"]),
        "side": side,
        "expected_action_intents": [str(value) for value in state["expected_action_intents"]],
        "expected_relation": [str(value) for value in pair.get("expected_relation", _expected_relation(pair))],
        "allowed_actions": [str(value) for value in state["allowed_actions"]],
    }


def build_rows(
    task_ids: list[int],
    atomic_pairs: list[dict],
    natural_pairs: list[dict],
    v2_pairs: list[dict],
    system_prompt: str,
    *,
    environment_repeat: int,
    environment_tasks: int = -1,
    price_pairs: int,
    option_pairs: int,
    nuisance_pairs: int,
    option_intervention_type: str = "option_goal_swap_natural",
    seed: int,
    explicit_price_budget: bool = False,
    excluded_tasks: set[int] | None = None,
    pair_block_size: int = 0,
    max_counterfactual_prompt_chars: int = -1,
) -> list[dict]:
    rng = random.Random(seed)
    excluded_tasks = excluded_tasks or set()
    available_environment_tasks = [
        task_id for task_id in task_ids if task_id not in excluded_tasks
    ]
    if environment_tasks >= 0:
        environment_rng = random.Random(seed + 1)
        environment_rng.shuffle(available_environment_tasks)
        available_environment_tasks = available_environment_tasks[:environment_tasks]
    environment_rows = [
        environment_row(task_id, system_prompt)
        for _ in range(environment_repeat)
        for task_id in available_environment_tasks
    ]
    groups = [
        select_pairs(
            atomic_pairs, "price_above_budget", price_pairs, rng, excluded_tasks,
            system_prompt, max_counterfactual_prompt_chars,
        ),
        select_pairs(
            natural_pairs, option_intervention_type, option_pairs, rng, excluded_tasks,
            system_prompt, max_counterfactual_prompt_chars,
        ),
        select_pairs(
            v2_pairs, "nuisance_display_note", nuisance_pairs, rng, excluded_tasks,
            system_prompt, max_counterfactual_prompt_chars,
        ),
    ]
    if explicit_price_budget:
        groups[0] = [make_budget_explicit(pair) for pair in groups[0]]
    paired_rows: list[list[dict]] = []
    for pairs in groups:
        for pair in pairs:
            paired_rows.append([
                counterfactual_row(pair, "original", system_prompt),
                counterfactual_row(pair, "counterfactual", system_prompt),
            ])

    if pair_block_size:
        if pair_block_size < 2 or pair_block_size % 2:
            raise ValueError("pair_block_size must be a positive even number")
        if len(environment_rows) % pair_block_size:
            raise ValueError("environment rows must divide evenly into pair blocks")
        pairs_per_block = pair_block_size // 2
        if len(paired_rows) % pairs_per_block:
            raise ValueError("selected pair count must divide evenly into pair blocks")
        blocks = [
            environment_rows[index:index + pair_block_size]
            for index in range(0, len(environment_rows), pair_block_size)
        ]
        blocks.extend(
            [row for pair in paired_rows[index:index + pairs_per_block] for row in pair]
            for index in range(0, len(paired_rows), pairs_per_block)
        )
        rng.shuffle(blocks)
        rows = [row for block in blocks for row in block]
    else:
        rows = environment_rows + [row for pair in paired_rows for row in pair]
        rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["extra_info"] = {"index": index}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo_prompts_1000.json")
    parser.add_argument("--atomic", type=Path, default=ROOT / "data/counterfactual/train_atomic_pairs_v1.jsonl")
    parser.add_argument("--natural", type=Path, default=ROOT / "data/counterfactual/train_natural_option_swap_v1.jsonl")
    parser.add_argument("--v2", type=Path, default=ROOT / "data/counterfactual/train_constraint_causal_v2.jsonl")
    parser.add_argument("--system", type=Path, default=ROOT / "configs/teacher_gpt-5.6-terra.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data/grpo_certified_train.parquet")
    parser.add_argument("--environment-repeat", type=int, default=4)
    parser.add_argument("--environment-tasks", type=int, default=-1,
                        help="number of distinct environment tasks; -1 keeps all")
    parser.add_argument("--price-pairs", type=int, default=1000)
    parser.add_argument("--option-pairs", type=int, default=500)
    parser.add_argument("--option-intervention-type", default="option_goal_swap_natural")
    parser.add_argument("--nuisance-pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--explicit-price-budget", action="store_true")
    parser.add_argument("--exclude-verbatim", action="store_true")
    parser.add_argument("--exclude-task-ids", action="append", type=Path, default=[])
    parser.add_argument("--pair-block-size", type=int, default=0,
                        help="shuffle fixed-size blocks while keeping pair sides together")
    parser.add_argument("--max-counterfactual-prompt-chars", type=int, default=-1,
                        help="exclude a whole pair if either rendered prompt exceeds this many characters")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    task_ids = [int(value) for value in json.loads(args.tasks.read_text())]
    system_prompt = yaml.safe_load(args.system.read_text())["system_prompt"]
    natural_pairs = load_jsonl(args.natural)
    if args.exclude_verbatim:
        natural_pairs = [
            pair for pair in natural_pairs
            if not pair["intervention"]["verification_detail"].get("catalog_verbatim")
        ]
    excluded_tasks = load_excluded_task_ids(args.exclude_task_ids)
    rows = build_rows(
        task_ids,
        load_jsonl(args.atomic),
        natural_pairs,
        load_jsonl(args.v2),
        system_prompt,
        environment_repeat=args.environment_repeat,
        environment_tasks=args.environment_tasks,
        price_pairs=args.price_pairs,
        option_pairs=args.option_pairs,
        nuisance_pairs=args.nuisance_pairs,
        option_intervention_type=args.option_intervention_type,
        seed=args.seed,
        explicit_price_budget=args.explicit_price_budget,
        excluded_tasks=excluded_tasks,
        pair_block_size=args.pair_block_size,
        max_counterfactual_prompt_chars=args.max_counterfactual_prompt_chars,
    )
    schema = pa.schema([
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("extra_info", pa.struct([("index", pa.int64())])),
        ("task_id", pa.int64()),
        ("data_source", pa.string()),
        ("sample_mode", pa.string()),
        ("pair_id", pa.string()),
        ("relation_id", pa.string()),
        ("intervention_type", pa.string()),
        ("side", pa.string()),
        ("expected_action_intents", pa.list_(pa.string())),
        ("expected_relation", pa.list_(pa.string())),
        ("allowed_actions", pa.list_(pa.string())),
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.out)
    counts = Counter(row["sample_mode"] for row in rows)
    interventions = Counter(row["intervention_type"] for row in rows if row["intervention_type"])
    print(json.dumps({
        "out": str(args.out),
        "rows": len(rows),
        "by_mode": dict(counts),
        "by_intervention_type": dict(interventions),
        "environment_tasks": len({row["task_id"] for row in rows if row["sample_mode"] == "environment"}),
        "excluded_task_count": len(excluded_tasks),
        "pair_block_size": args.pair_block_size,
        "max_counterfactual_prompt_chars": args.max_counterfactual_prompt_chars,
        "explicit_price_budget": args.explicit_price_budget,
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
