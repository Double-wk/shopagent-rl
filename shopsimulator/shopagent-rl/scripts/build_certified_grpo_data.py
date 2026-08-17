#!/usr/bin/env python3
"""Build a mixed environment/counterfactual parquet for Certified GRPO."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_pairs(records: list[dict], intervention_type: str, limit: int, rng: random.Random) -> list[dict]:
    selected = [record for record in records if record.get("intervention_type") == intervention_type]
    rng.shuffle(selected)
    return selected if limit < 0 else selected[:limit]


def environment_row(task_id: int, system_prompt: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": system_prompt}],
        "task_id": int(task_id),
        "data_source": "shopsim",
        "sample_mode": "environment",
        "pair_id": "",
        "intervention_type": "",
        "side": "",
        "expected_action_intents": [],
        "allowed_actions": [],
    }


def counterfactual_row(pair: dict, side: str, system_prompt: str) -> dict:
    state = pair[side]
    observation = state["observation"]
    if pair.get("intervention_type") == "price_above_budget":
        budget = float(pair["goal"]["price_upper"])
        observation += f"\n任务约束摘要: 预算上限={budget:g}元。"
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ],
        "task_id": int(pair["task_id"]),
        "data_source": "shopsim_counterfactual",
        "sample_mode": "counterfactual",
        "pair_id": str(pair["pair_id"]),
        "intervention_type": str(pair["intervention_type"]),
        "side": side,
        "expected_action_intents": [str(value) for value in state["expected_action_intents"]],
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
    price_pairs: int,
    option_pairs: int,
    nuisance_pairs: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows = [
        environment_row(task_id, system_prompt)
        for _ in range(environment_repeat)
        for task_id in task_ids
    ]
    groups = [
        select_pairs(atomic_pairs, "price_above_budget", price_pairs, rng),
        select_pairs(natural_pairs, "option_goal_swap_natural", option_pairs, rng),
        select_pairs(v2_pairs, "nuisance_display_note", nuisance_pairs, rng),
    ]
    for pairs in groups:
        for pair in pairs:
            for side in ("original", "counterfactual"):
                rows.append(counterfactual_row(pair, side, system_prompt))
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
    parser.add_argument("--price-pairs", type=int, default=1000)
    parser.add_argument("--option-pairs", type=int, default=500)
    parser.add_argument("--nuisance-pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    task_ids = [int(value) for value in json.loads(args.tasks.read_text())]
    system_prompt = yaml.safe_load(args.system.read_text())["system_prompt"]
    rows = build_rows(
        task_ids,
        load_jsonl(args.atomic),
        load_jsonl(args.natural),
        load_jsonl(args.v2),
        system_prompt,
        environment_repeat=args.environment_repeat,
        price_pairs=args.price_pairs,
        option_pairs=args.option_pairs,
        nuisance_pairs=args.nuisance_pairs,
        seed=args.seed,
    )
    schema = pa.schema([
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("extra_info", pa.struct([("index", pa.int64())])),
        ("task_id", pa.int64()),
        ("data_source", pa.string()),
        ("sample_mode", pa.string()),
        ("pair_id", pa.string()),
        ("intervention_type", pa.string()),
        ("side", pa.string()),
        ("expected_action_intents", pa.list_(pa.string())),
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
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
