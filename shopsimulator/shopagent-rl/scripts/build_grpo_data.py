"""Build the GRPO training parquet for veRL RLHFDataset.

One row per task_id. Columns (matching verl/utils/dataset/rl_dataset.py):
  * prompt     : list[dict] = [{"role":"system","content": <PROMPT_TEMPLATE_zh>}]
                 -> with data.return_raw_chat=True this becomes kwargs["raw_prompt"]
                 in ShopsimAgentLoop, which keeps the system msg and builds the
                 first user turn itself from the env reset observation.
  * extra_info : {"index": task_id}   # trajectory / GRPO-group tracking label
  * task_id    : int                  # ShopsimAgentLoop resets env on this
  * data_source: "shopsim"

rollout.n (configs/grpo.yaml, default 4) replicates each row -> GRPO groups of n.
Task pool = data/grpo_prompts_1000.json (1000 ids, range 1531..23408), which is
DISJOINT from data/final200.json (the 200 eval ids in [0,1459)) — verified.

The system prompt is read from configs/teacher_gpt-5.6-terra.yaml — the exact
prompt the SFT trajectories were collected (and trained) under — so GRPO
continues from SFT under an identical instruction framing.

Usage:
    python scripts/build_grpo_data.py
    # -> data/grpo_train.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SHOP_A = Path("/workspace/shopsimulator/shopagent-rl")
sys.path.insert(0, str(SHOP_A))

TASK_IDS_JSON = SHOP_A / "data" / "grpo_prompts_1000.json"
EVAL_JSON = SHOP_A / "data" / "final200.json"
TEACHER_YAML = SHOP_A / "configs" / "teacher_gpt-5.6-terra.yaml"
OUT_PARQUET = SHOP_A / "data" / "grpo_train.parquet"


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    task_ids: list[int] = json.loads(TASK_IDS_JSON.read_text())
    eval_ids = set(json.loads(EVAL_JSON.read_text()))
    overlap = set(task_ids) & eval_ids
    assert not overlap, f"GRPO train task_ids overlap eval final200: {sorted(overlap)[:10]}"

    sys_prompt = yaml.safe_load(TEACHER_YAML.read_text())["system_prompt"]
    assert sys_prompt and sys_prompt.strip(), "empty system_prompt in teacher yaml"

    rows = []
    for tid in task_ids:
        rows.append({
            "prompt": [{"role": "system", "content": sys_prompt}],
            "extra_info": {"index": int(tid)},
            "task_id": int(tid),
            "data_source": "shopsim",
        })

    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("extra_info", pa.struct([("index", pa.int64())])),
        ("task_id", pa.int64()),
        ("data_source", pa.string()),
    ]))

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT_PARQUET)

    print(f"wrote {len(rows)} rows -> {OUT_PARQUET}")
    print(f"  task_id range: {min(task_ids)}..{max(task_ids)}  | overlap with final200: {len(overlap)}")
    print(f"  system_prompt: {len(sys_prompt)} chars, head: {sys_prompt[:60]!r}")
    print(f"  sample row[0] task_id={rows[0]['task_id']} prompt_roles={[m['role'] for m in rows[0]['prompt']]}")

    # round-trip read to confirm schema survives parquet
    rt = pq.read_table(OUT_PARQUET)
    r0 = rt.to_pylist()[0]
    assert r0["prompt"][0]["role"] == "system" and r0["task_id"] == rows[0]["task_id"]
    print(f"  round-trip OK: {rt.num_rows} rows, cols={rt.column_names}")


if __name__ == "__main__":
    main()
