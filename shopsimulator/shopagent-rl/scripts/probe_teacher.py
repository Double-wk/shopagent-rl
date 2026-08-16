#!/usr/bin/env python
"""Run a reproducible, endpoint-isolated ShopSimulator teacher probe.

The default 30 task IDs are the historical teacher-comparison set.  Passing
``--endpoint-substring`` isolates one configured provider, so a 200 response
cannot be mistaken for evidence that another round-robin endpoint is good.

Example:
  python scripts/probe_teacher.py --config configs/teacher_gpt-5.6-terra.yaml \
    --endpoint-substring gptgod.online --limit 10 --out run/teacher_probe/gptgod_10.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.teacher.collect import run_one  # noqa: E402
from experiment.teacher.client_openai import AllKeysExhausted  # noqa: E402


HISTORICAL_TASK_IDS = [
    2785, 3875, 4566, 4695, 4758, 6022, 6038, 6274, 8616, 9667,
    9943, 10694, 11397, 11621, 12278, 13052, 13191, 14082, 14727, 15240,
    16930, 17076, 17381, 17996, 18212, 18910, 19802, 20575, 21182, 21721,
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint-substring", required=True,
                    help="only use endpoint whose base_url contains this string")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 1 <= args.limit <= len(HISTORICAL_TASK_IDS):
        raise SystemExit(f"--limit must be in [1, {len(HISTORICAL_TASK_IDS)}]")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    teacher_cfg = dict(cfg["teacher"])
    matching = [ep for ep in teacher_cfg.get("endpoints", [])
                if args.endpoint_substring in ep.get("base_url", "")]
    if len(matching) != 1:
        raise SystemExit(f"expected exactly one matching endpoint, got {len(matching)}")
    teacher_cfg["endpoints"] = matching
    teacher_cfg["system_prompt"] = cfg.get("system_prompt", "")

    task_ids = HISTORICAL_TASK_IDS[:args.limit]
    env_url = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task_id, teacher_cfg, env_url,
                               bool(cfg.get("if_persona", False))): task_id
                   for task_id in task_ids}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                record = future.result()
            except AllKeysExhausted as exc:
                for remaining in futures:
                    remaining.cancel()
                raise SystemExit(f"teacher keys exhausted: {exc}") from exc
            records.append(record)
            print(f"task={task_id} ok={record.get('ok')} strict={record.get('strict_success')} "
                  f"reward={record.get('reward')} steps={record.get('n_steps')}", flush=True)

    records.sort(key=lambda row: row["task_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed = [row for row in records if row.get("ok")]
    strict = sum(bool(row.get("strict_success")) for row in completed)
    reward = sum(float(row.get("reward", 0.0)) for row in completed) / max(1, len(completed))
    legal_steps = sum(row.get("n_steps", 0) - row.get("illegal_steps", 0) for row in completed)
    all_steps = sum(row.get("n_steps", 0) for row in completed)
    print(json.dumps({
        "endpoint": matching[0]["base_url"], "n": len(task_ids), "completed": len(completed),
        "strict": strict, "strict_rate": strict / max(1, len(completed)),
        "mean_reward": reward, "legal_ratio": legal_steps / max(1, all_steps), "out": str(out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
