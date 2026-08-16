"""Final-200 eval runner: Base / SFT / GRPO on the same 200 held-out tasks.

Same multi-turn rollout loop as teacher/collect.py::run_one, but driven by a
LOCAL vLLM policy (LocalVLLM) instead of the teacher API, and BATCHED across the
pack_api env pool:

  * tasks split into WAVES of W (<=16, the env16 pool size)
  * per wave: reset all W envs; then round-stepped — each round builds the next
    assistant prompt for every still-active task, calls engine.generate_batch()
    ONCE (vLLM batches internally), steps every active env, retires terminated
    tasks; loops until the wave is exhausted or max_turns.

This keeps the env pool saturated and gives vLLM a real batch each round, instead
of one sequential generation per task (which would ignore vLLM's strength).

Results per task: {task_id, done, purchased, reward, reward_detail, n_steps,
illegal_steps, reached_cap}. strict_success is derived in experiment.eval.metrics.aggregate.

Run via scripts/run_eval.sh (sources the local ROCm/vLLM environment).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

_ROOT = Path(__file__).resolve().parents[2]   # shop_A/
sys.path.insert(0, str(_ROOT))

from shop_env.client import ShopEnvClient          # noqa: E402
from shop_env.wrapper import ShopSimEnv, truncate_after_first_action  # noqa: E402
from shop_env.obs_format import format_observation  # noqa: E402
from experiment.eval.metrics import aggregate, print_report  # noqa: E402

BASE_URL = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")


def load_system_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]


def _reset_one(task_id: int, max_turns: int, retries: int = 4, backoff: float = 6.0) -> Tuple[ShopSimEnv, Dict[str, Any]]:
    """Create + reset one env with retry (absorbs transient pool exhaustion)."""
    env = ShopSimEnv(ShopEnvClient(BASE_URL), max_steps=max_turns)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            obs, _ = env.reset(task_id)
            return env, obs
        except RuntimeError as e:
            if "Unable to get available environment" not in str(e):
                raise
            last = e
            time.sleep(backoff * (attempt + 1))
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(backoff * (attempt + 1))
    raise last  # type: ignore[misc]


def _step_one(env: ShopSimEnv, response: str) -> Tuple[Any, Any, bool, Dict[str, Any], Optional[str]]:
    """Run one environment step, returning an exception instead of raising."""
    try:
        obs, reward, done, info = env.step(response)
        return obs, reward, done, info, None
    except Exception as e:  # noqa: BLE001
        return None, 0.0, False, {}, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final-200 eval: Base / SFT / GRPO.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (SFT/GRPO); omit for Base")
    ap.add_argument("--tasks", default=str(_ROOT / "data" / "final200.json"))
    ap.add_argument("--system_prompt", default=str(_ROOT / "configs" / "teacher_gpt-5.6-terra.yaml"))
    ap.add_argument("--out", required=True, help="output jsonl path (combined)")
    ap.add_argument("--official_dir", default=None,
                    help="per-task official-schema dir (get_score.py-compatible); "
                         "default: <out stem>_tasks/")
    ap.add_argument("--tag", required=True, help="label: Base / SFT / GRPO")
    ap.add_argument("--wave", type=int, default=16, help="concurrent envs per wave (<=16 env16 pool)")
    ap.add_argument("--reset_workers", type=int, default=1,
                    help="concurrent environment resets (1 avoids pack_api pool races)")
    ap.add_argument("--step_workers", type=int, default=4,
                    help="concurrent environment HTTP steps per turn")
    # Use a 20-turn horizon for the current SFT/GRPO comparison. Keep this as
    # a CLI option so older benchmark protocols remain reproducible on demand.
    ap.add_argument("--max_turns", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N task ids (protocol smoke only)")
    args = ap.parse_args()

    sys_prompt = load_system_prompt(args.system_prompt)
    task_ids: List[int] = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        task_ids = task_ids[:args.limit]
    print(f"[{args.tag}] {len(task_ids)} tasks | model={args.model} "
          f"adapter={args.adapter or '(none)'} | wave={args.wave}")

    # Engine MUST be built inside main() (spawn guard — see local_vllm docstring).
    from experiment.eval.local_vllm import LocalVLLM
    engine = LocalVLLM(args.model, adapter_path=args.adapter)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    official_dir = Path(args.official_dir) if args.official_dir else \
        out_path.parent / (out_path.stem + "_tasks")
    official_dir.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")
    results: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for wi in range(0, len(task_ids), args.wave):
        wave = task_ids[wi:wi + args.wave]
        print(f"  wave {wi // args.wave + 1}: tasks {wave[0]}..{wave[-1]} ({len(wave)})")

        # reset all envs in the wave (parallel; each task owns one env from the pool)
        envs: List[ShopSimEnv] = []
        reset_ok: List[bool] = []
        with ThreadPoolExecutor(max_workers=min(len(wave), args.reset_workers)) as ex:
            reset_outs = list(ex.map(lambda t: _reset_one(t, args.max_turns), wave))
        convs: List[List[dict]] = []
        recs: List[Dict[str, Any]] = []
        active: List[bool] = []
        for tid, ro in zip(wave, reset_outs):
            env, obs = ro
            envs.append(env); reset_ok.append(True)
            # format_observation(obs) already includes the instruction (obs is
            # built from r["instruction"]); do NOT prepend env.instruction or it
            # duplicates (matches the dedup'd SFT data — see validate._dedup_first_user).
            first_user = format_observation(obs)
            convs.append([{"role": "user", "content": first_user}])
            recs.append({
                "task_id": tid, "tag": args.tag, "done": False, "purchased": False,
                "reward": 0.0, "reward_detail": {}, "goal": {}, "purchase": {},
                "n_steps": 0, "illegal_steps": 0, "reached_cap": False,
            })
            active.append(True)

        # round-stepped generation
        for _turn in range(args.max_turns):
            idxs = [i for i, a in enumerate(active) if a]
            if not idxs:
                break
            batch = [[{"role": "system", "content": sys_prompt}] + convs[i] for i in idxs]
            try:
                responses = engine.generate_batch(
                    batch, max_tokens=args.max_tokens, temperature=args.temperature
                )
            except Exception as e:  # noqa: BLE001
                # vLLM batch failure (e.g. a too-long context) — fail active tasks softly
                print(f"    [warn] generate_batch failed: {e!r}; retiring {len(idxs)} active tasks")
                for i in idxs:
                    active[i] = False
                    recs[i]["error"] = repr(e)
                break
            # Environment transitions are independent HTTP requests. Submit
            # them concurrently; serial stepping made a 30-turn wave spend
            # minutes waiting on network/service latency after generation had
            # already completed.
            step_inputs = []
            for k, i in enumerate(idxs):
                # The environment executes the first action only.  Keep the
                # same boundary in the next model context; otherwise a sampled
                # continuation can impersonate future user/assistant turns and
                # contaminate the rest of the rollout.
                resp = truncate_after_first_action(responses[k])
                convs[i].append({"role": "assistant", "content": resp})
                recs[i]["n_steps"] += 1
                step_inputs.append((i, resp))
            with ThreadPoolExecutor(max_workers=min(len(step_inputs), args.step_workers)) as ex:
                step_outs = list(ex.map(lambda x: _step_one(envs[x[0]], x[1]), step_inputs))
            for (i, _resp), step_out in zip(step_inputs, step_outs):
                obs, _r, step_done, info, step_error = step_out
                if step_error is not None:
                    # A single stalled HTTP interaction must not discard the
                    # other tasks in the wave. Keep the generated turn and
                    # retire only this task; the final report records the
                    # exception under its `error` field.
                    print(f"    [warn] env step failed task={recs[i]['task_id']}: {step_error}")
                    active[i] = False
                    recs[i]["error"] = step_error
                    continue
                if not info.get("legal", True):
                    recs[i]["illegal_steps"] += 1
                if step_done:
                    recs[i]["done"] = True
                    recs[i]["reward"] = float(info.get("reward", _r))
                    recs[i]["reward_detail"] = info.get("reward_detail", {}) or {}
                    recs[i]["goal"] = info.get("goal", {}) or {}
                    recs[i]["purchase"] = info.get("purchase", {}) or {}
                    recs[i]["purchased"] = bool(info.get("env_done"))
                    active[i] = False
                else:
                    convs[i].append({"role": "user", "content": format_observation(obs)})

        # release envs; tasks still active hit the step cap
        for i, env in enumerate(envs):
            env.release()
            recs[i]["reached_cap"] = bool(active[i]) or not recs[i]["purchased"]

        # flush combined jsonl + per-task official-schema files (get_score.py-compatible)
        for i, rec in enumerate(recs):
            # Keep the compact summary fields, but include the complete raw
            # conversation in the combined output as well. The per-task files
            # remain for get_score.py compatibility; this makes the single
            # JSONL artifact self-contained.
            full_rec = {**rec, "conversation": convs[i]}
            fout.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
            full_results.append(full_rec)
            # official schema: {task_id, reward, reward_detail, goal, purchase, conversation}
            official = {
                "task_id": rec["task_id"],
                "reward": rec.get("reward", 0.0),
                "reward_detail": rec.get("reward_detail", {}) or {},
                "goal": rec.get("goal", {}) or {},
                "purchase": rec.get("purchase", {}) or {},
                "conversation": convs[i],
            }
            with open(official_dir / f"{rec['task_id']}.json", "w", encoding="utf-8") as fo:
                json.dump(official, fo, ensure_ascii=False)
        fout.flush()
        results.extend(recs)
        n_succ = sum(1 for r in recs if _strict(r))
        print(f"    wave done: {n_succ}/{len(recs)} strict | "
              f"total {len(results)}/{len(task_ids)} ({time.time() - t0:.0f}s)")

    fout.close()
    print()
    print_report(args.tag, results)

    # Official ShopSimulator scoring (rule-based, the canonical benchmark metric):
    # run get_score.calculate_metrics on our official-schema per-task files so the
    # headline numbers come from the SAME function the paper uses.
    official_metrics = None
    try:
        sys.path.insert(0, "/workspace/shopsimulator/ShopSimulator")
        from get_score import calculate_metrics, print_metrics  # type: ignore
        files = sorted(glob.glob(str(official_dir / "*.json")))
        if files:
            m = calculate_metrics(files)
            official_metrics = m
            print("\n[official ShopSimulator get_score.py]")
            print_metrics(m)
            # persist for the report
            with open(out_path.parent / (out_path.stem + "_official_metrics.json"), "w",
                      encoding="utf-8") as fm:
                json.dump({"tag": args.tag, "metrics": m,
                           "n_tasks": len(results)}, fm, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"(official get_score skipped: {e!r})")

    # One final, self-contained document: every task, every turn, aggregate
    # metrics, official metrics (when available), and the exact runner knobs.
    # This is intentionally separate from the compact *_official_metrics.json.
    full_report_path = out_path.parent / (out_path.stem + "_full_report.json")
    with open(full_report_path, "w", encoding="utf-8") as fr:
        json.dump(
            {
                "tag": args.tag,
                "model": args.model,
                "adapter": args.adapter,
                "tasks_file": args.tasks,
                "wave": args.wave,
                "reset_workers": args.reset_workers,
                "step_workers": args.step_workers,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "n_tasks": len(full_results),
                "aggregate_metrics": aggregate(results),
                "official_metrics": official_metrics,
                "tasks": full_results,
            },
            fr,
            ensure_ascii=False,
            indent=2,
        )
    print(f"full self-contained report -> {full_report_path}")
    print(f"\nresults -> {out_path}\nofficial-schema per-task -> {official_dir}")


def _strict(r: Dict[str, Any]) -> bool:
    from shop_env import reward as R
    return bool(R.strict_success(r.get("reward_detail", {}), r.get("reward", 0.0)))


if __name__ == "__main__":
    main()
