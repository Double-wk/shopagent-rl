"""Teacher trajectory collection over ShopSimulator tasks.

For each task: reset the env, loop the GPT-5.6-SOL teacher against it until terminal
or max_turns, and record the full conversation + terminal reward fields. Raw
trajectories are saved to data/trajectories_raw/<model>/trajectories_raw.jsonl (batch of 100);
teacher/validate.py later replays + filters to successful trajectories for SFT.

Run:
    python -m experiment.teacher.collect --config configs/teacher_gpt-5.6-terra.yaml --num 5000 --workers 24

Resumable: already-saved task ids are skipped (read from JSONL).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
from tqdm import tqdm

# make the shop_A root importable when run as a module/script
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from shop_env.client import ShopEnvClient          # noqa: E402
from shop_env.wrapper import ShopSimEnv            # noqa: E402
from shop_env.obs_format import format_observation  # noqa: E402
from shop_env import reward as R                    # noqa: E402
from experiment.teacher.client import TeacherClient           # noqa: E402
from experiment.teacher.client_openai import OpenAITeacherClient, AllKeysExhausted  # noqa: E402


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(task_id: int, teacher_cfg: Dict[str, Any], base_url: str, if_persona: bool,
            reset_retries: int = 4, reset_backoff: float = 6.0) -> Dict[str, Any]:
    """Collect one trajectory. Each worker owns its own env + teacher client."""
    env = ShopSimEnv(
        ShopEnvClient(base_url),
        max_steps=int(teacher_cfg.get("max_turns", 30)),
        if_persona=if_persona,
    )

    # 根据配置选择客户端类型
    client_type = teacher_cfg.get("client_type", "anthropic")
    if client_type == "openai":
        teacher = OpenAITeacherClient(
            model=teacher_cfg.get("model"),
            base_url=teacher_cfg.get("base_url"),
            api_key=teacher_cfg.get("api_key"),
            api_keys=teacher_cfg.get("api_keys"),
            endpoints=teacher_cfg.get("endpoints"),
            timeout=int(teacher_cfg.get("timeout", 120)),
            max_retries=int(teacher_cfg.get("max_retries", 5)),
            chat_completions_path=teacher_cfg.get("chat_completions_path", "/v1/chat/completions"),
        )
    else:  # default to Anthropic-compatible
        teacher = TeacherClient(
            model=teacher_cfg.get("model"),
            base_url=teacher_cfg.get("base_url"),
            api_key=teacher_cfg.get("api_key"),
            timeout=int(teacher_cfg.get("timeout", 120)),
            max_retries=int(teacher_cfg.get("max_retries", 5)),
        )

    sys_prompt = teacher_cfg["system_prompt"]
    max_tokens = int(teacher_cfg.get("max_tokens", 768))
    temperature = float(teacher_cfg.get("temperature", 0.5))

    rec: Dict[str, Any] = {
        "task_id": task_id, "ok": False, "messages": [], "n_steps": 0,
        "illegal_steps": 0, "reward": 0.0, "reward_detail": {},
        "purchase": {}, "goal": {}, "reached_cap": False, "strict_success": False,
    }
    try:
        # reset 带有限重试：吸收瞬时资源耗尽（池子暂时占满）与网络抖动。
        # "Unable to get available environment" 是临时错误——等几秒其他 worker
        # 释放槽后通常即可成功，不应直接判该 task 失败并写进数据集。
        obs = None
        for attempt in range(reset_retries):
            try:
                obs, _ = env.reset(task_id)
                break
            except RuntimeError as e:
                if "Unable to get available environment" not in str(e):
                    raise  # 非资源耗尽的 RuntimeError 不重试
                if attempt == reset_retries - 1:
                    raise
                time.sleep(reset_backoff * (attempt + 1))
            except requests.exceptions.RequestException:
                if attempt == reset_retries - 1:
                    raise
                time.sleep(reset_backoff * (attempt + 1))
        # format_observation(obs) already contains the instruction text (obs is
        # built from r["instruction"] via _compress), so prepending env.instruction
        # here would duplicate it. The single format_observation(obs) is the full
        # first user turn: instruction + search-availability + clickables.
        first_user = format_observation(obs)
        conv: List[Dict[str, str]] = [{"role": "user", "content": first_user}]
        done = False
        for _ in range(env.max_steps):
            resp = teacher.chat(conv, system=sys_prompt, max_tokens=max_tokens, temperature=temperature)
            obs, _r, step_done, info = env.step(resp)
            conv.append({"role": "assistant", "content": resp})
            rec["n_steps"] += 1
            if not info.get("legal", True):
                rec["illegal_steps"] += 1
            if step_done:
                done = True
                rec["reward"] = float(info.get("reward", _r))
                rec["reward_detail"] = info.get("reward_detail", {}) or {}
                rec["purchase"] = info.get("purchase", {}) or {}
                rec["goal"] = info.get("goal", {}) or {}
                break
            conv.append({"role": "user", "content": format_observation(obs)})

        rec["reached_cap"] = not done
        rec["strict_success"] = R.strict_success(rec["reward_detail"], rec["reward"])
        rec["messages"] = [{"role": "system", "content": sys_prompt}] + conv
        rec["ok"] = True
    except AllKeysExhausted:
        # 所有 key 余额/额度耗尽：上抛让 collect 熔断停整个 run（env 由 finally 释放），不在这里吞成 ok=False
        raise
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        env.release()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect teacher trajectories over ShopSimulator tasks.")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "teacher_gpt-5.6-terra.yaml"))
    ap.add_argument("--num", type=int, default=None, help="sample this many task ids (else full range)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="默认 data/trajectories_raw/<model>，按 config 的 model 自动分子目录")
    ap.add_argument("--targets", default=None,
                    help="json file: explicit task_id list to collect (overrides --num/--seed sampling; "
                         "resume-dedup against done_ids). Use to drive collection from "
                         "sample_sft_targets.py output so the set stays disjoint from GRPO/eval.")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    # system_prompt is a top-level key in the yaml (sibling of `teacher:`); run_one
    # reads it off teacher_cfg, so merge it in.
    teacher_cfg = {**cfg["teacher"], "system_prompt": cfg.get("system_prompt", "")}
    coll = cfg.get("collection", {})
    base_url = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")

    start = args.start if args.start is not None else coll.get("train_task_range", [1459, 23421])[0]
    end = args.end if args.end is not None else coll.get("train_task_range", [1459, 23421])[1]
    pool = list(range(start, end))

    if args.out is None:
        # 默认按 config 的 model 分子目录：不同 teacher 数据天然隔离，断点续采各自独立、永不会混
        out_dir = _ROOT / "data" / "trajectories_raw" / teacher_cfg.get("model", "unknown")
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ JSONL 格式保存（批量写入）
    jsonl_file = out_dir / "trajectories_raw.jsonl"
    save_batch_size = 50  # 每 50 条保存一次（更频繁落盘，减少中断时 buffer 丢失）

    # 读取已完成的 task_id
    done_ids = set()
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        done_ids.add(data.get("task_id"))
                    except json.JSONDecodeError:
                        continue

    pool = list(range(start, end))
    if args.targets:
        # Explicit task_id list (e.g. from sample_sft_targets.py) — collect exactly these,
        # minus whatever's already done (resume). Keeps the set disjoint from GRPO/eval
        # because the targets file was sampled that way; --num/--seed sampling is skipped.
        target_ids = json.loads(Path(args.targets).read_text(encoding="utf-8"))
        todo = [t for t in target_ids if t not in done_ids]
        print(f"[--targets] {len(target_ids)} targets, {len(done_ids)} already done -> collecting {len(todo)}")
    else:
        todo = [t for t in pool if t not in done_ids]
        if args.num and args.num < len(todo):
            rng = random.Random(args.seed)
            todo = rng.sample(todo, args.num)

    print(f"tasks: pool={len(pool)} already_done={len(done_ids)} sampling={len(todo)} workers={args.workers}")
    n_success = 0
    n_failed = 0   # 未持久化的（crashed 或 ok=False），下次运行会重试
    n_canceled = 0  # 熔断 cancel 掉的未启动任务（不算失败，断点续采下次接着跑）
    quota_stopped = False  # 是否因所有 key 余额/额度耗尽而提前熔断
    t0 = time.time()

    # 批量保存缓冲区（线程安全）
    buffer: List[Dict[str, Any]] = []
    buffer_lock = threading.Lock()

    def flush_buffer() -> None:
        """将缓冲区内容写入 JSONL 文件"""
        nonlocal buffer
        with buffer_lock:
            if not buffer:
                return
            with open(jsonl_file, "a", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            buffer.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, t, teacher_cfg, base_url, bool(cfg.get("if_persona", False))): t for t in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), desc="collect"):
            task_id = futures[fut]
            try:
                rec = fut.result()
            except AllKeysExhausted as e:
                # 所有 key 余额/额度耗尽 → 熔断：取消未启动任务，收尾在跑的，本次到此为止
                if not quota_stopped:
                    print(f"\n⚠️ {e} —— 取消剩余未启动任务，收尾在跑的，本次到此为止（断点续采下次接着跑）。")
                    quota_stopped = True
                    for f in list(futures):
                        f.cancel()
                n_failed += 1
                continue
            except CancelledError:
                # 熔断 cancel 掉的未启动任务，跳过（不算失败，断点续采下次接着跑）
                n_canceled += 1
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[task {task_id}] crashed: {e!r}")
                n_failed += 1
                continue

            # 只持久化完整跑完的轨迹（ok=True）。
            # 临时错误（资源耗尽/超时/teacher 异常）不写 jsonl：
            #   1) 避免空轨迹污染数据集；
            #   2) 避免把失败 task_id 标记为"已完成"——断点续传才能在下次重采这些任务。
            if not rec.get("ok"):
                n_failed += 1
                if rec.get("error"):
                    print(f"[task {task_id}] 未持久化(下次重试): {str(rec['error'])[:140]}")
                continue

            # 标记数据划分：eval(< train 起点) 仅作参考基线，绝不进训练；train 进训练
            train_start = coll.get("train_task_range", [1459, 23421])[0]
            rec["split"] = "eval" if task_id < train_start else "train"
            # 添加到缓冲区
            with buffer_lock:
                buffer.append(rec)

            # 每 100 条或最后一批时保存
            if len(buffer) >= save_batch_size:
                flush_buffer()

            if rec.get("strict_success"):
                n_success += 1

    # 保存剩余数据
    flush_buffer()

    n_persisted = len(todo) - n_failed - n_canceled
    msg = (f"done in {time.time() - t0:.0f}s | sampled={len(todo)} persisted(ok)={n_persisted} "
           f"strict_success={n_success} ({100*n_success/max(1,n_persisted):.1f}% of persisted) "
           f"| failed(retry next run)={n_failed} canceled={n_canceled}")
    if quota_stopped:
        msg += "  ⚠️ 因 key 余额/额度耗尽提前熔断，断点续采下次接着跑"
    print(msg)


if __name__ == "__main__":
    main()
