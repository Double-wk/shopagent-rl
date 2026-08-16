"""Validate raw teacher trajectories and build the SFT dataset.

A raw trajectory (teacher/collect.py) already carries the env's terminal
reward_detail + per-step legality, captured against the LIVE service during
collection. This script:

  1. keeps trajectories that STRICTLY succeed (all four reward dimensions
     satisfied) with an acceptable action-legality ratio and sane length;
  2. optionally replays the action sequence through a fresh env (--replay) as a
     structural sanity check. NOTE: product prices are randomized per reset
     (engine.generate_product_prices), so the price dimension can differ on
     replay — replay is diagnostic only and does NOT gate selection;
  3. writes per-task SFT records + a combined jsonl index to
     data/trajectories_sft/.

Target: ~6,800 of 10,000 collected. If more pass than --max_keep, the highest-reward
ones are kept.

Run:
    python -m experiment.teacher.validate --max_keep 6800
    python -m experiment.teacher.validate --max_keep 6800 --replay   # needs the live service
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from shop_env import reward as R

_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def passes(rec: Dict[str, Any], min_legal: float, min_steps: int, max_steps: int) -> bool:
    if not rec.get("ok", False):
        return False
    if not rec.get("strict_success"):
        return False
    n = rec.get("n_steps", 0)
    if n < min_steps or n > max_steps:
        return False
    illegal = rec.get("illegal_steps", 0)
    legal_ratio = 1.0 if n == 0 else 1.0 - illegal / n
    return legal_ratio >= min_legal


def replay(rec: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """Re-run the recorded assistant actions through a fresh env (diagnostic)."""
    from shop_env.client import ShopEnvClient
    from shop_env.wrapper import ShopSimEnv

    env = ShopSimEnv(ShopEnvClient(base_url), max_steps=42)
    out: Dict[str, Any] = {"reachable": False, "done": False, "reward": 0.0}
    try:
        msgs = rec["messages"]
        assistant_turns = [m["content"] for m in msgs if m["role"] == "assistant"]
        env.reset(rec["task_id"])
        done = False
        for a in assistant_turns:
            _obs, _r, done, _info = env.step(a)
            if done:
                break
        out["done"] = bool(done)
        out["reachable"] = bool(done)
    finally:
        env.release()
    return out


def _dedup_first_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a duplicated instruction prefix in the first user turn.

    collect.py historically built first_user = env.instruction + '\\n\\n' +
    format_observation(obs), but format_observation(obs) ALREADY contains the
    instruction (obs is built from it via _compress) -> the instruction text
    appeared twice. This turns 'INSTR\\n\\nINSTR\\n\\n搜索功能是否可用:...'
    back into 'INSTR\\n\\n搜索功能是否可用:...'. No-op if not duplicated, so it
    is safe on already-clean data (e.g. trajectories collected after the fix).
    """
    if len(messages) < 2 or messages[1].get("role") != "user":
        return messages
    content = messages[1].get("content", "")
    marker = "搜索功能是否可用:"
    idx = content.find(marker)
    if idx <= 0:
        return messages
    prefix = content[:idx]                 # 'INSTR\n\nINSTR\n\n' when duplicated
    half = len(prefix) // 2
    if half > 0 and prefix[:half] == prefix[half:]:
        content = prefix[:half] + content[idx:]
        return [messages[0], {"role": "user", "content": content}] + messages[2:]
    return messages


def to_sft(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": rec["task_id"],
        "messages": _dedup_first_user(rec["messages"]),  # [{system}, {user}, {assistant}, ...]
        "reward": rec.get("reward", 0.0),
        "reward_detail": rec.get("reward_detail", {}),
        "n_steps": rec.get("n_steps", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate raw trajectories -> SFT dataset.")
    ap.add_argument("--raw", default=str(_ROOT / "data" / "trajectories_raw" / "gpt-5.6-terra"))
    ap.add_argument("--out", default=str(_ROOT / "data" / "trajectories_sft"))
    ap.add_argument("--max_keep", type=int, default=6800)
    ap.add_argument("--min_legal", type=float, default=0.8)
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--replay", action="store_true", help="replay through live env (diagnostic)")
    ap.add_argument("--base_url", default="http://127.0.0.1:5000")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 读取轨迹数据（优先 JSONL 格式，向后兼容旧 JSON 格式）
    candidates: List[Dict[str, Any]] = []
    n_fail = 0
    n_raw = 0

    jsonl_file = raw_dir / "trajectories_raw.jsonl"
    if jsonl_file.exists():
        # JSONL 格式（当前 collect.py 的输出）
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_raw += 1
                if passes(rec, args.min_legal, args.min_steps, args.max_steps):
                    candidates.append(rec)
                else:
                    n_fail += 1
    else:
        # 旧 JSON 格式（向后兼容：每任务一个 .json 文件）
        files = sorted(raw_dir.glob("*.json"))
        for fp in files:
            rec = _load(fp)
            n_raw += 1
            if passes(rec, args.min_legal, args.min_steps, args.max_steps):
                candidates.append(rec)
            else:
                n_fail += 1

    # keep highest-reward first, then fewest steps, then cap
    candidates.sort(key=lambda r: (-r.get("reward", 0.0), r.get("n_steps", 0)))
    kept = candidates[: args.max_keep]

    if args.replay:
        print(f"replaying {len(kept)} trajectories through {args.base_url} ...")
        for rec in kept:
            rec["_replay"] = replay(rec, args.base_url)

    combined = out_dir.parent / "sft_train.jsonl"
    n_written = 0
    with open(combined, "w", encoding="utf-8") as fcomb:
        for rec in kept:
            sft = to_sft(rec)
            with open(out_dir / f"{rec['task_id']}.json", "w", encoding="utf-8") as f:
                json.dump(sft, f, ensure_ascii=False, indent=2)
            fcomb.write(json.dumps(sft, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"raw={n_raw} passed={len(candidates)} kept={n_written} (max_keep={args.max_keep})")
    print(f"SFT index -> {combined}")
    if candidates and len(candidates) < args.max_keep:
        print(f"WARNING: only {len(candidates)} passed strict_success — collect more to reach {args.max_keep}")


if __name__ == "__main__":
    main()
