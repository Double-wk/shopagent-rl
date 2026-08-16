"""Live smoke test of shop_env wrapper against the running service (:5000).

Validates the client wire protocol + wrapper parsing and dumps a REAL observation
so _compress/_extract_fields and obs_format can be refined against actual data.

Usage (run AFTER 00_start_env.sh; shopsimulator env):
    python scripts/test_wrapper.py [task_id]              # reset + dump
    python scripts/test_wrapper.py [task_id] "search[关键词]"   # reset then one step
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from shop_env.client import ShopEnvClient          # noqa: E402
from shop_env.obs_format import format_observation  # noqa: E402
from shop_env.wrapper import ShopSimEnv            # noqa: E402

BASE = "http://127.0.0.1:5000"


def dump(tag: str, obs: dict, info: dict | None = None) -> None:
    print(f"\n===== {tag} =====")
    print("has_search :", obs.get("has_search"), "| n_segments:", len(obs.get("page_segments", [])))
    print("clickables:", json.dumps(obs.get("clickables", []), ensure_ascii=False)[:400])
    print("fields     :", obs.get("fields"))
    segs = obs.get("page_segments", [])
    print("segments[0:3] (trimmed):")
    for s in segs[:3]:
        print("   ", s[:160])
    print("--- format_observation (first 700 chars) ---")
    print(format_observation(obs)[:700])
    if info:
        print("step info:", {k: info.get(k) for k in
              ("legal", "action_type", "matched_clickable", "done", "env_done", "reward", "steps")})
        if info.get("reward_detail"):
            print("reward_detail:", info["reward_detail"])


def main() -> None:
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = ShopSimEnv(ShopEnvClient(BASE), max_steps=30)
    try:
        obs, _ = env.reset(task_id)
        print("INSTRUCTION:", (env.instruction or "")[:600])
        dump("after reset", obs)
        if len(sys.argv) > 2:
            action = sys.argv[2]
            obs2, _r, done, info = env.step(action)
            dump(f"after step: {action!r}", obs2, info)
            print("done:", done)
    finally:
        env.release()


if __name__ == "__main__":
    main()
