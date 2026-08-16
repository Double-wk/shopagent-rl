"""HTTP client for the ShopSimulator env service (pack_api.py @ :5000).

Wire protocol — verified against upstream shop_env/shop_env/shop_agent.py and
pack_api.py:

  reset:    {"action": "reset",    "idx": task_id}                  -> {instruction, instruction_simple, goal_options, env_idx, idx, [user_persona, reason_key]}
  interact: {"action": "interact", "env_idx", "response": action}   -> {done, reward, instruction, reward_detail, purchase, goal, over, env_idx, idx}
  release_one: {"action": "release_one", "env_idx"}                 -> free one env slot
  release_all: {"action": "release_all"}                            -> free all env slots

Action string format (the env splits on "\\nAction: " and takes the suffix):
    search['查询词']        click['可点击项名称']

The service manages a fixed pool of gym envs (default 20); reset() obtains an
env_idx, and the slot is auto-released when `over` is True. `over` is set when
the task is done OR history length exceeds MAX_HISTORY_LENGTH (42).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

log = logging.getLogger(__name__)

API_PATH = "/api/shop_agent"
DEFAULT_TIMEOUT = 30


class ShopEnvClient:
    """Thin, synchronous HTTP client wrapping the ShopSimulator Flask service."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}{API_PATH}"
        self.timeout = timeout

    def _post(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Flask/pyserini can briefly return 502 while a pool slot is being
        # released.  A rollout must not turn that transient service hiccup into
        # a failed Ray batch. Retry only transport/5xx failures; 4xx and invalid
        # protocol responses remain actionable errors.
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(self.url, json=data, timeout=self.timeout)
                if resp.status_code >= 500:
                    resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                delay = 1.0 * (attempt + 1)
                log.warning("shop env request failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        else:  # pragma: no cover - loop either breaks or raises
            raise last_error or RuntimeError("shop env request failed")
        resp.raise_for_status()
        body = resp.json()
        if "result" not in body:
            raise ValueError(f"unexpected response (no 'result' key): {body}")
        result = body["result"]
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"shop env error: {result['error']}")
        return result

    # --- env lifecycle ---
    def reset(self, task_id: int, if_persona: bool = False) -> Dict[str, Any]:
        """Start a task; returns instruction + an assigned env_idx."""
        return self._post({"action": "reset", "idx": int(task_id), "if_persona": bool(if_persona)})

    def interact(self, env_idx: int, action_text: str) -> Dict[str, Any]:
        """Apply one action; returns next observation + reward fields + terminal flags."""
        return self._post({"action": "interact", "env_idx": int(env_idx), "response": action_text})

    def release(self, env_idx: int) -> Dict[str, Any]:
        return self._post({"action": "release_one", "env_idx": int(env_idx)})

    def release_all(self) -> Dict[str, Any]:
        return self._post({"action": "release_all"})


if __name__ == "__main__":
    # Minimal connectivity check against a running service:
    #   python -m shop_env.client
    import os
    from dotenv import load_dotenv

    load_dotenv()
    c = ShopEnvClient(os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000"))
    print("release_all ->", c.release_all())
    r = c.reset(0)
    print("reset(0) ->", {k: r.get(k) for k in ("env_idx", "idx", "message")})
    if "env_idx" in r:
        print("release ->", c.release(r["env_idx"]))
