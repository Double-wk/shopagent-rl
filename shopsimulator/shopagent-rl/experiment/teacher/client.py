"""Teacher LLM client: Zhipu GLM-4.6 via the Anthropic-compatible endpoint.

Verified working on 2026-08-06: POST {base_url}/v1/messages with headers
`x-api-key` + `anthropic-version: 2023-06-01` returns standard Anthropic
Messages JSON. Uses `requests` directly (no `anthropic` SDK dependency) and
bypasses the local mihomo proxy (the CN endpoint is directly reachable; the
proxy routes abroad and is not needed here).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import requests
log = logging.getLogger(__name__)
ANTHROPIC_VERSION = "2023-06-01"

# Credentials hardcoded in-source by request (no .env / env-var dependency).
# If this repo is ever pushed to a remote, rotate the key in the BigModel console.
TEACHER_BASE_URL = "https://open.bigmodel.cn/api/anthropic"
TEACHER_API_KEY = "546b387222144da2a6b65751c75ecaf9.uCtKSgXFtAXoZuGh"
TEACHER_MODEL = "glm-5.2"


class TeacherClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 5,
    ) -> None:
        self.base_url = (base_url or os.environ.get("TEACHER_BASE_URL") or TEACHER_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY") or TEACHER_API_KEY
        self.model = model or os.environ.get("TEACHER_MODEL") or TEACHER_MODEL
        self.url = f"{self.base_url}/v1/messages"
        self.timeout = timeout
        self.max_retries = max_retries

        # Direct connection; skip proxy env + cert verify (matches the verified probe).
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = False

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 768,
        temperature: float = 0.5,
    ) -> str:
        """Return the concatenated text blocks of the assistant message."""
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(self.url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}", response=r)
                # 401/403 are not transient (bad key / forbidden) — surface at once,
                # without retry/backoff, so a misconfigured key fails fast.
                if r.status_code in (401, 403):
                    raise PermissionError(f"teacher auth failed ({r.status_code}): {r.text[:160]}")
                r.raise_for_status()
                data = r.json()
                return "".join(
                    blk.get("text", "")
                    for blk in data.get("content", [])
                    if blk.get("type") == "text"
                )
            except PermissionError:
                raise  # non-retryable auth failure
            except Exception as e:  # noqa: BLE001 - retry any transient failure
                last_err = e
                wait = min(2 ** attempt, 20)
                log.warning("teacher call failed (attempt %d/%d): %s; retry in %ds", attempt, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"teacher call failed after {self.max_retries} attempts: {last_err}")


if __name__ == "__main__":
    # Smoke test (set TEACHER_API_KEY in your env first):
    #   python -m experiment.teacher.client
    tc = TeacherClient()
    print(tc.chat([{"role": "user", "content": "reply with exactly: OK"}], system=""))
