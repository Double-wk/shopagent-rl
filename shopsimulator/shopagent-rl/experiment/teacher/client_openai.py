"""Teacher LLM client: OpenAI-compatible endpoint(s).

支持多端点轮询：endpoints 参数传入多个 {base_url, api_keys}，所有 (端点, key) 对
跨线程 round-robin 轮询；任一 key 余额耗尽(401/402/403) 自动剔除该 (端点,key)，
该端点所有 key 都死则该端点停用；所有端点都死 → AllKeysExhausted 熔断。
向后兼容：仅传 base_url + api_keys/api_key 时退化为单端点多 key（原 deepseek 用法不变）。
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import requests

# 抑制 verify=False 的 InsecureRequestWarning（部分端点跳过证书校验，警告会刷屏淹没真实错误）
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# 可选 token 用量记账：设置环境变量 TEACHER_USAGE_LOG=<file> 后，每次调用把
# (model, prompt_tokens, completion_tokens, reasoning_tokens) 追加到该文件。
# 不设置则完全不记，零开销。
_USAGE_LOG_PATH = os.environ.get("TEACHER_USAGE_LOG", "")


def _log_usage(model: str, usage) -> None:
    # 运行时读 environ，不依赖上面模块级 _USAGE_LOG_PATH 缓存：`python -m experiment.teacher.collect`
    # 启动时模块级变量在某些 import 时序下会读到空(实测 collect 进程不写 usage、但同 import
    # 路径的新进程能写)。进程 environ 一直有 TEACHER_USAGE_LOG，每次现读(dict 查找零开销)。
    path = os.environ.get("TEACHER_USAGE_LOG", "")
    if not path or not usage:
        return
    try:
        comp_detail = usage.get("completion_tokens_details") or {}
        with open(path, "a") as f:
            f.write(f"{model}\t{usage.get('prompt_tokens', 0)}\t{usage.get('completion_tokens', 0)}"
                    f"\t{comp_detail.get('reasoning_tokens', 0)}\n")
    except Exception:
        pass


log = logging.getLogger(__name__)


class AllKeysExhausted(RuntimeError):
    """所有端点/key 都不可用（余额耗尽 / 当日额度用尽 / 鉴权失败）。

    上抛此异常让 collect.py 熔断：取消剩余未启动任务、收尾在跑的，本次 run 干净停止，
    而不是空转着对必失败的任务刷重试、浪费 token。下次（充值/额度重置后）断点续采即可。
    """


class OpenAITeacherClient:
    """OpenAI-compatible Teacher client，支持多端点 + 多 key 轮询。

    - endpoints: [{"base_url": ..., "api_keys": [...]}] —— 多端点，跨端点 round-robin
    - 向后兼容：仅 base_url + api_keys/api_key → 单端点多 key

    跨所有实例/线程共享 round-robin 计数器与 dead (url,key) 集合（进程级，下次 run 重置 →
    充值/额度重置后该 key 会被重新尝试）。
    """

    # 跨实例/线程共享：round-robin 计数器 + dead (url,key) 集合 + 短期冷却表
    _counter = itertools.count()
    _lock = threading.Lock()
    _dead: set = set()
    # (url,key) -> 恢复时间(monotonic)。5xx/超时类临时故障短期冷却(默认 60s)，
    # 期间 _candidates 跳过它、_next 自动换稳定端点；过期自动恢复再试。不永久剔除
    # (区别于 401/402/403 的 _dead)——服务端临时不稳，冷却完应重新尝试做冗余。
    _cooldown: dict = {}

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        endpoints: Optional[List[dict]] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 5,
        chat_completions_path: str = "/v1/chat/completions",
    ) -> None:
        eps: List[dict] = []
        # 新格式：endpoints 列表（多端点）
        for e in (endpoints or []):
            bu = (e.get("base_url") or "").rstrip("/")
            if not bu:
                continue
            keys: List[str] = []
            for k in (e.get("api_keys") or []):
                if k and k not in keys:
                    keys.append(k)
            ak = e.get("api_key")
            if ak and ak not in keys:
                keys.append(ak)
            if keys:
                # weight: 该端点在 round-robin 中的相对权重（默认 1）。稳定/快的端点
                # 调大、不稳定(常 503/慢)的调小，让请求自然偏向稳定端点、减少无效重试。
                weight = max(1, int(e.get("weight", 1) or 1))
                # Some OpenAI-compatible providers advertise a base URL ending
                # in `/v1`, while the project default path also starts with
                # `/v1`.  Normalize the join so either spelling works and we
                # never silently request `/v1/v1/chat/completions`.
                path = chat_completions_path
                if bu.endswith("/v1") and path.startswith("/v1/"):
                    path = path[len("/v1"):]
                eps.append({"url": bu + path, "api_keys": keys, "weight": weight,
                            "model": e.get("model") or ""})
        # 旧格式兜底：base_url + api_keys/api_key（向后兼容 deepseek 等单端点配置）
        if not eps:
            bu = (base_url or os.environ.get("OPENAI_TEACHER_BASE_URL", "https://fluxionai.space")).rstrip("/")
            keys = []
            for k in (api_keys or []):
                if k and k not in keys:
                    keys.append(k)
            if api_key and api_key not in keys:
                keys.append(api_key)
            if not keys:
                keys = [os.environ.get("OPENAI_TEACHER_API_KEY", "")]
            path = chat_completions_path
            if bu.endswith("/v1") and path.startswith("/v1/"):
                path = path[len("/v1"):]
            eps.append({"url": bu + path, "api_keys": keys})
        self.endpoints = eps
        # 按端点覆盖模型名：不同代理对同一模型的命名不同（如 seekai 的 gpt-5-6-terra
        # 是连字符写法）。url -> model；未覆盖的端点沿用全局 self.model。
        self._model_by_url = {ep["url"]: ep["model"] for ep in eps if ep.get("model")}
        self.chat_completions_path = chat_completions_path
        # 向后兼容属性（取第一个端点，供外部读取）
        self.url = eps[0]["url"]
        self.api_keys = eps[0]["api_keys"]
        self.api_key = eps[0]["api_keys"][0]
        if eps[0]["url"].endswith(chat_completions_path):
            self.base_url = eps[0]["url"][: -len(chat_completions_path)]
        else:
            self.base_url = eps[0]["url"]
        self.model = model or os.environ.get("OPENAI_TEACHER_MODEL", "gpt-5.6-sol")
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = False

    def _candidates(self, include_cooldown: bool = False) -> List[Tuple[str, str]]:
        """所有 live (url, key) 对（剔除已死的；冷却中的默认也跳过）。

        include_cooldown=True 时连冷却中的一并返回——仅在全部端点都在冷却中时
        作为降级 fallback 调用，避免 ziliao 也临时 503 时误判全死而熔断。
        """
        now = time.monotonic()
        out: List[Tuple[str, str]] = []
        for ep in self.endpoints:
            w = ep.get("weight", 1)
            for k in ep["api_keys"]:
                pair = (ep["url"], k)
                if pair in self._dead:
                    continue
                if not include_cooldown and self._cooldown.get(pair, 0) > now:
                    continue  # 冷却未到期，跳过
                # 按权重重复 (url,key)：weight=3 → 占 3 票，round-robin 自然偏向它。
                out.extend([pair] * w)
        return out

    def _next(self) -> Optional[Tuple[str, str]]:
        """round-robin 取下一个可用 (url, key)；全死返回 None。

        全部都在冷却中时降级：连冷却中的一并考虑（某个可能已恢复），不误判熔断。
        """
        cands = self._candidates()
        if not cands:
            cands = self._candidates(include_cooldown=True)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        with self._lock:
            idx = next(self._counter) % len(cands)
        return cands[idx]

    def _mark_dead(self, url: str, key: str, reason: str = "") -> None:
        """把某 (端点, key) 标记为不可用，后续 _next 自动跳过。"""
        with self._lock:
            if (url, key) in self._dead:
                return
            self._dead.add((url, key))
        total = sum(len(ep["api_keys"]) for ep in self.endpoints)
        n_dead = sum(1 for ep in self.endpoints for k in ep["api_keys"]
                     if (ep["url"], k) in self._dead)
        log.warning("端点 %s key ...%s 标记不可用(%s)。剩余可用 (端点,key): %d/%d",
                    url, key[-6:], reason[:80], total - n_dead, total)

    def _mark_cooldown(self, url: str, key: str, seconds: float = 60.0, reason: str = "") -> None:
        """把某 (端点, key) 短期冷却（5xx/超时类临时故障）：seconds 内 _next 跳过它、
        换稳定端点；到期自动恢复。用于避开反复 503 的不稳端点(如 191.96)，不永久剔除。
        """
        now = time.monotonic()
        with self._lock:
            self._cooldown[(url, key)] = now + seconds
        n_cooled = sum(1 for ep in self.endpoints for k in ep["api_keys"]
                       if self._cooldown.get((ep["url"], k), 0) > now)
        log.warning("端点 %s key ...%s 冷却 %ds(%s)。当前冷却中: %d 个 (url,key)",
                    url, key[-6:], int(seconds), reason[:60], n_cooled)

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant message content."""
        api_messages: List[Dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        body = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err: Optional[Exception] = None
        # 401/402/403 = 余额耗尽/额度用尽/鉴权失败/需付款 → 剔除该 (端点,key) 换可用的重试；全死则熔断。
        # tokenrhythm 用 402 Payment Required 表示余额不足，必须一并判定，否则死 key 不剔除、collect 不熔断。
        for attempt in range(1, self.max_retries + 1):
            nk = self._next()
            if nk is None:
                raise AllKeysExhausted(
                    f"所有端点/key 不可用，已剔除 {len(self._dead)} 个 (url,key)"
                )
            url, key = nk
            body["model"] = self._model_by_url.get(url, self.model)
            headers = {
                "Authorization": f"Bearer {key}",
                "content-type": "application/json",
            }
            try:
                r = self.session.post(url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code in (401, 402, 403):
                    self._mark_dead(url, key, f"{r.status_code} {r.text[:120]}")
                    last_err = PermissionError(f"...{key[-6:]} {r.status_code}: {r.text[:120]}")
                    continue
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}", response=r)
                r.raise_for_status()
                data = r.json()
                _log_usage(self.model, data.get("usage"))
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                # 5xx / 429 / 超时 / 连接错误 = 该端点临时不稳 → 短期冷却，retry 时 _next
                # 自动换稳定端点，不再反复撞这个 503 端点空等。区别于 401/402/403(永久 mark_dead)。
                if isinstance(e, (requests.Timeout, requests.ConnectionError)) or (
                    isinstance(e, requests.HTTPError)
                    and getattr(e, "response", None) is not None
                    and (e.response.status_code >= 500 or e.response.status_code == 429)
                ):
                    self._mark_cooldown(url, key, seconds=60, reason=repr(e)[:60])
                wait = min(2 ** attempt, 20)
                log.warning("teacher call failed (attempt %d/%d): %s; retry in %ds",
                           attempt, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"teacher call failed after {self.max_retries} attempts: {last_err}")


if __name__ == "__main__":
    # Smoke test:
    #   export OPENAI_TEACHER_API_KEY=your_key
    #   python -m experiment.teacher.client_openai
    client = OpenAITeacherClient()
    print(client.chat([{"role": "user", "content": "reply with exactly: OK"}], system=""))
