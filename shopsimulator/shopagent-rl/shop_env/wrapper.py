"""Gym-like wrapper around ShopEnvClient (the :5000 service).

Implements the project's environment-capsulation requirements:
  * search/click action space with a structured Action API
  * clickable validation (exact -> fuzzy match via stdlib difflib; flag illegal clicks)
  * observation field-structured compression (parse [SEP]-text + actions block)
  * max-step termination (env hard-caps history at 42; we cap lower, e.g. 30)

Observation wire format (text mode; from web_agent_text_env.convert_html_to_text
+ shop_agent._format_available_actions):

    <page text joined by ' [SEP] '>
    \n\n搜索功能是否可用: True|False
    \n\n可点击的按钮: [<json list of lowercase clickable strings>]

Action format (env.parse_action, matches upstream PROMPT_TEMPLATE_zh — NO inner
quotes):  search[keywords]   click[value]
The env lowercases the arg; a click matches only if it equals a key in
text_to_clickable. We validate before sending and record legality (used by the
teacher-trajectory filter and as an optional reward penalty).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from .client import ShopEnvClient


@dataclass
class Action:
    """A structured environment action."""
    type: str   # "search" | "click"
    value: str

    def to_env(self) -> str:
        return f"{self.type}[{self.value.strip()}]"


# Matches an action anywhere in the model response. SFT checkpoints sometimes
# continue generating a copied user/assistant turn after the intended action
# (e.g. `Action: click[...]具有战士 ...`). Requiring the action to be the final
# text made the evaluator mark otherwise executable actions as illegal.
_ACTION_RE = re.compile(r"(search|click)\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)


def parse_model_action(text: str) -> Optional[Action]:
    """Extract the intended Action from a raw model output string.

    Handles `<think>` blocks and the `Action:` marker used by the upstream env.
    Returns None if no parseable action is found.
    """
    if not text:
        return None
    t = text
    if "</think>" in t:                       # strip reasoning blocks
        t = t.split("</think>")[-1]
    if "Action:" in t:                        # use the first intended action
        t = t.split("Action:", 1)[1]
    elif "Action:" in text:
        # Qwen3 may hit a short per-turn token cap before closing a <think>
        # block. The explicit Action marker is still unambiguous in that case.
        t = text.split("Action:", 1)[1]
    t = t.strip().strip("`").strip('"').strip("'")
    m = _ACTION_RE.search(t)
    if not m:
        return None
    # Strip surrounding quotes from the value too: the model often writes
    # click['643436000957'] / search["关键词"], and the env matches click targets
    # EXACTLY (lowercased) — a quoted '643436000957' never equals the clickable
    # "643436000957", so the click silently no-ops and the model loops forever.
    val = m.group(2).strip().strip("`").strip('"').strip("'").strip()
    return Action(m.group(1).lower(), val)


def truncate_after_first_action(text: str) -> str:
    """Keep one model turn through its first dispatchable action.

    The environment consumes only the first action, but retaining a model's
    continuation (often a hallucinated next user/assistant turn) in the chat
    history corrupts all following decisions.  This is intentionally applied
    at the protocol boundary rather than relying on a model to emit a stop
    token.  If no complete action exists, preserve the response for diagnostics.
    """
    if not text:
        return text
    start = text.find("Action:")
    search_from = start + len("Action:") if start >= 0 else 0
    match = _ACTION_RE.search(text, pos=search_from)
    return text[:match.end()] if match else text


class ShopSimEnv:
    """Single-task wrapper. reset(task_id) -> step(model_output) loop."""

    ENV_MAX_HISTORY = 42  # upstream hard cap (shop_agent.MAX_HISTORY_LENGTH)

    def __init__(
        self,
        client: ShopEnvClient,
        max_steps: int = 30,
        if_persona: bool = False,
        fuzzy_cutoff: float = 0.8,
    ) -> None:
        self.client = client
        self.max_steps = min(max_steps, self.ENV_MAX_HISTORY)
        self.if_persona = if_persona
        self.fuzzy_cutoff = fuzzy_cutoff
        self.env_idx: Optional[int] = None
        self.task_id: Optional[int] = None
        self.steps = 0
        self.instruction: Optional[str] = None
        self.goal_options: Optional[Any] = None
        self._last_clickables: List[str] = []
        self._last_has_search = True

    # ------------------------------------------------------------------ lifecycle
    def reset(self, task_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r = self.client.reset(task_id, if_persona=self.if_persona)
        self.env_idx = r["env_idx"]
        self.task_id = task_id
        self.steps = 0
        self.instruction = r.get("instruction")
        self.goal_options = r.get("goal_options")
        obs = self._compress(r.get("instruction", ""))
        # At task start the agent is on the landing page where search is always
        # available — but reset returns only the instruction (no action block), so
        # _compress would read has_search=False and the model would think it cannot
        # search (observed: GLM-4.6 then opens with `Action: wait`). Override here.
        obs["has_search"] = True
        self._last_has_search = True
        return obs, r

    def release(self) -> None:
        if self.env_idx is not None:
            try:
                self.client.release(self.env_idx)
            finally:
                self.env_idx = None

    # ------------------------------------------------------------------ step
    def step(self, model_output: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply one model output. Returns (obs, reward, done, info).

        The raw model_output is forwarded to the env as-is (the env extracts the
        `Action:` itself). Legality of the parsed action is recorded in info.
        """
        action = parse_model_action(model_output)
        info: Dict[str, Any] = {
            "raw_action": action.to_env() if action else (model_output or "")[:120],
            "action_type": action.type if action else None,
            "legal": True,
            "matched_clickable": None,
        }
        dispatch_action = action
        if action is None:
            info["legal"] = False
        elif action.type == "click":
            matched = self._validate_click(action.value)
            if matched is None:
                info["legal"] = False
            else:
                info["matched_clickable"] = matched
                # The service accepts exact clickable text only.  Sending the
                # fuzzy source value after accepting it locally creates a hidden
                # no-op, so canonicalize the payload to the actual button.
                dispatch_action = Action("click", matched)
        elif action.type == "search" and not self._last_has_search:
            info["legal"] = False

        # Send the action in a form the env's parser ALWAYS extracts. Upstream
        # _extract_action_from_response only strips a "\nAction: " prefix (with a
        # leading newline), so a bare "Action: search[...]" at the start of the
        # output (no preceding newline) is NOT stripped -> parse_action reads
        # action_name="Action: search" -> the env silently no-ops and re-renders
        # the current page. A bare action string is parsed correctly in every case
        # (verified against the live service) and makes dispatch independent of
        # how the model formatted its "Action:" marker.
        env_payload = dispatch_action.to_env() if dispatch_action else (model_output or "")
        resp = self.client.interact(self.env_idx, env_payload)

        self.steps += 1
        env_done = bool(resp.get("done"))
        env_over = bool(resp.get("over"))
        reached_cap = self.steps >= self.max_steps
        done = env_done or env_over or reached_cap

        reward = float(resp.get("reward", 0.0))
        obs = self._compress(resp.get("instruction", ""))

        info.update(
            reward=reward,
            reward_detail=resp.get("reward_detail", {}) if env_done else {},
            purchase=resp.get("purchase", {}),
            goal=resp.get("goal", {}),
            total_reward=reward,
            env_done=env_done,
            env_over=env_over,
            reached_cap=reached_cap,
            steps=self.steps,
        )
        return obs, reward, done, info

    # ------------------------------------------------------------------ parsing
    def _validate_click(self, value: str) -> Optional[str]:
        """Return the exact clickable string this click resolves to, or None."""
        val = value.strip().lower()
        clics_lower = [c.lower() for c in self._last_clickables]
        if val in clics_lower:
            return self._last_clickables[clics_lower.index(val)]
        cand = get_close_matches(val, clics_lower, n=1, cutoff=self.fuzzy_cutoff)
        if cand:
            return self._last_clickables[clics_lower.index(cand[0])]
        return None

    def _compress(self, raw: str) -> Dict[str, Any]:
        """Parse the env instruction into structured fields + available actions."""
        body = raw or ""
        clickables: List[str] = []
        has_search = False

        if "可点击的按钮:" in body:
            body, action_part = body.split("可点击的按钮:", 1)
            try:
                clickables = [str(c) for c in json.loads(action_part.strip())]
            except (ValueError, json.JSONDecodeError):
                clickables = []
        if "搜索功能是否可用:" in body:
            body, flag_part = body.split("搜索功能是否可用:", 1)
            has_search = flag_part.strip().lower().startswith("true")

        self._last_clickables = clickables
        self._last_has_search = has_search
        segments = [s.strip() for s in body.split(" [SEP] ") if s.strip()]
        fields = self._extract_fields(segments)
        return {
            "raw": raw,
            "has_search": has_search,
            "clickables": clickables,
            "page_segments": segments,
            "fields": fields,
        }

    @staticmethod
    def _extract_fields(segments: List[str]) -> Dict[str, Any]:
        """Field extraction from page segments, refined against live observations.

        Two real price layouts seen from the service:
          * detail page  -> a "价格: 65.0" segment (explicit Chinese prefix)
          * results page -> a bare "65.0" or "18.0 to 26.0" segment sitting between
            the ASIN and the product title.
        Prices always carry a decimal point ("65.0", "819.0"), while the 12-digit
        product ASINs are bare integers (e.g. "713834844237") — so requiring a
        decimal cleanly separates prices from ASINs.
        """
        text = " ".join(segments)
        price: Optional[str] = None

        # 1) Detail page: explicit "价格: <num>" / "价格: <lo> to <hi>".
        m = re.search(
            r"价格\s*[:：]\s*(\d+(?:\.\d+)?(?:\s*(?:to|~-?|-|—)\s*\d+(?:\.\d+)?)?)",
            text,
        )
        if m:
            price = m.group(1).strip()
        else:
            # 2) Results page: a segment that is exactly a decimal price (range).
            for seg in segments:
                mm = re.fullmatch(r"(\d+\.\d+)(?:\s+to\s+(\d+\.\d+))?", seg.strip())
                if mm:
                    price = f"{mm.group(1)} to {mm.group(2)}" if mm.group(2) else mm.group(1)
                    break

        return {"price": price, "n_segments": len(segments)}
