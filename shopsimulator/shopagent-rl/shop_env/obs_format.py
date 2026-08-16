"""Render a wrapper observation into a compact text block for the model context.

The env already returns observations as `[SEP]`-joined text plus an actions block
(see wrapper._compress). This module trims that to a token budget — the
"Observation 字段化压缩" step that keeps long-horizon context within 8K. Shared by
teacher collection, SFT dataset build, and eval so the model always sees the same
formatting.
"""
from __future__ import annotations

import json
from typing import Any, Dict


def format_observation(obs: Dict[str, Any], max_chars: int = 3500) -> str:
    """Compact, budget-capped text representation of a wrapper observation."""
    segments = obs.get("page_segments", []) or []
    text = " [SEP] ".join(segments)
    if len(text) > max_chars:
        text = text[:max_chars] + " …(已截断)"

    clickables = obs.get("clickables", []) or []
    lines = [
        text,
        f"搜索功能是否可用: {bool(obs.get('has_search'))}",
        f"可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}",
    ]
    fields = obs.get("fields") or {}
    if fields.get("price") is not None:
        lines.append(f"当前价格: {fields['price']}")
    return "\n\n".join(lines)
