"""In-place dedup of duplicated instruction in trajectory data files.

Applies the SAME logic as teacher.validate._dedup_first_user (imported, not
re-implemented) to every record's first user message, rewriting each file in
place (atomic: write <file>.tmp then os.replace). Safe on already-clean data:
if a record's first user turn is not the exact 'INSTR\\n\\nINSTR\\n\\n' prefix,
it is left untouched.

Handles:
  *.jsonl   one JSON record per line (raw trajectories)
  *.json    single JSON record (per-task SFT files)

Usage:
  python scripts/dedup_trajectories.py <file1> <file2> ... <globN>
  (shell expands the globs; pass as many files as you like)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]   # shopagent-rl/
sys.path.insert(0, str(_ROOT))

from experiment.teacher.validate import _dedup_first_user  # noqa: E402  (single source of truth)


def _first_user_content(rec: Dict[str, Any]) -> str:
    msgs = rec.get("messages") or []
    if len(msgs) > 1 and msgs[1].get("role") == "user":
        return str(msgs[1].get("content", ""))
    return ""


def _dedup_record(rec: Dict[str, Any]) -> bool:
    """Dedup messages[1] in place; return True if it changed."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    before = _first_user_content(rec)
    new_msgs = _dedup_first_user(msgs)
    after = new_msgs[1].get("content", "") if len(new_msgs) > 1 else ""
    if after == before:
        return False
    rec["messages"] = new_msgs
    return True


def process_jsonl(path: str) -> Tuple[int, int]:
    """Stream-rewrite a jsonl: read line, dedup, write to .tmp, rename. (n, changed)."""
    tmp = path + ".dedup.tmp"
    n = changed = 0
    with open(path, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                fo.write(line)            # preserve blank lines
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fo.write(line)            # keep unparseable lines as-is
                continue
            n += 1
            if _dedup_record(rec):
                changed += 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return n, changed


def process_json(path: str) -> Tuple[int, int]:
    """Single-record JSON file (per-task SFT). Returns (1, changed)."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    changed = 1 if _dedup_record(rec) else 0
    tmp = path + ".dedup.tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return 1, changed


def main() -> None:
    files = sys.argv[1:]
    if not files:
        print("no files given", file=sys.stderr)
        sys.exit(2)
    tot_n = tot_changed = 0
    for fp in files:
        p = Path(fp)
        if not p.exists():
            print(f"  [skip] missing: {fp}")
            continue
        if p.suffix == ".jsonl":
            n, ch = process_jsonl(str(p))
        elif p.suffix == ".json":
            n, ch = process_json(str(p))
        else:
            print(f"  [skip] unknown ext: {fp}")
            continue
        tot_n += n
        tot_changed += ch
        flag = f"  deduped {ch:>5}/{n:<6}" if ch else f"  clean   {0:>5}/{n:<6}"
        print(f"{flag}  {fp}")
    print(f"\nTOTAL: deduped {tot_changed}/{tot_n} records across {len(files)} files")


if __name__ == "__main__":
    main()
