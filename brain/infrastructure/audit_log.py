"""Append-only audit log.

Records that an operation happened. It deliberately does **not** store a
pre-image, reconstructing prior state is git's job (H2), and duplicating it
here would be a second, weaker source of truth.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from brain.infrastructure.vault import append_jsonl


class AuditLog:
    def __init__(self, path: Path, author: str = "brain") -> None:
        self.path = path
        self.author = author

    def record(self, op: str, note_id: str | None = None, **detail: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "op": op,
            "author": self.author,
        }
        if note_id:
            entry["id"] = note_id
        entry.update({k: v for k, v in detail.items() if v is not None})
        append_jsonl(self.path, json.dumps(entry, ensure_ascii=False, default=str))
        return entry

    def entries(self, op: str | None = None) -> list[dict[str, Any]]:
        """Every entry, oldest first, optionally filtered by operation.

        `tail` answers "what just happened"; this answers "everything a given
        operation ever did", which is what an undo needs.
        """
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if op is None or entry.get("op") == op:
                out.append(entry)
        return out

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
