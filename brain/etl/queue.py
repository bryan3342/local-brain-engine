"""The ETL work queue: three append-only JSONL files plus a cursor.

JSONL rather than SQLite because the queue is small, is read by a skill as well
as by code, and benefits from being greppable when something looks wrong.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from brain.domain.errors import ValidationError
from brain.domain.models import Candidate, Rejection
from brain.infrastructure.vault import FileStamp, Vault, append_jsonl, atomic_write

QUEUE_DIR = "90-meta/queue"

_LOGGER = logging.getLogger(__name__)


class Queue:
    def __init__(self, vault: Vault) -> None:
        self.vault = vault
        self.root = vault.root / QUEUE_DIR

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.jsonl"

    @property
    def rejected_path(self) -> Path:
        return self.root / "rejected.jsonl"

    @property
    def cursor_path(self) -> Path:
        return self.root / "cursor.json"

    # ---- reading ----------------------------------------------------------

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file, skipping unparseable lines rather than failing.

        Skipped lines are logged, not silent, a queue that swallows corruption
        without a trace defeats the point of being greppable.
        """
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                _LOGGER.warning("%s: skipping unparseable JSON on line %d", path, lineno)
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
            else:
                _LOGGER.warning("%s: skipping non-dict JSON on line %d (got %s)",
                                 path, lineno, type(parsed).__name__)
        return out

    def pending(self, limit: int | None = None) -> list[Candidate]:
        """Pending candidates, skipping any line that will not construct.

        A line can parse as JSON and still not be a Candidate, the skill edits
        this file by hand, so that is the expected failure, not an exotic one.
        One bad line must not brick every drain, so it is logged and skipped,
        exactly as an unparseable line already is.
        """
        cands: list[Candidate] = []
        for row in self.read_jsonl(self.candidates_path):
            cid = row.get("candidate_id")
            if not cid:
                continue
            try:
                cands.append(Candidate.from_dict(row))
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                _LOGGER.warning("%s: skipping malformed candidate %s: %s",
                                self.candidates_path, cid, exc)
        return cands[:limit] if limit else cands

    def pending_ids(self) -> set[str]:
        """Ids of everything currently queued, without validating them.

        Deliberately does not build Candidates: callers use this to reason about
        what is already queued, and that answer must stay available even when a
        line in the file is malformed.
        """
        return {r["candidate_id"] for r in self.read_jsonl(self.candidates_path)
                if r.get("candidate_id")}

    def seen_hashes(self) -> set[str]:
        """Every candidate id ever queued or rejected.

        Rejections count: a candidate dropped once must not reappear on the
        next scan, or the filter would re-decide the same thing forever.
        """
        seen = {r["candidate_id"] for r in self.read_jsonl(self.candidates_path)
                if r.get("candidate_id")}
        seen |= {r["candidate_id"] for r in self.read_jsonl(self.rejected_path)
                 if r.get("candidate_id")}
        return seen

    def cursor(self) -> str | None:
        if not self.cursor_path.is_file():
            return None
        try:
            data = json.loads(self.cursor_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        value = data.get("last_exchange_id")
        return str(value) if value else None

    # ---- writing ----------------------------------------------------------

    def append(self, candidates: Iterable[Candidate]) -> int:
        existing = self.seen_hashes()
        added = 0
        for candidate in candidates:
            if candidate.candidate_id in existing:
                continue
            append_jsonl(self.candidates_path,
                         json.dumps(candidate.to_dict(), ensure_ascii=False))
            existing.add(candidate.candidate_id)
            added += 1
        return added

    def reject(self, rejections: Iterable[Rejection]) -> int:
        count = 0
        for rejection in rejections:
            append_jsonl(self.rejected_path,
                         json.dumps(rejection.to_dict(), ensure_ascii=False))
            count += 1
        return count

    def drop(self, candidate_ids: Iterable[str]) -> int:
        """Remove candidates from pending. Rewrites the file atomically.

        Stamps the file before reading it, then passes that stamp as `expect=`
        so a concurrent `append()` landing before the rewrite raises
        `ConcurrentModificationError` instead of being silently overwritten by
        our stale snapshot. The caller retrying is correct; clobbering is not.
        """
        drop_set = set(candidate_ids)
        stamp = FileStamp.of(self.candidates_path)
        rows = self.read_jsonl(self.candidates_path)
        keep = [r for r in rows if r.get("candidate_id") not in drop_set]
        removed = len(rows) - len(keep)
        if removed:
            body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
            atomic_write(self.candidates_path, body.encode("utf-8"), expect=stamp)
        return removed

    def advance(self, exchange_id: str) -> None:
        """Move the read cursor. Only ever called after a successful append."""
        atomic_write(self.cursor_path,
                     json.dumps({"last_exchange_id": exchange_id}).encode("utf-8"))
