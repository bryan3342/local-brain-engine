"""Stage 0: read the conversation index.

Reads the plugin's archive rather than the raw transcript directory, because
transcripts are pruned aggressively while the archive keeps every session. It is
also already deduplicated and indexed.

The cursor is the SQLite `rowid`, not `id`. `id` is a content hash and does not
sort in insertion order, so resuming by it would skip newly-inserted rows with
no error.

Read-only. This module never writes to that database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from brain.application.knowledge_api import KnowledgeApi
from brain.domain.errors import VaultError
from brain.etl.prefilter import prefilter
from brain.etl.queue import Queue

EPISODIC_DB = Path.home() / ".config/superpowers/conversation-index/db.sqlite"


def unseen_exchanges(
    db_path: Path | None = None,
    after_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Main-line exchanges ordered by rowid, starting after `after_id`.

    `after_id` is the previous call's last `rowid`, carried as a string
    because the queue cursor is stored as text; an unparseable value (for
    example a leftover content-hash cursor from before rowid was the cursor)
    is treated as "start from the beginning" rather than raised.

    Sidechains are excluded: subagent transcripts are the agent talking to
    itself, which is exactly the stratum both prior audits found worthless.
    """
    path = db_path or EPISODIC_DB
    if not Path(path).is_file():
        return []

    after: int | None = None
    if after_id:
        try:
            after = int(after_id)
        except (TypeError, ValueError):
            after = None

    sql = ("SELECT rowid AS rowid, id, project, timestamp, user_message,"
           " assistant_message, session_id, git_branch, cwd FROM exchanges"
           " WHERE COALESCE(is_sidechain, 0) = 0")
    args: list[Any] = []
    if after is not None:
        sql += " AND rowid > ?"
        args.append(after)
    sql += " ORDER BY rowid LIMIT ?"
    args.append(limit)

    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise VaultError(f"episodic-memory database at {path} could not be opened: {exc}") from exc
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args)]
    except sqlite3.Error as exc:
        raise VaultError(f"episodic-memory database at {path} could not be read: {exc}") from exc
    finally:
        con.close()


def reported_sessions(api: KnowledgeApi) -> set[str]:
    """Session ids that already have a note, so we do not mine them twice."""
    out: set[str] = set()
    for note in api.iter_notes(include_deleted=True):
        source = note.frontmatter.source or {}
        if str(source.get("type", "")).startswith("claude-code-session"):
            ref = str(source.get("ref") or "")
            if ref:
                out.add(ref)
    return out


def scan(
    api: KnowledgeApi,
    queue: Queue,
    db_path: Path | None = None,
    limit: int = 500,
) -> dict[str, int]:
    """One pass: read, filter, queue, log rejections, advance the cursor.

    The cursor advances only after a successful append, so a crash re-reads
    rather than skips.
    """
    rows = unseen_exchanges(db_path, after_id=queue.cursor(), limit=limit)
    if not rows:
        return {"read": 0, "queued": 0, "requeued": 0, "rejected": 0}

    # A candidate that is still pending is a re-read, not a duplicate. The cursor
    # advances only after a successful append, so a crash in between guarantees
    # these rows come back, and calling them `duplicate-content` would put a
    # lie in rejected.jsonl, which is the pipeline's audit trail. Excluding them
    # from `seen` lets classify keep them; append() then dedups them for free.
    pending_ids = queue.pending_ids()
    kept, dropped = prefilter(rows, seen=queue.seen_hashes() - pending_ids,
                              reported=reported_sessions(api),
                              rules=api.config.projects)
    requeued = sum(1 for c in kept if c.candidate_id in pending_ids)
    queued = queue.append(kept)
    rejected = queue.reject(dropped)
    queue.advance(str(rows[-1]["rowid"]))
    return {"read": len(rows), "queued": queued, "requeued": requeued,
            "rejected": rejected}
