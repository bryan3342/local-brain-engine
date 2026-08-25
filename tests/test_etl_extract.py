# tests/test_etl_extract.py
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.errors import VaultError
from brain.domain.identity import (
    ProjectRules,
    normalize_project,
    project_from_cwd,
)
from brain.etl.extract import reported_sessions, scan, unseen_exchanges
from brain.etl.prefilter import prefilter
from brain.etl.queue import Queue

LONG = "why does the backend deploy not fire automatically when we push to main?"


#: What a user would put in their own vault config. Nothing here ships in the
#: package; the tests need something to resolve against.
RULES = ProjectRules(
    aliases=(("acme", "acme"),),
    repos=("portfolio-site", "client-site", "outreach-agent",
           "local-brain", "structa", "brain"),
)


def make_db(
    path: Path, rows: list[tuple[str, str, str]], project: str = "proj", cwd: str = ""
) -> Path:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE exchanges (id TEXT PRIMARY KEY, project TEXT, timestamp TEXT,"
        " user_message TEXT, assistant_message TEXT, session_id TEXT,"
        " git_branch TEXT, cwd TEXT, is_sidechain BOOLEAN DEFAULT 0);")
    con.executemany(
        "INSERT INTO exchanges (id, project, timestamp, user_message, assistant_message,"
        " session_id, git_branch, cwd, is_sidechain) VALUES (?,?,?,?,?,?,?,?,0)",
        [(i, project, "2026-08-20T00:00:00Z", u, "reply", s, "main", cwd) for i, u, s in rows])
    con.commit()
    con.close()
    return path


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    a = KnowledgeApi(
        BrainConfig(root=tmp_path / "vault", git=False, projects=RULES), index=None
    )
    a.initialize_vault()
    return a


def test_unseen_exchanges_reads_rows(tmp_path: Path) -> None:
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1"), ("ex-2", LONG + "!", "s1")])
    rows = unseen_exchanges(db)
    assert [r["id"] for r in rows] == ["ex-1", "ex-2"]


def test_unseen_exchanges_starts_after_the_cursor(tmp_path: Path) -> None:
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1"), ("ex-2", LONG + "!", "s1")])
    first = unseen_exchanges(db, limit=1)
    cursor = str(first[0]["rowid"])
    assert [r["id"] for r in unseen_exchanges(db, after_id=cursor)] == ["ex-2"]


def test_unseen_exchanges_orders_by_rowid_not_by_id(tmp_path: Path) -> None:
    """`id` is a content hash (episodic-memory's dedup key) and does not sort in
    insertion order. `rowid` does. Under `ORDER BY id` this would return
    ["aa-2", "mm-3", "zz-1"] instead."""
    db = make_db(tmp_path / "e.db",
                 [("zz-1", LONG, "s1"), ("aa-2", LONG + "!", "s2"), ("mm-3", LONG + "??", "s3")])
    rows = unseen_exchanges(db)
    assert [r["id"] for r in rows] == ["zz-1", "aa-2", "mm-3"]


def test_unseen_exchanges_resumes_by_rowid_despite_non_monotonic_ids(tmp_path: Path) -> None:
    """A scan resuming after the first-inserted row must return the remaining
    two, in insertion order, even though their ids sort the other way."""
    db = make_db(tmp_path / "e.db",
                 [("zz-1", LONG, "s1"), ("aa-2", LONG + "!", "s2"), ("mm-3", LONG + "??", "s3")])
    first = unseen_exchanges(db, limit=1)
    assert [r["id"] for r in first] == ["zz-1"]
    cursor = str(first[0]["rowid"])
    remaining = unseen_exchanges(db, after_id=cursor)
    assert [r["id"] for r in remaining] == ["aa-2", "mm-3"]


def test_unseen_exchanges_ignores_an_unparseable_cursor(tmp_path: Path) -> None:
    """A cursor that is not an integer (e.g. left over from before rowid was the
    cursor) is treated as "start from the beginning", not as a crash."""
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1"), ("ex-2", LONG + "!", "s1")])
    rows = unseen_exchanges(db, after_id="not-an-integer")
    assert [r["id"] for r in rows] == ["ex-1", "ex-2"]


def test_missing_database_yields_nothing(tmp_path: Path) -> None:
    assert unseen_exchanges(tmp_path / "absent.db") == []


def test_corrupt_database_raises_vault_error(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"not a sqlite database, just garbage bytes")
    with pytest.raises(VaultError):
        unseen_exchanges(bad)


def test_reported_sessions_reads_source_refs(api: KnowledgeApi) -> None:
    api.create_note("A Session", "session-report",
                    source={"type": "claude-code-session", "ref": "sess-abc"})
    assert reported_sessions(api) == {"sess-abc"}


def test_scan_queues_survivors_and_logs_rejections(api: KnowledgeApi, tmp_path: Path) -> None:
    db = make_db(tmp_path / "e.db",
                 [("ex-1", LONG, "s1"), ("ex-2", "hi", "s1"), ("ex-3", LONG + " really", "s2")])
    queue = Queue(api.vault)
    stats = scan(api, queue, db_path=db)
    assert stats["queued"] == 2
    assert stats["rejected"] == 1
    assert len(queue.pending()) == 2
    # The cursor is the rowid of the last row read (3rd row inserted), not its
    # content-hash id, see test_unseen_exchanges_orders_by_rowid_not_by_id.
    assert queue.cursor() == "3"


def test_scan_is_resumable_and_does_not_requeue(api: KnowledgeApi, tmp_path: Path) -> None:
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1")])
    queue = Queue(api.vault)
    scan(api, queue, db_path=db)
    second = scan(api, queue, db_path=db)
    assert second["queued"] == 0
    assert len(queue.pending()) == 1


# ---- normalize_project ----



def test_normalize_project_collapses_alias_variants() -> None:
    for raw in ("-home-dev-code-notes-acme", "acme", "-home-dev-code-acme-apps-web"):
        assert normalize_project(raw, RULES) == "acme", raw


def test_normalize_project_maps_a_known_repo() -> None:
    assert normalize_project("-home-dev-code-client-site", RULES) == "client-site"


def test_normalize_project_maps_structa() -> None:
    assert normalize_project("-home-dev-code-structa", RULES) == "structa"


def test_normalize_project_empty_is_unknown() -> None:
    assert normalize_project("", RULES) == "unknown"


def test_normalize_project_unrecognised_is_unchanged() -> None:
    assert normalize_project("some-other-thing", RULES) == "some-other-thing"


def test_normalize_project_longer_repo_wins() -> None:
    """Repos resolve in order, so "local-brain" must precede "brain"."""
    assert normalize_project("-home-dev-code-local-brain", RULES) == "local-brain"


def test_normalize_project_bare_fragment_is_not_guessed() -> None:
    assert normalize_project("web", RULES) == "web"


def test_normalize_project_without_rules_returns_input() -> None:
    """No rules means no mapping. The package ships no names of its own."""
    assert normalize_project("-home-dev-code-acme") == "-home-dev-code-acme"


def test_scan_normalizes_candidate_project_through_the_real_pipeline(
    api: KnowledgeApi, tmp_path: Path
) -> None:
    """Not just the helper in isolation: the full read-filter-queue pipeline
    must hand back a Candidate whose `.project` is already normalized."""
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1")],
                 project="-home-dev-code-structa")
    queue = Queue(api.vault)
    scan(api, queue, db_path=db)
    [candidate] = queue.pending()
    assert candidate.project == "structa"


# ---- project_from_cwd ----
#
# The recorded `project` column is a slugified path and is often a fragment
# rather than a name, so `cwd` is the reliable source.

ORG_ROOT = "/home/dev/code/notes/work/acme"


def test_project_from_cwd_alias_root() -> None:
    assert project_from_cwd(ORG_ROOT, RULES) == "acme"


def test_project_from_cwd_alias_subdirectory() -> None:
    assert project_from_cwd(ORG_ROOT + "/apps/web", RULES) == "acme"
    assert project_from_cwd(ORG_ROOT + "/infra", RULES) == "acme"


def test_project_from_cwd_longer_repo_wins() -> None:
    assert project_from_cwd("/home/dev/code/local-brain", RULES) == "local-brain"


def test_project_from_cwd_maps_a_known_repo() -> None:
    assert project_from_cwd("/home/dev/code/client-site", RULES) == "client-site"


def test_project_from_cwd_empty_is_unknown() -> None:
    assert project_from_cwd("", RULES) == "unknown"
    assert project_from_cwd("   ", RULES) == "unknown"


def test_project_from_cwd_unknown_path_uses_final_segment() -> None:
    assert project_from_cwd("/tmp/scratch-dir", RULES) == "scratch-dir"


def test_project_from_cwd_without_rules_uses_final_segment() -> None:
    assert project_from_cwd("/home/dev/code/acme/infra") == "infra"


def test_scan_derives_candidate_project_from_cwd_over_the_mangled_project_column(
    api: KnowledgeApi, tmp_path: Path
) -> None:
    """The regression this task exists to fix, through the real extraction
    path: a row whose `project` column says "subagents" (not a project at
    all) but whose `cwd` is a real acme path must come out of `scan` as
    "acme", not "subagents"."""
    db = make_db(tmp_path / "e.db", [("ex-1", LONG, "s1")],
                 project="subagents", cwd=ORG_ROOT + "/infra")
    queue = Queue(api.vault)
    scan(api, queue, db_path=db)
    [candidate] = queue.pending()
    assert candidate.project == "acme"


# ---- the crash-retry path must not forge rejections -------------------------


def test_a_crash_before_advance_does_not_forge_duplicate_rejections(
    api: KnowledgeApi, tmp_path: Path
) -> None:
    """append() succeeded, advance() never ran, so the rows get re-read.

    Those rows are already queued, the correct end state. Logging them as
    `duplicate-content` corrupts rejected.jsonl, which is the pipeline's audit
    trail and one of its stated success criteria: after a crash it would claim
    three kept items had been filtered out.
    """
    db = make_db(tmp_path / "e.db",
                 [("ex-1", LONG, "s1"), ("ex-2", LONG + " two", "s1"),
                  ("ex-3", LONG + " three", "s2")])
    queue = Queue(api.vault)

    rows = unseen_exchanges(db, after_id=None, limit=500)
    kept, _ = prefilter(rows, seen=queue.seen_hashes(), reported=set())
    queue.append(kept)          # ... and then the process dies, before advance()
    assert queue.cursor() is None
    rejected_before = len(queue.read_jsonl(queue.rejected_path))

    stats = scan(api, queue, db_path=db)

    assert len(queue.pending()) == 3, "the pending set must be unchanged"
    assert stats["rejected"] == 0, "a re-read is not a rejection"
    assert stats["requeued"] == 3, "re-reads must still be accounted for"
    assert len(queue.read_jsonl(queue.rejected_path)) == rejected_before


def test_real_duplicate_detection_still_works(api: KnowledgeApi, tmp_path: Path) -> None:
    """Fixing the retry path must not disable the dedup paths that do work.

    Two of them could plausibly have broken: duplicates *within* one batch, and
    a candidate that was rejected on an earlier pass. Both must still be caught.
    """
    db = make_db(tmp_path / "e.db",
                 [("ex-1", LONG, "s1"), ("ex-2", LONG, "s1"), ("ex-3", "hi", "s2")])
    queue = Queue(api.vault)
    stats = scan(api, queue, db_path=db)

    # ex-2 repeats ex-1's content; ex-3 is too short.
    assert stats["queued"] == 1
    assert stats["rejected"] == 2
    reasons = [r.get("reason") for r in queue.read_jsonl(queue.rejected_path)]
    assert "duplicate-content" in reasons
    assert "too-short" in reasons

    # Re-reading a previously *rejected* row must reject it again, not queue it.
    queue.advance("0")
    again = scan(api, queue, db_path=db)
    assert again["queued"] == 0, "nothing new should be queued on a rewind"
    assert again["requeued"] == 1, "ex-1 is still pending, so it is a re-read"
