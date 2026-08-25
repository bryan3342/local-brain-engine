from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from brain.domain.errors import ConcurrentModificationError
from brain.domain.models import Candidate, Rejection
from brain.etl.queue import Queue
from brain.infrastructure.vault import Vault, append_jsonl


def make(cid: str, text: str = "a reasonably long user message here") -> Candidate:
    return Candidate(cid, "sess", "proj", "2026-08-20T00:00:00Z", text, "reply", {}, ())


@pytest.fixture
def queue(tmp_path: Path) -> Queue:
    v = Vault(tmp_path / "vault")
    v.initialize(git=False)
    return Queue(v)


def test_append_then_pending_returns_what_went_in(queue: Queue) -> None:
    queue.append([make("a"), make("b")])
    assert [c.candidate_id for c in queue.pending()] == ["a", "b"]


def test_pending_respects_a_limit(queue: Queue) -> None:
    queue.append([make("a"), make("b"), make("c")])
    assert len(queue.pending(limit=2)) == 2


def test_drop_removes_only_the_named_candidates(queue: Queue) -> None:
    queue.append([make("a"), make("b"), make("c")])
    assert queue.drop(["b"]) == 1
    assert [c.candidate_id for c in queue.pending()] == ["a", "c"]


def test_drop_raises_rather_than_clobbering_a_concurrent_append(
    queue: Queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate appended between drop()'s read and its write must survive.

    Simulates the race by having read_jsonl(), drop()'s only read of the
    file, land a concurrent append() immediately after it captures the rows
    drop() will act on, but before drop() gets to rewrite the file. The stamp
    drop() took up front no longer matches, so atomic_write must raise rather
    than silently overwrite the concurrent write with a stale snapshot.
    """
    queue.append([make("a"), make("b")])
    real_read_jsonl = Queue.read_jsonl

    def racing_read(path: Path) -> list[dict[str, Any]]:
        rows = real_read_jsonl(path)
        if path == queue.candidates_path:
            append_jsonl(path, json.dumps(make("c").to_dict(), ensure_ascii=False))
        return rows

    monkeypatch.setattr(Queue, "read_jsonl", staticmethod(racing_read))
    with pytest.raises(ConcurrentModificationError):
        queue.drop(["a"])
    monkeypatch.undo()

    # Nothing was lost: the concurrent append is still there, and "a" was not
    # dropped either, since the rewrite never happened.
    assert {c.candidate_id for c in queue.pending()} == {"a", "b", "c"}


def test_append_is_idempotent_on_candidate_id(queue: Queue) -> None:
    queue.append([make("a")])
    queue.append([make("a")])
    assert len(queue.pending()) == 1


def test_seen_hashes_includes_rejected_ones(queue: Queue) -> None:
    """A candidate rejected once must not be re-queued on the next scan."""
    queue.append([make("kept")])
    queue.reject([Rejection("dropped", "prefilter", "too-short")])
    assert queue.seen_hashes() == {"kept", "dropped"}


def test_append_does_not_readd_a_rejected_candidate(queue: Queue) -> None:
    """append() must dedup against rejected.jsonl too, not just pending()."""
    queue.reject([Rejection("dropped", "prefilter", "too-short")])
    assert queue.append([make("dropped")]) == 0
    assert queue.pending() == []


def test_cursor_starts_empty_and_advances(queue: Queue) -> None:
    assert queue.cursor() is None
    queue.advance("exchange-42")
    assert queue.cursor() == "exchange-42"


def test_a_corrupt_line_does_not_break_the_queue(queue: Queue) -> None:
    """One bad line must not make the whole queue unreadable."""
    queue.append([make("a")])
    with queue.candidates_path.open("a") as fh:
        fh.write("not json\n")
    queue.append([make("b")])
    assert {c.candidate_id for c in queue.pending()} == {"a", "b"}


def test_a_corrupt_line_logs_a_warning(
    queue: Queue, caplog: pytest.LogCaptureFixture
) -> None:
    """Corruption must be visible, not just survivable."""
    queue.append([make("a")])
    with queue.candidates_path.open("a") as fh:
        fh.write("not json\n")

    with caplog.at_level(logging.WARNING, logger="brain.etl.queue"):
        rows = queue.pending()

    assert {c.candidate_id for c in rows} == {"a"}
    assert str(queue.candidates_path) in caplog.text
    assert "line 2" in caplog.text


def test_a_malformed_line_is_skipped_rather_than_bricking_the_queue(
    queue: Queue, caplog: pytest.LogCaptureFixture
) -> None:
    """One bad line must not take the whole queue down.

    The skill hand-edits candidates.jsonl, so a line that parses as JSON but is
    not a valid Candidate is the expected failure, not an exotic one. The spec:
    "A malformed candidate is skipped and logged; the queue keeps draining."
    """
    queue.append([make("good")])
    append_jsonl(queue.candidates_path, json.dumps({"candidate_id": "bad", "session_id": "s"}))

    with caplog.at_level(logging.WARNING):
        got = queue.pending()

    assert [c.candidate_id for c in got] == ["good"]
    assert "bad" in caplog.text


def test_pending_ids_survives_a_malformed_line(queue: Queue) -> None:
    """pending_ids() must not validate, it exists to be safe on a broken file."""
    queue.append([make("good")])
    append_jsonl(queue.candidates_path, json.dumps({"candidate_id": "bad", "session_id": "s"}))
    assert queue.pending_ids() == {"good", "bad"}
