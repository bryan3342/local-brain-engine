# tests/test_etl_inbox.py
from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.etl.inbox import list_inbox, promote, reject
from brain.etl.queue import Queue
from brain.etl.reconcile import SupersedeArchiveError


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    a = KnowledgeApi(BrainConfig(root=tmp_path / "vault", git=False), index=None)
    a.initialize_vault()
    a.create_note("Deploy is manual", "reference", domain="client-site",
                  status="inbox", answer="Manual dispatch only.", folder="00-inbox")
    return a


def test_list_shows_the_answer_first(api: KnowledgeApi) -> None:
    """Ordered to make 'no' the cheap answer."""
    rows = list_inbox(api)
    assert len(rows) == 1
    assert list(rows[0])[:2] == ["answer", "id"]
    assert rows[0]["answer"] == "Manual dispatch only."


def test_list_ignores_notes_outside_the_inbox(api: KnowledgeApi) -> None:
    api.create_note("Elsewhere", "concept", status="evergreen")
    assert [r["id"] for r in list_inbox(api)] == ["reference_deploy-is-manual"]


def test_promote_moves_the_note_and_marks_it_evergreen(api: KnowledgeApi) -> None:
    result = promote(api, "reference_deploy-is-manual", to="30-work")
    note = api.get_note(result["id"])
    assert note.frontmatter.status.value == "evergreen"
    assert note.path.parent.name == "30-work"


def test_promote_preserves_the_id_and_the_body(api: KnowledgeApi) -> None:
    from brain.infrastructure.markdown import split

    before = split(api.get_note("reference_deploy-is-manual").path.read_bytes()).body
    promote(api, "reference_deploy-is-manual")
    after = api.get_note("reference_deploy-is-manual")
    assert after.id == "reference_deploy-is-manual"
    assert split(after.path.read_bytes()).body == before


def test_promote_leaves_the_note_visible_if_the_flip_fails_after_the_move(
    api: KnowledgeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Move-then-flip order: a crash between the two steps must not strand the
    note invisibly inside 00-inbox/. It must still surface in list_inbox."""

    def raise_after_move(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated failure after the move")

    monkeypatch.setattr(api, "update_note", raise_after_move)

    with pytest.raises(RuntimeError):
        promote(api, "reference_deploy-is-manual", to="30-work")

    moved = api.get_note("reference_deploy-is-manual")
    assert moved.path.parent.name == "30-work"
    assert moved.frontmatter.status.value == "inbox"
    assert [r["id"] for r in list_inbox(api)] == ["reference_deploy-is-manual"]


def test_reject_trashes_the_note_and_records_the_reason(api: KnowledgeApi) -> None:
    queue = Queue(api.vault)
    reject(api, queue, "reference_deploy-is-manual", "already covered elsewhere")
    assert [r["id"] for r in list_inbox(api)] == []
    logged = Queue.read_jsonl(queue.rejected_path)
    assert logged[-1]["reason"] == "already covered elsewhere"
    assert logged[-1]["stage"] == "promotion"


# ---- promotion is where a supersede is actually carried out -----------------


def test_promote_archives_what_the_candidate_supersedes(
    api: KnowledgeApi
) -> None:
    """The human gate is where a trusted note is retired, not before.

    apply() only records `supersedes::<id>`. Promotion is the human action that
    says "yes, this replaces that", so it is promotion that archives the
    original.
    """
    old = api.create_note("Deploy is automatic", "reference", domain="client-site",
                          answer="Push to main deploys.", status="evergreen")
    new = api.create_note("Deploy is manual now", "reference", domain="client-site",
                          answer="It is manual dispatch only.", folder="00-inbox",
                          status="inbox", relationships=[f"supersedes::{old.id}"])

    assert api.get_note(old.id).frontmatter.status.value == "evergreen"
    promote(api, new.id)

    assert api.get_note(old.id).frontmatter.status.value == "archived"
    assert api.get_note(new.id).frontmatter.status.value == "evergreen"


def test_promote_is_loud_if_archiving_the_superseded_note_fails(
    api: KnowledgeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-done supersede must never pass silently: both notes would be live
    and evergreen, asserting contradictory things with no signal."""
    old = api.create_note("Deploy is automatic", "reference", domain="client-site",
                          answer="Push to main deploys.", status="evergreen")
    new = api.create_note("Deploy is manual now", "reference", domain="client-site",
                          answer="It is manual dispatch only.", folder="00-inbox",
                          status="inbox", relationships=[f"supersedes::{old.id}"])

    real = api.update_note

    def boom(ref: str, **kwargs: object) -> object:
        if ref == old.id:
            raise RuntimeError("disk full")
        return real(ref, **kwargs)

    monkeypatch.setattr(api, "update_note", boom)

    with pytest.raises(SupersedeArchiveError) as exc:
        promote(api, new.id)
    assert old.id in str(exc.value) and new.id in str(exc.value)


def test_promote_without_a_supersedes_edge_is_unaffected(
    api: KnowledgeApi
) -> None:
    n = api.create_note("Plain candidate", "concept", domain="meta",
                        answer="nothing special", folder="00-inbox", status="inbox")
    assert promote(api, n.id)["status"] == "evergreen"
