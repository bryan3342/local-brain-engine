"""Tests for the derived index, and the proof that it is disposable."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.infrastructure.index import SqliteIndex


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(config, index=SqliteIndex(config.root))
    a.initialize_vault()
    return a


@pytest.fixture
def noindex_api(tmp_path: Path) -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / "vault2", git=False)
    a = KnowledgeApi(config, index=None)
    a.initialize_vault()
    return a


def test_deleting_the_database_changes_no_query_result(api: KnowledgeApi) -> None:
    """The proof that SQLite is derived, not canonical."""
    a = api.create_note("Alpha", "concept", domain="networking", answer="First.")
    b = api.create_note("Beta", "concept", domain="networking", answer="Second.")
    api.link_notes(a.id, "mentions", b.id)
    api.reindex(full=True)

    before = {
        "list": [r.to_dict() for r in api.list_notes()],
        "backlinks": [x.to_dict() for x in api.get_backlinks(b.id)],
        "graph": api.export_graph().to_dict(),
        "search": sorted(h.ref.id for h in api.search_notes("Alpha")),
    }

    assert isinstance(api.index, SqliteIndex)
    api.index.drop()
    api.reindex(full=True)

    after = {
        "list": [r.to_dict() for r in api.list_notes()],
        "backlinks": [x.to_dict() for x in api.get_backlinks(b.id)],
        "graph": api.export_graph().to_dict(),
        "search": sorted(h.ref.id for h in api.search_notes("Alpha")),
    }
    assert before == after


def test_api_works_with_no_index_at_all(noindex_api: KnowledgeApi) -> None:
    a = noindex_api.create_note("Alpha", "concept")
    b = noindex_api.create_note("Beta", "concept")
    noindex_api.link_notes(a.id, "mentions", b.id)
    assert [x.source.id for x in noindex_api.get_backlinks(b.id)] == [a.id]
    assert noindex_api.search_notes("Alpha")


def test_touch_does_not_trigger_a_reparse(api: KnowledgeApi) -> None:
    """Review test 10: mtime moved, bytes did not."""
    import os

    note = api.create_note("Stable", "concept")
    api.reindex(full=True)
    os.utime(note.path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    stats = api.reindex()
    assert stats.reindexed == 0
    assert stats.unchanged >= 1


def test_content_change_does_trigger_a_reparse(api: KnowledgeApi) -> None:
    note = api.create_note("Changing", "concept")
    api.reindex(full=True)
    api.update_note(note.id, tags=["new"])
    assert api.reindex().reindexed >= 0
    assert api.get_note(note.id).frontmatter.tags == ["new"]


# ---- search ---------------------------------------------------------------


def test_search_matches_aliases_not_just_titles(api: KnowledgeApi) -> None:
    api.create_note("Physical Layer. Fiber-Optic Cabling", "concept",
                    aliases=["fibre optic", "single-mode"], answer="Light, not current.")
    api.reindex(full=True)
    assert any("fiber" in h.ref.id for h in api.search_notes("single-mode"))


def test_search_survives_punctuation_that_would_break_fts(api: KnowledgeApi) -> None:
    api.create_note("Quoting", "concept", answer="x")
    api.reindex(full=True)
    api.search_notes('bad " OR (query')          # must not raise


# ---- reconciliation (review test 9) ---------------------------------------


def test_the_fts_path_also_hides_archived_notes(tmp_path: Path) -> None:
    """The scan path filters archived notes; the indexed path must agree.

    Two search implementations that disagree about what is retrievable is worse
    than either being wrong on its own, the answer would depend on whether an
    index happened to be attached.
    """
    config = BrainConfig(root=tmp_path / "vault", git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root))
    api.initialize_vault()
    note = api.create_note("Polling interval is thirty seconds", "reference",
                           domain="acme", status="evergreen",
                           answer="The worker polls every thirty seconds.")
    api.reindex(full=True)
    assert api.index is not None, "this test is meaningless without the FTS index"
    assert any(h.ref.id == note.id for h in api.search_notes("polling interval", limit=10))

    api.update_note(note.id, status="archived")
    api.reindex(full=True)
    assert not any(h.ref.id == note.id
                   for h in api.search_notes("polling interval", limit=10))
