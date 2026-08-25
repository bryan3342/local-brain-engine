"""End-to-end Knowledge API tests.

Weighted toward the review's verification list: body preservation and id
stability across rename.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.errors import (
    ConcurrentModificationError,
    DuplicateNoteError,
    NoteNotFoundError,
    ValidationError,
)
from brain.infrastructure.markdown import split

HAND_BODY = b"""# Hand written

Prose with a fenced block that contains a horizontal rule:

```yaml
---
id: decoy
---
```

Tail.
"""


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(config, index=None)
    a.initialize_vault()
    return a


def test_create_mints_prefixed_id_and_is_readable(api: KnowledgeApi) -> None:
    note = api.create_note("OSI Model", "concept", domain="networking",
                           answer="Seven layers.")
    assert note.id == "concept_osi-model"
    assert note.answer == "Seven layers."
    assert api.get_note("concept_osi-model").id == note.id
    assert api.get_note("OSI Model").id == note.id          # by title
    assert note.path.exists()


def test_create_refuses_same_type_same_title(api: KnowledgeApi) -> None:
    api.create_note("GraphRAG", "concept")
    with pytest.raises(DuplicateNoteError):
        api.create_note("Graph RAG", "concept")            # normalizes together


def test_dedupe_is_scoped_by_type(api: KnowledgeApi) -> None:
    api.create_note("Python", "concept")
    api.create_note("Python", "project")                   # a codename, not a duplicate
    assert len(api.list_notes()) == 2


def test_duplicate_can_be_forced(api: KnowledgeApi) -> None:
    api.create_note("Note", "concept")
    second = api.create_note("Note", "concept", allow_duplicate=True)
    assert second.id == "concept_note_2"                   # smallest free suffix


def test_create_rejects_empty_title(api: KnowledgeApi) -> None:
    with pytest.raises(ValidationError):
        api.create_note("   ", "concept")


def test_create_rejects_malformed_relationship(api: KnowledgeApi) -> None:
    with pytest.raises(ValidationError):
        api.create_note("X", "concept", relationships=["not-an-edge"])


def test_get_unknown_raises(api: KnowledgeApi) -> None:
    with pytest.raises(NoteNotFoundError):
        api.get_note("nope_nothing")


# ---- update: the body must never move -------------------------------------


def test_update_preserves_body_bytes_exactly(api: KnowledgeApi) -> None:
    """Review test 3, and the reason `splice` exists."""
    note = api.create_note("Fenced", "concept", body=HAND_BODY)
    before = split(note.path.read_bytes()).body

    api.update_note(note.id, title="Fenced Renamed", tags=["a", "b"], status="evergreen")

    after = split(note.path.read_bytes()).body
    assert after == before, "body changed during a frontmatter-only update"
    assert b"id: decoy" in after


def test_update_cannot_change_id_or_created(api: KnowledgeApi) -> None:
    note = api.create_note("Frozen", "concept")
    created = note.frontmatter.created
    updated = api.update_note(note.id, title="Totally Different Title")
    assert updated.id == "concept_frozen"                   # id did NOT follow the title
    assert updated.frontmatter.created == created


def test_update_detects_a_concurrent_writer(api: KnowledgeApi) -> None:
    note = api.create_note("Contested", "concept")
    stale = "0" * 64
    with pytest.raises(ConcurrentModificationError):
        api.update_note(note.id, title="X", expect_hash=stale)


def test_update_preserves_unmanaged_frontmatter_keys(api: KnowledgeApi) -> None:
    note = api.create_note("Extra", "concept")
    raw = note.path.read_bytes().replace(b"status:", b'module: "4 - Physical Layer"\nstatus:')
    note.path.write_bytes(raw)
    api.update_note(note.id, tags=["x"])
    assert b'module: "4 - Physical Layer"' in note.path.read_bytes()


# ---- rename: id stability (review test 7) ---------------------------------


def test_rename_keeps_id_and_backlinks_and_aliases_the_old_title(api: KnowledgeApi) -> None:
    target = api.create_note("GraphRAG", "concept")
    source = api.create_note("Project Brain", "project")
    api.link_notes(source.id, "mentions", target.id)

    renamed = api.rename_note(target.id, "Microsoft GraphRAG")

    assert renamed.id == target.id, "id must survive a rename"
    assert "GraphRAG" in renamed.frontmatter.aliases
    assert api.get_note("GraphRAG").id == target.id         # old name still resolves
    backlinks = api.get_backlinks(target.id)
    assert [b.source.id for b in backlinks] == [source.id]
    assert not target.path.exists()                         # file did move


def test_rename_can_move_folders(api: KnowledgeApi) -> None:
    note = api.create_note("Movable", "concept")
    moved = api.rename_note(note.id, "Movable", dest_folder="40-research")
    assert moved.path.parent.name == "40-research"
    assert moved.id == note.id


# ---- delete ---------------------------------------------------------------


def test_soft_delete_moves_to_trash_and_hides_from_listings(api: KnowledgeApi) -> None:
    note = api.create_note("Doomed", "concept")
    result = api.delete_note(note.id)
    assert result["hard"] is False
    assert not note.path.exists()
    assert (api.vault.trash / "doomed.md").exists()
    assert [r.id for r in api.list_notes()] == []


def test_deleted_note_becomes_an_unresolved_edge_not_a_ghost_node(api: KnowledgeApi) -> None:
    target = api.create_note("Target", "concept")
    source = api.create_note("Source", "concept")
    api.link_notes(source.id, "mentions", target.id)
    api.delete_note(target.id)

    graph = api.export_graph()
    assert target.id not in [n.id for n in graph.nodes]
    assert graph.unresolved and graph.unresolved[0]["target"] == target.id


# ---- relationships and graph ----------------------------------------------


def test_link_is_idempotent_and_unlink_removes(api: KnowledgeApi) -> None:
    a = api.create_note("A", "concept")
    b = api.create_note("B", "concept")
    api.link_notes(a.id, "mentions", b.id)
    api.link_notes(a.id, "mentions", b.id)
    assert [r.encode() for r in api.get_note(a.id).frontmatter.relationships] == \
        [f"mentions::{b.id}"]
    api.unlink_notes(a.id, "mentions", b.id)
    assert api.get_note(a.id).frontmatter.relationships == []


def test_link_allows_forward_reference(api: KnowledgeApi) -> None:
    """Validating targets on write would make linking O(vault) (H3)."""
    a = api.create_note("A", "concept")
    api.link_notes(a.id, "mentions", "concept_does-not-exist-yet")
    graph = api.export_graph()
    assert graph.unresolved[0]["target"] == "concept_does-not-exist-yet"


def test_neighbors_reports_both_directions(api: KnowledgeApi) -> None:
    a = api.create_note("A", "concept")
    b = api.create_note("B", "concept")
    api.link_notes(a.id, "part_of", b.id)
    assert api.get_neighbors(b.id)["inbound"][0]["source"]["id"] == a.id
    assert api.get_neighbors(a.id)["outbound"][0]["target"] == b.id


# ---- index disposability (review test 6) ----------------------------------


def test_every_mutation_is_audited(api: KnowledgeApi) -> None:
    note = api.create_note("Audited", "concept")
    api.update_note(note.id, tags=["t"])
    api.rename_note(note.id, "Audited Twice")
    api.delete_note(note.id)
    ops = [entry["op"] for entry in api.audit.tail(20)]
    assert [
        op for op in ops if op != "initialize_vault"
    ][-4:] == ["create_note", "update_note", "rename_note", "delete_note"]


def test_anchors_become_nodes_not_dangling_edges(api: KnowledgeApi) -> None:
    """A file or PR a session touched is a real graph node, not an error."""
    s = api.create_note("Session", "session-report",
                        relationships=["touched::file_src-app-ts", "shipped::pr_repo-42"])
    graph = api.export_graph()
    kinds = {n.type for n in graph.nodes}
    assert "anchor:file" in kinds and "anchor:pr" in kinds
    assert graph.unresolved == []
    assert len([e for e in graph.edges if e["source"] == s.id]) == 2


def test_edges_resolve_through_aliases(api: KnowledgeApi) -> None:
    """Regression: bridged notes referenced sessions by uuid, not by note id.

    78 provenance edges dangled until edge targets resolved the same way
    `get_note` does, id first, then alias.
    """
    session = api.create_note("Some Session", "session-report",
                              aliases=["session_abc-123"])
    memory = api.create_note("A Memory", "reference",
                             relationships=["derived_from::session_abc-123"])

    graph = api.export_graph()
    assert graph.unresolved == []
    edge = next(e for e in graph.edges if e["source"] == memory.id)
    assert edge["target"] == session.id, "alias target should resolve to the note id"
