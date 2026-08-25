"""The invariant that makes unattended operation safe."""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.etl.reconcile import DraftNote, apply, reconcile


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    a = KnowledgeApi(BrainConfig(root=tmp_path / "vault", git=False), index=None)
    a.initialize_vault()
    return a


def test_the_etl_cannot_write_outside_the_inbox(api: KnowledgeApi) -> None:
    for i in range(5):
        d = DraftNote(title=f"Candidate {i}", note_type="concept", domain="networking",
                      answer="Something.", source_ref="sess-1")
        apply(api, d, reconcile(api, d))
    written = [n for n in api.iter_notes()]
    assert written, "expected the ETL to write something"
    for note in written:
        assert note.path.parent.name == "00-inbox", f"{note.id} escaped the inbox"
        assert note.frontmatter.status.value == "inbox"


def test_every_etl_note_carries_provenance(api: KnowledgeApi) -> None:
    d = DraftNote(title="Traceable", note_type="concept", domain="networking",
                  answer="x", source_ref="sess-42")
    apply(api, d, reconcile(api, d))
    note = api.get_note("concept_traceable")
    assert note.frontmatter.source["ref"] == "sess-42"
    assert note.frontmatter.source["connector"] == "brain-etl"
