"""Two-writer reconciliation: scan reports, doctor repairs."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.application.reconcile import Reconciler
from brain.config import BrainConfig
from brain.infrastructure.markdown import split

HAND_BODY = b"""# Hand written

Prose with a fenced block:

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


def test_scan_flags_a_hand_written_file_and_doctor_adopts_it(api: KnowledgeApi) -> None:
    rogue = api.vault.root / "10-knowledge" / "rogue.md"
    rogue.write_bytes(HAND_BODY)

    report = Reconciler(api).scan()
    kinds = {f.kind for f in report.findings}
    assert "no-frontmatter" in kinds

    dry = Reconciler(api).doctor(dry_run=True)
    assert any(f.kind == "no-frontmatter" for f in dry.repaired)
    assert split(rogue.read_bytes()).frontmatter_text is None, "dry run must not write"

    Reconciler(api).doctor(dry_run=False)
    doc = split(rogue.read_bytes())
    assert doc.frontmatter_text is not None
    assert doc.body == HAND_BODY, "adoption must not touch the body"
    assert api.get_note("Hand written").frontmatter.type.value == "concept"


def test_scan_reports_duplicate_ids_without_merging(api: KnowledgeApi) -> None:
    note = api.create_note("Original", "concept")
    clone = api.vault.root / "10-knowledge" / "clone.md"
    clone.write_bytes(note.path.read_bytes())

    findings = [f for f in Reconciler(api).scan().findings if f.kind == "duplicate-id"]
    assert findings and not findings[0].repairable

    applied = Reconciler(api).doctor(dry_run=False)
    assert any(f.kind == "duplicate-id" for f in applied.skipped)
    assert clone.exists(), "doctor must never delete a duplicate on its own"


def test_scan_reports_dangling_edges(api: KnowledgeApi) -> None:
    a = api.create_note("A", "concept")
    api.link_notes(a.id, "mentions", "concept_ghost")
    assert any(f.kind == "dangling-edge" for f in Reconciler(api).scan().findings)


def test_scan_tolerates_unparseable_frontmatter(api: KnowledgeApi) -> None:
    bad = api.vault.root / "10-knowledge" / "bad.md"
    bad.write_bytes(b"---\nid: [unclosed\n---\n# body\n")
    report = Reconciler(api).scan()               # must not raise
    assert any(f.kind in {"unparseable-frontmatter", "invalid-frontmatter"}
               for f in report.findings)


# ---- audit ----------------------------------------------------------------


def test_scan_hides_advisory_findings_by_default(api: KnowledgeApi) -> None:
    api.create_note("Elsewhere", "concept", folder="40-research")
    assert not any(f.kind == "type-folder-mismatch" for f in Reconciler(api).scan().findings)
    assert any(f.kind == "type-folder-mismatch"
               for f in Reconciler(api).scan(include_advisory=True).findings)
