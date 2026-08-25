# tests/test_etl_e2e.py
"""The whole pipeline on a synthetic vault, plus the regression gate."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain.application.evaluate import evaluate, load_cases
from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.etl.extract import scan
from brain.etl.inbox import list_inbox, promote
from brain.etl.queue import Queue
from brain.etl.reconcile import DraftNote, apply, reconcile
from brain.infrastructure.index import SqliteIndex

LONG = "why does the backend deploy not fire automatically when we push to main?"


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    cfg = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(cfg, index=SqliteIndex(cfg.root))
    a.initialize_vault()
    return a


def seed_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE exchanges (id TEXT PRIMARY KEY, project TEXT, timestamp TEXT,"
        " user_message TEXT, assistant_message TEXT, session_id TEXT,"
        " git_branch TEXT, is_sidechain BOOLEAN DEFAULT 0, cwd TEXT);")
    # `cwd` is the real source of project identity, see project_from_cwd. The
    # column must exist here or this fixture stops resembling the live schema.
    con.executemany(
        "INSERT INTO exchanges VALUES (?,?,?,?,?,?,?,0,?)",
        [("ex-1", "p", "2026-08-20T00:00:00Z", LONG, "manual dispatch", "s1", "main",
          "/Users/x/Developer/structa"),
         ("ex-2", "p", "2026-08-20T00:01:00Z", "ok", "sure", "s1", "main",
          "/Users/x/Developer/structa"),
         ("ex-3", "p", "2026-08-20T00:02:00Z", "<bash-stdout>noise here now</bash-stdout>",
          "x", "s1", "main", "/Users/x/Developer/structa")])
    con.commit()
    con.close()
    return path


def test_full_pipeline_scan_draft_promote(api: KnowledgeApi, tmp_path: Path) -> None:
    queue = Queue(api.vault)
    stats = scan(api, queue, db_path=seed_db(tmp_path / "e.db"))
    assert stats["queued"] == 1 and stats["rejected"] == 2

    candidate = queue.pending()[0]
    d = DraftNote(title="Backend deploy is manual", note_type="reference",
                  domain="client-site", answer="Manual dispatch only; push does not deploy.",
                  aliases=["why does the deploy not fire on push"],
                  source_ref=candidate.session_id)
    apply(api, d, reconcile(api, d))
    queue.drop([candidate.candidate_id])

    assert [r["id"] for r in list_inbox(api)] == ["reference_backend-deploy-is-manual"]
    assert queue.pending() == []

    promote(api, "reference_backend-deploy-is-manual", to="30-work")
    assert list_inbox(api) == []
    note = api.get_note("reference_backend-deploy-is-manual")
    assert note.frontmatter.status.value == "evergreen"


def test_a_batch_must_not_lower_precision_at_1(api: KnowledgeApi, tmp_path: Path) -> None:
    """The real acceptance criterion from the spec.

    Adding notes must not make existing questions harder to answer.

    The batch below is deliberately not lexically inert. A batch of drafts
    that shares no vocabulary with the golden query could never lower its
    precision@1: FTS5's MATCH excludes a non-matching row from the result set
    entirely rather than merely ranking it lower, so such a batch would leave
    the golden note as the sole (and therefore rank-1) hit no matter how badly
    ranking was broken. A gate built only from such filler would pass
    unconditionally and would be worthless.

    So three of the ten drafts here mention "wifi" or "slower" in a context
    that has nothing to do with the golden question (a guest-network
    password, a laptop fan, a slow cooker), genuine BM25 competitors, not
    just extra rows. This is also the realistic case: real mined transcripts
    share incidental vocabulary with existing notes without being duplicates
    of them. See the report for the mutation test that confirms this
    assertion actually fires when ranking regresses.
    """
    api.create_note("Wireless Media", "concept", domain="networking",
                    aliases=["why does wifi get slower"], answer="Half-duplex shared medium.")
    api.reindex(full=True)
    golden = api.vault.root / "90-meta/eval/golden.jsonl"
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text('{"q": "why does wifi get slower", "expect": ["concept_wireless-media"]}\n')
    before = evaluate(api, load_cases(golden)).precision_at_1
    assert before == 1.0, "fixture is broken: the golden note must be findable before the batch"

    competitors = [
        ("Guest Wifi Password", "It is on the sticker under the office router."),
        ("Laptop Fan Getting Slower", "Compressed air cleared the dust; nothing to do with wifi."),
        ("Why The Crockpot Cooks Slower On Low", "Lower wattage; nothing to do with networking."),
    ]
    for i, (title, answer) in enumerate(competitors):
        d = DraftNote(title=title, note_type="reference", domain="client-site",
                      answer=answer, source_ref=f"competitor-{i}")
        apply(api, d, reconcile(api, d))
    for i in range(len(competitors), 10):
        d = DraftNote(title=f"Unrelated Candidate {i}", note_type="reference",
                      domain="client-site", answer="Something unrelated entirely.",
                      source_ref="s1")
        apply(api, d, reconcile(api, d))
    api.reindex(full=True)

    after = evaluate(api, load_cases(golden)).precision_at_1
    assert after >= before, f"precision@1 regressed: {before:.1%} -> {after:.1%}"
