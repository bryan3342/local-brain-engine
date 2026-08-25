from __future__ import annotations

from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.errors import NoteNotFoundError
from brain.etl.reconcile import DraftNote, Verdict, apply, reconcile


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    a = KnowledgeApi(BrainConfig(root=tmp_path / "vault", git=False), index=None)
    a.initialize_vault()
    return a


def draft(**over: object) -> DraftNote:
    base = dict(title="Deploy is manual", note_type="reference", domain="client-site",
                answer="The backend deploy is manual-dispatch only.",
                aliases=["why no auto deploy"], tags=["deploy"], relationships=[],
                source_ref="sess-1", supersedes=None)
    base.update(over)
    return DraftNote(**base)  # type: ignore[arg-type]


def test_new_material_is_an_add(api: KnowledgeApi) -> None:
    assert reconcile(api, draft()).verdict is Verdict.ADD


def test_an_existing_equivalent_note_is_a_noop(api: KnowledgeApi) -> None:
    """The anti-duplication mechanism: the upstream service re-derived facts 3-5 times.

    The existing note carries draft()'s own alias, so this is a genuine
    re-derivation of material already on file -- not a draft bringing a new
    alias to a bare note, which is the (distinct) update case below.
    """
    api.create_note("Deploy is manual", "reference", domain="client-site",
                    answer="The backend deploy is manual-dispatch only.",
                    aliases=["why no auto deploy"])
    decision = reconcile(api, draft())
    assert decision.verdict is Verdict.NOOP
    assert decision.existing_id == "reference_deploy-is-manual"


def test_new_aliases_on_an_existing_note_is_an_update(api: KnowledgeApi) -> None:
    api.create_note("Deploy is manual", "reference", domain="client-site",
                    answer="The backend deploy is manual-dispatch only.")
    decision = reconcile(api, draft(aliases=["a phrasing not yet recorded"]))
    assert decision.verdict is Verdict.UPDATE


def test_an_explicit_supersedes_is_honoured(api: KnowledgeApi) -> None:
    old = api.create_note("Deploy is automatic", "reference", domain="client-site",
                          answer="Push to main deploys.")
    decision = reconcile(api, draft(supersedes=old.id))
    assert decision.verdict is Verdict.SUPERSEDE
    assert decision.existing_id == old.id


def test_apply_add_creates_in_the_inbox_only(api: KnowledgeApi) -> None:
    result = apply(api, draft(), reconcile(api, draft()))
    note = api.get_note(result["id"])
    assert note.path.parent.name == "00-inbox"
    assert note.frontmatter.status.value == "inbox"
    assert note.frontmatter.source["ref"] == "sess-1"


def test_apply_update_only_adds_aliases_and_leaves_the_body_alone(api: KnowledgeApi) -> None:
    from brain.infrastructure.markdown import split

    existing = api.create_note("Deploy is manual", "reference", domain="client-site",
                               answer="The backend deploy is manual-dispatch only.")
    before = split(existing.path.read_bytes()).body
    d = draft(aliases=["brand new phrasing"])
    apply(api, d, reconcile(api, d))
    after = api.get_note(existing.id)
    assert "brand new phrasing" in after.frontmatter.aliases
    assert split(after.path.read_bytes()).body == before


def test_apply_supersede_records_the_edge_but_does_not_archive_yet(
    api: KnowledgeApi,
) -> None:
    """The ETL proposes; it does not retire a trusted note on its own.

    The replacement lands in 00-inbox as an unreviewed candidate. Archiving the
    original here would let a proposal retire a verified fact, and if the human
    then rejected the candidate, the vault would have lost knowledge with
    nothing to restore it. The `supersedes::` edge records the intent; promote()
    executes it.
    """
    old = api.create_note("Deploy is automatic", "reference", domain="client-site",
                          answer="Push to main deploys.")
    d = draft(supersedes=old.id)
    result = apply(api, d, reconcile(api, d))

    assert api.get_note(old.id).frontmatter.status.value != "archived", (
        "an unreviewed candidate must not retire a live note")
    new = api.get_note(result["id"])
    assert new.frontmatter.status.value == "inbox"
    assert f"supersedes::{old.id}" in [r.encode() for r in new.frontmatter.relationships]


def test_reconcile_refuses_a_supersedes_target_that_does_not_exist(
    api: KnowledgeApi,
) -> None:
    """Silently downgrading to ADD would drop the model's stated intent: the
    contradiction goes unrecorded and the stale note stays live, exactly what
    supersede exists to prevent. The id is model-composed, so typos are likely
    and cheap to correct."""
    with pytest.raises(NoteNotFoundError, match="concept_no-such-note"):
        reconcile(api, draft(supersedes="concept_no-such-note"))


def test_apply_supersede_same_title_is_not_blocked_by_the_duplicate_guard(
    api: KnowledgeApi,
) -> None:
    """allow_duplicate=True is load-bearing exactly here (Finding 2 review):
    replacing a note with another of the identical title would otherwise trip
    create_note's own duplicate guard, which scans by (type, normalized title)."""
    old = api.create_note("Deploy is manual", "reference", domain="client-site",
                          answer="The backend deploy is manual-dispatch only.")
    d = draft(title="Deploy is manual", supersedes=old.id)
    result = apply(api, d, reconcile(api, d))
    new = api.get_note(result["id"])
    assert new.id != old.id
    assert new.frontmatter.title == "Deploy is manual"
    assert f"supersedes::{old.id}" in [r.encode() for r in new.frontmatter.relationships]


def test_apply_noop_writes_nothing(api: KnowledgeApi) -> None:
    api.create_note("Deploy is manual", "reference", domain="client-site",
                    answer="The backend deploy is manual-dispatch only.",
                    aliases=["why no auto deploy"])
    before = len(api.list_notes())
    apply(api, draft(), reconcile(api, draft()))
    assert len(api.list_notes()) == before


def test_alias_collision_with_an_existing_title_is_not_a_fresh_add(api: KnowledgeApi) -> None:
    """Reviewer's finding: `_find_equivalent` searched the drafted title only.

    A draft phrased differently from an existing note, but carrying that
    note's exact title as one of its own aliases, sailed through as ADD --
    two notes for one fact. It must search the drafted title *and* aliases.
    """
    api.create_note("Why the backend deploy is manual", "reference",
                    domain="client-site", answer="Manual dispatch only.")
    d = draft(title="Backend deploy uses manual dispatch",
              aliases=["Why the backend deploy is manual"])
    decision = reconcile(api, d)
    assert decision.verdict is not Verdict.ADD
    apply(api, d, decision)
    assert len(api.list_notes()) == 1


def test_a_superseded_note_stops_competing_in_search(api: KnowledgeApi) -> None:
    """Archived means retired, and retired knowledge must not be retrievable.

    Superseding was unreachable in production until #22, so no vault had
    archived notes and this never bit. Making it reachable means the ETL now
    manufactures them, and serving a fact the vault has explicitly marked wrong
    is worse than serving a duplicate.
    """
    old = api.create_note("Polling interval is thirty seconds", "reference",
                          domain="acme", status="evergreen",
                          answer="The worker polls every thirty seconds.")
    api.reindex(full=True)
    assert any(h.ref.id == old.id for h in api.search_notes("polling interval", limit=10))

    api.update_note(old.id, status="archived")
    api.reindex(full=True)
    assert not any(h.ref.id == old.id for h in api.search_notes("polling interval", limit=10)), (
        "an archived note must not be returned by search")


# ---- enrichment must be attributable and reversible ------------------------


def test_update_records_which_aliases_it_added_and_which_session_justified_them(
    api: KnowledgeApi,
) -> None:
    """The ETL enriches trusted notes unattended. That is allowed, aliases are
    additive and reversible, but only because it is recorded well enough to
    reverse. Without attribution, a precision regression is untraceable and
    permanent.

    The attribution lives in the audit log, not as a `derived_from::` edge on
    the note: the note was NOT derived from that session, only one alias was,
    and asserting otherwise would put a false edge in the graph.
    """
    old = api.create_note("Deploy is manual", "reference", domain="acme",
                          answer="Manual dispatch only.", aliases=["how do we deploy"],
                          status="evergreen")
    d = draft(title="Deploy is manual",
              aliases=["how do we deploy", "is the deploy automatic", "can I push"],
              source_ref="sess-abc")
    result = apply(api, d, reconcile(api, d))

    assert result["aliases_added"] == ["is the deploy automatic", "can I push"], (
        "only genuinely new aliases count as added")

    entry = next(e for e in api.audit.tail(50) if e.get("op") == "etl_enrich")
    assert entry["id"] == old.id
    assert entry["session"] == "sess-abc"
    assert entry["aliases_added"] == ["is the deploy automatic", "can I push"]

    assert not any("derived_from::sess" in r.encode()
                   for r in api.get_note(old.id).frontmatter.relationships), (
        "the note was not derived from that session; only an alias was")


def test_enrichment_of_a_trusted_note_is_flagged_as_such(api: KnowledgeApi) -> None:
    """Enriching an unreviewed inbox candidate is a proposal editing a proposal.
    Enriching an evergreen note is a proposal editing verified knowledge. The
    audit entry must distinguish them, or the log cannot answer the only
    question that matters when precision drops: what did this touch?"""
    ever = api.create_note("Trusted thing", "reference", domain="acme",
                           answer="x", status="evergreen")
    d = draft(title="Trusted thing", aliases=["new phrasing"], source_ref="s1")
    apply(api, d, reconcile(api, d))
    entry = next(e for e in api.audit.tail(50) if e.get("op") == "etl_enrich")
    assert entry["trusted"] is True

    cand = api.create_note("Pending thing", "reference", domain="acme", answer="x",
                           folder="00-inbox", status="inbox")
    d2 = draft(title="Pending thing", aliases=["another phrasing"], source_ref="s2")
    apply(api, d2, reconcile(api, d2))
    entry2 = next(e for e in api.audit.tail(50)
                  if e.get("op") == "etl_enrich" and e.get("id") == cand.id)
    assert entry2["trusted"] is False
    assert ever.id != cand.id
