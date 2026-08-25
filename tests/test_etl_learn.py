# tests/test_etl_learn.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.application.evaluate import load_cases
from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.models import EvalCase
from brain.etl.learn import coverage, teach
from brain.infrastructure.index import SqliteIndex


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    cfg = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(cfg, index=SqliteIndex(cfg.root))
    a.initialize_vault()
    a.create_note("Wireless Media", "concept", domain="networking",
                  answer="Radio and microwave frequencies.")
    a.reindex(full=True)
    return a


def golden(api: KnowledgeApi) -> Path:
    p = api.vault.root / "90-meta/eval/golden.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"q": "existing", "expect": ["concept_wireless-media"]}\n')
    return p


def test_teach_appends_a_golden_case(api: KnowledgeApi) -> None:
    path = golden(api)
    teach(api, "why does wifi get slower", "concept_wireless-media", golden_path=path)
    cases = load_cases(path)
    assert any(c.query == "why does wifi get slower" for c in cases)


def test_teach_adds_the_query_as_an_alias(api: KnowledgeApi) -> None:
    teach(api, "why does wifi get slower", "concept_wireless-media",
          golden_path=golden(api))
    assert "why does wifi get slower" in api.get_note("concept_wireless-media").frontmatter.aliases


def test_teach_leaves_the_body_byte_identical(api: KnowledgeApi) -> None:
    from brain.infrastructure.markdown import split

    before = split(api.get_note("concept_wireless-media").path.read_bytes()).body
    teach(api, "a new phrasing entirely", "concept_wireless-media", golden_path=golden(api))
    after = split(api.get_note("concept_wireless-media").path.read_bytes()).body
    assert after == before


def test_teach_is_idempotent(api: KnowledgeApi) -> None:
    path = golden(api)
    teach(api, "same question", "concept_wireless-media", golden_path=path)
    teach(api, "same question", "concept_wireless-media", golden_path=path)
    assert sum(1 for c in load_cases(path) if c.query == "same question") == 1


def test_teach_rejects_an_unknown_note(api: KnowledgeApi) -> None:
    from brain.domain.errors import NoteNotFoundError

    with pytest.raises(NoteNotFoundError):
        teach(api, "q", "concept_does-not-exist", golden_path=golden(api))


def test_teach_reports_the_resulting_metrics(api: KnowledgeApi) -> None:
    result = teach(api, "why does wifi get slower", "concept_wireless-media",
                   golden_path=golden(api))
    assert "precision@1" in result and "cases" in result


def test_coverage_reports_notes_no_query_reaches(api: KnowledgeApi) -> None:
    api.create_note("Never Asked About", "concept", domain="networking", answer="x")
    api.reindex(full=True)
    cases = [EvalCase(query="Wireless Media", expect=("concept_wireless-media",))]
    assert "concept_never-asked-about" in coverage(api, cases)


def test_cli_coverage_json_reports_stale_expectations(
    api: KnowledgeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    """`eval --coverage` must not discard the stale-expectations it already computed:

    a golden case pointing at a deleted or renamed note is a defect in the golden
    set, not in retrieval, and coverage mode is exactly when someone is auditing
    the golden set and most needs to be told.
    """
    from brain.cli import main

    path = golden(api)
    with path.open("a") as fh:
        fh.write(json.dumps({"q": "gone", "expect": ["concept_does-not-exist"]}) + "\n")

    assert main(["--vault", str(api.vault.root), "eval", "--coverage", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "concept_does-not-exist" in payload["stale_expectations"]


def test_coverage_honours_the_k_depth(api: KnowledgeApi) -> None:
    """`brain eval --coverage -k 10` must actually search 10 deep.

    coverage() hardcoded limit=5, so -k was silently ignored on the coverage
    path while the plain eval path honoured it, two answers from one flag.
    """
    from brain.etl.learn import coverage

    seen: list[int] = []
    real = api.search_notes

    def spy(query: str, limit: int = 5) -> object:
        seen.append(limit)
        return real(query, limit=limit)

    api.search_notes = spy  # type: ignore[method-assign]
    coverage(api, [EvalCase(query="anything at all", expect=("concept_x",))], limit=10)
    assert seen and set(seen) == {10}


def test_coverage_ignores_inbox_candidates(api: KnowledgeApi) -> None:
    """An unpromoted inbox note is not 'unreached knowledge', it is a proposal.

    Reporting it as unreached buries the real signal (evergreen notes no query
    finds) under every candidate awaiting triage.
    """
    from brain.etl.learn import coverage

    api.create_note("A pending candidate", "concept", domain="meta",
                    answer="not promoted yet", folder="00-inbox", status="inbox")
    api.reindex(full=True)
    unreached = coverage(api, [EvalCase(query="something unrelated entirely", expect=("x",))])
    assert not any("pending-candidate" in note_id for note_id in unreached)


# ---- reverting a batch that hurt retrieval ---------------------------------


def test_revert_session_removes_only_that_sessions_aliases(api: KnowledgeApi) -> None:
    """Undo must be surgical. A batch that lowered precision has to come out
    without taking authored aliases or another batch's work with it."""
    from brain.etl.learn import revert_session
    from brain.etl.reconcile import DraftNote, apply, reconcile

    note = api.create_note("Deploy is manual", "reference", domain="acme",
                           answer="Manual dispatch only.",
                           aliases=["how do we deploy"], status="evergreen")

    def enrich(session: str, aliases: list[str]) -> None:
        d = DraftNote(title="Deploy is manual", note_type="reference", domain="acme",
                      answer="Manual dispatch only.", aliases=aliases,
                      source_ref=session)
        apply(api, d, reconcile(api, d))

    enrich("sess-good", ["can I push to deploy"])
    enrich("sess-bad", ["deployment", "release"])
    assert api.get_note(note.id).frontmatter.aliases == [
        "how do we deploy", "can I push to deploy", "deployment", "release"]

    result = revert_session(api, "sess-bad")

    assert result["notes_touched"] == 1
    assert result["aliases_removed"] == 2
    assert api.get_note(note.id).frontmatter.aliases == [
        "how do we deploy", "can I push to deploy"], (
        "the authored alias and the good batch must both survive")


def test_revert_session_is_idempotent(api: KnowledgeApi) -> None:
    """Running it twice must not remove more than it should, reverting an
    already-reverted batch is a no-op, not a second bite."""
    from brain.etl.learn import revert_session
    from brain.etl.reconcile import DraftNote, apply, reconcile

    note = api.create_note("A note", "reference", domain="acme", answer="x",
                           aliases=["original"], status="evergreen")
    d = DraftNote(title="A note", note_type="reference", domain="acme", answer="x",
                  aliases=["added"], source_ref="s1")
    apply(api, d, reconcile(api, d))

    assert revert_session(api, "s1")["aliases_removed"] == 1
    assert revert_session(api, "s1")["aliases_removed"] == 0
    assert api.get_note(note.id).frontmatter.aliases == ["original"]


def test_revert_an_unknown_session_changes_nothing(api: KnowledgeApi) -> None:
    from brain.etl.learn import revert_session

    api.create_note("A note", "reference", domain="acme", answer="x",
                    aliases=["original"], status="evergreen")
    assert revert_session(api, "never-happened") == {
        "session": "never-happened", "notes_touched": 0, "aliases_removed": 0}


# ---- the baseline: retrieval quality as a ratchet ---------------------------


def test_baseline_round_trips_and_check_detects_a_drop(tmp_path: Path) -> None:
    """The regression gate only ever ran in CI, against a synthetic fixture.
    In production nothing compared a batch's effect to what came before, so an
    autonomous pipeline could quietly make retrieval worse forever.

    Uses the real FTS index rather than the fallback scan: the scan path scores
    a title hit and a body hit identically, so two notes tie and the "winner" is
    whichever iterated first, a coin flip is no basis for a regression test.
    """
    from brain.etl.learn import check_against_baseline, save_baseline
    from brain.infrastructure.index import SqliteIndex

    config = BrainConfig(root=tmp_path / "vault", git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root))
    api.initialize_vault()
    api.create_note("Wireless Media", "concept", domain="networking",
                    answer="Radio and microwave frequencies.")
    api.reindex(full=True)
    cases = [EvalCase(query="Wireless Media", expect=("concept_wireless-media",))]

    saved = save_baseline(api, cases)
    assert saved["precision_at_1"] == 1.0

    ok = check_against_baseline(api, cases)
    assert ok["regressed"] is False
    assert ok["baseline_precision_at_1"] == 1.0

    # A note whose TITLE is the query outranks one that merely matches it,
    # because title carries the heaviest bm25 weight.
    api.create_note("Wireless Media Standards Overview", "reference",
                    domain="networking", answer="A decoy.",
                    aliases=["Wireless Media", "wireless media"])
    api.reindex(full=True)

    bad = check_against_baseline(api, cases)
    assert bad["regressed"] is True, (
        f"a drop below the baseline must be reported (got {bad})")
    assert bad["precision_at_1"] < 1.0
    assert bad["delta"] is not None and bad["delta"] < 0


def test_check_without_a_baseline_does_not_claim_a_regression(api: KnowledgeApi) -> None:
    """No baseline means no comparison, reporting a regression against nothing
    would train the user to ignore the signal."""
    from brain.etl.learn import check_against_baseline

    out = check_against_baseline(api, [EvalCase(query="Wireless Media",
                                                expect=("concept_wireless-media",))])
    assert out["regressed"] is False
    assert out["baseline_precision_at_1"] is None


# ---- paired significance: is a delta real, or one lucky question? -----------


def test_check_reports_a_paired_test_not_just_a_delta(tmp_path: Path) -> None:
    """35 queries give a 31-point-wide CI on precision@1, so a raw delta is not
    evidence. The same queries run every time, so the comparison is *paired*, McNemar's exact test on the win/loss table is the honest statistic
    (Smucker, Allan & Carterette, CIKM 2007).

    A one-question flip must NOT come back significant. If it ever does, the
    gate is lying and every future tuning decision inherits the lie.
    """
    from brain.etl.learn import check_against_baseline, save_baseline
    from brain.infrastructure.index import SqliteIndex

    config = BrainConfig(root=tmp_path / "vault", git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root))
    api.initialize_vault()
    for i in range(12):
        api.create_note(f"Topic Number {i}", "concept", domain="meta",
                        answer=f"An answer about topic number {i}.")
    api.reindex(full=True)
    cases = [EvalCase(query=f"Topic Number {i}", expect=(f"concept_topic-number-{i}",))
             for i in range(12)]

    save_baseline(api, cases)
    out = check_against_baseline(api, cases)
    assert out["improved"] == 0 and out["worsened"] == 0
    assert out["significant"] is False
    assert out["p_value"] == 1.0

    # Break exactly one query by burying it under a same-titled competitor.
    api.create_note("Topic Number 3 Extended Discussion", "reference", domain="meta",
                    answer="decoy", aliases=["Topic Number 3"])
    api.reindex(full=True)

    one = check_against_baseline(api, cases)
    assert one["worsened"] + one["improved"] == 1, one
    assert one["significant"] is False, (
        "a single flipped query must never be reported as significant")
    assert one["p_value"] == 1.0
