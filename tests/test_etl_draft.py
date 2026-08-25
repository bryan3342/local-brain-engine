"""`etl-draft`: the only supported way the skill writes a note.

Before this, the skill wrote notes by calling the generic `brain create` with a
model-authored `--folder`/`--status` pair, so the vault's central invariant --
the ETL cannot write outside `00-inbox/` -- was enforced by prose, not code.
`etl-draft` has no such flags: placement is `apply()`'s decision, never the
caller's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.cli import build_parser, main
from brain.config import BrainConfig
from brain.domain.models import Candidate
from brain.etl.queue import Queue
from brain.infrastructure.vault import Vault


@pytest.fixture
def vault(tmp_path: Path) -> str:
    root = str(tmp_path / "vault")
    assert main(["--vault", root, "init"]) == 0
    return root


def _queue(vault: str) -> Queue:
    return Queue(Vault(Path(vault)))


def _api(vault: str) -> KnowledgeApi:
    return KnowledgeApi(BrainConfig(root=Path(vault), git=False), index=None)


def _seed(vault: str, **over: object) -> Candidate:
    base: dict[str, object] = dict(
        candidate_id="cand-1",
        session_id="sess-99",
        project="local-brain",
        timestamp="2026-08-20T00:00:00Z",
        user_text="how does the retry backoff work",
        assistant_text="It doubles each attempt up to a cap.",
        tool_summary={},
        branches=(),
    )
    base.update(over)
    candidate = Candidate(**base)  # type: ignore[arg-type]
    _queue(vault).append([candidate])
    return candidate


def out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def json_out(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(out(capsys))


# ---- the invariant ---------------------------------------------------------


def test_etl_draft_has_no_argparse_path_to_override_placement() -> None:
    """Assert against the parser itself: no flag lets `etl-draft` write outside
    `00-inbox/`. This is what makes the guarantee structural rather than a
    matter of code-review convention -- if a future edit adds `--folder` or
    `--status` back to this subparser, this test fails immediately."""
    parser = build_parser()
    base = ["etl-draft", "--candidate-id", "c1", "--title", "T",
            "--type", "concept", "--domain", "d", "--answer", "a"]

    try:
        parser.parse_args(base)
    except SystemExit as exc:
        pytest.fail(f"the legitimate etl-draft surface must parse; got SystemExit({exc.code})")

    for extra in (["--folder", "10-knowledge"], ["--status", "evergreen"]):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, *extra])


# ---- behaviour --------------------------------------------------------------


def test_etl_draft_creates_in_the_inbox_and_drains_the_queue(
    vault: str, capsys: pytest.CaptureFixture[str],
) -> None:
    seed = _seed(vault)
    assert main(["--vault", vault, "--json", "etl-draft",
                 "--candidate-id", seed.candidate_id,
                 "--title", "Retry Backoff Doubles", "--type", "concept",
                 "--domain", "reliability",
                 "--answer", "It doubles each attempt up to a cap."]) == 0
    payload = json_out(capsys)
    assert payload["verdict"] == "add"

    note = _api(vault).get_note(payload["id"])
    assert note.path.parent.name == "00-inbox"
    assert note.frontmatter.status.value == "inbox"
    assert _queue(vault).pending() == []


def test_etl_draft_note_carries_session_provenance(
    vault: str, capsys: pytest.CaptureFixture[str],
) -> None:
    seed = _seed(vault, session_id="sess-provenance-42")
    assert main(["--vault", vault, "--json", "etl-draft",
                 "--candidate-id", seed.candidate_id,
                 "--title", "Traceable Fact", "--type", "concept",
                 "--domain", "reliability", "--answer", "x"]) == 0
    payload = json_out(capsys)

    note = _api(vault).get_note(payload["id"])
    assert note.frontmatter.source["ref"] == "sess-provenance-42"
    assert note.frontmatter.source["type"] == "claude-code-session"
    assert note.frontmatter.source["connector"] == "brain-etl"
    edges = [r.encode() for r in note.frontmatter.relationships]
    assert "derived_from::session_sess-provenance-42" in edges


def test_unknown_candidate_id_writes_nothing_and_exits_nonzero(
    vault: str, capsys: pytest.CaptureFixture[str],
) -> None:
    before = len(_api(vault).list_notes())
    code = main(["--vault", vault, "etl-draft",
                 "--candidate-id", "does-not-exist",
                 "--title", "T", "--type", "concept",
                 "--domain", "d", "--answer", "a"])
    assert code != 0
    err = capsys.readouterr().err
    assert "does-not-exist" in err
    assert len(_api(vault).list_notes()) == before


def test_alias_collision_does_not_add_a_second_note_via_the_cli(
    vault: str, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--vault", vault, "create", "Why the backend deploy is manual",
                 "--type", "reference", "--domain", "client-site",
                 "--answer", "Manual dispatch only."]) == 0
    capsys.readouterr()

    seed = _seed(vault, candidate_id="cand-collision")
    assert main(["--vault", vault, "--json", "etl-draft",
                 "--candidate-id", seed.candidate_id,
                 "--title", "Backend deploy uses manual dispatch",
                 "--type", "reference", "--domain", "client-site",
                 "--answer", "Manual dispatch only.",
                 "--alias", "Why the backend deploy is manual"]) == 0
    payload = json_out(capsys)
    assert payload["verdict"] != "add"
    assert len(_api(vault).list_notes()) == 1
    assert _queue(vault).pending() == []


def test_noop_still_drains_the_queue(
    vault: str, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--vault", vault, "create", "Deploy is manual",
                 "--type", "reference", "--domain", "client-site",
                 "--answer", "Manual dispatch only.",
                 "--alias", "why no auto deploy"]) == 0
    capsys.readouterr()

    seed = _seed(vault, candidate_id="cand-noop")
    assert main(["--vault", vault, "--json", "etl-draft",
                 "--candidate-id", seed.candidate_id,
                 "--title", "Deploy is manual", "--type", "reference",
                 "--domain", "client-site", "--answer", "Manual dispatch only.",
                 "--alias", "why no auto deploy"]) == 0
    payload = json_out(capsys)
    assert payload["verdict"] == "noop"
    assert _queue(vault).pending() == []


# ---- --supersedes: the fourth verdict, reachable at last --------------------


def test_etl_draft_supersedes_produces_a_supersede_verdict(
    vault: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before this flag existed, SUPERSEDE was structurally unreachable from the
    only production caller, the ETL could add and enrich, but never record that
    a new fact contradicts an old one."""
    _seed(vault)
    old = _api(vault).create_note("Deploy is automatic", "reference",
                                  domain="client-site",
                                  answer="Push to main deploys.", status="evergreen")

    rc = main(["--vault", vault, "--json", "etl-draft", "--candidate-id", "cand-1",
               "--title", "Deploy is manual now", "--type", "reference",
               "--domain", "client-site", "--answer", "It is manual dispatch only.",
               "--supersedes", old.id])
    assert rc == 0
    payload = json.loads(out(capsys))
    assert payload["verdict"] == "supersede"
    assert payload["existing_id"] == old.id

    api = _api(vault)
    new = api.get_note(payload["id"])
    assert f"supersedes::{old.id}" in [r.encode() for r in new.frontmatter.relationships]
    assert new.frontmatter.status.value == "inbox"
    assert api.get_note(old.id).frontmatter.status.value == "evergreen", (
        "the ETL must not retire a trusted note; promotion does that")


def test_etl_draft_supersedes_an_unknown_id_writes_nothing(vault: str) -> None:
    _seed(vault)
    before = len(_api(vault).list_notes())
    rc = main(["--vault", vault, "etl-draft", "--candidate-id", "cand-1",
               "--title", "Anything", "--type", "concept", "--domain", "meta",
               "--answer", "x", "--supersedes", "concept_does-not-exist"])
    assert rc != 0
    assert len(_api(vault).list_notes()) == before, (
        "a bad supersede id must not leave a note behind")
    assert _queue(vault).pending(), "nor may it silently drain the candidate"
