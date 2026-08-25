"""CLI tests.

The CLI is the user-facing surface and the place every bit of formatting lives,
so it needs its own tests rather than riding on the API's. These call `main()`
directly with an argv list, no subprocess, so failures give real tracebacks.

Exit codes matter: they are the contract a script (or a future MCP adapter)
depends on to tell "not found" from "duplicate" from "invalid".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.cli import main


@pytest.fixture
def vault(tmp_path: Path) -> str:
    root = str(tmp_path / "vault")
    assert main(["--vault", root, "init"]) == 0
    return root


def run(vault: str, *args: str) -> int:
    return main(["--vault", vault, *args])


def out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def json_out(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(out(capsys))


# ---- happy paths ----------------------------------------------------------


def test_init_creates_the_skeleton(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "v"
    assert main(["--vault", str(root), "--json", "init"]) == 0
    payload = json_out(capsys)
    assert "10-knowledge" in payload["created"]
    assert (root / "90-meta").is_dir()


def test_create_then_get(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(vault, "--json", "create", "OSI Model", "--type", "concept",
               "--domain", "networking", "--answer", "Seven layers.") == 0
    created = json_out(capsys)
    assert created["id"] == "concept_osi-model"

    assert run(vault, "--json", "get", "concept_osi-model") == 0
    fetched = json_out(capsys)
    assert fetched["title"] == "OSI Model"
    assert fetched["answer"] == "Seven layers."


def test_global_flags_work_after_the_subcommand(vault: str,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """`brain create X --json` is what people actually type."""
    assert run(vault, "create", "Late Flag", "--json") == 0
    assert json_out(capsys)["id"] == "concept_late-flag"


def test_list_and_search(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "Fiber Optic Cabling", "--answer", "Light, not current.",
        "--alias", "fibre")
    capsys.readouterr()
    assert run(vault, "--json", "list") == 0
    assert len(json.loads(out(capsys))) == 1
    assert run(vault, "--json", "search", "fibre") == 0
    assert json.loads(out(capsys))[0]["id"] == "concept_fiber-optic-cabling"


def test_link_backlinks_and_neighbors(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "A")
    run(vault, "create", "B")
    capsys.readouterr()
    assert run(vault, "link", "concept_a", "mentions", "concept_b") == 0
    capsys.readouterr()
    assert run(vault, "--json", "backlinks", "concept_b") == 0
    links = json.loads(out(capsys))
    assert links[0]["source"]["id"] == "concept_a"
    assert run(vault, "neighbors", "concept_a") == 0
    assert json_out(capsys)["outbound"][0]["target"] == "concept_b"


def test_rename_preserves_id_via_cli(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "GraphRAG")
    capsys.readouterr()
    assert run(vault, "--json", "rename", "concept_graphrag", "Microsoft GraphRAG") == 0
    renamed = json_out(capsys)
    assert renamed["id"] == "concept_graphrag"
    assert "GraphRAG" in renamed["aliases"]


def test_update_then_delete(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "Doomed")
    capsys.readouterr()
    assert run(vault, "--json", "update", "concept_doomed", "--status", "evergreen") == 0
    assert json_out(capsys)["status"] == "evergreen"
    assert run(vault, "--json", "delete", "concept_doomed") == 0
    assert json_out(capsys)["deleted"] is True


def test_export_graph_is_always_json(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "A", "--rel", "touched::file_x-ts")
    capsys.readouterr()
    assert run(vault, "export-graph") == 0
    graph = json_out(capsys)
    assert {n["type"] for n in graph["nodes"]} == {"concept", "anchor:file"}


def test_reindex_and_no_index_agree(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "Alpha")
    capsys.readouterr()
    assert run(vault, "--json", "reindex", "--full") == 0
    assert json_out(capsys)["notes"] == 1
    run(vault, "--json", "list")
    indexed = out(capsys)
    assert main(["--vault", vault, "--no-index", "--json", "list"]) == 0
    assert out(capsys) == indexed


def test_scan_and_doctor_dry_run_writes_nothing(vault: str,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    rogue = Path(vault) / "10-knowledge" / "rogue.md"
    rogue.write_text("# Hand written\n\nbody\n")
    assert run(vault, "--json", "scan") == 0
    assert any(f["kind"] == "no-frontmatter" for f in json_out(capsys)["findings"])

    assert run(vault, "--json", "doctor") == 0
    assert json_out(capsys)["dry_run"] is True
    assert rogue.read_text().startswith("# Hand written"), "dry run must not write"

    assert run(vault, "--json", "doctor", "--apply") == 0
    capsys.readouterr()
    assert rogue.read_text().startswith("---"), "apply should have adopted it"


def test_scan_advisory_flag(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "Elsewhere", "--folder", "40-research")
    capsys.readouterr()
    run(vault, "--json", "scan")
    assert not any(f["kind"] == "type-folder-mismatch" for f in json_out(capsys)["findings"])
    run(vault, "--json", "scan", "--advisory")
    assert any(f["kind"] == "type-folder-mismatch" for f in json_out(capsys)["findings"])


def test_audit_records_every_mutation(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "Tracked")
    run(vault, "update", "concept_tracked", "--status", "evergreen")
    capsys.readouterr()
    assert run(vault, "--json", "audit") == 0
    ops = [e["op"] for e in json.loads(out(capsys))]
    assert "create_note" in ops and "update_note" in ops


# ---- error contract -------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (("get", "nope_missing"), 3),                    # NoteNotFoundError
        (("create", "   "), 5),                          # ValidationError
        (("create", "X", "--rel", "not-an-edge"), 5),     # ValidationError
    ],
)
def test_errors_map_to_stable_exit_codes(vault: str, args: tuple[str, ...], code: int,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    assert run(vault, *args) == code
    assert "error:" in capsys.readouterr().err


def test_duplicate_exits_four(vault: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(vault, "create", "GraphRAG")
    capsys.readouterr()
    assert run(vault, "create", "Graph RAG") == 4
    assert run(vault, "create", "Graph RAG", "--allow-duplicate") == 0


def test_operating_on_an_uninitialized_vault_exits_ten(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--vault", str(tmp_path / "empty"), "create", "X"]) == 10
    assert "error:" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error(vault: str) -> None:
    with pytest.raises(SystemExit) as exc:
        run(vault, "frobnicate")
    assert exc.value.code == 2


# ---- eval ------------------------------------------------------------------


def eval_vault(tmp_path: Path) -> str:
    """A vault holding both eval sets and one note each can find."""
    root = tmp_path / "vault"
    assert main(["--vault", str(root), "init"]) == 0
    assert main(["--vault", str(root), "create", "Fiber-Optic Cabling",
                 "--type", "concept", "--answer", "Light, not current."]) == 0
    assert main(["--vault", str(root), "reindex"]) == 0
    eval_dir = root / "90-meta" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "golden.jsonl").write_text(
        json.dumps({"q": "fiber", "expect": ["concept_fiber-optic-cabling"]}) + "\n")
    (eval_dir / "golden-neutral.jsonl").write_text(
        json.dumps({"q": "what carries light instead of current",
                    "expect": ["concept_fiber-optic-cabling"]}) + "\n")
    return str(root)


def test_eval_reports_every_set_not_just_the_tuned_one(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Scoring only golden.jsonl flatters retrieval, so the default reports both.

    Would fail if `eval` kept emitting a single flat report: `sets` would be
    absent, and the neutral score would stay invisible unless asked for.
    """
    root = eval_vault(tmp_path)
    capsys.readouterr()

    main(["--vault", root, "--json", "eval"])
    payload = json_out(capsys)

    assert sorted(payload["sets"]) == ["golden", "golden-neutral"]
    for name in ("golden", "golden-neutral"):
        assert "precision@1" in payload["sets"][name]
        assert payload["sets"][name]["cases"] == 1


def test_eval_with_an_explicit_set_keeps_the_flat_shape(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A script naming its own set is parsing one report, not a map of them.

    Would fail if the multi-set shape were applied unconditionally.
    """
    root = eval_vault(tmp_path)
    capsys.readouterr()

    main(["--vault", root, "--json", "eval",
          "--golden", str(Path(root) / "90-meta" / "eval" / "golden-neutral.jsonl")])
    payload = json_out(capsys)

    assert "sets" not in payload
    assert payload["cases"] == 1
    assert "precision@1" in payload


def test_eval_fails_when_any_set_has_a_miss(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A set the vault cannot answer must not be hidden by one it can.

    Would fail if the exit code were taken from a single report rather than
    every scored set.
    """
    root = eval_vault(tmp_path)
    (Path(root) / "90-meta" / "eval" / "golden-neutral.jsonl").write_text(
        json.dumps({"q": "nothing here answers this", "expect": ["concept_absent"]}) + "\n")
    capsys.readouterr()

    assert main(["--vault", root, "--json", "eval"]) == 1
    payload = json_out(capsys)
    assert payload["sets"]["golden"]["precision@1"] == 1.0
    assert payload["sets"]["golden-neutral"]["precision@1"] == 0.0


def test_reindex_reports_vectors_when_an_embedder_is_configured(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI has to hand the configured embedder to the index it builds.

    Would fail if `_api` kept constructing a bare SqliteIndex: every command
    would run keyword-only no matter what the environment asked for.
    """
    from tests.test_vectors import StubEmbedder

    monkeypatch.setattr("brain.infrastructure.embedder.embedder_from_env",
                        lambda: StubEmbedder())
    root = str(tmp_path / "vault")
    assert main(["--vault", root, "init"]) == 0
    assert main(["--vault", root, "create", "Fiber", "--type", "concept",
                 "--answer", "Light."]) == 0
    capsys.readouterr()

    main(["--vault", root, "--json", "reindex", "--full"])

    assert json_out(capsys)["vectors"] == 1
