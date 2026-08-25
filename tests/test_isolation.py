"""The guards in conftest must actually fire.

A safety net nobody tests is a safety net nobody has. These assert the guard
catches the exact incident it was written for: a CLI invocation whose `--vault`
went missing and fell back to the ambient environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.cli import main
from brain.config import BrainConfig, default_root

from .conftest import assert_tmp_root


def test_ambient_brain_vault_is_not_visible_to_tests() -> None:
    """Guard 1: the session fixture dropped it."""
    assert os.environ.get("BRAIN_VAULT") is None


def test_default_root_reads_the_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A module constant would have frozen this at import, that was the bug."""
    monkeypatch.setenv("BRAIN_VAULT", str(tmp_path / "late"))
    assert default_root() == tmp_path / "late"
    monkeypatch.delenv("BRAIN_VAULT")
    assert default_root() == Path("~/Developer/brain")


def test_guard_rejects_a_real_vault_path() -> None:
    """Guard 2: anything outside tmp is refused."""
    with pytest.raises(AssertionError, match="outside tmp"):
        assert_tmp_root("~/Developer/brain")
    with pytest.raises(AssertionError, match="outside tmp"):
        assert_tmp_root(Path.home() / "Documents" / "vault")


def test_guard_accepts_a_tmp_path(tmp_path: Path) -> None:
    assert assert_tmp_root(tmp_path) == tmp_path


def test_cli_without_vault_flag_is_blocked_not_silently_redirected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident, reproduced.

    A CLI call with no resolvable `--vault` used to fall through to the real
    vault. It must now hit the guard instead of writing anywhere real.
    """
    monkeypatch.delenv("BRAIN_VAULT", raising=False)
    with pytest.raises(AssertionError, match="outside tmp"):
        main(["create", "Should Never Be Written"])


def test_explicit_tmp_vault_still_works(tmp_path: Path) -> None:
    root = str(tmp_path / "vault")
    assert main(["--vault", root, "init"]) == 0
    assert main(["--vault", root, "create", "Fine"]) == 0
    assert (tmp_path / "vault" / "10-knowledge" / "fine.md").exists()


def test_config_load_honours_an_explicit_root(tmp_path: Path) -> None:
    cfg = BrainConfig.load(tmp_path / "explicit")
    assert cfg.root == tmp_path / "explicit"
