"""Test isolation.

A test suite must never be able to touch a real vault. This one once could: the
CLI's subparser reset `--vault` to `None`, which fell back to `$BRAIN_VAULT`, and
a run of *failing* tests wrote a dozen fixture notes into the live vault.

Two guards, because either alone is insufficient:

1. `$BRAIN_VAULT` is dropped for the whole session, so an omitted or reset
   `--vault` resolves to the fallback rather than to whatever the developer's
   shell happened to export.
2. Every `BrainConfig` built during a test is checked to live under the system
   temp dir. That catches what the first guard cannot, any path resolved some
   other way, including the hard-coded fallback.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.config import BrainConfig


@pytest.fixture(autouse=True, scope="session")
def _drop_ambient_vault_env() -> Iterator[None]:
    """Never inherit a developer's BRAIN_VAULT into the suite."""
    saved = os.environ.pop("BRAIN_VAULT", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["BRAIN_VAULT"] = saved


def assert_tmp_root(root: Path | str) -> Path:
    """Raise unless `root` is under the system temp dir."""
    resolved = Path(root).expanduser()
    tmp = Path(tempfile.gettempdir()).resolve()
    try:
        resolved.resolve().relative_to(tmp)
    except ValueError:
        raise AssertionError(
            f"test tried to use a vault outside tmp: {resolved}. "
            "Tests must pass an explicit tmp_path root."
        ) from None
    return resolved


@pytest.fixture(autouse=True)
def _forbid_real_vault_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test aims a BrainConfig anywhere but tmp."""
    real_load = BrainConfig.load.__func__  # type: ignore[attr-defined]

    def guarded(cls: type[BrainConfig], root: Path | str | None = None,
                **overrides: object) -> BrainConfig:
        cfg = real_load(cls, root, **overrides)
        assert_tmp_root(cfg.root)
        return cfg

    monkeypatch.setattr(BrainConfig, "load", classmethod(guarded))
