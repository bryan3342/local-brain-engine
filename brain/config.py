"""Configuration. Explicit, injected, never global."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from brain.domain.identity import ProjectRules

FALLBACK_ROOT = "~/Developer/brain"


def default_root() -> Path:
    """Resolve the default vault, reading the environment *at call time*.

    Deliberately a function, not a module constant. Captured at import, the
    value freezes before a caller (or a test fixture) can change it, which is
    exactly how a test run once wrote a dozen fixture notes into a real vault.
    """
    return Path(os.environ.get("BRAIN_VAULT") or FALLBACK_ROOT)


@dataclass(frozen=True)
class BrainConfig:
    """Everything the Knowledge API needs to know about where it is running.

    Passed in explicitly so the same process can operate two vaults, and so
    tests need no monkeypatching. There is no module-level singleton.
    """

    root: Path = field(default_factory=lambda: default_root().expanduser())
    git: bool = True
    autocommit: bool = False
    author: str = "brain"
    projects: ProjectRules = field(default_factory=ProjectRules)

    @classmethod
    def load(cls, root: Path | str | None = None, **overrides: object) -> BrainConfig:
        """Read `90-meta/brain.toml` if present; explicit args win over it."""
        base = Path(root).expanduser() if root else default_root().expanduser()
        settings: dict[str, object] = {}
        config_file = base / "90-meta" / "brain.toml"
        if config_file.is_file():
            try:
                settings = dict(tomllib.loads(config_file.read_text()).get("brain", {}))
            except (OSError, tomllib.TOMLDecodeError):
                settings = {}
        settings.update({k: v for k, v in overrides.items() if v is not None})
        return cls(
            root=base,
            git=bool(settings.get("git", True)),
            autocommit=bool(settings.get("autocommit", False)),
            author=str(settings.get("author", "brain")),
            projects=_read_project_rules(config_file),
        )


def _read_project_rules(config_file: Path) -> ProjectRules:
    """Read the `[projects]` table. Absent or malformed gives empty rules.

    These live in the vault, not in the package, so no organisation or repo
    name is committed here.
    """
    if not config_file.is_file():
        return ProjectRules()
    try:
        table = tomllib.loads(config_file.read_text()).get("projects", {})
    except (OSError, tomllib.TOMLDecodeError):
        return ProjectRules()
    aliases = tuple(
        (str(k).lower(), str(v)) for k, v in dict(table.get("aliases", {})).items()
    )
    return ProjectRules(aliases=aliases, repos=tuple(str(r) for r in table.get("repos", [])))
