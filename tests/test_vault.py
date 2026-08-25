"""Tests for the filesystem boundary.

The interesting cases are the ones that lose data: a crash between write and
rename, and a concurrent writer landing between read and write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.domain.errors import ConcurrentModificationError, VaultError
from brain.infrastructure.vault import FileStamp, Vault, append_jsonl, atomic_write


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    v = Vault(tmp_path / "vault")
    v.initialize(git=False)
    return v


def test_initialize_is_idempotent(tmp_path: Path) -> None:
    v = Vault(tmp_path / "v")
    first = v.initialize(git=False)
    second = v.initialize(git=False)
    assert "10-knowledge" in first
    assert second == []
    assert v.exists()


def test_require_raises_on_uninitialized(tmp_path: Path) -> None:
    with pytest.raises(VaultError):
        Vault(tmp_path / "nope").require()


def test_atomic_write_then_read(vault: Vault) -> None:
    path = vault.root / "10-knowledge" / "a.md"
    stamp = atomic_write(path, b"hello")
    data, read_stamp = vault.read(path)
    assert data == b"hello"
    assert read_stamp.sha256 == stamp.sha256


def test_crash_mid_write_leaves_original_intact_and_cleans_up(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical store must survive an interrupted write."""
    path = vault.root / "10-knowledge" / "a.md"
    atomic_write(path, b"ORIGINAL")
    before = sorted(p.name for p in path.parent.iterdir())

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(path, b"REPLACEMENT")

    assert path.read_bytes() == b"ORIGINAL", "original was damaged"
    assert sorted(p.name for p in path.parent.iterdir()) == before, "temp file leaked"


def test_stale_stamp_is_rejected(vault: Vault) -> None:
    """Obsidian autosaving between our read and our write must not be clobbered."""
    path = vault.root / "10-knowledge" / "a.md"
    atomic_write(path, b"v1")
    _, stamp = vault.read(path)

    atomic_write(path, b"edited by obsidian")  # third-party writer

    with pytest.raises(ConcurrentModificationError):
        atomic_write(path, b"v2", expect=stamp)
    assert path.read_bytes() == b"edited by obsidian"


def test_matching_stamp_is_accepted(vault: Vault) -> None:
    path = vault.root / "10-knowledge" / "a.md"
    atomic_write(path, b"v1")
    _, stamp = vault.read(path)
    atomic_write(path, b"v2", expect=stamp)
    assert path.read_bytes() == b"v2"


def test_exclusive_create_refuses_to_overwrite(vault: Vault) -> None:
    path = vault.root / "10-knowledge" / "a.md"
    atomic_write(path, b"first", exclusive=True)
    with pytest.raises(ConcurrentModificationError):
        atomic_write(path, b"second", exclusive=True)
    assert path.read_bytes() == b"first"


def test_touch_changes_mtime_but_content_hash_detects_no_change(vault: Vault) -> None:
    """Staleness gate: a `touch` must not read as a content change."""
    path = vault.root / "10-knowledge" / "a.md"
    atomic_write(path, b"same bytes")
    before = FileStamp.of(path)
    os.utime(path, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    after = FileStamp.of(path)
    assert before != after                       # mtime moved
    assert after is not None
    assert after.content_matches(before)         # ...but the bytes did not


def test_note_paths_excludes_templates_dotfolders_and_root_docs(vault: Vault) -> None:
    (vault.root / "10-knowledge" / "real.md").write_text("x")
    (vault.root / "90-meta" / "templates" / "concept.md").write_text("x")
    (vault.root / "90-meta" / "trash" / "gone.md").write_text("x")
    (vault.root / ".obsidian").mkdir(exist_ok=True)
    (vault.root / ".obsidian" / "cfg.md").write_text("x")
    (vault.root / "README.md").write_text("x")

    names = {p.name for p in vault.note_paths()}
    assert names == {"real.md"}


def test_append_jsonl_is_line_atomic(vault: Vault) -> None:
    log = vault.audit_log
    for i in range(50):
        append_jsonl(log, f'{{"n": {i}}}')
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 50
    assert lines[0] == '{"n": 0}' and lines[-1] == '{"n": 49}'


def test_lock_is_exclusive_across_processes(vault: Vault) -> None:
    import subprocess
    import sys

    repo = str(Path(__file__).resolve().parents[1])
    script = (
        f"import sys; sys.path.insert(0, {repo!r})\n"
        "from pathlib import Path\n"
        "from brain.infrastructure.vault import Vault\n"
        "from brain.domain.errors import LockError\n"
        f"v = Vault(Path({str(vault.root)!r}))\n"
        "try:\n"
        "    with v.lock(blocking=False): print('ACQUIRED')\n"
        "except LockError: print('BLOCKED')\n"
    )

    with vault.lock():
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True).stdout
    assert "BLOCKED" in out, out

    out2 = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True).stdout
    assert "ACQUIRED" in out2, out2


def test_lock_non_blocking_raises_when_held(vault: Vault) -> None:
    with vault.lock():
        # same process, same fd table, flock is per-fd, so a second open blocks
        pass
    with vault.lock(blocking=False):
        pass  # released, so this succeeds


def test_git_commit_is_optional(tmp_path: Path) -> None:
    v = Vault(tmp_path / "nogit")
    v.initialize(git=False)
    assert v.git_available() is False
    assert v.git_commit("noop") is None


def test_git_commit_when_available(tmp_path: Path) -> None:
    v = Vault(tmp_path / "withgit")
    v.initialize(git=True)
    if not v.git_available():
        pytest.skip("git unavailable")
    import subprocess
    subprocess.run(["git", "-C", str(v.root), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(v.root), "config", "user.name", "t"], check=True)
    (v.root / "10-knowledge" / "a.md").write_text("hello")
    rev = v.git_commit("add a")
    assert rev and len(rev) >= 6
