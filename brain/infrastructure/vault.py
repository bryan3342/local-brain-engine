"""Filesystem boundary: atomic writes, optimistic concurrency, locking, git.

Everything here exists because the Markdown vault is declared *canonical*. A
canonical store that can be left half-written by a crash, or silently clobbered
by Obsidian autosaving mid-update, is not canonical (C3).

Three writers exist, this package, Obsidian, and the human. This module cannot
make them coordinate. What it can do is make *its own* writes atomic and refuse
to overwrite a change it did not see. Reconciliation of the rest is
`application/reconcile.py`.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from brain.domain.errors import ConcurrentModificationError, LockError, VaultError

#: Top-level folders. Numeric prefixes encode trust tier, not just ordering.
TIERS: tuple[str, ...] = (
    "00-inbox", "10-knowledge", "20-projects", "30-work", "40-research", "90-meta",
)
META = "90-meta"
INDEX_DIR = f"{META}/indexes"
INDEX_DB = f"{INDEX_DIR}/brain.db"
LOCK_FILE = f"{META}/.brain.lock"
AUDIT_LOG = f"{META}/audit.jsonl"
TRASH_DIR = f"{META}/trash"
TEMPLATE_DIR = f"{META}/templates"


@dataclass(frozen=True, slots=True)
class FileStamp:
    """What a file looked like when we read it.

    `mtime_ns` and `size` make the common case cheap; `sha256` is the backstop,
    because some sync providers preserve mtime across a content change and a
    bare `touch` changes mtime without changing bytes.
    """

    mtime_ns: int
    size: int
    sha256: str

    @classmethod
    def of(cls, path: Path) -> FileStamp | None:
        """Stamp a file, or None if it does not exist."""
        try:
            st = path.stat()
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        return cls(st.st_mtime_ns, st.st_size, hashlib.sha256(data).hexdigest())

    @classmethod
    def of_bytes(cls, path: Path, data: bytes) -> FileStamp:
        st = path.stat()
        return cls(st.st_mtime_ns, st.st_size, hashlib.sha256(data).hexdigest())

    def content_matches(self, other: FileStamp | None) -> bool:
        return other is not None and self.sha256 == other.sha256


def atomic_write(
    path: Path,
    data: bytes,
    expect: FileStamp | None = None,
    *,
    exclusive: bool = False,
) -> FileStamp:
    """Write `data` to `path` atomically.

    temp file in the same directory -> flush -> fsync -> `os.replace`, which is
    atomic on POSIX. A crash mid-write therefore leaves either the old file or
    the new one, never a truncated one.

    `expect` is an optimistic-concurrency guard: if the file no longer matches
    the stamp taken at read time, raise instead of clobbering. `exclusive=True`
    asserts the file does not exist yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if exclusive and path.exists():
        raise ConcurrentModificationError(f"{path} already exists")
    if expect is not None and FileStamp.of(path) != expect:
        raise ConcurrentModificationError(f"{path} changed on disk since it was read")

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".brain-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # Re-check immediately before the swap to narrow the race as far as a
        # single process can. A writer that lands between here and os.replace
        # is what the vault lock is for.
        if expect is not None and FileStamp.of(path) != expect:
            raise ConcurrentModificationError(f"{path} changed on disk during write")
        if exclusive and path.exists():
            raise ConcurrentModificationError(f"{path} appeared during write")
        os.replace(tmp, path)
        tmp = None  # type: ignore[assignment]
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)

    _fsync_dir(path.parent)
    return FileStamp.of_bytes(path, data)


def _fsync_dir(directory: Path) -> None:
    """Persist the rename itself, not just the file contents."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def append_jsonl(path: Path, line: str) -> None:
    """Append one record with a single `write()` under `O_APPEND`.

    One syscall per record keeps concurrent appenders from interleaving
    partial lines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (line.rstrip("\n") + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


class Vault:
    """The vault directory, and the only code allowed to touch it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    # ---- layout -----------------------------------------------------------

    @property
    def index_db(self) -> Path:
        return self.root / INDEX_DB

    @property
    def audit_log(self) -> Path:
        return self.root / AUDIT_LOG

    @property
    def trash(self) -> Path:
        return self.root / TRASH_DIR

    @property
    def templates(self) -> Path:
        return self.root / TEMPLATE_DIR

    def exists(self) -> bool:
        return self.root.is_dir() and (self.root / META).is_dir()

    def require(self) -> None:
        if not self.exists():
            raise VaultError(f"{self.root} is not an initialized brain vault")

    def initialize(self, *, git: bool = True) -> list[str]:
        """Create the folder skeleton. Idempotent."""
        created: list[str] = []
        for rel in (*TIERS, INDEX_DIR, TRASH_DIR, TEMPLATE_DIR):
            path = self.root / rel
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(rel)
        if git and not (self.root / ".git").exists():
            try:
                subprocess.run(["git", "init", "-q", "-b", "main", str(self.root)],
                               check=True, capture_output=True)
                created.append(".git")
            except (OSError, subprocess.CalledProcessError):
                pass  # git is the rollback story, not a hard dependency
        return created

    # ---- notes ------------------------------------------------------------

    def note_paths(self) -> Iterator[Path]:
        """Every candidate note, excluding dotfolders and templates.

        Templates are scaffolding with placeholder ids; indexing them would
        collide with real notes.
        """
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if rel.parts[:2] == (META, "templates"):
                continue
            if rel.parts[:2] == (META, "trash"):
                continue
            if len(rel.parts) == 1 and rel.name in {"README.md", "CLAUDE.md", "INDEX.md"}:
                continue
            yield path

    def read(self, path: Path) -> tuple[bytes, FileStamp]:
        data = path.read_bytes()
        return data, FileStamp.of_bytes(path, data)

    def write(
        self, path: Path, data: bytes, expect: FileStamp | None = None, *, exclusive: bool = False
    ) -> FileStamp:
        return atomic_write(path, data, expect, exclusive=exclusive)

    def relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    # ---- locking ----------------------------------------------------------

    @contextmanager
    def lock(self, *, blocking: bool = True) -> Iterator[None]:
        """One coarse advisory lock, serializing API-vs-API writes.

        Deliberately coarse: contention is a single-user desktop scenario, and a
        per-note lock would buy nothing while adding deadlock surface.
        """
        lock_path = self.root / LOCK_FILE
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            try:
                fcntl.flock(fd, flags)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise LockError("another brain process holds the vault lock") from exc
                raise
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ---- git --------------------------------------------------------------

    def git_available(self) -> bool:
        return (self.root / ".git").is_dir()

    def git_commit(self, message: str, paths: list[Path] | None = None) -> str | None:
        """Commit. Git is the rollback story (H2), the audit log has no pre-image."""
        if not self.git_available():
            return None
        try:
            targets = [str(p) for p in paths] if paths else ["-A"]
            subprocess.run(["git", "-C", str(self.root), "add", *targets],
                           check=True, capture_output=True)
            result = subprocess.run(
                ["git", "-C", str(self.root), "commit", "-q", "-m", message],
                capture_output=True, text=True)
            if result.returncode != 0:
                return None
            rev = subprocess.run(["git", "-C", str(self.root), "rev-parse", "--short", "HEAD"],
                                 check=True, capture_output=True, text=True)
            return rev.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
