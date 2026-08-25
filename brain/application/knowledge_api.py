"""The Knowledge API: the only writer that maintains the vault's invariants.

Note the wording. The spec originally said "the only write path", but that is
not true and cannot be made true (C2): Obsidian writes, and so does the human.
What this class guarantees is narrower and achievable, *its own* writes are
atomic, id-stable, body-preserving, and audited, and it refuses to overwrite a
change it did not see. `reconcile.scan/doctor` converges everything else.

No printing, no argparse, no global state. Every method returns a
JSON-serializable dataclass and raises a typed `BrainError`, because the Phase-2
seam is an MCP server where each method becomes a tool.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from brain.application.mapper import (
    apply_to_mapping,
    parse_note,
    render_new_note,
)
from brain.application.validators import find_duplicate, validate
from brain.config import BrainConfig
from brain.domain.enums import DEFAULT_FOLDER, NoteStatus, NoteType
from brain.domain.errors import (
    ConcurrentModificationError,
    DuplicateNoteError,
    ImmutableFieldError,
    NoteNotFoundError,
    ValidationError,
)
from brain.domain.identity import (
    anchor_kind,
    is_anchor,
    mint_id,
    normalize_title,
    slugify,
)
from brain.domain.models import (
    Backlink,
    Frontmatter,
    GraphExport,
    IndexStats,
    Note,
    NoteRef,
    Relationship,
    SearchHit,
)
from brain.infrastructure.audit_log import AuditLog
from brain.infrastructure.markdown import splice
from brain.infrastructure.templates import render_body
from brain.infrastructure.vault import FileStamp, Vault


class NoteIndex(Protocol):
    """What the API needs from an index. Optional by contract.

    Every method here has a full-scan fallback, which is what makes the index
    provably disposable: delete the database and everything still works, only
    slower.
    """

    def sync(self, vault: Vault, *, full: bool = False) -> IndexStats: ...
    def by_id(self, note_id: str) -> str | None: ...
    def refs(self) -> list[NoteRef]: ...
    def backlinks(self, note_id: str) -> list[tuple[str, str]]: ...
    def search(self, query: str, limit: int
               ) -> list[tuple[str, str, float, float | None]]: ...


class KnowledgeApi:
    def __init__(self, config: BrainConfig | None = None, index: NoteIndex | None = None) -> None:
        self.config = config or BrainConfig.load()
        self.vault = Vault(self.config.root)
        self.index = index
        self.audit = AuditLog(self.vault.audit_log, author=self.config.author)

    # ---- vault ------------------------------------------------------------

    def initialize_vault(self) -> dict[str, Any]:
        created = self.vault.initialize(git=self.config.git)
        if created:
            self.audit.record("initialize_vault", created=created)
        return {"root": str(self.vault.root), "created": created}

    # ---- reading ----------------------------------------------------------

    def iter_notes(self, *, include_deleted: bool = False) -> Iterator[Note]:
        """Every parseable note. Unparseable files are skipped, not raised on.

        Tolerance is the point: a hand-written file must not break `list`,
        `search`, or `export-graph`. `scan()` is where such files are reported.
        """
        for path in self.vault.note_paths():
            try:
                note = parse_note(path, path.read_bytes())
            except (ValidationError, OSError):
                continue
            if not include_deleted and note.frontmatter.status is NoteStatus.DELETED:
                continue
            yield note

    def get_note(self, ref: str) -> Note:
        """Resolve by id, then alias, then path, then normalized title.

        Never by re-slugifying a title, that is the C1 bug.
        """
        if self.index is not None:
            hit = self.index.by_id(ref)
            if hit:
                path = self.vault.root / hit
                if path.is_file():
                    return parse_note(path, path.read_bytes())

        candidate = Path(ref)
        for probe in (candidate, self.vault.root / ref):
            if probe.is_file() and probe.suffix == ".md":
                return parse_note(probe, probe.read_bytes())

        by_alias: Note | None = None
        by_title: Note | None = None
        target = normalize_title(ref)
        for note in self.iter_notes(include_deleted=True):
            if note.id == ref:
                return note
            if by_alias is None and any(normalize_title(a) == target
                                        for a in note.frontmatter.aliases):
                by_alias = note
            if by_title is None and normalize_title(note.frontmatter.title) == target:
                by_title = note
        resolved = by_alias or by_title
        if resolved is None:
            raise NoteNotFoundError(f"no note resolves for {ref!r}")
        return resolved

    def list_notes(
        self,
        *,
        note_type: str | None = None,
        domain: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[NoteRef]:
        out: list[NoteRef] = []
        for note in self.iter_notes():
            fm = note.frontmatter
            if note_type and fm.type.value != note_type:
                continue
            if domain and fm.domain != domain:
                continue
            if status and fm.status.value != status:
                continue
            out.append(self._ref(note))
            if limit and len(out) >= limit:
                break
        return out

    # ---- writing ----------------------------------------------------------

    def create_note(
        self,
        title: str,
        note_type: str | NoteType = NoteType.CONCEPT,
        *,
        domain: str | None = None,
        status: str | NoteStatus = NoteStatus.DRAFT,
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
        source: dict[str, Any] | None = None,
        relationships: list[str] | None = None,
        answer: str = "",
        body: bytes | None = None,
        folder: str | None = None,
        allow_duplicate: bool = False,
    ) -> Note:
        """Create a note. The id is minted here, once, and frozen forever."""
        self.vault.require()
        ntype = NoteType(note_type)
        nstatus = NoteStatus(status)
        if not title.strip():
            raise ValidationError("title is required")

        with self.vault.lock():
            existing = [(n.id, n.frontmatter.type.value, n.frontmatter.title,
                         n.frontmatter.aliases)
                        for n in self.iter_notes(include_deleted=True)]
            taken = {row[0] for row in existing}

            note_id = mint_id(ntype, title, taken)
            fm = Frontmatter(
                id=note_id,
                title=title.strip(),
                type=ntype,
                status=nstatus,
                domain=domain,
                tags=list(tags or []),
                aliases=list(aliases or []),
                source=dict(source or {}),
                relationships=self._edges(relationships),
                created=_dt.date.today(),
            )
            problems = validate(fm)
            if problems:
                raise ValidationError("; ".join(problems))
            if not allow_duplicate:
                dup = find_duplicate(fm, existing)
                if dup:
                    raise DuplicateNoteError(
                        f"a {ntype.value} note titled {title!r} already exists as {dup}; "
                        "pass allow_duplicate=True to create it anyway")

            rel_dir = folder or DEFAULT_FOLDER[ntype]
            path = self.vault.root / rel_dir / f"{slugify(title)}.md"
            n = 2
            while path.exists():
                path = path.with_name(f"{slugify(title)}-{n}.md")
                n += 1

            content = body if body is not None else render_body(
                self.vault.templates, ntype.value, title=fm.title, answer=answer)
            data = render_new_note(fm, content)
            self.vault.write(path, data, exclusive=True)
            self.audit.record("create_note", note_id, path=self.vault.relative(path),
                              type=ntype.value)
            self.refresh_index()
            return parse_note(path, data)

    def update_note(
        self,
        ref: str,
        *,
        title: str | None = None,
        status: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
        source: dict[str, Any] | None = None,
        expect_hash: str | None = None,
    ) -> Note:
        """Update frontmatter only. The body is re-emitted byte-for-byte.

        `id` and `created` are immutable. Templates are never applied here, an
        update that re-rendered a body from a template would shred human edits
        (H1b).
        """
        self.vault.require()
        with self.vault.lock():
            note = self.get_note(ref)
            raw, stamp = self.vault.read(note.path)
            if expect_hash and stamp.sha256 != expect_hash:
                raise ConcurrentModificationError(
                    f"{note.path} has hash {stamp.sha256[:12]}, expected {expect_hash[:12]}")

            fm = note.frontmatter
            if title is not None:
                fm.title = title.strip()
            if status is not None:
                fm.status = NoteStatus(status)
            if domain is not None:
                fm.domain = domain
            if tags is not None:
                fm.tags = list(tags)
            if aliases is not None:
                fm.aliases = list(aliases)
            if source is not None:
                fm.source = dict(source)
            fm.updated = _dt.date.today()

            problems = validate(fm)
            if problems:
                raise ValidationError("; ".join(problems))

            original_id, original_created = note.frontmatter.id, note.frontmatter.created

            def mutate(data: Any) -> None:
                if str(data.get("id", original_id)) != original_id:
                    raise ImmutableFieldError("id is frozen at creation")
                apply_to_mapping(fm, data)
                data["id"] = original_id
                if original_created is not None:
                    data["created"] = original_created.isoformat()

            new_raw = splice(raw, mutate)
            self.vault.write(note.path, new_raw, expect=stamp)
            self.audit.record("update_note", note.id, path=self.vault.relative(note.path))
            self.refresh_index()
            return parse_note(note.path, new_raw)

    def rename_note(
        self, ref: str, new_title: str, *, dest_folder: str | None = None
    ) -> Note:
        """Retitle and optionally move. The id does not change.

        The old title is appended to `aliases`, so anything that referred to the
        note by its old name still resolves. This is the operation that proves
        id-stability: after it, every inbound edge must still resolve.
        """
        self.vault.require()
        with self.vault.lock():
            note = self.get_note(ref)
            raw, stamp = self.vault.read(note.path)
            old_title = note.frontmatter.title
            fm = note.frontmatter
            fm.title = new_title.strip()
            renamed = normalize_title(old_title) != normalize_title(new_title)
            if old_title and renamed and old_title not in fm.aliases:
                fm.aliases = [*fm.aliases, old_title]
            fm.updated = _dt.date.today()

            frozen_id, frozen_created = fm.id, fm.created

            def mutate(data: Any) -> None:
                apply_to_mapping(fm, data)
                data["id"] = frozen_id
                if frozen_created is not None:
                    data["created"] = frozen_created.isoformat()

            new_raw = splice(raw, mutate)
            target_dir = (self.vault.root / dest_folder) if dest_folder else note.path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            new_path = target_dir / f"{slugify(new_title)}.md"
            n = 2
            while new_path.exists() and new_path != note.path:
                new_path = target_dir / f"{slugify(new_title)}-{n}.md"
                n += 1

            if new_path == note.path:
                self.vault.write(note.path, new_raw, expect=stamp)
            else:
                self.vault.write(new_path, new_raw, exclusive=True)
                note.path.unlink()
            self.audit.record("rename_note", fm.id, was=old_title, now=fm.title,
                              path=self.vault.relative(new_path))
            self.refresh_index()
            return parse_note(new_path, new_raw)

    def delete_note(self, ref: str, *, hard: bool = False) -> dict[str, Any]:
        """Soft-delete by default (M3): `status: deleted` plus a move to trash.

        Inbound edges are left alone and surface as `unresolved` in
        `export_graph`, rather than being silently rewritten.
        """
        self.vault.require()
        with self.vault.lock():
            note = self.get_note(ref)
            _, stamp = self.vault.read(note.path)
            if hard:
                if FileStamp.of(note.path) != stamp:
                    raise ConcurrentModificationError(f"{note.path} changed during delete")
                note.path.unlink()
                self.audit.record("delete_note", note.id, hard=True)
                self.refresh_index()
                return {"id": note.id, "deleted": True, "hard": True}

            raw, _ = self.vault.read(note.path)
            fm = note.frontmatter
            fm.status = NoteStatus.DELETED
            frozen_id, frozen_created = fm.id, fm.created

            def mutate(data: Any) -> None:
                apply_to_mapping(fm, data)
                data["id"] = frozen_id
                if frozen_created is not None:
                    data["created"] = frozen_created.isoformat()

            new_raw = splice(raw, mutate)
            dest = self.vault.trash / note.path.name
            n = 2
            while dest.exists():
                dest = self.vault.trash / f"{note.path.stem}-{n}.md"
                n += 1
            self.vault.write(dest, new_raw, exclusive=True)
            # Guard the unlink too: if the original changed after we copied it to
            # trash, we would otherwise destroy an edit we never saw.
            if FileStamp.of(note.path) != stamp:
                dest.unlink(missing_ok=True)
                raise ConcurrentModificationError(
                    f"{note.path} changed while being deleted; nothing was removed")
            note.path.unlink()
            self.audit.record("delete_note", note.id, hard=False,
                              trashed=self.vault.relative(dest))
            self.refresh_index()
            return {"id": note.id, "deleted": True, "hard": False,
                    "trash": self.vault.relative(dest)}

    # ---- relationships ----------------------------------------------------

    def link_notes(self, src: str, rel: str, dst: str) -> Note:
        """Add one typed edge. Idempotent.

        The target is *not* validated by a full scan, that would make linking
        O(n) (H3). A dangling target is reported by `export_graph` and `doctor`.
        """
        note = self.get_note(src)
        target = dst
        # Forward references are allowed: linking must not be O(vault) (H3), and
        # a dangling target is reported by export_graph and doctor.
        with contextlib.suppress(NoteNotFoundError):
            target = self.get_note(dst).id
        edges = [r.encode() for r in note.frontmatter.relationships]
        new_edge = Relationship(rel, target).encode()
        if new_edge not in edges:
            edges.append(new_edge)
        return self._write_edges(note, edges, "link_notes", rel=rel, target=target)

    def unlink_notes(self, src: str, rel: str, dst: str) -> Note:
        note = self.get_note(src)
        drop = Relationship(rel, dst).encode()
        edges = [r.encode() for r in note.frontmatter.relationships if r.encode() != drop]
        return self._write_edges(note, edges, "unlink_notes", rel=rel, target=dst)

    def get_backlinks(self, ref: str) -> list[Backlink]:
        note = self.get_note(ref)
        if self.index is not None:
            out: list[Backlink] = []
            for src_id, rel in self.index.backlinks(note.id):
                try:
                    out.append(Backlink(source=self._ref(self.get_note(src_id)), rel=rel))
                except NoteNotFoundError:
                    continue
            return out
        return [
            Backlink(source=self._ref(other), rel=edge.rel)
            for other in self.iter_notes()
            for edge in other.frontmatter.relationships
            if edge.target == note.id
        ]

    def get_neighbors(self, ref: str) -> dict[str, Any]:
        note = self.get_note(ref)
        outbound = [{"rel": e.rel, "target": e.target} for e in note.frontmatter.relationships]
        inbound = [b.to_dict() for b in self.get_backlinks(note.id)]
        return {"id": note.id, "outbound": outbound, "inbound": inbound}

    def export_graph(self) -> GraphExport:
        """Nodes and edges. Renderer-agnostic, no Sigma/Cytoscape shape baked in.

        Deleted notes are excluded and inbound edges to them are reported as
        `unresolved` rather than emitted as ghost nodes (M3).
        """
        graph = GraphExport()
        known: dict[str, NoteRef] = {}
        notes = list(self.iter_notes())
        for note in notes:
            ref = self._ref(note)
            known[note.id] = ref
            graph.nodes.append(ref)

        # Edges resolve by id, then by alias, the same order get_note uses. A
        # note may be referred to by an identifier it used to have, or by the
        # session uuid it was derived from.
        by_alias: dict[str, str] = {}
        for note in notes:
            for alias in note.frontmatter.aliases:
                by_alias.setdefault(alias, note.id)

        anchors: dict[str, NoteRef] = {}
        for note in notes:
            for edge in note.frontmatter.relationships:
                target = edge.target if edge.target in known else by_alias.get(
                    edge.target, edge.target)
                record = {"source": note.id, "rel": edge.rel, "target": target}
                if target in known:
                    graph.edges.append(record)
                elif is_anchor(target):
                    # A file, PR, skill or domain. Not a note, but a real node, # and the join that makes multi-hop queries work, because two
                    # sessions that touched the same file connect through it.
                    if target not in anchors:
                        kind = anchor_kind(target)
                        anchors[target] = NoteRef(
                            id=target,
                            title=target[len(kind) + 1:].replace("-", " "),
                            type=f"anchor:{kind}",
                            path="",
                            status="anchor",
                        )
                    graph.edges.append(record)
                else:
                    graph.unresolved.append(record)
        graph.nodes.extend(anchors.values())
        return graph

    def search_notes(self, query: str, *, limit: int = 20) -> list[SearchHit]:
        """Full-text search. Uses FTS5 when an index is attached, else scans."""
        if self.index is not None:
            hits: list[SearchHit] = []
            for note_id, snippet, rank, similarity in self.index.search(query, limit):
                try:
                    note = self.get_note(note_id)
                except NoteNotFoundError:
                    continue
                hits.append(SearchHit(ref=self._ref(note), answer=note.answer,
                                      snippet=snippet, rank=rank,
                                      similarity=similarity))
            return hits

        needle = query.lower()
        out: list[SearchHit] = []
        for note in self.iter_notes():
            fm = note.frontmatter
            # Same rule as the FTS path: archived means retired, and retired
            # knowledge must not be served. Both paths need it -- this one is
            # what `--no-index` runs.
            if fm.status in (NoteStatus.ARCHIVED, NoteStatus.DELETED):
                continue
            haystacks = [fm.title, *fm.aliases, *fm.tags]
            score = 3.0 if any(needle in h.lower() for h in haystacks) else 0.0
            body_text = note.body.decode("utf-8", "replace")
            if needle in body_text.lower():
                score += 1.0
            if score:
                idx = body_text.lower().find(needle)
                window = body_text[max(0, idx - 60): idx + 120] if idx >= 0 else ""
                snippet = " ".join(window.split())
                out.append(SearchHit(ref=self._ref(note), answer=note.answer,
                                     snippet=snippet, rank=score))
            if len(out) >= limit * 4:
                break
        out.sort(key=lambda h: -h.rank)
        return out[:limit]

    # ---- index ------------------------------------------------------------

    def reindex(self, *, full: bool = False) -> IndexStats:
        if self.index is None:
            return IndexStats()
        stats = self.index.sync(self.vault, full=full)
        self.audit.record("reindex", None, full=full, stats=stats.to_dict())
        return stats

    # ---- internals --------------------------------------------------------

    def refresh_index(self) -> None:
        if self.index is not None:
            self.index.sync(self.vault)

    def _write_edges(self, note: Note, edges: list[str], op: str, **detail: Any) -> Note:
        raw, stamp = self.vault.read(note.path)

        def mutate(data: Any) -> None:
            data["relationships"] = edges

        new_raw = splice(raw, mutate)
        self.vault.write(note.path, new_raw, expect=stamp)
        self.audit.record(op, note.id, **detail)
        self.refresh_index()
        return parse_note(note.path, new_raw)

    @staticmethod
    def _edges(items: list[str] | None) -> list[Relationship]:
        from brain.domain.identity import parse_edge

        out: list[Relationship] = []
        for item in items or []:
            parsed = parse_edge(item)
            if parsed is None:
                raise ValidationError(f"malformed relationship: {item!r} (expected rel::target)")
            out.append(Relationship(*parsed))
        return out

    def _ref(self, note: Note) -> NoteRef:
        return NoteRef(
            id=note.id,
            title=note.frontmatter.title,
            type=note.frontmatter.type.value,
            path=self.vault.relative(note.path),
            status=note.frontmatter.status.value,
        )
