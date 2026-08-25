"""Two-writer reconciliation: `scan` reports, `doctor` repairs.

The spec's "all writes go through the API" boundary is a fiction, and pretending
otherwise is what C2 flagged. Obsidian writes; the human writes; this package
writes. Three writers, and until now zero reconciliation story.

So the boundary is restated honestly: the Knowledge API is the only writer that
*maintains invariants*, and this module converges everything else. It is a
one-shot pass the user runs, not a daemon and not continuous crawling, both of
which the spec excludes, and neither of which this is.

`scan` never raises on a bad file. A malformed note is a finding; the sweep
continues.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from pathlib import Path
from typing import Any

from brain.application.knowledge_api import KnowledgeApi
from brain.application.mapper import apply_to_mapping, parse_note
from brain.application.validators import validate
from brain.domain.enums import NoteStatus, NoteType
from brain.domain.errors import FrontmatterError, ValidationError
from brain.domain.identity import is_anchor, mint_id
from brain.domain.models import DoctorReport, Finding, Frontmatter, ScanReport
from brain.infrastructure.markdown import (
    dump_frontmatter,
    join,
    load_frontmatter,
    splice,
    split,
)

#: Kinds `doctor` knows how to fix without a human deciding anything.
REPAIRABLE = {"no-frontmatter", "missing-schema-version", "missing-status", "missing-id"}


class Reconciler:
    def __init__(self, api: KnowledgeApi) -> None:
        self.api = api
        self.vault = api.vault

    # ---- scan -------------------------------------------------------------

    def scan(self, *, include_advisory: bool = False) -> ScanReport:
        report = ScanReport()
        ids: dict[str, list[str]] = defaultdict(list)
        known_ids: set[str] = set()
        alias_ids: set[str] = set()
        edges: list[tuple[str, str, str]] = []

        for path in self.vault.note_paths():
            rel = self.vault.relative(path)
            try:
                raw = path.read_bytes()
            except OSError as exc:
                report.findings.append(Finding(rel, "unreadable", str(exc)))
                continue

            doc = split(raw)
            if not doc.has_frontmatter:
                report.findings.append(Finding(
                    rel, "no-frontmatter",
                    "hand-written file the API never saw; doctor can adopt it",
                    repairable=True))
                continue

            try:
                data = load_frontmatter(doc.frontmatter_text or "")
            except FrontmatterError as exc:
                report.findings.append(Finding(rel, "unparseable-frontmatter", str(exc)))
                continue

            if not str(data.get("id") or "").strip():
                report.findings.append(Finding(
                    rel, "missing-id", "no id; doctor can mint one", repairable=True))
                continue
            if "schema_version" not in data:
                report.findings.append(Finding(
                    rel, "missing-schema-version", "doctor can add schema_version: 1",
                    repairable=True))
            if "status" not in data:
                report.findings.append(Finding(
                    rel, "missing-status", "doctor can add status: draft", repairable=True))

            try:
                note = parse_note(path, raw)
            except ValidationError as exc:
                report.findings.append(Finding(rel, "invalid-frontmatter", str(exc)))
                continue

            problems = validate(note.frontmatter)
            if problems:
                report.findings.append(Finding(rel, "invalid-frontmatter", "; ".join(problems)))
                continue

            ids[note.id].append(rel)
            known_ids.add(note.id)
            alias_ids.update(note.frontmatter.aliases)
            edges.extend((note.id, e.rel, e.target) for e in note.frontmatter.relationships)

            # Advisory only (M1): type is authoritative, folder is cosmetic, and
            # `update_note` deliberately never moves a file on retype.
            expected = _default_folder(note.frontmatter.type)
            # Advisory and off by default: folder is cosmetic by design and type
            # is authoritative, so a mismatch is almost never actionable. Left in
            # for anyone deliberately tidying folders.
            if include_advisory and expected and not rel.startswith(expected):
                report.findings.append(Finding(
                    rel, "type-folder-mismatch",
                    f"type {note.frontmatter.type.value} usually lives under {expected}/ "
                    "(advisory, type is authoritative)"))
            report.valid += 1

        for note_id, paths in ids.items():
            if len(paths) > 1:
                report.findings.append(Finding(
                    paths[0], "duplicate-id",
                    f"id {note_id} is claimed by {len(paths)} files: {', '.join(paths)}. "
                    "Resolution is a human decision, never auto-merged."))

        for src, rel_type, dst in edges:
            if dst in known_ids or dst in alias_ids:
                continue
            if is_anchor(dst):
                continue  # anchors are nodes, not dangling references
            report.findings.append(Finding(
                src, "dangling-edge", f"{rel_type}::{dst} resolves to nothing"))

        return report

    # ---- doctor -----------------------------------------------------------

    def doctor(self, *, dry_run: bool = True) -> DoctorReport:
        """Repair what is mechanically repairable. Everything else is reported.

        Deliberately conservative: duplicate ids and unparseable frontmatter are
        never touched, because both need a human to decide what was meant.
        """
        report = DoctorReport(dry_run=dry_run)
        scan = self.scan()
        taken = {n.id for n in self.api.iter_notes(include_deleted=True)}

        for finding in scan.findings:
            if finding.kind not in REPAIRABLE:
                report.skipped.append(finding)
                continue
            path = self.vault.root / finding.path
            if not path.is_file():
                report.skipped.append(finding)
                continue
            if dry_run:
                report.repaired.append(finding)
                continue
            try:
                if finding.kind == "no-frontmatter":
                    self._adopt(path, taken)
                elif finding.kind == "missing-id":
                    self._mint_missing_id(path, taken)
                else:
                    self._fill_defaults(path)
                report.repaired.append(finding)
                self.api.audit.record("doctor_repair", path=finding.path, kind=finding.kind)
            except Exception as exc:
                report.skipped.append(Finding(finding.path, finding.kind,
                                              f"repair failed: {exc}"))
        if not dry_run:
            self.api.refresh_index()
        return report

    # ---- repairs ----------------------------------------------------------

    def _adopt(self, path: Path, taken: set[str]) -> None:
        """Give a hand-written file frontmatter, without touching its body."""
        raw = path.read_bytes()
        doc = split(raw)
        title = _title_from_body(doc.body) or path.stem.replace("-", " ").strip().capitalize()
        note_type = _type_from_path(self.vault.relative(path))
        note_id = mint_id(note_type, title, taken)
        taken.add(note_id)

        fm = Frontmatter(
            id=note_id, title=title, type=note_type, status=NoteStatus.DRAFT,
            domain=_domain_from_path(self.vault.relative(path)),
            source={"type": "adopted", "connector": "doctor",
                    "imported_at": _dt.date.today().isoformat()},
            created=_dt.date.fromtimestamp(path.stat().st_mtime),
        )
        data = load_frontmatter("{}")
        apply_to_mapping(fm, data)
        _, stamp = self.api.vault.read(path)
        self.api.vault.write(path, join(doc, dump_frontmatter(data)), expect=stamp)

    def _mint_missing_id(self, path: Path, taken: set[str]) -> None:
        raw, stamp = self.api.vault.read(path)
        doc = split(raw)
        data = load_frontmatter(doc.frontmatter_text or "")
        title = str(data.get("title") or _title_from_body(doc.body) or path.stem)
        raw_type = str(data.get("type") or "").strip()
        note_type = NoteType(raw_type) if raw_type in set(NoteType) else _type_from_path(
            self.vault.relative(path))
        note_id = mint_id(note_type, title, taken)
        taken.add(note_id)

        def mutate(payload: Any) -> None:
            payload["id"] = note_id
            payload.setdefault("type", note_type.value)
            payload.setdefault("title", title)
            payload.setdefault("schema_version", 1)
            payload.setdefault("status", NoteStatus.DRAFT.value)

        self.api.vault.write(path, splice(raw, mutate), expect=stamp)

    def _fill_defaults(self, path: Path) -> None:
        raw, stamp = self.api.vault.read(path)

        def mutate(payload: Any) -> None:
            payload.setdefault("schema_version", 1)
            payload.setdefault("status", NoteStatus.DRAFT.value)

        self.api.vault.write(path, splice(raw, mutate), expect=stamp)


# ---- heuristics -----------------------------------------------------------


def _title_from_body(body: bytes) -> str | None:
    for line in body.decode("utf-8", "replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _type_from_path(rel: str) -> NoteType:
    """Folder is only ever a *hint*, used when adopting a file that has no type."""
    if rel.startswith("20-projects/sessions"):
        return NoteType.SESSION_REPORT
    if rel.startswith("40-research"):
        return NoteType.RESEARCH
    if rel.startswith("90-meta/profile"):
        return NoteType.FEEDBACK
    if rel.startswith("20-projects"):
        return NoteType.PROJECT
    if rel.startswith("30-work"):
        return NoteType.REFERENCE
    return NoteType.CONCEPT


def _domain_from_path(rel: str) -> str | None:
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[0] == "10-knowledge":
        return parts[1]
    return None


def _default_folder(note_type: NoteType) -> str | None:
    from brain.domain.enums import DEFAULT_FOLDER

    return DEFAULT_FOLDER.get(note_type)

