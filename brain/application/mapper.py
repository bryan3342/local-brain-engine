"""Frontmatter mapping <-> domain models.

An addition beyond the review's module layout, for the same reason `identity.py`
was added: it isolates the one place where loose YAML becomes typed data, so
tolerance lives here rather than being sprinkled through the API.

Reading is deliberately forgiving (C2. Obsidian and the human co-author these
files). Writing touches only managed keys, leaving every other key, its order,
and its quoting exactly as found.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from brain.domain.enums import NoteStatus, NoteType
from brain.domain.errors import ValidationError
from brain.domain.identity import parse_edge
from brain.domain.models import KNOWN_FIELDS, Frontmatter, Note, Relationship
from brain.infrastructure.markdown import (
    SplitDoc,
    as_list,
    coerce_date,
    dump_frontmatter,
    join,
    load_frontmatter,
    split,
)

#: Key order used when creating a note. Existing notes keep whatever order they
#: already have, reordering someone's file is churn, not a fix.
CANONICAL_ORDER: tuple[str, ...] = (
    "schema_version", "id", "title", "type", "domain", "status",
    "tags", "aliases", "source", "relationships", "created", "updated",
)


def frontmatter_from_mapping(data: Any) -> Frontmatter:
    """Build a `Frontmatter` from a parsed YAML mapping.

    Unknown `type`/`status` values raise rather than silently defaulting: a typo
    like `type: refernce` should surface as a finding, not quietly become a
    concept.
    """
    note_id = str(data.get("id") or "").strip()
    if not note_id:
        raise ValidationError("missing required field: id")

    raw_type = str(data.get("type") or "").strip()
    try:
        note_type = NoteType(raw_type)
    except ValueError as exc:
        raise ValidationError(f"unknown type: {raw_type!r}") from exc

    raw_status = str(data.get("status") or NoteStatus.DRAFT.value).strip()
    try:
        status = NoteStatus(raw_status)
    except ValueError as exc:
        raise ValidationError(f"unknown status: {raw_status!r}") from exc

    relationships: list[Relationship] = []
    for item in as_list(data.get("relationships")):
        parsed = parse_edge(item)
        if parsed is not None:
            relationships.append(Relationship(*parsed))

    source = data.get("source")
    if not isinstance(source, dict):
        source = {"value": str(source)} if source is not None else {}

    try:
        schema_version = int(data.get("schema_version") or 1)
    except (TypeError, ValueError):
        schema_version = 1

    return Frontmatter(
        id=note_id,
        title=str(data.get("title") or note_id).strip().strip('"'),
        type=note_type,
        status=status,
        schema_version=schema_version,
        domain=(str(data["domain"]).strip() if data.get("domain") else None),
        tags=as_list(data.get("tags")),
        aliases=as_list(data.get("aliases")),
        source=dict(source),
        relationships=relationships,
        created=coerce_date(data.get("created")),
        updated=coerce_date(data.get("updated")),
        extra={k: v for k, v in data.items() if k not in KNOWN_FIELDS},
    )


def parse_note(path: Path, raw: bytes) -> Note:
    """Parse a file into a `Note`. Raises `ValidationError` if it is not one."""
    doc = split(raw)
    if doc.frontmatter_text is None:
        raise ValidationError("no frontmatter block")
    data = load_frontmatter(doc.frontmatter_text)
    return Note(frontmatter=frontmatter_from_mapping(data), body=doc.body, path=path)


def apply_to_mapping(fm: Frontmatter, data: Any) -> None:
    """Write managed fields back into an existing mapping, in place.

    Only the managed keys are assigned. Anything else the file carries, a
    `module:`, a `project:`, a hand-added field, is left untouched, as is the
    key order ruamel captured.
    """
    data["schema_version"] = fm.schema_version
    data["id"] = fm.id
    data["title"] = fm.title
    data["type"] = fm.type.value
    if fm.domain is not None:
        data["domain"] = fm.domain
    data["status"] = fm.status.value
    data["tags"] = list(fm.tags)
    data["aliases"] = list(fm.aliases)
    if fm.source:
        data["source"] = dict(fm.source)
    data["relationships"] = [r.encode() for r in fm.relationships]
    if fm.created is not None:
        data["created"] = fm.created.isoformat()
    if fm.updated is not None:
        data["updated"] = fm.updated.isoformat()


def render_new_note(fm: Frontmatter, body: bytes) -> bytes:
    """Serialize a brand-new note with canonical key order.

    Only ever used at creation. Updates go through `splice`, which preserves the
    file's own ordering.
    """
    data = load_frontmatter("{}")
    apply_to_mapping(fm, data)

    ordered = load_frontmatter("{}")
    for key in CANONICAL_ORDER:
        if key in data:
            ordered[key] = data[key]
    for key in data:
        if key not in ordered:
            ordered[key] = data[key]

    doc = SplitDoc(prefix=b"", frontmatter_text="", body=body, newline=b"\n")
    return join(doc, dump_frontmatter(ordered))


def today() -> _dt.date:
    return _dt.date.today()
