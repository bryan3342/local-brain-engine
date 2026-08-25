"""Metadata validation and duplicate detection.

Folded together per the review: each was one function, and a module per function
is abstraction tax.
"""

from __future__ import annotations

from collections.abc import Iterable

from brain.domain.identity import dedupe_key, is_valid_id
from brain.domain.models import Frontmatter

REQUIRED = ("schema_version", "id", "title", "type", "status")


def validate(fm: Frontmatter) -> list[str]:
    """Return a list of problems. Empty means valid.

    Returns rather than raises, because `scan` needs to classify a whole vault
    without the first bad file aborting the sweep (C2).
    """
    problems: list[str] = []
    if not fm.id:
        problems.append("missing id")
    elif not is_valid_id(fm.id):
        problems.append(f"malformed id: {fm.id!r}")
    if not fm.title.strip():
        problems.append("missing title")
    if fm.schema_version < 1:
        problems.append(f"bad schema_version: {fm.schema_version}")
    for rel in fm.relationships:
        if not rel.rel or not rel.target:
            problems.append(f"malformed relationship: {rel.encode()!r}")
    if fm.source and not isinstance(fm.source, dict):
        problems.append("source must be a mapping")
    return problems


def find_duplicate(
    fm: Frontmatter,
    existing: Iterable[tuple[str, str, str, list[str]]],
) -> str | None:
    """Return the id of a same-type note with the same normalized title.

    `existing` yields `(id, type, title, aliases)`.

    Scoped by type on purpose (M2): two people named "John Smith" are a real
    collision; a concept "Python" and a project codenamed "Python" are not.
    Aliases are cross-checked within type so "GraphRAG" and "Graph RAG" find
    each other. The result is surfaced as a soft conflict for a human, never
    auto-merged.
    """
    key = dedupe_key(fm.type, fm.title)
    for other_id, other_type, other_title, other_aliases in existing:
        if other_id == fm.id:
            continue
        if other_type != fm.type.value:
            continue
        if dedupe_key(fm.type, other_title) == key:
            return other_id
        for alias in other_aliases:
            if dedupe_key(fm.type, alias) == key:
                return other_id
    return None
