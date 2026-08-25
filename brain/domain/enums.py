"""Controlled vocabularies.

Extensible by design: `NoteType` and `RelationshipType` mirror what the vault
actually uses today. Adding a member is a one-line change plus a
`schema_version` bump only if the *encoding* changes, not the vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class NoteType(StrEnum):
    """What a note *is*. Authoritative, the folder is a cosmetic default (M1)."""

    CONCEPT = "concept"
    PROCEDURE = "procedure"
    REFERENCE = "reference"
    RESEARCH = "research"
    DECISION = "decision"
    SESSION_REPORT = "session-report"
    FEEDBACK = "feedback"
    PROJECT = "project"


class NoteStatus(StrEnum):
    """Lifecycle. `DELETED` is soft-delete only (M3), the file moves to trash."""

    INBOX = "inbox"
    DRAFT = "draft"
    EVERGREEN = "evergreen"
    ARCHIVED = "archived"
    DELETED = "deleted"


class RelationshipType(StrEnum):
    """Typed edges. Encoded as flat `type::target` strings, never nested YAML (H1d)."""

    PART_OF = "part_of"
    MENTIONS = "mentions"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    TOUCHED = "touched"
    SHIPPED = "shipped"
    USED_SKILL = "used_skill"
    HIT = "hit"


#: Folder each type lands in at create time. Cosmetic only, never infer type
#: from a path, and `update_note` must never move a file because of a retype.
DEFAULT_FOLDER: dict[NoteType, str] = {
    NoteType.CONCEPT: "10-knowledge",
    NoteType.PROCEDURE: "10-knowledge",
    NoteType.REFERENCE: "10-knowledge",
    NoteType.RESEARCH: "40-research",
    NoteType.DECISION: "10-knowledge",
    NoteType.SESSION_REPORT: "20-projects/sessions",
    NoteType.FEEDBACK: "90-meta/profile",
    NoteType.PROJECT: "20-projects",
}
