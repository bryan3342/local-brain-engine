"""Typed error hierarchy.

Every Knowledge API method raises one of these and never prints. The CLI maps
them to exit codes; a future MCP adapter maps them to structured tool errors.
That mapping is the whole reason the hierarchy is typed rather than a bag of
`ValueError`s.
"""

from __future__ import annotations


class BrainError(Exception):
    """Base for everything this package raises deliberately."""


class VaultError(BrainError):
    """The vault itself is missing, malformed, or unusable."""


class NoteNotFoundError(BrainError):
    """No note resolves for the given id, alias, or path."""


class CandidateNotFoundError(BrainError):
    """No pending candidate matches the given id.

    A candidate lives in the ETL queue, not the vault -- it is not a note
    until `etl-draft` writes one, so this is distinct from NoteNotFoundError.
    """


class DuplicateNoteError(BrainError):
    """A note with the same (type, normalized title) already exists.

    Surfaced as a soft conflict for a human to resolve, never auto-merged (M2).
    """


class ValidationError(BrainError):
    """Frontmatter failed schema validation."""


class ConcurrentModificationError(BrainError):
    """The file changed on disk between read and write.

    Obsidian and the human are co-authors; this is the guard that stops a
    read-modify-write from clobbering their edit (C3).
    """


class FrontmatterError(BrainError):
    """The frontmatter block is unparseable or unterminated."""


class ImmutableFieldError(BrainError):
    """Attempted to change a field that is frozen for the life of the note.

    `id` and `created` are minted once. Re-deriving `id` from a changed title
    is the specific bug this exists to prevent (C1).
    """


class LockError(BrainError):
    """Could not acquire the vault lock."""
