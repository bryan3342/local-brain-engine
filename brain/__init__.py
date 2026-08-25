"""Brain, a local-first personal knowledge OS.

The Obsidian Markdown vault is canonical. Indexes, backlinks, graph exports and
any future embeddings are derived and disposable.
"""

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.enums import NoteStatus, NoteType, RelationshipType
from brain.domain.errors import (
    BrainError,
    ConcurrentModificationError,
    DuplicateNoteError,
    FrontmatterError,
    ImmutableFieldError,
    LockError,
    NoteNotFoundError,
    ValidationError,
    VaultError,
)

__all__ = [
    "BrainConfig", "BrainError", "ConcurrentModificationError", "DuplicateNoteError",
    "FrontmatterError", "ImmutableFieldError", "KnowledgeApi", "LockError",
    "NoteNotFoundError", "NoteStatus", "NoteType", "RelationshipType",
    "ValidationError", "VaultError",
]
__version__ = "0.1.0"
