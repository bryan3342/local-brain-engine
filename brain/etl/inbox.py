"""Stage 5: the human gate.

The listing is ordered Answer-first on purpose. Promotion should take seconds
and the common answer should be no, a queue that is tedious to review becomes a
graveyard, and then the whole pipeline has quietly stopped working.
"""

from __future__ import annotations

from typing import Any

from brain.application.knowledge_api import KnowledgeApi
from brain.domain.enums import DEFAULT_FOLDER, NoteStatus, NoteType
from brain.domain.models import Note, Rejection
from brain.etl.queue import Queue
from brain.etl.reconcile import SupersedeArchiveError

INBOX_FOLDER = "00-inbox"


def list_inbox(api: KnowledgeApi) -> list[dict[str, Any]]:
    """Pending candidates. Answer first, then identity, then provenance."""
    rows: list[dict[str, Any]] = []
    for note in api.iter_notes():
        if note.frontmatter.status is not NoteStatus.INBOX:
            continue
        rows.append({
            "answer": note.answer or "",
            "id": note.id,
            "title": note.frontmatter.title,
            "type": note.frontmatter.type.value,
            "domain": note.frontmatter.domain,
            "source": note.frontmatter.source.get("ref", ""),
        })
    return rows


def promote(api: KnowledgeApi, ref: str, to: str | None = None) -> dict[str, Any]:
    """Move a candidate into a trusted tier and mark it evergreen.

    The id does not change, so anything already pointing at the candidate keeps
    resolving. The move happens *before* the status flip, on purpose: if the
    move fails, nothing has changed and the note is still visible in the
    inbox. If the flip fails after a successful move, the note merely sits in
    its new folder still marked `inbox`, so `list_inbox` still shows it and
    re-running `promote` is safe. Every failure mode leaves the note visible,
    rather than stranding it invisibly inside the untrusted folder.
    """
    note = api.get_note(ref)
    destination = to or DEFAULT_FOLDER[NoteType(note.frontmatter.type.value)]
    moved = api.rename_note(note.id, note.frontmatter.title, dest_folder=destination)
    updated = api.update_note(moved.id, status=NoteStatus.EVERGREEN.value)

    # Promotion is where a supersede is actually carried out. The ETL only
    # recorded the intent as an edge, because a proposal must not retire a
    # verified note; saying yes to the replacement is the human action that
    # earns the archive. Done after the promotion so a failure here leaves the
    # replacement live rather than losing both.
    superseded = _supersedes_targets(updated)
    for target in superseded:
        try:
            api.update_note(target, status=NoteStatus.ARCHIVED.value)
        except Exception as exc:
            raise SupersedeArchiveError(
                f"promoted {updated.id}, but archiving the note it supersedes "
                f"({target}) failed ({exc}); both are now live and evergreen and "
                f"will assert contradictory things -- archive {target} by hand."
            ) from exc

    api.audit.record("inbox_promote", note.id, to=destination, superseded=superseded)
    return {"id": updated.id, "path": api.vault.relative(updated.path),
            "status": updated.frontmatter.status.value, "superseded": superseded}


def _supersedes_targets(note: Note) -> list[str]:
    """Ids this note claims to replace, read from its `supersedes::` edges."""
    out: list[str] = []
    for rel in note.frontmatter.relationships:
        text = rel.encode()
        if text.startswith("supersedes::"):
            out.append(text.split("::", 1)[1])
    return out


def reject(api: KnowledgeApi, queue: Queue, ref: str, reason: str) -> dict[str, Any]:
    """Soft-delete a candidate and record why, so the filter stays auditable."""
    note = api.get_note(ref)
    source_ref = str(note.frontmatter.source.get("ref") or note.id)
    result = api.delete_note(note.id)
    queue.reject([Rejection(candidate_id=source_ref, stage="promotion", reason=reason,
                            detail=note.frontmatter.title)])
    api.audit.record("inbox_reject", note.id, reason=reason)
    return {"id": note.id, "rejected": True, "reason": reason, **result}
