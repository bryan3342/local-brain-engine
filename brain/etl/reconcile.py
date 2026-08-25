"""Stage 3: decide what a drafted note does to what already exists.

Nothing is written blindly. Without consulting the corpus first, the same
fact gets re-derived and stored several times over.

NOOP fires only when every alias the draft carries already matches one on
the matched note. New phrasing of a known fact becomes an UPDATE, since the
phrasing is worth keeping. Neither outcome ever creates a second note for
material that already has one.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from brain.application.knowledge_api import KnowledgeApi
from brain.domain.enums import NoteStatus, NoteType
from brain.domain.errors import BrainError, CandidateNotFoundError
from brain.domain.identity import normalize_title
from brain.domain.models import Candidate
from brain.etl.queue import Queue


class Verdict(StrEnum):
    ADD = "add"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    NOOP = "noop"


@dataclass
class DraftNote:
    """What the skill produced. Not yet a note."""

    title: str
    note_type: str
    domain: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    source_ref: str = ""
    supersedes: str | None = None


@dataclass(frozen=True, slots=True)
class Reconciliation:
    verdict: Verdict
    existing_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "existing_id": self.existing_id,
                "reason": self.reason}


class SupersedeArchiveError(BrainError):
    """A SUPERSEDE created its replacement but failed to archive the original.

    Create-then-archive fails safe (Finding 1 review): the worst case is two
    active notes, never a lost one. So the order stays; what was missing was
    noise. This makes the partial state loud instead of silent, so a human
    can finish the archive by hand.
    """


def _find_equivalent(api: KnowledgeApi, draft: DraftNote) -> str | None:
    """An existing note of the same type whose title or an alias, once
    normalized, matches the drafted title or any of the drafted aliases.

    Searching the drafted title alone left a second line of defence open: a
    fresh title phrased differently from an existing note, but carrying that
    note's exact title as one of its own aliases, sailed through as ADD --
    two notes for one fact (reviewer finding). Both sides of the comparison
    now include aliases.
    """
    drafted = {normalize_title(draft.title), *(normalize_title(a) for a in draft.aliases)}
    for note in api.iter_notes():
        if note.frontmatter.type.value != draft.note_type:
            continue
        existing = {normalize_title(note.frontmatter.title),
                    *(normalize_title(a) for a in note.frontmatter.aliases)}
        if drafted & existing:
            return note.id
    return None


def reconcile(api: KnowledgeApi, draft: DraftNote) -> Reconciliation:
    if draft.supersedes:
        # Deliberately not caught. Downgrading to ADD would drop the stated
        # intent: the contradiction goes unrecorded and the stale note stays
        # live, which is the one outcome supersede exists to prevent. The id is
        # model-composed, so a typo is likely -- and cheap to correct.
        existing = api.get_note(draft.supersedes)
        return Reconciliation(Verdict.SUPERSEDE, existing.id, "explicitly supersedes")

    match = _find_equivalent(api, draft)
    if match is None:
        return Reconciliation(Verdict.ADD, None, "no equivalent note")

    known = {a.lower() for a in api.get_note(match).frontmatter.aliases}
    fresh = [a for a in draft.aliases if a.lower() not in known]
    if fresh:
        return Reconciliation(Verdict.UPDATE, match,
                              f"adds {len(fresh)} alias(es) to an existing note")
    return Reconciliation(Verdict.NOOP, match, "already covered")


def apply(api: KnowledgeApi, draft: DraftNote, decision: Reconciliation) -> dict[str, Any]:
    """Carry out a reconciliation. Returns a JSON-serializable summary."""
    if decision.verdict is Verdict.NOOP:
        return {**decision.to_dict(), "id": decision.existing_id, "wrote": False}

    if decision.verdict is Verdict.UPDATE and decision.existing_id:
        existing = api.get_note(decision.existing_id)
        known = {a.lower() for a in existing.frontmatter.aliases}
        added = [a for a in draft.aliases if a.lower() not in known]
        merged = [*existing.frontmatter.aliases, *added]
        api.update_note(existing.id, aliases=merged)

        # The ETL enriches trusted notes unattended, unlike SUPERSEDE which waits
        # for a human. That asymmetry is deliberate -- an alias is additive and
        # reversible where an archive is destructive -- but "reversible" is only
        # true if we record enough to reverse it. Aliases are the query-shaped
        # retrieval surface, so a bad one costs precision; without this entry the
        # damage would be untraceable and permanent.
        #
        # Attribution goes here rather than onto the note as a `derived_from::`
        # edge: the note was not derived from that session, only one alias was,
        # and a false edge in the graph is worse than none.
        api.audit.record("etl_enrich", existing.id, aliases_added=added,
                         session=draft.source_ref,
                         trusted=existing.frontmatter.status is not NoteStatus.INBOX)
        return {**decision.to_dict(), "id": existing.id, "wrote": True,
                "aliases_added": added}

    relationships = list(draft.relationships)
    if decision.verdict is Verdict.SUPERSEDE and decision.existing_id:
        relationships.append(f"supersedes::{decision.existing_id}")

    note = api.create_note(
        draft.title,
        NoteType(draft.note_type),
        domain=draft.domain or None,
        status=NoteStatus.INBOX,
        aliases=list(draft.aliases),
        tags=list(draft.tags),
        relationships=relationships,
        answer=draft.answer,
        source={"type": "claude-code-session", "ref": draft.source_ref,
                "connector": "brain-etl",
                "imported_at": _dt.date.today().isoformat()},
        folder="00-inbox",
        # reconcile() already made the duplicate call against LIVE notes
        # (_find_equivalent scans iter_notes(), not include_deleted=True).
        # create_note's own guard also checks trashed notes; bypassing it here
        # is deliberate -- a soft-deleted note must never block a legitimate
        # re-add (Finding 2 review).
        allow_duplicate=True,
    )
    # The archive is NOT done here. The replacement is an unreviewed candidate
    # in 00-inbox; letting it retire a verified note would put a proposal in
    # charge of forgetting, and a later reject would leave the vault short a
    # fact with nothing to restore it. The `supersedes::` edge above records the
    # intent, and promote() -- the human action -- carries it out.
    return {**decision.to_dict(), "id": note.id, "wrote": True}


def _find_pending(queue: Queue, candidate_id: str) -> Candidate:
    """The one point where an unknown candidate id becomes a named, typed
    error, before anything has been built or written."""
    for candidate in queue.pending():
        if candidate.candidate_id == candidate_id:
            return candidate
    raise CandidateNotFoundError(f"no candidate {candidate_id!r} in the queue")


def draft_from_candidate(
    api: KnowledgeApi,
    queue: Queue,
    candidate_id: str,
    *,
    title: str,
    note_type: str,
    domain: str,
    answer: str,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    relationships: list[str] | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Turn one queued candidate into a note. Stages 3 and 4.

The caller supplies judgement: title, type, domain, answer, aliases, tags,
relationships. It does not supply placement. `apply()` hardcodes
`folder="00-inbox"` and `status=INBOX`.

Provenance comes from the candidate, not the caller, so notes never ship
with an empty `source`.

The candidate leaves the queue on every verdict including NOOP, but stays
if `reconcile()` rejects the draft, so bad input retries rather than
vanishing.
"""
    candidate = _find_pending(queue, candidate_id)
    edge = f"derived_from::session_{candidate.session_id}"
    draft = DraftNote(
        title=title,
        note_type=note_type,
        domain=domain,
        answer=answer,
        aliases=list(aliases or []),
        tags=list(tags or []),
        relationships=[*(relationships or []), edge],
        source_ref=candidate.session_id,
        supersedes=supersedes,
    )
    decision = reconcile(api, draft)
    result = apply(api, draft, decision)
    queue.drop([candidate_id])
    return result
