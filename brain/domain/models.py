"""Domain models. Pure data, no I/O, no formatting.

Every model is JSON-serializable via `to_dict()`, because the Phase-2 seam is an
MCP server: each Knowledge API method becomes a tool, and its return value must
survive a JSON round-trip without a custom encoder.

`Note.body` is `bytes` on purpose. It is an **opaque byte range**, never parsed,
never re-rendered, only ever copied verbatim. That is what makes byte
preservation provable rather than hoped-for.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.domain.enums import NoteStatus, NoteType
from brain.domain.errors import ValidationError
from brain.domain.identity import format_edge

#: Fields that may never change after creation.
IMMUTABLE_FIELDS: frozenset[str] = frozenset({"id", "created"})

#: Fields the API manages explicitly; everything else in the frontmatter is
#: preserved untouched under `Frontmatter.extra`.
KNOWN_FIELDS: frozenset[str] = frozenset({
    "schema_version", "id", "title", "type", "domain", "status",
    "tags", "aliases", "source", "relationships", "created", "updated",
})


@dataclass(frozen=True, slots=True)
class Relationship:
    """One typed edge, stored flat as `rel::target`."""

    rel: str
    target: str

    def encode(self) -> str:
        return format_edge(self.rel, self.target)

    def to_dict(self) -> dict[str, str]:
        return {"rel": self.rel, "target": self.target}


@dataclass
class Frontmatter:
    """The structured half of a note.

    `extra` holds every key the API does not manage, so a hand-written or
    Obsidian-written field survives a splice untouched.
    """

    id: str
    title: str
    type: NoteType
    status: NoteStatus
    schema_version: int = 1
    domain: str | None = None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    created: _dt.date | None = None
    updated: _dt.date | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "domain": self.domain,
            "status": self.status.value,
            "tags": list(self.tags),
            "aliases": list(self.aliases),
            "source": dict(self.source),
            "relationships": [r.to_dict() for r in self.relationships],
            "created": self.created.isoformat() if self.created else None,
            "updated": self.updated.isoformat() if self.updated else None,
            **{k: v for k, v in self.extra.items() if k not in KNOWN_FIELDS},
        }


@dataclass
class Note:
    """A note as it exists on disk."""

    frontmatter: Frontmatter
    body: bytes
    path: Path

    @property
    def id(self) -> str:
        return self.frontmatter.id

    @property
    def answer(self) -> str | None:
        """The `> **Answer:**` block, unwrapped to a single line.

        This is the payload a grep hit is supposed to return, so it is worth
        having as a first-class projection rather than a caller's regex.
        """
        import re

        match = re.search(rb"^>\s*\*\*Answer:\*\*\s*(.+(?:\n>.*)*)", self.body, re.M)
        if not match:
            return None
        text = match.group(1).decode("utf-8", "replace")
        return " ".join(re.sub(r"^\s*>\s?", "", text, flags=re.M).split())

    def to_dict(self, include_body: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": str(self.path),
            "answer": self.answer,
            **self.frontmatter.to_dict(),
        }
        if include_body:
            out["body"] = self.body.decode("utf-8", "replace")
        return out


@dataclass(frozen=True, slots=True)
class NoteRef:
    """A lightweight pointer, for listings and graph nodes."""

    id: str
    title: str
    type: str
    path: str
    status: str = NoteStatus.EVERGREEN.value

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "type": self.type,
                "path": self.path, "status": self.status}


@dataclass(frozen=True, slots=True)
class SearchHit:
    ref: NoteRef
    answer: str | None
    snippet: str
    rank: float
    #: Cosine similarity to the query, or None when no dense retrieval ran.
    #: None rather than 0.0 on purpose: a gate reading an unmeasured signal as
    #: "definitely irrelevant" would go permanently silent wherever no model is
    #: reachable, which is exactly the case that must degrade gracefully.
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.ref.to_dict(), "answer": self.answer,
                "snippet": self.snippet, "rank": self.rank,
                "similarity": self.similarity}


@dataclass(frozen=True, slots=True)
class Backlink:
    """An inbound edge: `source` points at the queried note via `rel`."""

    source: NoteRef
    rel: str

    def to_dict(self) -> dict[str, Any]:
        return {"rel": self.rel, "source": self.source.to_dict()}


@dataclass
class GraphExport:
    """Nodes and edges. Renderer-agnostic on purpose, no Sigma/Cytoscape shape."""

    nodes: list[NoteRef] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": list(self.edges),
            "unresolved": list(self.unresolved),
        }


@dataclass
class Finding:
    """One thing `scan`/`doctor` noticed about one file."""

    path: str
    kind: str          # untracked | invalid-frontmatter | missing-field | duplicate-id | ...
    detail: str
    repairable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind,
                "detail": self.detail, "repairable": self.repairable}


@dataclass
class ScanReport:
    """Two-writer reconciliation result. Tolerant by design (C2): a hand-written
    file is a *finding*, never an exception that kills the whole sweep."""

    valid: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def quarantined(self) -> int:
        return sum(1 for f in self.findings if not f.repairable)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "repairable": sum(1 for f in self.findings if f.repairable),
                "quarantined": self.quarantined,
                "findings": [f.to_dict() for f in self.findings]}


@dataclass
class DoctorReport:
    """What `doctor` did, or would do with `--dry-run` off."""

    dry_run: bool = True
    repaired: list[Finding] = field(default_factory=list)
    skipped: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"dry_run": self.dry_run,
                "repaired": [f.to_dict() for f in self.repaired],
                "skipped": [f.to_dict() for f in self.skipped]}


@dataclass
class IndexStats:
    notes: int = 0
    edges: int = 0
    vectors: int = 0
    reindexed: int = 0
    unchanged: int = 0
    removed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"notes": self.notes, "edges": self.edges, "vectors": self.vectors,
                "reindexed": self.reindexed, "unchanged": self.unchanged,
                "removed": self.removed}


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One golden question and the note(s) that should answer it.

    `expect` is a list because some questions are legitimately answered by more
    than one note; a hit on any of them counts.
    """

    query: str
    expect: tuple[str, ...]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "expect": list(self.expect), "note": self.note}


@dataclass(frozen=True, slots=True)
class EvalResult:
    """What retrieval actually did for one case.

    `rank` is 1-based, or None for a miss. `top` is recorded even on a hit,
    because a correct answer at rank 4 behind three irrelevant notes is a
    different problem from a clean rank-1 hit, and both are worth seeing.
    """

    case: EvalCase
    rank: int | None
    returned: tuple[str, ...]

    @property
    def hit_at_1(self) -> bool:
        return self.rank == 1

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.case.to_dict(),
            "rank": self.rank,
            "hit@1": self.hit_at_1,
            "returned": list(self.returned),
        }


@dataclass
class EvalReport:
    """Aggregate retrieval quality. The number that has to move."""

    k: int = 5
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hits_at_1(self) -> int:
        return sum(1 for r in self.results if r.hit_at_1)

    @property
    def hits_at_k(self) -> int:
        return sum(1 for r in self.results if r.rank is not None)

    @property
    def precision_at_1(self) -> float:
        return self.hits_at_1 / self.total if self.total else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.hits_at_k / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        if not self.total:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.total

    @property
    def misses(self) -> list[EvalResult]:
        return [r for r in self.results if r.rank is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "cases": self.total,
            "precision@1": round(self.precision_at_1, 4),
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "misses": [r.to_dict() for r in self.misses],
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """One exchange that survived the deterministic pre-filter.

    `candidate_id` is a content hash, not a database id, so the same text
    arriving twice from different sessions is recognised as a duplicate.
    """

    candidate_id: str
    session_id: str
    project: str
    timestamp: str
    user_text: str
    assistant_text: str
    tool_summary: dict[str, int] = field(default_factory=dict, hash=False)
    branches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "project": self.project,
            "timestamp": self.timestamp,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "tool_summary": dict(self.tool_summary),
            "branches": list(self.branches),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        candidate_id = str(data["candidate_id"])
        user_text = str(data.get("user_text", "")).strip()
        if not user_text:
            raise ValidationError(
                f"candidate {candidate_id}: user_text is required and must not be blank"
            )
        return cls(
            candidate_id=candidate_id,
            session_id=str(data.get("session_id", "")),
            project=str(data.get("project", "")),
            timestamp=str(data.get("timestamp", "")),
            user_text=user_text,
            assistant_text=str(data.get("assistant_text", "")),
            tool_summary=dict(data.get("tool_summary") or {}),
            branches=tuple(data.get("branches") or ()),
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    """Why a candidate did not become a note.

    Never pruned. Without this you cannot tell over-filtering from
    under-filtering, which is the blindness that made 909 stored conclusions
    unusable.
    """

    candidate_id: str
    stage: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "stage": self.stage,
                "reason": self.reason, "detail": self.detail}
