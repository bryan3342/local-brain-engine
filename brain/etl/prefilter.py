"""Stage 1: drop what cannot be signal, by fixed rules and for free.

Deliberately permissive. A false drop is invisible and permanent; a false keep
costs one line of skill judgement. When in doubt, keep.

Every decision emits either a Candidate or a Rejection, never neither, never
both. That total accounting is what makes the filter auditable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from brain.domain.identity import (
    NO_PROJECT_RULES,
    ProjectRules,
    normalize_project,
    project_from_cwd,
)
from brain.domain.models import Candidate, Rejection

#: Below this, a user message cannot carry a durable, reusable claim.
MIN_USER_CHARS = 40

#: Markers that identify harness plumbing rather than something a human typed.
_TOOL_MARKERS = (
    "<task-notification>", "<bash-stdout>", "<bash-stderr>",
    "<system-reminder>", "<local-command-stdout>", "<command-name>",
    "[Image:", "[Request interrupted",
)

#: Harness-injected preamble text that arrives as the "user" turn but was
#: never typed by a human, skill/system scaffolding prose, not tool output.
#: Distinct from `_TOOL_MARKERS` (and checked first in `classify`): one entry,
#: `<system-reminder>`, overlaps a `_TOOL_MARKERS` entry, and must resolve to
#: reason "injected-preamble" rather than the older, less specific
#: "tool-output-only".
_INJECTED_PREAMBLE_MARKERS = (
    "base directory for this skill:",
    "<system-reminder>",
    "caveat: the messages below were generated",
)
_WS = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Hash normalized text, so whitespace differences are not new content."""
    return "sha256:" + hashlib.sha256(
        _WS.sub(" ", text).strip().encode("utf-8")).hexdigest()


def _is_tool_output(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(m) for m in _TOOL_MARKERS)


def _is_injected_preamble(text: str) -> bool:
    stripped = text.lstrip().lower()
    return any(stripped.startswith(m) for m in _INJECTED_PREAMBLE_MARKERS)


def classify(
    row: Mapping[str, Any],
    seen: set[str],
    reported: set[str],
    rules: ProjectRules = NO_PROJECT_RULES,
) -> tuple[Candidate | None, Rejection | None]:
    """Decide one exchange. Returns exactly one of (candidate, rejection)."""
    exchange_id = str(row.get("id") or "")
    user_text = str(row.get("user_message") or "")
    session_id = str(row.get("session_id") or "")

    def drop(reason: str, detail: str = "") -> tuple[None, Rejection]:
        return None, Rejection(
            candidate_id=content_hash(user_text) if user_text.strip()
            else f"exchange:{exchange_id}",
            stage="prefilter", reason=reason, detail=detail)

    if not user_text.strip():
        return drop("no-human-content")
    if _is_injected_preamble(user_text):
        return drop("injected-preamble", user_text[:80])
    if _is_tool_output(user_text):
        return drop("tool-output-only", user_text[:80])
    if len(user_text.strip()) < MIN_USER_CHARS:
        return drop("too-short", f"{len(user_text.strip())} chars")
    if session_id and session_id in reported:
        return drop("session-already-reported", session_id)

    digest = content_hash(user_text)
    if digest in seen:
        return drop("duplicate-content")

    branch = str(row.get("git_branch") or "")
    cwd = str(row.get("cwd") or "")
    project = (
        project_from_cwd(cwd, rules) if cwd.strip()
        else normalize_project(str(row.get("project") or ""), rules)
    )
    return Candidate(
        candidate_id=digest,
        session_id=session_id,
        project=project,
        timestamp=str(row.get("timestamp") or ""),
        user_text=user_text,
        assistant_text=str(row.get("assistant_message") or ""),
        tool_summary=dict(row.get("tool_summary") or {}),
        branches=(branch,) if branch and branch not in ("", "HEAD") else (),
    ), None


def prefilter(
    rows: Iterable[Mapping[str, Any]],
    seen: set[str],
    reported: set[str],
    rules: ProjectRules = NO_PROJECT_RULES,
) -> tuple[list[Candidate], list[Rejection]]:
    """Classify a batch. `seen` grows as we go, so duplicates inside one batch
    are caught too."""
    kept: list[Candidate] = []
    dropped: list[Rejection] = []
    running = set(seen)
    for row in rows:
        candidate, rejection = classify(row, running, reported, rules)
        if not ((candidate is None) ^ (rejection is None)):
            row_id = str(row.get("id") or "unknown")
            msg = "classify must return exactly one of (Candidate, Rejection), "
            msg += f"not both or neither; row id: {row_id}"
            raise ValueError(msg)
        if candidate is not None:
            kept.append(candidate)
            running.add(candidate.candidate_id)
        elif rejection is not None:
            dropped.append(rejection)
    return kept, dropped
