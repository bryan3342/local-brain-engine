"""Stage 6: turn a retrieval failure into a permanent improvement.

One correction does two things at once. It becomes a golden case, so the failure
can never silently return, and it becomes an alias, so the note is findable by
the words that actually failed. That is the whole self-improvement loop, and it
only ever *adds* aliases, it never rewrites what you wrote.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from brain.application.evaluate import GOLDEN_PATH, evaluate, load_cases
from brain.application.knowledge_api import KnowledgeApi
from brain.domain.errors import NoteNotFoundError
from brain.domain.models import EvalCase
from brain.infrastructure.vault import append_jsonl, atomic_write


def teach(
    api: KnowledgeApi,
    query: str,
    note_id: str,
    golden_path: Path | None = None,
) -> dict[str, Any]:
    """Record that `query` should retrieve `note_id`.

    Raises NoteNotFoundError if the note does not exist, rather than writing a
    golden case that can never pass.
    """
    note = api.get_note(note_id)
    path = golden_path or (api.vault.root / GOLDEN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    already = False
    if path.is_file():
        for case in load_cases(path):
            if case.query.strip().lower() == query.strip().lower():
                already = True
                break
    if not already:
        append_jsonl(path, json.dumps(
            {"q": query, "expect": [note.id], "note": "learned"}, ensure_ascii=False))

    aliases = list(note.frontmatter.aliases)
    if query not in aliases:
        api.update_note(note.id, aliases=[*aliases, query])
        api.refresh_index()

    report = evaluate(api, load_cases(path))
    return {"query": query, "note": note.id, "already_known": already,
            **report.to_dict()}


def coverage(api: KnowledgeApi, cases: Iterable[EvalCase], limit: int = 5) -> list[str]:
    """Note ids that no golden query retrieves, searching `limit` deep.

    Advisory only. A note here is either dead weight or badly aliased, and only
    a human can say which.

    Inbox notes are excluded. They are proposals awaiting a human, not knowledge
    that failed to be found, counting them would bury the real signal under
    every candidate still waiting on triage.
    """
    reached: set[str] = set()
    for case in cases:
        for hit in api.search_notes(case.query, limit=limit):
            reached.add(hit.ref.id)
    inbox = {ref.id for ref in api.list_notes(status="inbox")}
    return sorted({ref.id for ref in api.list_notes()} - reached - inbox)


def revert_session(api: KnowledgeApi, session: str) -> dict[str, Any]:
    """Remove exactly the aliases one ETL session added, and nothing else.

    The counterpart to unattended enrichment. The ETL is allowed to enrich a
    trusted note without asking because an alias is additive and reversible --
    this is the half that makes "reversible" true rather than aspirational.

    Surgical by construction: it removes only alias strings the audit log
    records that session adding, so authored aliases and other batches' work
    survive. Idempotent, because an alias already gone is simply not found.
    """
    removed = 0
    touched: set[str] = set()
    for entry in api.audit.entries("etl_enrich"):
        if entry.get("session") != session:
            continue
        note_id = entry.get("id")
        added = entry.get("aliases_added") or []
        if not note_id or not added:
            continue
        try:
            note = api.get_note(note_id)
        except NoteNotFoundError:
            continue
        drop = {a.lower() for a in added}
        keep = [a for a in note.frontmatter.aliases if a.lower() not in drop]
        if len(keep) == len(note.frontmatter.aliases):
            continue
        removed += len(note.frontmatter.aliases) - len(keep)
        touched.add(note_id)
        api.update_note(note_id, aliases=keep)

    if touched:
        api.audit.record("etl_revert", session=session, notes=sorted(touched),
                         aliases_removed=removed)
    return {"session": session, "notes_touched": len(touched), "aliases_removed": removed}


#: Where the last accepted retrieval score lives. Kept in the vault, not the
#: index, because it is a decision about quality -- not derived data that
#: `reindex` may rebuild.
BASELINE_PATH = "90-meta/eval/baseline.json"


def _baseline_file(api: KnowledgeApi) -> Path:
    return api.vault.root / BASELINE_PATH


def save_baseline(api: KnowledgeApi, cases: Iterable[EvalCase]) -> dict[str, Any]:
    """Record the current retrieval score as the bar to beat.

    Saving is deliberately a separate, explicit act: a baseline that updated
    itself on every run would ratchet downward silently, which is exactly the
    failure it exists to catch.
    """
    report = evaluate(api, list(cases))
    payload = {
        "precision_at_1": report.precision_at_1,
        "recall_at_k": report.recall_at_k,
        "cases": report.total,
        # Per-query outcomes, not just the aggregate. The same queries run every
        # time, so a comparison is paired -- and a paired test needs to know
        # WHICH queries changed, not merely how many.
        "hits_at_1": {r.case.query: bool(r.hit_at_1) for r in report.results},
        "saved_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }
    path = _baseline_file(api)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    api.audit.record("eval_baseline", None, **payload)
    return payload


def read_baseline(api: KnowledgeApi) -> dict[str, Any] | None:
    path = _baseline_file(api)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def check_against_baseline(api: KnowledgeApi, cases: Iterable[EvalCase]) -> dict[str, Any]:
    """Score now, and say whether it is worse than the last accepted score.

    This is the gate that lets enrichment run unattended: the ETL may add
    aliases without asking, and this is what notices when a batch made
    retrieval worse. With no baseline it reports no regression rather than
    inventing one -- a false alarm teaches people to ignore the real ones.
    """
    report = evaluate(api, list(cases))
    base = read_baseline(api)
    base_p = base.get("precision_at_1") if base else None
    regressed = base_p is not None and report.precision_at_1 < base_p

    # Paired comparison. A raw delta is not evidence at this sample size: with
    # 35 queries the 95% CI on precision@1 is ~31 points wide, so one flipped
    # question moves the number 2.9pp and means nothing. McNemar's exact test
    # asks the only answerable question -- of the queries whose outcome CHANGED,
    # is the split lopsided enough to be more than chance?
    prior = (base or {}).get("hits_at_1") or {}
    now = {r.case.query: bool(r.hit_at_1) for r in report.results}
    improved = sum(1 for q, hit in now.items() if hit and not prior.get(q, hit))
    worsened = sum(1 for q, hit in now.items() if not hit and prior.get(q, hit))
    p_value = _mcnemar_exact(improved, worsened)

    return {
        "precision_at_1": report.precision_at_1,
        "recall_at_k": report.recall_at_k,
        "cases": report.total,
        "baseline_precision_at_1": base_p,
        "baseline_recall_at_k": base.get("recall_at_k") if base else None,
        "regressed": regressed,
        "delta": (report.precision_at_1 - base_p) if base_p is not None else None,
        "improved": improved,
        "worsened": worsened,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _mcnemar_exact(improved: int, worsened: int) -> float:
    """Two-sided exact McNemar p-value for a paired win/loss split.

    Only discordant pairs carry information: queries that were right before and
    after say nothing about which config is better. Under the null the split of
    n discordant pairs is Binomial(n, 0.5), so this is an exact binomial test --
    no scipy, no approximation, and correct for the tiny n it will actually see.
    """
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))
