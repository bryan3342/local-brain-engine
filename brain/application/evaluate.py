"""Retrieval evaluation.

Turns "is the vault good at answering questions?" into a number that moves,
so enrichment is judged by measurement rather than feel.

The golden set is vault-specific and lives at `90-meta/eval/golden.jsonl`.
This module is the generic harness.

Three metrics: precision@1 (what an agent actually consumes, since it reads
the top hit and stops), recall@k (a gap from precision@1 means ranking, not
coverage), and MRR (so a slip from rank 1 to 3 stays visible).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from brain.application.knowledge_api import KnowledgeApi
from brain.domain.errors import ValidationError
from brain.domain.models import EvalCase, EvalReport, EvalResult

GOLDEN_PATH = "90-meta/eval/golden.jsonl"
EVAL_DIR = "90-meta/eval"

#: Eval sets in reporting order. `golden.jsonl` was written alongside the alias
#: work, so it knows the vocabulary the vault already indexes and scores well
#: for that reason alone; `golden-neutral.jsonl` was not. The arbiter reads last
#: because that is the number a human should walk away with.
EVAL_SETS: tuple[str, ...] = ("golden.jsonl", "golden-neutral.jsonl")


def discover_eval_sets(root: Path) -> list[tuple[str, Path]]:
    """Every eval set present in a vault, named and in reporting order.

    Absent sets are skipped rather than raising: a vault that has never had a
    neutral set is not broken, it is just scored on less evidence.
    """
    eval_dir = root / EVAL_DIR
    return [(name.removesuffix(".jsonl"), eval_dir / name)
            for name in EVAL_SETS if (eval_dir / name).is_file()]


def load_cases(path: Path) -> list[EvalCase]:
    """Read a golden set. One JSON object per line.

    A malformed line raises rather than being skipped: a silently dropped case
    is a silently inflated score.
    """
    if not path.is_file():
        raise ValidationError(f"no golden set at {path}")
    cases: list[EvalCase] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        query = str(data.get("q") or data.get("query") or "").strip()
        expect = data.get("expect") or data.get("expects") or []
        if isinstance(expect, str):
            expect = [expect]
        if not query or not expect:
            raise ValidationError(f"{path}:{lineno} needs both 'q' and 'expect'")
        cases.append(EvalCase(query=query, expect=tuple(str(e) for e in expect),
                              note=str(data.get("note", ""))))
    if not cases:
        raise ValidationError(f"{path} contains no cases")
    return cases


def run_case(api: KnowledgeApi, case: EvalCase, k: int) -> EvalResult:
    hits = api.search_notes(case.query, limit=k)
    returned = tuple(h.ref.id for h in hits)
    rank: int | None = None
    for position, note_id in enumerate(returned, start=1):
        if note_id in case.expect:
            rank = position
            break
    return EvalResult(case=case, rank=rank, returned=returned)


def evaluate(api: KnowledgeApi, cases: Iterable[EvalCase], k: int = 5) -> EvalReport:
    report = EvalReport(k=k)
    for case in cases:
        report.results.append(run_case(api, case, k))
    return report


def unresolved_expectations(api: KnowledgeApi, cases: Iterable[EvalCase]) -> Iterator[str]:
    """Yield expected ids that do not exist in the vault.

    A golden set that points at deleted or renamed notes scores badly for a
    reason that has nothing to do with retrieval, so this is checked separately
    and reported as a defect in the *test set* rather than in the vault.
    """
    known = {ref.id for ref in api.list_notes()}
    seen: set[str] = set()
    for case in cases:
        for note_id in case.expect:
            if note_id not in known and note_id not in seen:
                seen.add(note_id)
                yield note_id
