"""Tests for the retrieval eval harness.

The harness is what every later tuning decision gets judged against, so its
arithmetic has to be right. A scorer that silently inflates is worse than none:
it would certify a regression as an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.application.evaluate import (
    evaluate,
    load_cases,
    run_case,
    unresolved_expectations,
)
from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.errors import ValidationError
from brain.domain.models import EvalCase, EvalReport, EvalResult
from brain.infrastructure.index import SqliteIndex


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(config, index=SqliteIndex(config.root))
    a.initialize_vault()
    a.create_note("Fiber-Optic Cabling", "concept", domain="networking",
                  aliases=["fibre", "single-mode"], answer="Light, not current.")
    a.create_note("Copper Cabling", "concept", domain="networking",
                  aliases=["crosstalk"], answer="Electrical pulses; attenuates.")
    a.reindex(full=True)
    return a


def write_golden(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# ---- loading --------------------------------------------------------------


def test_load_accepts_q_or_query_and_scalar_expect(tmp_path: Path) -> None:
    p = write_golden(tmp_path / "g.jsonl", [
        {"q": "one", "expect": ["concept_a"]},
        {"query": "two", "expect": "concept_b"},
    ])
    cases = load_cases(p)
    assert [c.query for c in cases] == ["one", "two"]
    assert cases[1].expect == ("concept_b",)


def test_load_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text('# a comment\n\n{"q": "one", "expect": ["x"]}\n\n')
    assert len(load_cases(p)) == 1


def test_malformed_line_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    """A silently dropped case is a silently inflated score."""
    p = tmp_path / "g.jsonl"
    p.write_text('{"q": "ok", "expect": ["x"]}\nnot json at all\n')
    with pytest.raises(ValidationError, match="not valid JSON"):
        load_cases(p)


def test_case_without_expect_raises(tmp_path: Path) -> None:
    p = write_golden(tmp_path / "g.jsonl", [{"q": "lonely"}])
    with pytest.raises(ValidationError, match="needs both"):
        load_cases(p)


def test_missing_or_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no golden set"):
        load_cases(tmp_path / "absent.jsonl")
    empty = tmp_path / "e.jsonl"
    empty.write_text("# only a comment\n")
    with pytest.raises(ValidationError, match="no cases"):
        load_cases(empty)


# ---- scoring arithmetic ---------------------------------------------------


def _result(rank: int | None) -> EvalResult:
    case = EvalCase(query="q", expect=("concept_a",))
    return EvalResult(case=case, rank=rank, returned=("concept_a",) if rank else ())


def test_metrics_on_a_known_mix() -> None:
    report = EvalReport(k=5, results=[_result(1), _result(1), _result(3), _result(None)])
    assert report.total == 4
    assert report.precision_at_1 == 0.5                      # 2 of 4 at rank 1
    assert report.recall_at_k == 0.75                        # 3 of 4 found at all
    assert report.mrr == pytest.approx((1 + 1 + 1 / 3 + 0) / 4)
    assert len(report.misses) == 1


def test_empty_report_does_not_divide_by_zero() -> None:
    report = EvalReport()
    assert report.precision_at_1 == 0.0
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_rank_is_one_based_not_zero_based() -> None:
    assert _result(1).reciprocal_rank == 1.0
    assert _result(2).reciprocal_rank == 0.5
    assert _result(None).reciprocal_rank == 0.0
    assert _result(1).hit_at_1 and not _result(2).hit_at_1


# ---- running against a real vault -----------------------------------------


def test_hit_is_scored_by_position(api: KnowledgeApi) -> None:
    case = EvalCase(query="fibre", expect=("concept_fiber-optic-cabling",))
    result = run_case(api, case, k=5)
    assert result.rank == 1
    assert result.hit_at_1


def test_any_expected_id_counts_as_a_hit(api: KnowledgeApi) -> None:
    """Some questions are fairly answered by more than one note."""
    case = EvalCase(query="crosstalk",
                    expect=("concept_nonexistent", "concept_copper-cabling"))
    assert run_case(api, case, k=5).rank is not None


def test_a_miss_records_what_came_back_instead(api: KnowledgeApi) -> None:
    """The returned list is the diagnostic, it says *why* the case failed."""
    case = EvalCase(query="fibre", expect=("concept_something-else",))
    result = run_case(api, case, k=5)
    assert result.rank is None
    assert "concept_fiber-optic-cabling" in result.returned


def test_k_bounds_how_deep_a_hit_counts(api: KnowledgeApi) -> None:
    case = EvalCase(query="cabling", expect=("concept_copper-cabling",))
    deep = run_case(api, case, k=5)
    shallow = run_case(api, case, k=1)
    if deep.rank and deep.rank > 1:
        assert shallow.rank is None, "k must actually truncate"


def test_evaluate_aggregates_every_case(api: KnowledgeApi) -> None:
    cases = [
        EvalCase(query="fibre", expect=("concept_fiber-optic-cabling",)),
        EvalCase(query="crosstalk", expect=("concept_copper-cabling",)),
        EvalCase(query="zzzz nothing", expect=("concept_absent",)),
    ]
    report = evaluate(api, cases, k=5)
    assert report.total == 3
    assert len(report.misses) == 1
    assert 0.0 < report.precision_at_1 < 1.0


def test_stale_expectations_are_reported_separately(api: KnowledgeApi) -> None:
    """A golden set pointing at deleted notes scores badly for the wrong reason."""
    cases = [
        EvalCase(query="fibre", expect=("concept_fiber-optic-cabling",)),
        EvalCase(query="x", expect=("concept_deleted-long-ago",)),
    ]
    stale = list(unresolved_expectations(api, cases))
    assert stale == ["concept_deleted-long-ago"]


def test_report_serializes_for_json_output(api: KnowledgeApi) -> None:
    report = evaluate(api, [EvalCase(query="fibre", expect=("concept_fiber-optic-cabling",))])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["cases"] == 1
    assert "precision@1" in payload and "recall@5" in payload and "mrr" in payload


# ---- eval sets ------------------------------------------------------------


def test_discover_finds_every_eval_set_with_the_arbiter_last(tmp_path: Path) -> None:
    """Both sets are reported, and the uncontaminated one is reported last.

    golden.jsonl was written alongside the alias work and knows the vocabulary
    the vault already indexes; golden-neutral.jsonl does not. Scoring only the
    first flatters retrieval, so discovery must surface both, and the arbiter
    reads last because that is the number a human should walk away with.
    """
    from brain.application.evaluate import discover_eval_sets

    eval_dir = tmp_path / "90-meta" / "eval"
    eval_dir.mkdir(parents=True)
    write_golden(eval_dir / "golden.jsonl", [{"q": "fiber", "expect": ["x"]}])
    write_golden(eval_dir / "golden-neutral.jsonl", [{"q": "copper", "expect": ["y"]}])

    found = discover_eval_sets(tmp_path)

    assert [name for name, _ in found] == ["golden", "golden-neutral"]
    assert [p.name for _, p in found] == ["golden.jsonl", "golden-neutral.jsonl"]


def test_discover_skips_a_set_the_vault_does_not_have(tmp_path: Path) -> None:
    """A vault with no neutral set is scored on less evidence, not broken."""
    from brain.application.evaluate import discover_eval_sets

    eval_dir = tmp_path / "90-meta" / "eval"
    eval_dir.mkdir(parents=True)
    write_golden(eval_dir / "golden.jsonl", [{"q": "fiber", "expect": ["x"]}])

    assert [name for name, _ in discover_eval_sets(tmp_path)] == ["golden"]


def test_discover_returns_nothing_when_the_eval_directory_is_absent(tmp_path: Path) -> None:
    from brain.application.evaluate import discover_eval_sets

    assert discover_eval_sets(tmp_path) == []
