from __future__ import annotations

import pytest

from brain.domain.models import Candidate, Rejection


def test_candidate_roundtrips_through_dict() -> None:
    c = Candidate(
        candidate_id="sha256:abc",
        session_id="sess-1",
        project="client-site",
        timestamp="2026-07-15T14:02:11Z",
        user_text="why does the deploy not fire on push?",
        assistant_text="Because the workflow is manual-dispatch only.",
        tool_summary={"Bash": 12, "Edit": 4},
        branches=("main",),
    )
    assert Candidate.from_dict(c.to_dict()) == c


def test_candidate_to_dict_is_json_serializable() -> None:
    import json
    c = Candidate("id", "s", "p", "t", "u", "a", {"Bash": 1}, ("main",))
    assert json.loads(json.dumps(c.to_dict()))["candidate_id"] == "id"


def test_rejection_carries_a_reason() -> None:
    r = Rejection(candidate_id="id", stage="prefilter", reason="too-short")
    assert r.to_dict()["reason"] == "too-short"
    assert r.to_dict()["stage"] == "prefilter"


def test_candidate_is_hashable() -> None:
    c = Candidate("id", "s", "p", "t", "u", "a", {"Bash": 1}, ("main",))
    h = hash(c)
    assert isinstance(h, int)


def test_candidate_with_different_tool_summary_not_equal() -> None:
    c1 = Candidate("id", "s", "p", "t", "u", "a", {"Bash": 1}, ("main",))
    c2 = Candidate("id", "s", "p", "t", "u", "a", {"Bash": 2}, ("main",))
    assert c1 != c2


def test_candidate_from_dict_requires_user_text() -> None:
    from brain.domain.errors import ValidationError

    with pytest.raises(ValidationError):
        Candidate.from_dict({"candidate_id": "id"})


def test_candidate_from_dict_rejects_blank_user_text() -> None:
    from brain.domain.errors import ValidationError

    with pytest.raises(ValidationError):
        Candidate.from_dict(
            {"candidate_id": "id", "user_text": "   "}
        )


def test_candidate_from_dict_tolerates_missing_metadata() -> None:
    c = Candidate.from_dict({
        "candidate_id": "id",
        "user_text": "question",
    })
    assert c.candidate_id == "id"
    assert c.user_text == "question"
    assert c.project == ""
    assert c.session_id == ""
