from __future__ import annotations

from brain.domain.identity import ProjectRules
from brain.etl.prefilter import MIN_USER_CHARS, classify, content_hash, prefilter

LONG = "why does the backend deploy not fire automatically on push to main?"


RULES = ProjectRules(aliases=(("acme", "acme"),), repos=("structa", "local-brain", "brain"))


def row(**over: object) -> dict[str, object]:
    base = {
        "id": "ex-1", "session_id": "s1", "project": "client-site",
        "timestamp": "2026-07-15T14:02:11Z", "user_message": LONG,
        "assistant_message": "Because the workflow is manual-dispatch only.",
        "git_branch": "main",
    }
    base.update(over)
    return base


def test_a_substantive_exchange_is_kept() -> None:
    candidate, rejection = classify(row(), seen=set(), reported=set())
    assert rejection is None
    assert candidate is not None
    assert candidate.project == "client-site"
    assert candidate.branches == ("main",)


def test_short_messages_are_dropped_with_a_reason() -> None:
    candidate, rejection = classify(row(user_message="ok"), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "too-short"


def test_empty_user_message_is_dropped() -> None:
    candidate, rejection = classify(row(user_message="   "), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "no-human-content"


def test_tool_output_is_dropped() -> None:
    for noise in ("<task-notification>done</task-notification>",
                  "<bash-stdout>lots of output here indeed</bash-stdout>",
                  "[Image: original 2186x1292, displayed at 2000x1182 pixels]"):
        candidate, rejection = classify(row(user_message=noise), seen=set(), reported=set())
        assert candidate is None, noise
        assert rejection is not None and rejection.reason == "tool-output-only"


def test_already_seen_content_is_dropped() -> None:
    seen = {content_hash(LONG)}
    candidate, rejection = classify(row(), seen=seen, reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "duplicate-content"


def test_already_reported_session_is_dropped() -> None:
    candidate, rejection = classify(row(), seen=set(), reported={"s1"})
    assert candidate is None
    assert rejection is not None and rejection.reason == "session-already-reported"


def test_content_hash_ignores_whitespace_differences() -> None:
    assert content_hash("a  b\n c") == content_hash("a b c")


def test_prefilter_is_deterministic() -> None:
    """Same input twice must give byte-identical output, no model, no discretion."""
    rows = [row(id=f"ex-{i}", user_message=f"{LONG} number {i}") for i in range(5)]
    rows.append(row(id="ex-noise", user_message="hi"))
    first = prefilter(rows, seen=set(), reported=set())
    second = prefilter(rows, seen=set(), reported=set())
    assert [c.to_dict() for c in first[0]] == [c.to_dict() for c in second[0]]
    assert [r.to_dict() for r in first[1]] == [r.to_dict() for r in second[1]]


def test_prefilter_never_silently_drops() -> None:
    """Every input row appears in exactly one of the two output lists."""
    rows = [row(id="a"), row(id="b", user_message="no"), row(id="c", user_message=LONG + "!")]
    kept, dropped = prefilter(rows, seen=set(), reported=set())
    assert len(kept) + len(dropped) == len(rows)


def test_min_user_chars_is_a_documented_constant() -> None:
    assert MIN_USER_CHARS == 40


def test_invariant_exactly_one_result_is_enforced(monkeypatch: object) -> None:
    """Invariant: classify returns exactly one of (Candidate, Rejection)."""
    import brain.etl.prefilter as pf

    def bad_classify(*args: object, **kwargs: object) -> object:
        from brain.domain.models import Candidate, Rejection
        return (Candidate(
            candidate_id="test", session_id="s1", project="p",
            timestamp="t", user_text="u", assistant_text="a",
        ), Rejection(candidate_id="test", stage="s", reason="r"))

    monkeypatch.setattr(pf, "classify", bad_classify)
    rows = [row()]
    try:
        prefilter(rows, seen=set(), reported=set())
        raise AssertionError("prefilter should have raised ValueError")
    except ValueError as e:
        assert "exactly one" in str(e) and "ex-1" in str(e)


def test_intra_batch_dedup_catches_duplicates() -> None:
    """Duplicate user_message within one batch is caught by running set."""
    shared_msg = LONG + " shared"
    rows = [
        row(id="a", user_message=shared_msg),
        row(id="b", user_message=shared_msg),
    ]
    kept, dropped = prefilter(rows, seen=set(), reported=set())
    assert len(kept) == 1
    assert len(dropped) == 1
    assert dropped[0].reason == "duplicate-content"


def test_tool_marker_must_start_message_not_be_mentioned() -> None:
    """Message quoting a marker mid-sentence is kept, not dropped."""
    msg = f'Why does {LONG} show "<bash-stdout>" in my output?'
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is not None, "Message mentioning marker should not be dropped"
    assert rejection is None


def test_tool_marker_at_start_is_dropped() -> None:
    """Message starting with a marker is tool output."""
    msg = "<bash-stdout>output here</bash-stdout>"
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "tool-output-only"


def test_injected_skill_preamble_is_dropped() -> None:
    msg = "Base directory for this skill: /home/dev/code/Obsidian/vault"
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "injected-preamble"


def test_injected_caveat_preamble_is_dropped() -> None:
    msg = ("Caveat: The messages below were generated by the user while running "
           "local commands. DO NOT respond to them as if the user typed them.")
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "injected-preamble"


def test_injected_system_reminder_is_dropped_as_injected_preamble_not_tool_output() -> None:
    """`<system-reminder>` is also one of `_TOOL_MARKERS`, so it would already
    be dropped as "tool-output-only" by the older, more generic rule. The new
    rule must be checked first: the reason must be exactly "injected-preamble",
    not the older marker's reason. This pins the ordering so a future reorder
    of the rules cannot silently change the reason without a test noticing."""
    msg = "<system-reminder>The date has changed. Today's date is 2026-08-20.</system-reminder>"
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "injected-preamble"


def test_injected_preamble_match_is_case_insensitive() -> None:
    msg = "BASE DIRECTORY FOR THIS SKILL: /home/dev/code/Obsidian/vault"
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "injected-preamble"


def test_injected_preamble_after_leading_whitespace_is_still_dropped() -> None:
    msg = "   Base directory for this skill: /home/dev/code/Obsidian/vault"
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is None
    assert rejection is not None and rejection.reason == "injected-preamble"


def test_message_mentioning_injected_preamble_mid_sentence_is_kept() -> None:
    """A human talking *about* the harness preamble is not the preamble
    itself: the match is start-anchored, not "mentions somewhere"."""
    msg = f'{LONG} I noticed the transcript literally said "Base directory for this skill:" there.'
    candidate, rejection = classify(row(user_message=msg), seen=set(), reported=set())
    assert candidate is not None, "Message mentioning the marker mid-sentence should be kept"
    assert rejection is None


def test_classify_prefers_cwd_derived_project_over_the_mangled_project_column() -> None:
    """The read-side fix this task exists for: episodic-memory's `project`
    column can be a non-project slug ("subagents", 594 of 3,896 live rows),
    while `cwd` carries the unmangled path for every row. `cwd` must win."""
    cwd = ("/home/dev/code/Obsidian/Desktop/Agentic_AI_Factory/"
           "07_Work_Notes/Acme/acme/infra")
    candidate, rejection = classify(row(project="subagents", cwd=cwd), seen=set(), reported=set(), rules=RULES)
    assert rejection is None
    assert candidate is not None
    assert candidate.project == "acme"


def test_classify_falls_back_to_normalize_project_when_cwd_is_absent() -> None:
    """When `cwd` is empty or missing, `project` still goes through the
    existing `normalize_project` fallback, a mangled slug is normalized, not
    passed through raw."""
    candidate, rejection = classify(
        row(project="-home-dev-code-structa"), seen=set(), reported=set(), rules=RULES)
    assert rejection is None
    assert candidate is not None
    assert candidate.project == "structa"


def test_whitespace_only_messages_have_distinct_ids() -> None:
    """Two whitespace-only rows must have distinct rejection ids."""
    rows = [
        row(id="a", user_message="   "),
        row(id="b", user_message="\t\n  "),
    ]
    kept, dropped = prefilter(rows, seen=set(), reported=set())
    assert len(kept) == 0
    assert len(dropped) == 2
    assert dropped[0].candidate_id != dropped[1].candidate_id
    assert dropped[0].candidate_id == "exchange:a"
    assert dropped[1].candidate_id == "exchange:b"
