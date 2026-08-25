"""Tests for the load-bearing contract.

These target the bugs that cause silent data loss, not the happy path. The
code-fence corpus is seeded from the real vault, where the pattern already
occurs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from brain.domain.errors import FrontmatterError
from brain.infrastructure.markdown import (
    as_list,
    coerce_date,
    dump_frontmatter,
    join,
    load_frontmatter,
    splice,
    split,
)

FM = b"---\nid: concept_x\ntitle: \"X\"\n---\n"

DANGEROUS_BODY = b"""# Title

Here is a fenced block that contains a horizontal rule:

```markdown
---
id: not_real_frontmatter
---
```

And a thematic break:

---

And a YAML document-end marker inside a fence:

```yaml
foo: bar
...
```

Trailing prose.
"""


def test_roundtrip_unchanged_is_byte_identical() -> None:
    raw = FM + DANGEROUS_BODY
    doc = split(raw)
    assert join(doc, doc.frontmatter_text) == raw


def test_dash_dash_dash_inside_code_fence_does_not_truncate_body() -> None:
    """The bug this module exists to make impossible."""
    raw = FM + DANGEROUS_BODY
    doc = split(raw)
    assert doc.has_frontmatter
    assert doc.body == DANGEROUS_BODY
    assert b"not_real_frontmatter" in doc.body
    assert b"Trailing prose." in doc.body


def test_splice_preserves_body_bytes_exactly() -> None:
    raw = FM + DANGEROUS_BODY

    def mutate(data: object) -> None:
        data["status"] = "evergreen"  # type: ignore[index]

    out = splice(raw, mutate)
    assert split(out).body == DANGEROUS_BODY
    assert b"status: evergreen" in out


def test_unterminated_frontmatter_is_treated_as_absent_not_as_eof() -> None:
    """A malformed header must never swallow the body."""
    raw = b"---\nid: concept_x\ntitle: no closing marker\n\n# Real heading\n\nbody\n"
    doc = split(raw)
    assert doc.frontmatter_text is None
    assert doc.body == raw
    assert join(doc, None) == raw


def test_frontmatter_only_recognized_at_byte_zero() -> None:
    raw = b"\n---\nid: x\n---\nbody\n"
    doc = split(raw)
    assert doc.frontmatter_text is None
    assert doc.body == raw


def test_dot_dot_dot_closes_frontmatter() -> None:
    raw = b"---\nid: concept_x\n...\nbody here\n"
    doc = split(raw)
    assert doc.has_frontmatter
    assert doc.body == b"body here\n"


def test_crlf_roundtrip() -> None:
    raw = b"---\r\nid: concept_x\r\ntitle: X\r\n---\r\n# Body\r\n\r\ntext\r\n"
    doc = split(raw)
    assert doc.newline == b"\r\n"
    assert doc.has_frontmatter
    assert doc.body == b"# Body\r\n\r\ntext\r\n"
    assert join(doc, doc.frontmatter_text) == raw


def test_bom_is_preserved() -> None:
    raw = b"\xef\xbb\xbf---\nid: concept_x\n---\nbody\n"
    doc = split(raw)
    assert doc.prefix == b"\xef\xbb\xbf"
    assert doc.has_frontmatter
    assert join(doc, doc.frontmatter_text) == raw


def test_empty_frontmatter_block() -> None:
    doc = split(b"---\n---\nbody\n")
    assert doc.has_frontmatter
    assert doc.body == b"body\n"


def test_no_trailing_newline_after_closing_marker() -> None:
    raw = b"---\nid: x\n---\n"
    doc = split(raw)
    assert doc.body == b""
    assert join(doc, doc.frontmatter_text) == raw


def test_splice_without_frontmatter_raises() -> None:
    with pytest.raises(FrontmatterError):
        splice(b"# just a body\n", lambda d: None)


def test_load_rejects_non_mapping() -> None:
    with pytest.raises(FrontmatterError):
        load_frontmatter("- a\n- b\n")


def test_dump_preserves_key_order_and_quotes() -> None:
    text = 'zebra: 1\nid: concept_x\ntitle: "Quoted, em dash"\ntags: [a, b]\n'
    data = load_frontmatter(text)
    out = dump_frontmatter(data)
    assert out.index("zebra") < out.index("id") < out.index("title")
    assert '"Quoted, em dash"' in out


def test_dump_does_not_reflow_long_scalars() -> None:
    """Default ruamel width would wrap this and churn the diff on every write."""
    long = "x" * 400
    data = load_frontmatter(f'description: "{long}"\n')
    assert dump_frontmatter(data).count("\n") == 1


@pytest.mark.parametrize(
    "value",
    ["2026-08-19", '"2026-08-19"', "2026-08-19T10:30:00"],
)
def test_coerce_date_accepts_every_shape_obsidian_might_leave(value: str) -> None:
    import datetime as dt

    assert coerce_date(value) == dt.date(2026, 8, 19)


def test_coerce_date_accepts_already_coerced_objects() -> None:
    import datetime as dt

    assert coerce_date(dt.date(2026, 8, 19)) == dt.date(2026, 8, 19)
    assert coerce_date(dt.datetime(2026, 8, 19, 3, 4)) == dt.date(2026, 8, 19)
    assert coerce_date(None) is None
    assert coerce_date("nonsense") is None


def test_as_list_normalizes() -> None:
    assert as_list(None) == []
    assert as_list("one") == ["one"]
    assert as_list(["a", "b"]) == ["a", "b"]
    assert as_list("") == []


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(body=st.binary(max_size=2000))
def test_property_arbitrary_body_bytes_survive_roundtrip(body: bytes) -> None:
    """Any byte sequence at all, including invalid UTF-8, must come back intact."""
    raw = FM + body
    doc = split(raw)
    if doc.has_frontmatter:
        assert doc.body == body
        assert join(doc, doc.frontmatter_text) == raw


@settings(max_examples=100)
@given(title=st.text(min_size=1, max_size=60))
def test_property_splice_never_alters_body(title: str) -> None:
    raw = FM + DANGEROUS_BODY

    def mutate(data: object) -> None:
        data["title"] = title  # type: ignore[index]

    assert split(splice(raw, mutate)).body == DANGEROUS_BODY


REAL_VAULT = Path(os.path.expanduser("~/Developer/brain"))


@pytest.mark.skipif(not REAL_VAULT.is_dir(), reason="real vault not present")
def test_every_note_in_the_real_vault_roundtrips_byte_identical() -> None:
    """The corpus test. If this fails, a write would corrupt a real note."""
    checked = fenced = 0
    for path in sorted(REAL_VAULT.rglob("*.md")):
        if any(p.startswith(".") for p in path.relative_to(REAL_VAULT).parts):
            continue
        raw = path.read_bytes()
        doc = split(raw)
        assert join(doc, doc.frontmatter_text) == raw, f"round-trip changed {path}"
        checked += 1
        if b"```" in doc.body and b"\n---" in doc.body:
            fenced += 1
    assert checked > 200, f"expected the full vault, only saw {checked}"
    assert fenced >= 1, "corpus should include notes with a --- inside a code fence"
