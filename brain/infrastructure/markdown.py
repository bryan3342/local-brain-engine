"""Frontmatter in, body bytes out, unchanged.

Two rules make the dangerous bug impossible rather than unlikely.

Head-anchored split: frontmatter exists only if byte 0 is `---`, and closes at
the first later line that is exactly `---` or `...`. Everything after is an
opaque byte range. A parser that splits the whole file on `---` truncates any
note containing a fenced code block with `---` inside it.

Unterminated means absent: an opening `---` with no close is treated as no
frontmatter, so a malformed header cannot swallow a body.
"""

from __future__ import annotations

import codecs
import datetime as _dt
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from brain.domain.errors import FrontmatterError

_CLOSERS = (b"---", b"...")


def _yaml() -> YAML:
    """Round-trip YAML tuned to leave the file alone.

    `width` is set absurdly high because the default reflows long scalars, which
    would rewrite every `description:` line on any write and bury real changes in
    diff noise.
    """
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.width = 1 << 20
    y.indent(mapping=2, sequence=4, offset=2)
    y.allow_duplicate_keys = False
    return y


@dataclass(frozen=True, slots=True)
class SplitDoc:
    """A file cleaved into its frontmatter text and its opaque body.

    Invariant: when nothing is changed, `join(doc, doc.frontmatter_text)`
    reproduces the original bytes exactly.
    """

    prefix: bytes                 # UTF-8 BOM, if the file had one
    frontmatter_text: str | None  # raw YAML between the markers; None if absent
    body: bytes                   # opaque, never parsed, never re-rendered
    newline: bytes                # b"\n" or b"\r\n", as used by the header

    @property
    def has_frontmatter(self) -> bool:
        return self.frontmatter_text is not None


def split(raw: bytes) -> SplitDoc:
    """Cleave a file. Tolerant: anything unparseable is reported as body-only."""
    prefix = b""
    if raw[:3] == codecs.BOM_UTF8:
        prefix, raw = raw[:3], raw[3:]

    if raw.startswith(b"---\r\n"):
        newline = b"\r\n"
    elif raw.startswith(b"---\n"):
        newline = b"\n"
    else:
        return SplitDoc(prefix=prefix, frontmatter_text=None, body=raw, newline=b"\n")

    opener = b"---" + newline
    pos = len(opener)
    while pos <= len(raw):
        nl_at = raw.find(newline, pos)
        line = raw[pos:nl_at] if nl_at != -1 else raw[pos:]
        if line.rstrip(b"\r") in _CLOSERS:
            text = raw[len(opener):pos].decode("utf-8", "replace")
            body = raw[nl_at + len(newline):] if nl_at != -1 else b""
            return SplitDoc(prefix=prefix, frontmatter_text=text, body=body, newline=newline)
        if nl_at == -1:
            break
        pos = nl_at + len(newline)

    # Opening marker with no closing marker: treat as no frontmatter, never as
    # frontmatter-to-EOF. A malformed header must not be able to eat a body.
    return SplitDoc(prefix=prefix, frontmatter_text=None, body=raw, newline=newline)


def join(doc: SplitDoc, frontmatter_text: str | None) -> bytes:
    """Reassemble. `doc.body` is copied through untouched."""
    if frontmatter_text is None:
        return doc.prefix + doc.body
    text = frontmatter_text
    if not text.endswith("\n"):
        text += "\n"
    encoded = text.encode("utf-8")
    if doc.newline != b"\n":
        encoded = encoded.replace(b"\r\n", b"\n").replace(b"\n", doc.newline)
    return doc.prefix + b"---" + doc.newline + encoded + b"---" + doc.newline + doc.body


def load_frontmatter(text: str) -> Any:
    """Parse frontmatter YAML in round-trip mode (order and quoting preserved)."""
    try:
        data = _yaml().load(text or "")
    except YAMLError as exc:
        raise FrontmatterError(f"unparseable frontmatter: {exc}") from exc
    if data is None:
        data = _yaml().load("{}")
    if not hasattr(data, "keys"):
        raise FrontmatterError("frontmatter is not a mapping")
    return data


def dump_frontmatter(data: Any) -> str:
    """Serialize back, preserving key order and quotes ruamel captured."""
    stream = io.StringIO()
    _yaml().dump(data, stream)
    return stream.getvalue()


def splice(raw: bytes, mutate: Callable[[Any], None]) -> bytes:
    """Apply `mutate` to the frontmatter mapping only.

    The body is re-emitted byte-for-byte. This is the *only* sanctioned way to
    change an existing note: it cannot touch prose, code fences, callouts, or
    embeds, because it never decodes them.
    """
    doc = split(raw)
    if doc.frontmatter_text is None:
        raise FrontmatterError("note has no frontmatter block to splice")
    data = load_frontmatter(doc.frontmatter_text)
    mutate(data)
    return join(doc, dump_frontmatter(data))


def coerce_date(value: Any) -> _dt.date | None:
    """Read a date leniently.

    Obsidian's Properties UI strips quotes when it re-serializes, so a value this
    package wrote as `"2026-08-19"` may come back as a bare scalar that ruamel
    has already turned into a `date`. All three forms must read identically
    (H1e). Never assume the on-disk shape is the one we wrote.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return _dt.datetime.strptime(text[:len(fmt) + 8], fmt).date()
        except ValueError:
            continue
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def as_list(value: Any) -> list[str]:
    """Normalize a scalar-or-list frontmatter field to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if hasattr(value, "__iter__"):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]
