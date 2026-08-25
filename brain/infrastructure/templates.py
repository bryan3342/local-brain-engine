"""Note templates. **Create-only**, this is a hard rule, not a convention.

Re-rendering a body from a template on update would shred human edits, code
fences, callouts and embeds (H1b). `update_note` is a frontmatter splice; it
never calls anything in this module.

Substitution is stdlib string replacement, not Jinja: templates are a handful of
placeholders, and a template engine would be a second dependency for no gain.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_BODY = """# {title}

> **Answer:** {answer}

## Details

## Connections
"""


def render_body(
    templates_dir: Path,
    note_type: str,
    *,
    title: str,
    answer: str = "",
) -> bytes:
    """Render a body for a NEW note.

    A `90-meta/templates/<type>.md` file, if present, supplies the body; only
    its body is used, since the API composes frontmatter itself.
    """
    candidate = templates_dir / f"{note_type}.md"
    body_text = DEFAULT_BODY
    if candidate.is_file():
        from brain.infrastructure.markdown import split

        doc = split(candidate.read_bytes())
        text = doc.body.decode("utf-8", "replace").lstrip("\n")
        if text.strip():
            body_text = text
    rendered = (
        body_text.replace("{title}", title)
        .replace("TITLE", title)
        .replace("{answer}", answer or "TODO, one to three sentences that answer this note.")
    )
    return rendered.encode("utf-8")
