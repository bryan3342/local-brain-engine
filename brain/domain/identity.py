"""Identity: slugs, id minting, edge encoding, dedupe keys.

The highest-risk pure surface in the package, because of C1: an `id` that looks
like a slug but must behave like an opaque key.

    id = f"{prefix(type)}_{slugify(title)}"   minted ONCE at creation, then frozen.

It is readable and hand-typeable (so `part_of::concept_osi-model` can be written
by a human in Obsidian), but it is **never regenerated**. Obsidian rewrites
filenames and `[[wikilinks]]` on rename; it does not touch `id:` or
`relationships:`. Any code path that re-slugs a changed title writes an edge to
a node that does not exist, and the graph silently forks.

Nothing in this module performs I/O.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from collections.abc import Iterable

from brain.domain.enums import NoteType

MAX_SLUG = 70

#: Id prefix per type. Mostly the type name, but `session-report` shortens to
#: `session` to match ids already frozen in the vault.
_ID_PREFIX: dict[NoteType, str] = {t: t.value.replace("-", "_") for t in NoteType}
_ID_PREFIX[NoteType.SESSION_REPORT] = "session"

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_[a-z0-9][a-z0-9._-]*$")
_EDGE_RE = re.compile(r"^([a-z_]+)::(.+)$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DASHES = re.compile(r"-{2,}")


def slugify(text: str, max_len: int = MAX_SLUG) -> str:
    """Lowercase ASCII slug.

    NFKD-folds accents so `Café` -> `cafe`. Text that folds away entirely
    (CJK, emoji-only, punctuation-only) falls back to a stable content hash
    rather than returning an empty string, so every title yields a usable slug.
    """
    folded = unicodedata.normalize("NFKD", text)
    ascii_text = folded.encode("ascii", "ignore").decode("ascii").lower()
    slug = _DASHES.sub("-", _NON_ALNUM.sub("-", ascii_text).strip("-"))
    if not slug:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return f"n-{digest}"
    return slug[:max_len].strip("-") or f"n-{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def id_prefix(note_type: NoteType) -> str:
    """The `{prefix}_` an id of this type carries."""
    return _ID_PREFIX[note_type]


def mint_id(note_type: NoteType, title: str, taken: Iterable[str] = ()) -> str:
    """Mint a new id. Call this exactly once per note, at creation.

    Collisions take the smallest free numeric suffix (`concept_graphrag_2`), so
    two notes may share a title without sharing an identity.

    This function must never be called to *look up* an existing note. Resolution
    goes through the index (see `resolve_order`), never by re-slugging a title.
    """
    base = f"{id_prefix(note_type)}_{slugify(title)}"
    taken_set = set(taken)
    if base not in taken_set:
        return base
    n = 2
    while f"{base}_{n}" in taken_set:
        n += 1
    return f"{base}_{n}"


def is_valid_id(value: str) -> bool:
    """Whether a string is shaped like a mintable id."""
    return bool(_ID_RE.match(value))


def normalize_title(title: str) -> str:
    """Fold a title for duplicate detection.

    Deliberately aggressive: `GraphRAG`, `Graph RAG`, and `graph-rag` all
    normalize together, because the false *negative* (two notes for one concept,
    fragmenting the graph) is worse than the false positive, which a human
    resolves once (M2).
    """
    folded = unicodedata.normalize("NFKD", title)
    ascii_text = folded.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_ALNUM.sub("", ascii_text)


def dedupe_key(note_type: NoteType, title: str) -> tuple[str, str]:
    """Duplicate identity is scoped by type.

    A person named "Python" and a language named "Python" are not duplicates;
    two `concept` notes titled "Python" are.
    """
    return (note_type.value, normalize_title(title))


def format_edge(rel: str, target: str) -> str:
    """Encode one typed edge as the flat string Obsidian can render as a List."""
    return f"{rel}::{target}"


def parse_edge(value: str) -> tuple[str, str] | None:
    """Decode `type::target`. Returns None for anything malformed."""
    match = _EDGE_RE.match(value.strip())
    if not match:
        return None
    rel, target = match.group(1), match.group(2).strip()
    return (rel, target) if target else None


#: Edge targets that are legitimate graph nodes but are not notes: a file a
#: session touched, a PR it shipped, a skill it used, the domain it belongs to.
#: They give the graph its connective tissue, sessions link to each other
#: *through* the files and PRs they share, so `export_graph` emits them as
#: synthetic nodes rather than reporting them as dangling.
ANCHOR_PREFIXES: tuple[str, ...] = (
    "domain_", "module_", "project_", "file_", "pr_", "skill_", "gotcha_",
)


def is_anchor(target: str) -> bool:
    """Whether an edge target names an anchor rather than a note."""
    return target.startswith(ANCHOR_PREFIXES)


def anchor_kind(target: str) -> str:
    """The anchor's category, e.g. `file` for `file_admin-app-js`."""
    for prefix in ANCHOR_PREFIXES:
        if target.startswith(prefix):
            return prefix.rstrip("_")
    return "unknown"


#: Order in which a user-supplied reference is resolved to a note.
#: Never includes "re-slugify the title", that is the C1 bug.
resolve_order: tuple[str, ...] = ("id", "alias", "path", "normalized-title")


#: Repos whose names contain dashes, so a trailing-segment split cannot
#: recover them from the path slug. Longest first, `client-site` must be
#: tested before `structa` would ever match a substring of it.
@dataclass(frozen=True)
class ProjectRules:
    """How a working directory maps to a project name.

    Passed in by the caller so no organisation or repo name is baked into the
    package. `aliases` maps a path fragment to a name and wins over `repos`,
    which is checked in order, so list longer names first.
    """

    aliases: tuple[tuple[str, str], ...] = ()
    repos: tuple[str, ...] = ()


NO_PROJECT_RULES = ProjectRules()


def normalize_project(raw: str, rules: ProjectRules = NO_PROJECT_RULES) -> str:
    """Map a slugified working directory to a project name.

    Anything unrecognised comes back lowercased and unchanged. A wrong name
    forks the graph; an ugly one does not.
    """
    if not raw:
        return "unknown"
    value = raw.lower()
    for fragment, name in rules.aliases:
        if fragment in value:
            return name
    for repo in rules.repos:
        if value == repo or value.endswith("-" + repo) or f"-{repo}-" in value:
            return repo
    return value


def project_from_cwd(cwd: str, rules: ProjectRules = NO_PROJECT_RULES) -> str:
    """Derive a project name from the session\'s real `cwd`.

    Order: alias fragment, then the first matching repo, then the final path
    segment. Empty input gives "unknown".
    """
    if not cwd or not cwd.strip():
        return "unknown"
    value = cwd.strip().lower()
    for fragment, name in rules.aliases:
        if "/" + fragment in value:
            return name
    for repo in rules.repos:
        if f"/{repo}" in value:
            return repo
    tail = cwd.strip().rstrip("/").rsplit("/", 1)[-1]
    slug = _DASHES.sub("-", _NON_ALNUM.sub("-", tail.lower()).strip("-"))
    return slug or "unknown"
