"""Derived SQLite and FTS5 index.

Markdown stays the source of truth; this is a disposable projection, and a test
proves it: delete the database, rebuild, and every query returns identical
results.

It exists because the naive design is O(n) per read. `get_backlinks` scans every
note, and so does validating a relationship target on create.

Staleness is `(mtime_ns, size)` with a content hash as tiebreak, so a `touch`
does not trigger a reparse but a real edit under a mtime-preserving sync still
does.
"""

from __future__ import annotations

import array
import hashlib
import math
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from brain.application.mapper import parse_note
from brain.domain.errors import ValidationError
from brain.domain.models import IndexStats, NoteRef
from brain.infrastructure.embedder import Embedder
from brain.infrastructure.vault import INDEX_DB, Vault

#: bm25 column weights, in table order: id, title, aliases, tags, squashed, body.
#: `id` is UNINDEXED and can never match, but it still consumes a position.
#: Chosen by sweeping against the golden set: body weight is the dominant lever
#: (1.0 -> 0.2 moved precision@1 by ~6 points), while title weight made no
#: measurable difference above ~10, so it is left modest rather than tuned to
#: the last decimal on 35 cases.
BM25_WEIGHTS = "0.0, 20.0, 15.0, 10.0, 15.0, 0.2"

#: Weight of the dense ranking in the reciprocal-rank sum, keyword being 1.0.
#: Swept over both eval sets: the tuned set peaks at 1.0 and the neutral set at
#: 2.0, and the neutral set is the one whose vocabulary retrieval was not built
#: around. Above 2.0 nothing moved on either.
VECTOR_WEIGHT = 2.0

#: RRF's rank-smoothing constant, from the original paper. Damps the difference
#: between adjacent positions so one retriever's confident first place cannot
#: alone outvote the other's agreement further down.
RRF_K = 60

#: How deep each retriever runs before fusion. Fusion can only promote what one
#: side already surfaced, so this has to exceed the limit actually returned.
FUSION_POOL = 50

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path      TEXT PRIMARY KEY,
    mtime_ns  INTEGER NOT NULL,
    size      INTEGER NOT NULL,
    hash      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id       TEXT PRIMARY KEY,
    path     TEXT NOT NULL UNIQUE,
    title    TEXT NOT NULL,
    type     TEXT NOT NULL,
    domain   TEXT,
    status   TEXT NOT NULL,
    created  TEXT,
    answer   TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL,
    PRIMARY KEY (src, rel, dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS aliases (id TEXT NOT NULL, alias TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias);
CREATE TABLE IF NOT EXISTS tags (id TEXT NOT NULL, tag TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
    id  TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
    USING fts5(id UNINDEXED, title, aliases, tags, squashed, body,
               tokenize='porter unicode61');
"""


def fts5_available() -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        return True
    except sqlite3.OperationalError:
        return False


class SqliteIndex:
    """Disposable read accelerator."""

    def __init__(self, root: Path, db_path: Path | None = None,
                 embedder: Embedder | None = None) -> None:
        self.root = Path(root)
        self.db_path = db_path or (self.root / INDEX_DB)
        self.embedder = embedder
        #: Trips on the first embedding failure and stays tripped for the life
        #: of this index. Scoped per instance, so the next command retries, a
        #: model that was down a minute ago is worth one more request, but not
        #: one per note in the middle of a rebuild.
        self._embedder_down = False
        self._con: sqlite3.Connection | None = None

    # ---- lifecycle --------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._con is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(self.db_path))
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.executescript(SCHEMA)
            self._migrate_if_stale(con)
            self._con = con
        return self._con

    @staticmethod
    def _migrate_if_stale(con: sqlite3.Connection) -> None:
        """Rebuild the FTS table when its columns no longer match SCHEMA.

        `CREATE VIRTUAL TABLE IF NOT EXISTS` silently keeps an older shape, so a
        new column would otherwise fail at INSERT time on every existing
        database. Dropping is safe precisely because this index is derived: the
        next sync rebuilds it from the Markdown, which is the only source of
        truth.
        """
        try:
            cols = {row[1] for row in con.execute("PRAGMA table_info(notes_fts)")}
        except sqlite3.OperationalError:
            return
        if cols and "squashed" not in cols:
            con.executescript(
                "DROP TABLE IF EXISTS notes_fts;"
                "DELETE FROM files;"          # force a full re-sync of every note
            )
            con.executescript(SCHEMA)
            con.commit()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def drop(self) -> None:
        """Delete the database. Always safe, it is derived."""
        self.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    # ---- sync -------------------------------------------------------------

    def sync(self, vault: Vault, *, full: bool = False) -> IndexStats:
        """Bring the index up to date. Incremental unless `full`."""
        con = self.connect()
        stats = IndexStats()
        if full:
            con.executescript(
                "DELETE FROM files; DELETE FROM notes; DELETE FROM edges; "
                "DELETE FROM aliases; DELETE FROM tags; DELETE FROM notes_fts; "
                "DELETE FROM vectors;")

        known = {row["path"]: row for row in con.execute("SELECT * FROM files")}
        seen: set[str] = set()

        for path in vault.note_paths():
            rel = vault.relative(path)
            seen.add(rel)
            st = path.stat()
            prior = known.get(rel)
            if prior and prior["mtime_ns"] == st.st_mtime_ns and prior["size"] == st.st_size:
                stats.unchanged += 1
                continue

            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if prior and prior["hash"] == digest:
                # mtime moved but bytes did not, a `touch`. Refresh the stamp so
                # the next sweep is cheap again, but do not reparse.
                con.execute("UPDATE files SET mtime_ns=?, size=? WHERE path=?",
                            (st.st_mtime_ns, st.st_size, rel))
                stats.unchanged += 1
                continue

            try:
                note = parse_note(path, data)
            except (ValidationError, OSError):
                self._forget_path(con, rel)
                con.execute(
                    "INSERT INTO files(path, mtime_ns, size, hash) VALUES (?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET mtime_ns=excluded.mtime_ns, "
                    "size=excluded.size, hash=excluded.hash",
                    (rel, st.st_mtime_ns, st.st_size, digest))
                continue

            self._forget_path(con, rel)
            fm = note.frontmatter
            con.execute(
                "INSERT INTO notes(id, path, title, type, domain, status, created, answer) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET path=excluded.path, title=excluded.title, "
                "type=excluded.type, domain=excluded.domain, status=excluded.status, "
                "created=excluded.created, answer=excluded.answer",
                (fm.id, rel, fm.title, fm.type.value, fm.domain, fm.status.value,
                 fm.created.isoformat() if fm.created else None, note.answer))
            con.executemany("INSERT OR IGNORE INTO edges(src, rel, dst) VALUES (?,?,?)",
                            [(fm.id, e.rel, e.target) for e in fm.relationships])
            con.executemany("INSERT INTO aliases(id, alias) VALUES (?,?)",
                            [(fm.id, a) for a in fm.aliases])
            con.executemany("INSERT INTO tags(id, tag) VALUES (?,?)",
                            [(fm.id, t) for t in fm.tags])
            con.execute(
                "INSERT INTO notes_fts(id, title, aliases, tags, squashed, body) "
                "VALUES (?,?,?,?,?,?)",
                (fm.id, fm.title, " ".join(fm.aliases), " ".join(fm.tags),
                 squash(" ".join([fm.title, *fm.aliases, *fm.tags])),
                 note.body.decode("utf-8", "replace")))
            self._store_vector(con, fm.id, embed_text(note))
            con.execute(
                "INSERT INTO files(path, mtime_ns, size, hash) VALUES (?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET mtime_ns=excluded.mtime_ns, "
                "size=excluded.size, hash=excluded.hash",
                (rel, st.st_mtime_ns, st.st_size, digest))
            stats.reindexed += 1

        for gone in set(known) - seen:
            self._forget_path(con, gone)
            con.execute("DELETE FROM files WHERE path=?", (gone,))
            stats.removed += 1

        # An embedder attached to an index built without one finds every file
        # stamp current, so the loop above skips all of them and no vector is
        # ever written. A missing vector is its own kind of stale.
        if self.embedder is not None:
            self._backfill_vectors(con, vault)

        con.commit()
        stats.notes = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        stats.edges = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        stats.vectors = con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return stats

    def _backfill_vectors(self, con: sqlite3.Connection, vault: Vault) -> None:
        """Embed notes the index already knows but holds no vector for.

        Deliberately not counted as a reparse: nothing about the note changed,
        only what this index is now able to record about it.
        """
        rows = con.execute(
            "SELECT id, path FROM notes WHERE id NOT IN (SELECT id FROM vectors)"
        ).fetchall()
        for row in rows:
            path = vault.root / row["path"]
            try:
                note = parse_note(path, path.read_bytes())
            except (ValidationError, OSError):
                continue
            self._store_vector(con, row["id"], embed_text(note))

    def _store_vector(self, con: sqlite3.Connection, note_id: str, text: str) -> None:
        """Embed and persist one note, or do nothing at all.

        Failure here is deliberately not fatal. Vectors are an accelerator on a
        derived index; losing them costs ranking quality, while raising would
        cost the user their ability to write a note because a model was down.
        """
        if self.embedder is None or self._embedder_down:
            return
        try:
            vector = self.embedder.embed([text])[0]
        except Exception:
            self._embedder_down = True
            return
        con.execute(
            "INSERT INTO vectors(id, dim, vec) VALUES (?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (note_id, len(vector), array.array("f", vector).tobytes()))

    @staticmethod
    def _forget_path(con: sqlite3.Connection, rel: str) -> None:
        row = con.execute("SELECT id FROM notes WHERE path=?", (rel,)).fetchone()
        if row is None:
            return
        note_id = row["id"]
        for table in ("edges", "aliases", "tags"):
            column = "src" if table == "edges" else "id"
            con.execute(f"DELETE FROM {table} WHERE {column}=?", (note_id,))
        con.execute("DELETE FROM notes_fts WHERE id=?", (note_id,))
        con.execute("DELETE FROM vectors WHERE id=?", (note_id,))
        con.execute("DELETE FROM notes WHERE id=?", (note_id,))

    # ---- queries ----------------------------------------------------------

    def by_id(self, note_id: str) -> str | None:
        row = self.connect().execute("SELECT path FROM notes WHERE id=?", (note_id,)).fetchone()
        return str(row["path"]) if row else None

    def refs(self) -> list[NoteRef]:
        return [
            NoteRef(id=r["id"], title=r["title"], type=r["type"],
                    path=r["path"], status=r["status"])
            for r in self.connect().execute("SELECT * FROM notes ORDER BY id")
        ]

    def backlinks(self, note_id: str) -> list[tuple[str, str]]:
        """O(indegree), not O(vault), the whole reason this file exists."""
        return [
            (r["src"], r["rel"])
            for r in self.connect().execute(
                "SELECT src, rel FROM edges WHERE dst=? ORDER BY src", (note_id,))
        ]

    def _search_lexical(self, query: str, limit: int) -> list[tuple[str, str, float]]:
        con = self.connect()
        try:
            rows = con.execute(
                "SELECT id, snippet(notes_fts, 5, '', '', '…', 16) AS snip, "
                # bm25 weights are positional over EVERY column, including UNINDEXED ones,
                # so `id` must be given a weight or all the others shift one column
                # left and the largest lands on a column that can never match.
                # Body is damped hard: bodies are long and mention everything
                # incidentally, so at equal weight they drown the structured
                # metadata that the answer-first design exists to put up front.
                # Tuned against the golden set, see BM25_WEIGHTS.
                f"bm25(notes_fts, {BM25_WEIGHTS}) AS score "
                "FROM notes_fts WHERE notes_fts MATCH ? "
                # Archived means retired. A superseded note is knowledge the
                # vault has explicitly marked wrong, and serving it is worse
                # than serving a duplicate -- the reader cannot tell which.
                # Written as NOT IN rather than a join on purpose: a join drops
                # any hit missing from `notes`, so an indexing gap would silently
                # delete results. This only removes ids KNOWN to be retired.
                "AND notes_fts.id NOT IN "
                "(SELECT id FROM notes WHERE status IN ('archived', 'deleted')) "
                "ORDER BY score LIMIT ?",
                (_fts_query(query), limit)).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS expression (stray quote, bare operator), fall back to
            # a LIKE scan rather than surfacing a SQL error to the caller.
            like = f"%{query}%"
            rows = con.execute(
                "SELECT n.id AS id, substr(n.answer, 1, 160) AS snip, 0.0 AS score "
                "FROM notes n WHERE (n.title LIKE ? OR n.answer LIKE ?) "
                "AND n.status NOT IN ('archived', 'deleted') LIMIT ?",
                (like, like, limit)).fetchall()
        return [(r["id"], r["snip"] or "", -float(r["score"] or 0.0)) for r in rows]

    # ---- fusion -----------------------------------------------------------

    def search(self, query: str, limit: int = 20
               ) -> list[tuple[str, str, float, float | None]]:
        """Keyword and dense retrieval, fused by reciprocal rank.

        Additive by construction: with no embedder, or no vectors, or a model
        that has gone away, this is exactly the keyword search it replaced.

        The returned score stays the BM25 score and never becomes the fusion
        score. Callers calibrate absolute thresholds against it. RRF sums
        rank positions, so its output is near-identical for a perfect match and
        a hopeless one, and substituting it would silently disable every gate
        downstream. A hit found only by the dense side scores 0.0 here, which
        is honest: BM25 did not find it.
        """
        lexical = self._search_lexical(query, max(limit, FUSION_POOL))
        dense = self._search_dense(query, max(limit, FUSION_POOL))
        if not dense:
            return [(i, s, r, None) for i, s, r in lexical[:limit]]

        fused: dict[str, float] = {}
        for position, (note_id, _, _) in enumerate(lexical, start=1):
            fused[note_id] = fused.get(note_id, 0.0) + 1.0 / (RRF_K + position)
        for position, (note_id, _) in enumerate(dense, start=1):
            fused[note_id] = fused.get(note_id, 0.0) + VECTOR_WEIGHT / (RRF_K + position)

        known = {note_id: (snippet, score) for note_id, snippet, score in lexical}
        similarity = dict(dense)
        ordered = sorted(fused, key=lambda i: -fused[i])[:limit]
        return [(note_id, *known.get(note_id, (self._answer_snippet(note_id), 0.0)),
                 similarity.get(note_id))
                for note_id in ordered]

    def _search_dense(self, query: str, limit: int) -> list[tuple[str, float]]:
        """(id, cosine) pairs, nearest first. Empty when unavailable."""
        if self.embedder is None or self._embedder_down:
            return []
        con = self.connect()
        rows = con.execute(
            "SELECT v.id AS id, v.vec AS vec FROM vectors v JOIN notes n ON n.id = v.id "
            "WHERE n.status NOT IN ('archived', 'deleted')").fetchall()
        if not rows:
            return []
        try:
            query_vector = _unit(self.embedder.embed([query])[0])
        except Exception:
            self._embedder_down = True
            return []
        scored: list[tuple[float, str]] = []
        for row in rows:
            stored = array.array("f")
            stored.frombytes(row["vec"])
            # A vector written by a different model is not comparable. Skipping
            # beats scoring nonsense: the index rebuilds on the next full sync.
            if len(stored) != len(query_vector):
                continue
            similarity = sum(a * b for a, b in zip(query_vector, _unit(stored), strict=True))
            scored.append((similarity, row["id"]))
        scored.sort(reverse=True)
        return [(note_id, score) for score, note_id in scored[:limit]]

    def _answer_snippet(self, note_id: str) -> str:
        row = self.connect().execute(
            "SELECT answer FROM notes WHERE id=?", (note_id,)).fetchone()
        return str(row["answer"] or "")[:160] if row else ""


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else [0.0] * len(vector)



#: Words carrying no retrieval signal. A query OR-ing these matches essentially
#: every note, and the genuinely rare terms get buried under the pile, which is
#: how "can I push straight to main?" returned an unrelated PR note.
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did",
    "do", "does", "doesn", "for", "from", "get", "got", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "like", "me", "my", "no", "not", "of",
    "on", "or", "our", "should", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "until", "up", "use",
    "used", "using", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
})


#: How much note body reaches the embedder. Long notes dilute a single vector, #: every extra paragraph pulls the centroid toward whatever the note mentions in
#: passing, and embedding models degrade past their trained window anyway. Four
#: thousand bytes covers the whole of almost every note in a vault of atomic
#: claims; per-section vectors are the answer for the rest, not a bigger cap.
EMBED_BODY_BYTES = 4000


def embed_text(note: Any) -> str:
    """The text that represents a note to an embedder.

    Structured fields lead because they are the note's own account of what it
    is about, and the Answer block is the sentence the vault was built to
    return. The body follows as context rather than as the primary signal.
    """
    fm = note.frontmatter
    body = note.body.decode("utf-8", "replace")[:EMBED_BODY_BYTES]
    parts = [fm.title, " ".join(fm.aliases), " ".join(fm.tags), note.answer or "", body]
    return "\n".join(p for p in parts if p)


def squash(text: str) -> str:
    """Collapse separators *inside* a word: `Wi-Fi` -> `wifi`.

    FTS5's tokenizer splits `Wi-Fi` into `wi` and `fi`, so a query for `wifi`
    can never match it however good the aliases are. Indexing a squashed copy of
    the title, aliases and tags, and squashing query terms the same way, makes
    the two sides meet. The transform is symmetric, so `wi-fi`, `Wi-Fi` and
    `wifi` all resolve to the same token.
    """
    words: list[str] = []
    for piece in re.split(r"[\s,;/|]+", text):
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", piece)
        if cleaned:
            words.append(cleaned.lower())
    return " ".join(words)


#: A trailing contraction fragment: the `t` of "doesn't", the `s` of "what's".
_CONTRACTION = re.compile(r"[\u2019'](?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)


def _fts_query(raw: str) -> str:
    """Turn a human question into a safe FTS5 expression.

    Every term is quoted, so punctuation a user types cannot become an operator.
    Stopwords are dropped, and each surviving term also contributes its squashed
    form so hyphenated names match unhyphenated queries.
    """
    # Strip contraction suffixes before tokenizing. The character pass below
    # turns "doesn't" into "doesn t", and that stray "t" then went into the
    # query as a search term -- it matches 77 of 297 notes in the real vault,
    # and "s" matches 138. Deliberately keyed on the apostrophe rather than on
    # token length: `ip`, `v1` and `am` are legitimate two-character search
    # terms here, and dropping those would break networking and version queries.
    cleaned = _CONTRACTION.sub("", raw)
    terms = [t for t in "".join(c if c.isalnum() or c in "-_" else " "
                                for c in cleaned).split() if t]
    content = [t for t in terms if t.lower() not in STOPWORDS]
    # A query made entirely of stopwords still deserves an answer; fall back
    # rather than returning nothing.
    chosen = content or terms
    if not chosen:
        return '""'

    variants: list[str] = []
    for term in chosen:
        lowered = term.lower()
        variants.append(lowered)
        squashed = squash(term)
        if squashed and squashed != lowered:
            variants.append(squashed)
    unique = list(dict.fromkeys(variants))
    return " OR ".join(f'"{v}"' for v in unique)
