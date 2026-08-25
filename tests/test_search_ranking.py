"""Regression tests for the three query bugs the eval harness exposed.

None of these would have been found by unit-testing search in isolation, they
only showed up as bad *rankings* against real questions. Each one is pinned here
so it cannot come back silently.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.infrastructure.index import (
    BM25_WEIGHTS,
    SCHEMA,
    STOPWORDS,
    SqliteIndex,
    _fts_query,
    squash,
)


@pytest.fixture
def api(tmp_path: Path) -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / "vault", git=False)
    a = KnowledgeApi(config, index=SqliteIndex(config.root))
    a.initialize_vault()
    return a


# ---- bug 1: stopword dilution ---------------------------------------------


def test_stopwords_are_dropped_from_the_query() -> None:
    """`can I push straight to main?` used to OR in `can`, `I` and `to`,
    so every note containing `to` matched and the signal terms drowned."""
    expr = _fts_query("can I push straight to main?")
    assert '"push"' in expr and '"main"' in expr
    for noise in ('"can"', '"i"', '"to"'):
        assert noise not in expr


def test_a_query_of_only_stopwords_still_searches() -> None:
    """Dropping every term would turn a odd question into zero results."""
    expr = _fts_query("what is it")
    assert expr != '""'
    assert '"what"' in expr


def test_empty_query_is_safe() -> None:
    assert _fts_query("   ") == '""'
    assert _fts_query("!!! ???") == '""'


def test_stopwords_do_not_swallow_domain_terms() -> None:
    """A stopword list that ate real vocabulary would be worse than none."""
    for word in ("switch", "main", "push", "fiber", "memory", "index", "note"):
        assert word not in STOPWORDS


# ---- bug 2: tokenization mismatch -----------------------------------------


def test_squash_collapses_separators_inside_a_word() -> None:
    assert squash("Wi-Fi") == "wifi"
    assert squash("single-mode") == "singlemode"
    assert squash("Wi-Fi, WLAN") == "wifi wlan"
    assert squash("") == ""


def test_squash_is_symmetric_across_spellings() -> None:
    """Both sides must land on the same token or the fix does nothing."""
    assert squash("Wi-Fi") == squash("wi-fi") == squash("WIFI") == "wifi"


def test_query_carries_both_raw_and_squashed_forms() -> None:
    expr = _fts_query("wi-fi coverage")
    assert '"wi-fi"' in expr and '"wifi"' in expr


def test_hyphenated_alias_is_findable_by_the_unhyphenated_query(
    api: KnowledgeApi,
) -> None:
    """The end-to-end case: notes say `Wi-Fi`, people type `wifi`.

    FTS5 tokenizes `Wi-Fi` into `wi` + `fi`, so before the squashed column this
    was unreachable however good the aliases were.
    """
    api.create_note("Wireless Media", "concept", domain="networking",
                    aliases=["Wi-Fi", "WLAN"], answer="Radio and microwave frequencies.")
    api.create_note("Copper Cabling", "concept", domain="networking",
                    answer="Electrical pulses over copper.")
    api.reindex(full=True)

    hits = api.search_notes("wifi", limit=5)
    assert hits, "query for 'wifi' found nothing"
    assert hits[0].ref.id == "concept_wireless-media"


# ---- bug 3: bm25 weights misaligned by the UNINDEXED column ----------------


def test_bm25_weight_count_matches_the_column_count() -> None:
    """The off-by-one that silently broke title ranking.

    bm25() weights are positional over *every* column, including UNINDEXED
    ones. Passing one weight too few shifts them all a column left, so the
    largest weight landed on `id`, which can never match, and every other
    column was scored as something it is not.
    """
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    columns = [row[1] for row in con.execute("PRAGMA table_info(notes_fts)")]
    weights = [w.strip() for w in BM25_WEIGHTS.split(",")]
    assert len(weights) == len(columns), (
        f"{len(weights)} bm25 weights for {len(columns)} columns "
        f"({', '.join(columns)}), weights would shift"
    )
    assert columns[0] == "id" and float(weights[0]) == 0.0, (
        "the UNINDEXED id column must be given a zero weight placeholder"
    )


def test_body_is_damped_relative_to_title() -> None:
    """Bodies are long and mention everything incidentally.

    At equal weight they drown the structured metadata that answer-first
    retrieval depends on; damping body moved precision@1 by ~6 points.
    """
    weights = [float(w) for w in BM25_WEIGHTS.split(",")]
    _, title, aliases, _tags, squashed, body = weights
    assert body < 1.0
    assert title > body * 10
    assert aliases > body and squashed > body


# ---- schema migration -----------------------------------------------------


def test_adding_a_column_rebuilds_a_stale_index(tmp_path: Path) -> None:
    """`CREATE VIRTUAL TABLE IF NOT EXISTS` keeps an older shape silently.

    Without a migration every INSERT against an existing database would fail on
    the new column. Dropping is safe because the index is derived.
    """
    config = BrainConfig(root=tmp_path / "vault", git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root))
    api.initialize_vault()
    api.create_note("Alpha", "concept", answer="First.")
    api.reindex(full=True)

    index = api.index
    assert isinstance(index, SqliteIndex)
    con = index.connect()
    con.executescript(
        "DROP TABLE notes_fts;"
        "CREATE VIRTUAL TABLE notes_fts USING fts5("
        "  id UNINDEXED, title, aliases, tags, body, tokenize='porter unicode61');"
    )
    con.commit()
    index.close()

    fresh = SqliteIndex(config.root)
    cols = [r[1] for r in fresh.connect().execute("PRAGMA table_info(notes_fts)")]
    assert "squashed" in cols, "stale schema was not rebuilt"

    rebuilt = KnowledgeApi(config, index=fresh)
    rebuilt.reindex(full=True)
    assert rebuilt.search_notes("Alpha"), "search broken after migration"


def test_contraction_fragments_do_not_become_search_terms() -> None:
    """`doesn't` tokenized to `doesn` + `t`, and the bare `t` went into the
    query as a term, matching 77 of 297 notes in the real vault. A grammatical
    fragment is not a search term."""
    from brain.infrastructure.index import _fts_query

    expr = _fts_query("why doesn't fiber suffer from electrical interference?")
    assert '"t"' not in expr
    assert '"fiber"' in expr

    for raw, fragment in [("what's the schema", '"s"'), ("we're deploying", '"re"'),
                          ("I've configured it", '"ve"'), ("they'll deploy", '"ll"')]:
        assert fragment not in _fts_query(raw), raw


def test_legitimate_short_tokens_survive() -> None:
    """The fix must not become "drop short tokens", `ip`, `v1` and `am` are
    real search terms in this corpus and dropping them would break networking
    and versioning queries."""
    from brain.infrastructure.index import _fts_query

    assert '"ip"' in _fts_query("how do I convert an ip address into binary?")
    assert '"v1"' in _fts_query("are we still allowed to use neurogenesis v1?")
    assert '"am"' in _fts_query("am I allowed to use mocks and fakes in tests?")
