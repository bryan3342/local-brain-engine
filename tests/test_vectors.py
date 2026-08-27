"""Dense vectors as a derived, optional projection of the vault.

Two properties carry all the weight here. Vectors must be *optional*, with no
embedder attached, every path has to behave exactly as it did before one
existed, because a hook that fires on every prompt cannot depend on a model
being reachable. And they must be *derived*, deleting the database and
rebuilding has to reproduce them byte for byte, the same guarantee the FTS
index already holds.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.infrastructure.index import SqliteIndex


class StubEmbedder:
    """A real embedder, not a mock: deterministic, offline, and total.

    Hashing text to a unit vector gives the one property the tests actually
    need, identical text always yields an identical vector, without pulling a
    model into the suite. Calls are recorded so a test can prove that an
    unchanged note is not re-embedded.
    """

    dim = 8

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [b / 255.0 for b in digest[: self.dim]]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / norm for x in raw])
        return out


def build(tmp_path: Path, embedder: object | None, name: str = "vault") -> KnowledgeApi:
    config = BrainConfig(root=tmp_path / name, git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root, embedder=embedder))
    api.initialize_vault()
    return api


def populate(api: KnowledgeApi) -> None:
    api.create_note("Fiber-Optic Cabling", "concept", domain="networking",
                    aliases=["fibre"], answer="Light, not current.")
    api.create_note("Copper Cabling", "concept", domain="networking",
                    answer="Electrical pulses; attenuates over distance.")


# ---- optionality -----------------------------------------------------------


def test_sync_stores_a_vector_per_note_when_an_embedder_is_attached(tmp_path: Path) -> None:
    api = build(tmp_path, StubEmbedder())
    populate(api)

    assert api.reindex(full=True).vectors == 2


def test_sync_stores_no_vectors_when_no_embedder_is_attached(tmp_path: Path) -> None:
    """The null path. No model reachable must mean no vectors, not an error."""
    api = build(tmp_path, None)
    populate(api)

    assert api.reindex(full=True).vectors == 0


# ---- derived, not canonical ------------------------------------------------


def test_deleting_the_database_reproduces_identical_vectors(tmp_path: Path) -> None:
    """The same guarantee the FTS index already carries, extended to vectors.

    Would fail if a vector were ever derived from anything but the Markdown,
    an insertion counter, a timestamp, a row id.
    """
    api = build(tmp_path, StubEmbedder())
    populate(api)
    api.reindex(full=True)
    index = api.index
    assert isinstance(index, SqliteIndex)

    before = dict(index.connect().execute("SELECT id, vec FROM vectors").fetchall())
    index.drop()
    api.reindex(full=True)
    after = dict(index.connect().execute("SELECT id, vec FROM vectors").fetchall())

    assert before and before == after


def test_an_unchanged_note_is_not_re_embedded(tmp_path: Path) -> None:
    """Embedding is the expensive step; the staleness stamp must gate it too.

    Would fail if `_store_vector` ran outside the reparse branch, every sync
    would re-embed the whole vault.
    """
    embedder = StubEmbedder()
    api = build(tmp_path, embedder)
    populate(api)

    before_full = len(embedder.seen)
    api.reindex(full=True)
    embedded_by_full = len(embedder.seen) - before_full

    settled = len(embedder.seen)
    api.reindex()

    assert embedded_by_full == 2, "a full rebuild embeds every note exactly once"
    assert len(embedder.seen) == settled, "an incremental sync embeds nothing new"


def test_attaching_an_embedder_later_backfills_the_missing_vectors(tmp_path: Path) -> None:
    """The upgrade path, and the one every existing vault will take.

    An index built before embeddings existed has fresh file stamps for every
    note, so an incremental sync skips them all and no vector is ever written.
    Retrieval would silently stay keyword-only until someone thought to force a
    full rebuild. Missing vector is its own kind of stale.
    """
    config = BrainConfig(root=tmp_path / "vault", git=False)
    plain = SqliteIndex(config.root)
    api = KnowledgeApi(config, index=plain)
    api.initialize_vault()
    populate(api)
    api.reindex(full=True)
    plain.close()

    embedder = StubEmbedder()
    upgraded = SqliteIndex(config.root, embedder=embedder)
    stats = KnowledgeApi(config, index=upgraded).reindex()

    assert stats.vectors == 2
    assert stats.reindexed == 0, "backfilling a vector must not count as a reparse"


def test_a_failing_embedder_is_not_retried_for_every_note(tmp_path: Path) -> None:
    """One dead model must not cost one timeout per note.

    A full rebuild of a real vault is hundreds of notes. At a ten-second HTTP
    timeout each, retrying a model that is simply not running turns a sync into
    an hour of waiting. Fail once, then stop asking.
    """

    class DeadEmbedder:
        dim = 8

        def __init__(self) -> None:
            self.attempts = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.attempts += 1
            raise OSError("connection refused")

    embedder = DeadEmbedder()
    config = BrainConfig(root=tmp_path / "vault", git=False)
    api = KnowledgeApi(config, index=SqliteIndex(config.root, embedder=embedder))
    api.initialize_vault()
    populate(api)

    stats = api.reindex(full=True)

    assert stats.notes == 2, "notes still index when embedding is unavailable"
    assert stats.vectors == 0
    assert embedder.attempts == 1, "the dead model is asked once, not once per note"


# ---- wiring ----------------------------------------------------------------


def test_no_embedder_is_built_unless_one_is_asked_for(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense retrieval is opt-in, so the default install gains no dependency.

    Would fail if the CLI constructed an embedder eagerly: every `brain create`
    in a vault on a machine with no model would pay a failed HTTP round trip,
    and the suite would stop being hermetic.
    """
    from brain.infrastructure.embedder import embedder_from_env

    monkeypatch.delenv("BRAIN_EMBEDDER", raising=False)
    assert embedder_from_env() is None

    monkeypatch.setenv("BRAIN_EMBEDDER", "none")
    assert embedder_from_env() is None


def test_asking_for_ollama_builds_an_ollama_embedder(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.infrastructure.embedder import OllamaEmbedder, embedder_from_env

    monkeypatch.setenv("BRAIN_EMBEDDER", "ollama")
    built = embedder_from_env()

    assert isinstance(built, OllamaEmbedder)
    assert built.dim == 768


# ---- fusion ----------------------------------------------------------------


TOPIC_WORDS = {
    "optical": ("fiber", "fibre", "light", "photon", "glass", "optic"),
    "electrical": ("copper", "current", "voltage", "electric", "pulse"),
}


class TopicEmbedder:
    """A real embedder over a two-topic vocabulary.

    Enough semantics to write an honest test: a query can land near a note it
    shares no literal word with, which is exactly the case keyword retrieval
    cannot serve and the only reason to add vectors at all.
    """

    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            low = text.lower()
            raw = [1.0 if any(w in low for w in words) else 0.0
                   for words in TOPIC_WORDS.values()]
            norm = math.sqrt(sum(x * x for x in raw))
            out.append([x / norm for x in raw] if norm else raw)
        return out


def test_fusion_finds_a_note_that_shares_no_word_with_the_query(tmp_path: Path) -> None:
    """The whole case for dense retrieval, in one query.

    "photons" appears nowhere in the vault, so BM25 has nothing to match and
    returns the note only by accident or not at all. Would fail if search
    ranked on the keyword index alone.
    """
    api = build(tmp_path, TopicEmbedder())
    populate(api)
    api.reindex(full=True)

    hits = [h.ref.id for h in api.search_notes("photons travelling through glass")]

    assert hits and hits[0] == "concept_fiber-optic-cabling"


def test_search_without_an_embedder_is_unchanged(tmp_path: Path) -> None:
    """Fusion is additive. With no vectors, results must be exactly as before.

    Would fail if the fusion path ran unconditionally and reordered keyword
    results, or crashed looking for a vector table it never populated.
    """
    api = build(tmp_path, None)
    populate(api)
    api.reindex(full=True)

    assert [h.ref.id for h in api.search_notes("fibre")] == ["concept_fiber-optic-cabling"]
    assert api.search_notes("photons travelling through glass") == []


# ---- gate signal -----------------------------------------------------------


def test_a_fused_hit_carries_the_similarity_that_found_it(tmp_path: Path) -> None:
    """The gate needs a signal that means something in absolute terms.

    BM25 ranks are unnormalised and RRF scores encode only position, so
    neither can tell an answerable prompt from an unanswerable one. Cosine can,
    but only if search reports it. Would fail while `SearchHit` carried nothing
    but the keyword score.
    """
    api = build(tmp_path, TopicEmbedder())
    populate(api)
    api.reindex(full=True)

    hits = api.search_notes("photons travelling through glass")

    scores = {h.ref.id: h.similarity for h in hits}
    assert all(v is not None for v in scores.values())
    assert scores["concept_fiber-optic-cabling"] > scores["concept_copper-cabling"], (
        "the optical note must sit nearer an optical query than the electrical one"
    )


def test_similarity_is_absent_when_nothing_dense_ran(tmp_path: Path) -> None:
    """`None` rather than 0.0, an unmeasured signal is not a weak one.

    A gate reading 0.0 as 'definitely irrelevant' would go permanently silent
    on a machine with no model, which is precisely the fallback case.
    """
    api = build(tmp_path, None)
    populate(api)
    api.reindex(full=True)

    assert api.search_notes("fibre")[0].similarity is None
