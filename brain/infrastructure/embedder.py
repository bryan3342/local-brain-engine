"""The seam between the vault and whatever produces dense vectors.

Kept as a protocol with no default implementation on purpose. The index, the
CLI and the retrieval hook all have to work when nothing here is reachable,
the hook fires on every prompt typed, and a personal knowledge base that stops
answering because a model is down has traded a real guarantee for a marginal
one. So `None` is a first-class value everywhere an embedder is accepted, and
absence degrades to keyword retrieval rather than raising.

`embed` takes and returns a list so a provider that supports batching can use
it. Ollama does not, and loops; the interface does not need to know that.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a fixed-width vector."""

    #: Width of every vector this embedder returns. Stored alongside the data so
    #: a model swap is detectable rather than silently producing garbage scores.
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Local embeddings over Ollama's HTTP API.

    Local rather than hosted because this runs inside a hook on every prompt:
    a network round trip to a third party would put someone else's availability
    in front of the user's own notes.
    """

    def __init__(self, model: str = "nomic-embed-text",
                 host: str = "http://localhost:11434", dim: int = 768,
                 timeout: float = 10.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embeddings", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            vector = json.loads(response.read())["embedding"]
        return [float(x) for x in vector]

    def available(self) -> bool:
        """Whether this embedder can be reached right now.

        Callers use this to decide between dense and keyword retrieval before
        doing any work, so it must never raise, an unreachable model is an
        ordinary condition, not an error.
        """
        try:
            self.embed(["ping"])
        except (urllib.error.URLError, OSError, ValueError, KeyError, TimeoutError):
            return False
        return True


def embedder_from_env() -> Embedder | None:
    """Build whatever `BRAIN_EMBEDDER` asks for, or nothing.

    Opt-in rather than autodetected. Probing for a model on every command would
    make the common case, a vault on a machine with nothing running, pay a
    failed round trip to discover what the environment already knows, and it
    would quietly add a runtime dependency to a project that has exactly one.
    """
    choice = os.environ.get("BRAIN_EMBEDDER", "none").strip().lower()
    if choice in ("", "none", "off", "0", "false"):
        return None
    if choice == "ollama":
        return OllamaEmbedder(
            model=os.environ.get("BRAIN_EMBED_MODEL", "nomic-embed-text"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    raise ValueError(
        f"unknown BRAIN_EMBEDDER {choice!r}, expected 'ollama' or 'none'")
