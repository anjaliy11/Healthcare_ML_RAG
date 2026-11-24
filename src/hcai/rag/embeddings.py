"""Embedders behind one protocol. All vectors are float32 and L2-normalised (cosine = dot)."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "do",
        "does",
        "can",
        "i",
        "my",
        "me",
        "you",
        "your",
        "should",
        "about",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int
    is_semantic: bool  # False for the offline hashing embedder (different score scale)

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing vectors. Lexical only: for tests and offline demos,
    never a substitute for a real semantic embedder."""

    is_semantic = False

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-bow-{dim}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += (1.0 + math.log(c)) * (1 if (h >> 63) == 0 else -1)
        return _normalise(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


class SentenceTransformerEmbedder:
    """BAAI/bge-small-en-v1.5 (384-d, CPU). Queries get the model's retrieval instruction."""

    is_semantic = True

    def __init__(self, model_name: str, query_prefix: str, batch_size: int = 32) -> None:
        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model: Any = None
        self.dim = 384 if "bge-small" in model_name else 0  # refined after first load

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            # Windows without Developer Mode cannot symlink; the warning is noise, the cache works.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

            self._model = SentenceTransformer(self.name, device="cpu")
            # Renamed in sentence-transformers 5.x; accept both so neither emits a FutureWarning.
            getter = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self.dim = int(getter())
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])
