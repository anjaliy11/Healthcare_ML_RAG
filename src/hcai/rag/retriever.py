"""Top-k dense retrieval plus the score-threshold layer of the 'not in documents' logic."""

from __future__ import annotations

from dataclasses import dataclass, field

from hcai.errors import IndexCorruptError, InvalidQueryError
from hcai.rag.chunking import Chunk
from hcai.rag.embeddings import Embedder
from hcai.rag.store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


@dataclass
class Retrieval:
    hits: list[Hit]  # top-k, regardless of threshold (used by evaluation)
    top_score: float
    threshold: float
    query_truncated: bool = False
    usable: list[Hit] = field(default_factory=list)  # hits with score >= threshold

    @property
    def answerable(self) -> bool:
        return bool(self.usable)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        threshold: float,
        max_query_chars: int,
    ) -> None:
        if store.embedding_model != embedder.name:
            raise IndexCorruptError(
                f"Index was built with '{store.embedding_model}' but the active embedder is "
                f"'{embedder.name}'. Rebuild with `python -m hcai index`."
            )
        self.embedder, self.store = embedder, store
        self.top_k, self.threshold, self.max_chars = top_k, threshold, max_query_chars

    def retrieve(self, query: str) -> Retrieval:
        query = " ".join(query.split())
        if not query:
            raise InvalidQueryError("Please enter a question.")
        truncated = len(query) > self.max_chars
        if truncated:
            query = query[: self.max_chars]
        vec = self.embedder.embed_queries([query])[0]
        found = self.store.search(vec, self.top_k)
        hits = [Hit(c, s, i + 1) for i, (c, s) in enumerate(found)]
        usable = [h for h in hits if h.score >= self.threshold]
        return Retrieval(
            hits=hits,
            top_score=hits[0].score if hits else 0.0,
            threshold=self.threshold,
            query_truncated=truncated,
            usable=usable,
        )
