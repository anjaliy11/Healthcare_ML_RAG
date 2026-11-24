"""Retrieval: dense, BM25 or hybrid (RRF), optional rerank, per-document diversity cap, then the
score-threshold layer of the 'not in documents' logic.

The answerability gate always uses the DENSE cosine score (the only calibrated scale); fusion and
reranking only change the order and which chunks fill the top-k."""

from __future__ import annotations

from dataclasses import dataclass, field

from hcai.errors import IndexCorruptError, InvalidQueryError
from hcai.rag.bm25 import BM25Index
from hcai.rag.chunking import Chunk
from hcai.rag.embeddings import Embedder
from hcai.rag.rerank import Reranker
from hcai.rag.store import VectorStore

MODES = ("dense", "bm25", "hybrid")
MAX_DENSE_CANDIDATES = 5000  # exact search over everything is fine at this corpus size


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # dense cosine similarity (calibrated scale used by the threshold)
    rank: int  # 1-based final rank
    fused: float = 0.0  # reciprocal-rank-fusion score (0 in dense mode)
    rerank: float | None = None


@dataclass
class Retrieval:
    hits: list[Hit]  # final top-k, regardless of threshold (used by evaluation)
    top_score: float  # best dense score over the whole corpus
    threshold: float
    query_truncated: bool = False
    usable: list[Hit] = field(default_factory=list)  # hits with dense score >= threshold

    @property
    def answerable(self) -> bool:
        return bool(self.usable)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over ranked id lists: score(d) = sum 1 / (k + rank_in_list)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def cap_per_doc(items: list[Hit], limit: int | None, top_k: int) -> list[Hit]:
    """Keep order, allow at most `limit` chunks per document, return up to top_k."""
    if limit is None:
        return items[:top_k]
    seen: dict[str, int] = {}
    out: list[Hit] = []
    for h in items:
        if seen.get(h.chunk.doc_id, 0) >= limit:
            continue
        seen[h.chunk.doc_id] = seen.get(h.chunk.doc_id, 0) + 1
        out.append(h)
        if len(out) == top_k:
            break
    return out


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        threshold: float,
        max_query_chars: int,
        *,
        mode: str = "dense",
        rrf_k: int = 60,
        max_per_doc: int | None = None,
        reranker: Reranker | None = None,
        rerank_top_n: int = 20,
    ) -> None:
        if store.embedding_model != embedder.name:
            raise IndexCorruptError(
                f"Index was built with '{store.embedding_model}' but the active embedder is "
                f"'{embedder.name}'. Rebuild with `python -m hcai index`."
            )
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from {MODES}.")
        self.embedder, self.store = embedder, store
        self.top_k, self.threshold, self.max_chars = top_k, threshold, max_query_chars
        self.mode, self.rrf_k, self.max_per_doc = mode, rrf_k, max_per_doc
        self.reranker, self.rerank_top_n = reranker, rerank_top_n
        self.bm25: BM25Index | None = None
        if mode != "dense":
            chunks = getattr(store, "chunks", None)
            if chunks is None:
                raise ValueError("BM25 modes need a store that exposes `.chunks`.")
            self.bm25 = BM25Index(chunks)

    def retrieve(self, query: str) -> Retrieval:
        query = " ".join(query.split())
        if not query:
            raise InvalidQueryError("Please enter a question.")
        truncated = len(query) > self.max_chars
        if truncated:
            query = query[: self.max_chars]

        vec = self.embedder.embed_queries([query])[0]
        n = min(max(len(self.store), 1), MAX_DENSE_CANDIDATES)
        dense = self.store.search(vec, n)  # sorted best-first
        dense_score = {c.chunk_id: s for c, s in dense}
        by_id = {c.chunk_id: c for c, _ in dense}
        top_score = dense[0][1] if dense else 0.0

        fused: dict[str, float] = {}
        if self.mode == "dense":
            order = [c.chunk_id for c, _ in dense]
        else:
            assert self.bm25 is not None
            lexical = self.bm25.search(query)
            by_id.update({c.chunk_id: c for c, _ in lexical})
            lex_ids = [c.chunk_id for c, _ in lexical]
            if self.mode == "bm25":
                order = lex_ids or [c.chunk_id for c, _ in dense]  # no term overlap: fall back
            else:
                fused_list = rrf_fuse([[c.chunk_id for c, _ in dense], lex_ids], self.rrf_k)
                fused = dict(fused_list)
                order = [cid for cid, _ in fused_list]

        rerank_scores: dict[str, float] = {}
        if self.reranker is not None and order:
            head = order[: self.rerank_top_n]
            scores = self.reranker.score(query, [by_id[c].text for c in head])
            rerank_scores = dict(zip(head, scores, strict=True))
            head.sort(key=lambda cid: -rerank_scores[cid])  # stable: ties keep fused order
            order = head + order[self.rerank_top_n :]

        candidates = [
            Hit(
                by_id[cid],
                dense_score.get(cid, 0.0),
                i + 1,
                fused.get(cid, 0.0),
                rerank_scores.get(cid),
            )
            for i, cid in enumerate(order)
        ]
        picked = cap_per_doc(candidates, self.max_per_doc, self.top_k)
        hits = [Hit(h.chunk, h.score, i + 1, h.fused, h.rerank) for i, h in enumerate(picked)]
        usable = [h for h in hits if h.score >= self.threshold]
        return Retrieval(hits, top_score, self.threshold, truncated, usable)
