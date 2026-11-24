"""Small in-memory BM25 (Okapi) so lexical matching needs no extra dependency.

Why: medical queries hinge on exact terms (hypertension, HbA1c, mmHg) that dense embeddings can
blur. BM25 is rebuilt from the stored chunks at load time (milliseconds for < 10k chunks)."""

from __future__ import annotations

import math
from collections import Counter

from hcai.rag.chunking import Chunk
from hcai.rag.embeddings import tokenize


def stem(token: str) -> str:
    """Very light plural folding, applied identically to documents and queries."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def bm25_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(chunks):
            toks = bm25_tokens(c.embed_text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((i, tf))
        n = len(chunks)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """Chunks with a positive score, best first. Chunks sharing no term are not returned."""
        scores: dict[int, float] = {}
        for term in set(bm25_tokens(query)):
            for i, tf in self.postings.get(term, []):
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.chunks[i], s) for i, s in ranked[:k]]
