"""End-to-end RAG: safety gate -> retrieve (threshold) -> generate -> validate citations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from hcai import DISCLAIMER
from hcai.config import Config
from hcai.logging import get_logger, log_event, new_request_id
from hcai.monitoring.metrics import MetricsSink, NullMetrics
from hcai.rag.chunking import chunk_document
from hcai.rag.embeddings import Embedder
from hcai.rag.generator import Generator
from hcai.rag.ingest import load_corpus
from hcai.rag.retriever import Retriever
from hcai.rag.safety import check_emergency
from hcai.rag.store import FaissStore

log = get_logger("rag.pipeline")

REFUSAL_MESSAGE = (
    "I can't answer that from the documents I have. The indexed health documents don't cover "
    "it, so I won't guess. Please ask a clinician or pharmacist."
)


@dataclass(frozen=True)
class Source:
    cite: str  # "C1"
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    license: str
    score: float
    snippet: str


@dataclass
class Answer:
    request_id: str
    text: str
    mode: str  # "llm" | "extractive" | "refusal" | "emergency"
    refused: bool
    sources: list[Source] = field(default_factory=list)
    top_score: float = 0.0
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class RagPipeline:
    def __init__(
        self, retriever: Retriever, generator: Generator, metrics: MetricsSink | None = None
    ) -> None:
        self.retriever, self.generator = retriever, generator
        self.metrics = metrics or NullMetrics()

    def ask(self, question: str) -> Answer:
        rid, t0 = new_request_id(), time.perf_counter()

        def done(ans: Answer, **extra: object) -> Answer:
            ans.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.metrics.record(
                request_id=rid, kind="ask", latency_ms=ans.latency_ms, top_score=ans.top_score,
                refused=ans.refused, mode=ans.mode, n_sources=len(ans.sources),
                fallback=any(w.startswith("llm_fallback") for w in ans.warnings),
                emergency=ans.mode == "emergency", **extra,
            )  # fmt: skip
            log_event(
                log, "ask_done", mode=ans.mode, refused=ans.refused,
                top_score=round(ans.top_score, 3), latency_ms=ans.latency_ms,
                question_chars=len(question),
            )  # fmt: skip
            return ans

        alert = check_emergency(question)
        if alert:
            return done(Answer(rid, alert.message, "emergency", refused=False))
        retrieval = self.retriever.retrieve(question)  # raises InvalidQueryError on empty input
        warnings = ["query_truncated"] if retrieval.query_truncated else []
        if not retrieval.answerable:
            return done(
                Answer(
                    rid,
                    REFUSAL_MESSAGE,
                    "refusal",
                    True,
                    top_score=retrieval.top_score,
                    warnings=warnings,
                )
            )
        gen = self.generator.generate(question, retrieval.usable)
        warnings += gen.warnings
        tokens = {"input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens}
        if gen.mode == "refusal" or not gen.cited:
            return done(
                Answer(
                    rid,
                    REFUSAL_MESSAGE,
                    "refusal",
                    True,
                    top_score=retrieval.top_score,
                    warnings=warnings,
                ),
                **tokens,
            )
        sources = [
            Source(
                cite=f"C{i}",
                chunk_id=(h := retrieval.usable[i - 1]).chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                section=h.chunk.section,
                url=h.chunk.url,
                license=h.chunk.license,
                score=round(h.score, 4),
                snippet=h.chunk.text[:300],
            )
            for i in gen.cited
        ]
        return done(
            Answer(rid, gen.text, gen.mode, False, sources, retrieval.top_score, warnings=warnings),
            **tokens,
        )


def build_index(cfg: Config, embedder: Embedder) -> FaissStore:
    """Ingest -> chunk -> embed -> FAISS, saved to cfg.paths.index_dir."""
    docs = load_corpus(cfg.paths.docs_dir)
    chunks = [
        c
        for d in docs
        for c in chunk_document(
            d, cfg.rag.chunk_target_words, cfg.rag.chunk_overlap_words, cfg.rag.chunk_max_words
        )
    ]
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store = FaissStore(vectors.shape[1], embedder.name)
    store.add(vectors, chunks)
    store.save(cfg.paths.index_dir)
    log_event(log, "index_built", documents=len(docs), chunks=len(chunks), model=embedder.name)
    return store
