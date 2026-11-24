"""BM25, rank fusion, diversity cap, reranking and the ablation harness (all offline)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hcai.config import load_config
from hcai.eval.rag_eval import GoldenItem, ablation_markdown, run_ablation
from hcai.rag.bm25 import BM25Index, bm25_tokens, stem
from hcai.rag.chunking import Chunk, chunk_document
from hcai.rag.embeddings import HashingEmbedder
from hcai.rag.ingest import load_corpus
from hcai.rag.rerank import CrossEncoderReranker
from hcai.rag.retriever import Hit, Retriever, cap_per_doc, rrf_fuse
from hcai.rag.store import FaissStore
from tests.rag_helpers import build_corpus


def _chunk(doc: str, i: int, text: str) -> Chunk:
    return Chunk(f"{doc}:{i:03d}", doc, doc, "u", "S", i, "l", text)


class NoiseEmbedder:
    """Dense retrieval that carries no signal (deterministic random vectors): isolates BM25."""

    is_semantic = True
    dim = 64
    name = "noise-64"

    def _v(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
            v = np.random.default_rng(seed).normal(size=self.dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)

    embed_documents = embed_queries = _v  # type: ignore[assignment]


def _store(tmp_path: Path, emb) -> FaissStore:  # type: ignore[no-untyped-def]
    docs = load_corpus(build_corpus(tmp_path / "docs"))
    chunks = [c for d in docs for c in chunk_document(d, 300, 45, 380)]
    store = FaissStore(emb.dim, emb.name)
    store.add(emb.embed_documents([c.embed_text for c in chunks]), chunks)
    return store


# ------------------------------------------------------------------ BM25
def test_stem_and_tokens_fold_plurals_and_drop_stopwords():
    assert stem("risks") == "risk" and stem("studies") == "study" and stem("glass") == "glass"
    assert bm25_tokens("What are the Risk Factors?") == ["risk", "factor"]


def test_bm25_ranks_exact_term_match_first_and_skips_non_matches():
    chunks = [
        _chunk("a", 0, "Hypertension raises stroke risk. Hypertension is common."),
        _chunk("b", 0, "Diabetes causes thirst and frequent urination."),
        _chunk("c", 0, "Healthy diet includes fruit and vegetables."),
    ]
    idx = BM25Index(chunks)
    hits = idx.search("hypertension")
    assert [c.doc_id for c, _ in hits] == ["a"]  # only chunks sharing a term are returned
    assert idx.search("zebra quantum") == [] and idx.search("") == []
    assert idx.search("thirst diabetes")[0][0].doc_id == "b"
    assert len(idx.search("diet fruit diabetes", k=1)) == 1


def test_bm25_idf_prefers_rare_terms_and_handles_empty_corpus():
    chunks = [_chunk("a", i, f"blood pressure note {i}") for i in range(5)] + [
        _chunk("z", 0, "blood pressure and albuminuria")
    ]
    assert BM25Index(chunks).search("blood albuminuria")[0][0].doc_id == "z"
    assert BM25Index([]).search("anything") == []


# ------------------------------------------------------------------ fusion / diversity
def test_rrf_rewards_agreement_and_is_deterministic():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b" and ids[1] == "a"  # present in both lists
    assert set(ids) == {"a", "b", "c", "d"}
    assert rrf_fuse([["a", "b"], ["b", "a"]]) == rrf_fuse([["a", "b"], ["b", "a"]])
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
    score = dict(fused)
    assert score["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_cap_per_doc_limits_and_preserves_order():
    hits = [Hit(_chunk(d, i, "t"), 0.5, 0) for i, d in enumerate("aaabab")]
    out = cap_per_doc(hits, 2, 10)
    assert [h.chunk.doc_id for h in out] == ["a", "a", "b", "b"]
    assert len(cap_per_doc(hits, None, 3)) == 3
    assert len(cap_per_doc(hits, 1, 5)) == 2


# ------------------------------------------------------------------ Retriever modes
def test_hybrid_recovers_the_right_document_when_dense_has_no_signal(tmp_path: Path):
    emb = NoiseEmbedder()
    store = _store(tmp_path, emb)
    q = "hypertension risk factors family history salt"

    def rank(mode: str) -> int:
        r = Retriever(emb, store, 5, -1.0, 500, mode=mode)
        ids = [h.chunk.doc_id for h in r.retrieve(q).hits]
        return ids.index("fx-hypertension") + 1 if "fx-hypertension" in ids else 99

    assert rank("bm25") == 1
    assert rank("hybrid") <= 2  # lexical signal survives fusion with an uninformative dense list
    assert rank("hybrid") <= rank("dense")


def test_hits_carry_dense_and_fused_scores_and_gate_uses_dense(tmp_path: Path):
    emb = HashingEmbedder()
    store = _store(tmp_path, emb)
    r = Retriever(emb, store, 5, 0.15, 500, mode="hybrid")
    res = r.retrieve("What are the risk factors for high blood pressure?")
    assert res.answerable and res.hits[0].fused > 0
    assert all(h.score >= 0.15 for h in res.usable)
    assert res.top_score == pytest.approx(
        max(
            s
            for _, s in store.search(
                emb.embed_queries(["What are the risk factors for high blood pressure?"])[0], 50
            )
        )
    )
    far = r.retrieve("zebra quantum volleyball")
    assert not far.answerable  # no lexical or semantic support: still refused
    dense = Retriever(emb, store, 5, 0.15, 500, mode="dense").retrieve("zebra quantum volleyball")
    assert dense.hits and dense.hits[0].fused == 0.0  # dense mode records no fusion score


def test_diversity_cap_spreads_hits_across_documents(tmp_path: Path):
    emb = HashingEmbedder()
    store = _store(tmp_path, emb)
    q = "salt blood pressure diet diabetes symptoms"
    uncapped = Retriever(emb, store, 6, -1.0, 500, mode="hybrid").retrieve(q).hits
    capped = Retriever(emb, store, 6, -1.0, 500, mode="hybrid", max_per_doc=1).retrieve(q).hits
    d_un, d_cap = (len({h.chunk.doc_id for h in x}) for x in (uncapped, capped))
    assert d_cap == len(capped) and d_cap >= d_un
    assert [h.rank for h in capped] == list(range(1, len(capped) + 1))


class KeywordReranker:
    name = "kw"

    def __init__(self, word: str) -> None:
        self.word, self.seen = word, 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.seen = len(texts)
        return [float(t.lower().count(self.word)) for t in texts]


def test_reranker_reorders_candidates_but_not_the_answerability_gate(tmp_path: Path):
    emb = HashingEmbedder()
    store = _store(tmp_path, emb)
    q = "blood pressure and diabetes"
    base = Retriever(emb, store, 5, -1.0, 500, mode="hybrid").retrieve(q)
    rr = KeywordReranker("diabetes")
    res = Retriever(emb, store, 5, -1.0, 500, mode="hybrid", reranker=rr, rerank_top_n=10).retrieve(
        q
    )
    assert res.hits[0].chunk.text.lower().count("diabetes") >= base.hits[
        0
    ].chunk.text.lower().count("diabetes")
    assert res.hits[0].rerank is not None and rr.seen <= 10
    assert res.top_score == base.top_score  # gate scale unchanged by reranking
    gated = Retriever(emb, store, 5, 0.99, 500, mode="hybrid", reranker=rr).retrieve(q)
    assert not gated.answerable


def test_cross_encoder_wrapper_pairs_query_with_texts_without_downloading():
    class FakeCE:
        def predict(self, pairs):
            self.pairs = pairs
            return np.array([0.1, 0.9])

    ce = CrossEncoderReranker("BAAI/bge-reranker-base")
    ce._model = FakeCE()
    assert ce.score("q", ["a", "b"]) == [pytest.approx(0.1), pytest.approx(0.9)]
    assert ce._model.pairs == [("q", "a"), ("q", "b")]
    assert ce.score("q", []) == []


def test_retriever_rejects_bad_mode_and_stores_without_chunks(tmp_path: Path):
    emb = HashingEmbedder()
    store = _store(tmp_path, emb)
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        Retriever(emb, store, 5, 0.1, 500, mode="magic")

    class Bare:
        embedding_model = emb.name

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="chunks"):
        Retriever(emb, Bare(), 5, 0.1, 500, mode="hybrid")  # type: ignore[arg-type]


# ------------------------------------------------------------------ ablation harness
def test_ablation_reports_every_variant_with_valid_metrics(tmp_path: Path):
    emb = HashingEmbedder()
    store = _store(tmp_path, emb)
    items = [
        GoldenItem(
            "a", "risk factors for high blood pressure", "answerable", "answer", ["fx-hypertension"]
        ),
        GoldenItem("b", "symptoms of diabetes", "answerable", "answer", ["fx-diabetes"]),
        GoldenItem(
            "c", "who won the cup", "unanswerable", "refuse", []
        ),  # ignored: no expected doc
    ]
    rag = load_config().rag
    rows = run_ablation(emb, store, items, rag)
    assert [r["variant"] for r in rows] == [
        "dense",
        "bm25",
        "hybrid (RRF)",
        "hybrid + diversity cap",
    ]
    for r in rows:
        assert r["n"] == 2
        assert 0 <= r["hit@1"] <= r["hit@3"] <= r["hit@5"] <= 1 and 0 <= r["mrr"] <= 1
        assert r["distinct_docs@k"] >= 1
    assert rows[0]["hit@1"] == 1.0  # sanity: the easy fixture questions are solved by dense

    with_rr = run_ablation(emb, store, items, rag, KeywordReranker("diabetes"))
    assert with_rr[-1]["variant"] == "hybrid + diversity + rerank"
    md = ablation_markdown(with_rr)
    assert "Retrieval ablation" in md and "| dense |" in md and "rerank" in md
    assert ablation_markdown(None) == "" and ablation_markdown([]) == ""


def test_default_config_uses_dense_with_cap_and_rerank_off():
    rag = load_config().rag
    assert rag.retrieval_mode == "dense" and rag.max_per_doc == 3 and rag.rerank is False
    assert rag.rrf_k == 60
