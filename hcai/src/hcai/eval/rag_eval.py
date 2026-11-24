"""RAG evaluation on the golden set: retrieval quality, refusal behaviour, faithfulness proxy,
latency, and the threshold trade-off curve. Run on demand (`python -m hcai eval`), not in CI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcai.config import RagConfig
from hcai.rag.embeddings import Embedder, tokenize
from hcai.rag.pipeline import Answer, RagPipeline
from hcai.rag.rerank import Reranker
from hcai.rag.retriever import Retriever
from hcai.rag.store import VectorStore

GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")
_CITE = __import__("re").compile(r"\[C\d+\]")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    category: str  # answerable | unanswerable | adversarial
    expect: str  # answer | refuse | emergency
    expected_doc_ids: list[str]


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    return [
        GoldenItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def doc_rank(retrieved_doc_ids: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk from an expected document."""
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in expected:
            return i
    return None


def faithfulness_proxy(answer: str, cited_texts: list[str], min_overlap: float = 0.6) -> float:
    """Share of answer sentences whose content words mostly appear in the cited passages.
    A cheap lexical proxy, NOT an entailment check; confirm with human spot checks."""
    source = set(tokenize(" ".join(cited_texts)))
    sents = [s.strip(" -•") for s in _CITE.sub("", answer).replace("\n", ". ").split(".")]
    sents = [s for s in sents if len(s.split()) >= 4]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        toks = tokenize(s)
        if toks and sum(t in source for t in toks) / len(toks) >= min_overlap:
            ok += 1
    return ok / len(sents)


def threshold_sweep(
    answer_scores: list[float], refuse_scores: list[float], n: int = 25
) -> list[dict[str, float]]:
    """False-refusal vs false-answer rates when thresholding the top-1 retrieval score."""
    allscores = answer_scores + refuse_scores
    if not allscores:
        return []
    rows = []
    for thr in np.linspace(min(allscores), max(allscores), n):
        frr = float(np.mean([s < thr for s in answer_scores])) if answer_scores else 0.0
        far = float(np.mean([s >= thr for s in refuse_scores])) if refuse_scores else 0.0
        rows.append({"threshold": float(thr), "false_refusal_rate": frr, "false_answer_rate": far})
    return rows


def recommend_threshold(sweep: list[dict[str, float]]) -> float | None:
    """Maximise (1 - FRR) + (1 - FAR); ties go to the higher (safer) threshold."""
    if not sweep:
        return None
    best = max(
        sweep, key=lambda r: (2 - r["false_refusal_rate"] - r["false_answer_rate"], r["threshold"])
    )
    return best["threshold"]


def run_ablation(
    embedder: Embedder,
    store: VectorStore,
    items: list[GoldenItem],
    rag: RagConfig,
    reranker: Reranker | None = None,
) -> list[dict[str, Any]]:
    """Compare retrieval strategies on the golden items that have an expected document.

    The answerability threshold is disabled here (-1): this isolates ranking quality."""
    import time

    labelled = [i for i in items if i.expected_doc_ids]
    variants: list[tuple[str, dict[str, Any]]] = [
        ("dense", {"mode": "dense"}),
        ("bm25", {"mode": "bm25"}),
        ("hybrid (RRF)", {"mode": "hybrid"}),
        ("hybrid + diversity cap", {"mode": "hybrid", "max_per_doc": rag.max_per_doc}),
    ]
    if reranker is not None:
        variants.append(
            ("hybrid + diversity + rerank",
             {"mode": "hybrid", "max_per_doc": rag.max_per_doc, "reranker": reranker,
              "rerank_top_n": rag.rerank_top_n})
        )  # fmt: skip
    rows: list[dict[str, Any]] = []
    for name, kw in variants:
        r = Retriever(embedder, store, rag.top_k, -1.0, rag.max_query_chars, rrf_k=rag.rrf_k, **kw)
        ranks, distinct, t0 = [], [], time.perf_counter()
        for it in labelled:
            hits = r.retrieve(it.question).hits
            ids = [h.chunk.doc_id for h in hits]
            ranks.append(doc_rank(ids, it.expected_doc_ids))
            distinct.append(len(set(ids)))
        ms = (time.perf_counter() - t0) * 1000 / max(len(labelled), 1)
        rows.append(
            {
                "variant": name,
                "n": len(labelled),
                "hit@1": float(np.mean([bool(x and x <= 1) for x in ranks])) if ranks else 0.0,
                "hit@3": float(np.mean([bool(x and x <= 3) for x in ranks])) if ranks else 0.0,
                "hit@5": float(np.mean([bool(x and x <= 5) for x in ranks])) if ranks else 0.0,
                "mrr": float(np.mean([1 / x if x else 0.0 for x in ranks])) if ranks else 0.0,
                "distinct_docs@k": float(np.mean(distinct)) if distinct else 0.0,
                "ms_per_query": round(ms, 1),
            }
        )
    return rows


def _chunk_texts(pipe: RagPipeline) -> dict[str, str]:
    chunks = getattr(pipe.retriever.store, "chunks", [])
    return {c.chunk_id: c.text for c in chunks}


def run_eval(pipe: RagPipeline, items: list[GoldenItem]) -> dict[str, Any]:
    texts = _chunk_texts(pipe)
    rows: list[dict[str, Any]] = []
    for it in items:
        ans: Answer = pipe.ask(it.question)
        retrieved = pipe.retriever.retrieve(it.question)
        ids = [h.chunk.doc_id for h in retrieved.hits]
        rows.append(
            {
                "item": it,
                "answer": ans,
                "rank": doc_rank(ids, it.expected_doc_ids) if it.expected_doc_ids else None,
                "top1": retrieved.top_score,
                "retrieved_docs": ids,
            }
        )

    with_docs = [r for r in rows if r["item"].expected_doc_ids]
    ranks = [r["rank"] for r in with_docs]
    k = max(len(r["retrieved_docs"]) for r in rows) if rows else 5
    retrieval = {
        "n": len(with_docs),
        **{
            f"hit@{n}": float(np.mean([bool(r and r <= n) for r in ranks]))
            for n in (1, 3, 5)
            if n <= k
        },
        "mrr": float(np.mean([1 / r if r else 0.0 for r in ranks])) if ranks else 0.0,
    }

    should_refuse = [r for r in rows if r["item"].expect == "refuse"]
    should_answer = [r for r in rows if r["item"].expect == "answer"]
    refused = [r for r in rows if r["answer"].mode == "refusal"]
    tp = sum(r in should_refuse for r in refused)
    refusal = {
        "precision": tp / len(refused) if refused else None,
        "recall": tp / len(should_refuse) if should_refuse else None,
        "false_refusal_rate": (
            sum(r["answer"].mode == "refusal" for r in should_answer) / len(should_answer)
            if should_answer
            else None
        ),
        "false_answer_rate": (
            sum(r["answer"].mode in {"llm", "extractive"} for r in should_refuse)
            / len(should_refuse)
            if should_refuse
            else None
        ),
    }
    emergencies = [r for r in rows if r["item"].expect == "emergency"]
    safety = {
        "emergency_recall": (
            float(np.mean([r["answer"].mode == "emergency" for r in emergencies]))
            if emergencies
            else None
        ),
        "false_alarms": sum(
            r["answer"].mode == "emergency" for r in rows if r["item"].expect != "emergency"
        ),
    }

    answered = [r for r in rows if r["answer"].mode in {"llm", "extractive"}]
    llm_answers = [r for r in answered if r["answer"].mode == "llm"]
    faith = [
        faithfulness_proxy(
            r["answer"].text, [texts.get(s.chunk_id, "") for s in r["answer"].sources]
        )
        for r in llm_answers
    ]
    lat = [r["answer"].latency_ms for r in rows]
    generation = {
        "answered": len(answered),
        "llm_answers": len(llm_answers),
        "extractive_answers": len(answered) - len(llm_answers),
        "faithfulness_proxy_llm": float(np.mean(faith)) if faith else None,
        "invalid_citation_events": sum(
            any(w.startswith("invalid_citations_stripped") for w in r["answer"].warnings)
            for r in rows
        ),
        "llm_fallbacks": sum(
            any(w.startswith("llm_fallback") for w in r["answer"].warnings) for r in rows
        ),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
        if lat
        else {},
    }

    sweep = threshold_sweep([r["top1"] for r in should_answer], [r["top1"] for r in should_refuse])
    worst = sorted(
        (r for r in with_docs if r["rank"] is None or r["rank"] > 1),
        key=lambda r: (r["rank"] is not None, -(r["rank"] or 0), r["top1"]),
    )[:3]
    return {
        "n_items": len(rows),
        "threshold_used": pipe.retriever.threshold,
        "retrieval": retrieval,
        "refusal": refusal,
        "safety": safety,
        "generation": generation,
        "threshold_sweep": sweep,
        "recommended_threshold": recommend_threshold(sweep),
        "failures_for_analysis": [
            {
                "id": r["item"].id,
                "question": r["item"].question,
                "expected": r["item"].expected_doc_ids,
                "rank_of_expected": r["rank"],
                "retrieved_docs": r["retrieved_docs"],
                "top_score": round(r["top1"], 3),
            }
            for r in worst
        ],
        "per_item": [
            {
                "id": r["item"].id,
                "mode": r["answer"].mode,
                "rank": r["rank"],
                "top1": round(r["top1"], 3),
            }
            for r in rows
        ],
    }


def ablation_markdown(rows: list[dict[str, Any]] | None) -> str:
    if not rows:
        return ""
    head = (
        "\n\n## Retrieval ablation (ranking quality only, answerability threshold disabled)\n"
        f"n = {rows[0]['n']} golden items with an expected document. Adopt a variant only if it beats "
        "`dense` here; with this few items, treat differences under ~0.05 as noise.\n\n"
        "| variant | hit@1 | hit@3 | hit@5 | MRR | distinct docs in top-k | ms/query |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(
        f"| {r['variant']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.2f} "
        f"| {r['distinct_docs@k']:.1f} | {r['ms_per_query']} |"
        for r in rows
    )


def write_report(
    result: dict[str, Any],
    reports_dir: Path,
    embedder: str,
    llm_used: bool,
    ablation: list[dict[str, Any]] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    result = {**result, "ablation": ablation or []}
    (reports_dir / "rag_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    r, f, g = result["retrieval"], result["refusal"], result["generation"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    md = (
        f"""# RAG evaluation (generated by `python -m hcai eval`; do not edit by hand)

embedder `{embedder}` · answer mode: {"LLM" if llm_used else "extractive only (no API key)"} · threshold in use {result["threshold_used"]:.3f} · {result["n_items"]} golden items

## Retrieval (items with an expected document, n={r["n"]})
hit@1 {pct(r.get("hit@1"))} · hit@3 {pct(r.get("hit@3"))} · hit@5 {pct(r.get("hit@5"))} · MRR {pct(r["mrr"])}

## Refusal behaviour
precision {pct(f["precision"])} · recall {pct(f["recall"])} · false-refusal rate {pct(f["false_refusal_rate"])} · false-answer rate {pct(f["false_answer_rate"])}

## Safety gate
emergency recall {pct(result["safety"]["emergency_recall"])} · false alarms on non-emergency items {result["safety"]["false_alarms"]}

## Generation
answered {g["answered"]} (LLM {g["llm_answers"]}, extractive {g["extractive_answers"]}) · faithfulness proxy (LLM answers, lexical) {pct(g["faithfulness_proxy_llm"])}
· invalid-citation events {g["invalid_citation_events"]} · LLM fallbacks {g["llm_fallbacks"]} · latency {g["latency_ms"]}

## Threshold trade-off (top-1 retrieval score; answerable vs. out-of-scope questions)
Recommended threshold (maximises 1-FRR + 1-FAR): **{pct(result["recommended_threshold"])}** — set `rag.score_threshold` in `configs/default.yaml`.

| threshold | false refusal | false answer |
|---|---|---|
"""
        + "\n".join(
            f"| {s['threshold']:.3f} | {s['false_refusal_rate']:.2f} | {s['false_answer_rate']:.2f} |"
            for s in result["threshold_sweep"]
        )
        + ablation_markdown(ablation)
        + """

## Failures to analyse (write your own paragraph: what failed, why, what you would fix)
"""
        + "\n".join(
            f'- {x["id"]} "{x["question"]}" expected {x["expected"]}, first expected hit at rank '
            f"{x['rank_of_expected']}, retrieved {x['retrieved_docs']}, top score {x['top_score']}"
            for x in result["failures_for_analysis"]
        )
    )
    out = reports_dir / "rag_eval.md"
    out.write_text(md + "\n", encoding="utf-8")
    return out
