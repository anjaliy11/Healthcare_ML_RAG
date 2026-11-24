"""Composition root + the combined ML -> RAG flow used by the CLI and Streamlit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from hcai import DISCLAIMER
from hcai.config import Config
from hcai.errors import HcaiError
from hcai.logging import get_logger, log_event, new_request_id
from hcai.ml.predict import Factor, RiskPredictor, RiskResult
from hcai.ml.schema import PatientInput, parse_patient
from hcai.monitoring.metrics import MetricsLog
from hcai.rag.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from hcai.rag.generator import AnthropicClient, GeminiClient, Generator, LLMClient
from hcai.rag.pipeline import Answer, RagPipeline
from hcai.rag.rerank import CrossEncoderReranker
from hcai.rag.retriever import Retriever
from hcai.rag.store import FaissStore

log = get_logger("app.service")

# Each risk factor becomes an informational (non-personal) retrieval query.
FACTOR_QUERIES: dict[str, str] = {
    "age": "How does age affect the risk of heart disease and high blood pressure?",
    "sex": "How does sex affect the risk of cardiovascular disease?",
    "cp": "What does chest pain mean for heart disease risk and when should a person see a doctor?",
    "trestbps": "How does high blood pressure raise the risk of heart disease?",
    "chol": "How does high cholesterol affect cardiovascular risk?",
    "fbs": "How does high blood sugar or diabetes affect heart disease risk?",
    "restecg": "What are the risk factors and warning signs of heart disease?",
    "thalach": "How does physical activity and heart fitness affect cardiovascular health?",
    "exang": "What are the warning signs of heart disease during physical activity?",
    "oldpeak": "What are the risk factors and warning signs of heart disease?",
    "slope": "What are the risk factors and warning signs of heart disease?",
    "ca": "What are the main risk factors for cardiovascular disease?",
    "thal": "What are the main risk factors for cardiovascular disease?",
}


def make_embedder(cfg: Config, model_name: str | None = None) -> Embedder:
    name = model_name or cfg.rag.embedding_model
    if name.startswith("hashing"):
        return HashingEmbedder(int(name.rsplit("-", 1)[1]))
    return SentenceTransformerEmbedder(name, cfg.rag.query_prefix)


def make_llm(cfg: Config, enabled: bool = True) -> LLMClient | None:
    """LLM client for the configured provider when its key exists; otherwise None (extractive)."""
    if not enabled:
        return None
    if cfg.llm.provider == "gemini" and (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        return GeminiClient(cfg.llm)
    if cfg.llm.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient(cfg.llm)
    return None


@dataclass
class FactorExplanation:
    factor: Factor
    answer: Answer | None
    note: str = ""


@dataclass
class Assessment:
    risk: RiskResult
    explanations: list[FactorExplanation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class Services:
    """Lazily loads the model and the RAG pipeline; each can fail without taking down the other."""

    def __init__(self, cfg: Config, use_llm: bool = True) -> None:
        self.cfg, self.use_llm = cfg, use_llm
        self.metrics = MetricsLog(cfg.paths.metrics_log)
        self._predictor: RiskPredictor | None = None
        self._pipeline: RagPipeline | None = None

    @property
    def predictor(self) -> RiskPredictor:
        if self._predictor is None:
            self._predictor = RiskPredictor.load(self.cfg)
        return self._predictor

    @property
    def pipeline(self) -> RagPipeline:
        if self._pipeline is None:
            store = FaissStore.load(self.cfg.paths.index_dir)
            embedder = make_embedder(self.cfg, store.embedding_model)
            threshold = (
                self.cfg.rag.score_threshold
                if embedder.is_semantic
                else self.cfg.rag.offline_score_threshold
            )
            r = self.cfg.rag
            retriever = Retriever(
                embedder, store, r.top_k, threshold, r.max_query_chars,
                mode=r.retrieval_mode, rrf_k=r.rrf_k, max_per_doc=r.max_per_doc,
                reranker=CrossEncoderReranker(r.reranker_model) if r.rerank else None,
                rerank_top_n=r.rerank_top_n,
            )  # fmt: skip
            gen = Generator(make_llm(self.cfg, self.use_llm), self.cfg.rag.extractive_sentences)
            self._pipeline = RagPipeline(retriever, gen, self.metrics)
        return self._pipeline

    def provenance(self) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            out["model_version"] = self.predictor.meta["model_version"]
        except HcaiError as exc:
            out["model_version"] = f"unavailable ({exc})"
        try:
            p = self.pipeline
            out["embedding_model"] = p.retriever.embedder.name
            out["index_chunks"] = str(len(p.retriever.store))
            out["retrieval"] = p.retriever.mode + ("+rerank" if p.retriever.reranker else "")
            out["answer_mode"] = "LLM" if p.generator.llm else "extractive (no LLM key)"
        except HcaiError as exc:
            out["embedding_model"] = f"unavailable ({exc})"
        return out

    def ask(self, question: str) -> Answer:
        return self.pipeline.ask(question)

    def assess(self, patient: dict[str, object] | PatientInput) -> Assessment:
        """Validate -> predict -> explain top risk-increasing factors from the documents."""
        rid = new_request_id()
        p = parse_patient(patient)
        risk = self.predictor.predict([p])[0]
        self.metrics.record(
            request_id=rid, kind="assess", risk_band=risk.band, range_flags=risk.range_flags
        )
        log_event(log, "assess_done", band=risk.band, range_flags=risk.range_flags)
        result = Assessment(risk=risk)
        if risk.range_flags:
            result.notes.append(
                "Some values fall outside the range of the training data, so the estimate is "
                "an extrapolation: " + ", ".join(risk.range_flags) + "."
            )
        if risk.ignored_fields:
            result.notes.append(
                "This model was trained on a dataset without: "
                + ", ".join(risk.ignored_fields)
                + ". Those values were ignored."
            )
        if risk.imputed_fields:
            result.notes.append(
                "Not provided and filled with typical training values: "
                + ", ".join(risk.imputed_fields)
                + "."
            )
        seen: set[str] = set()
        for f in risk.top_factors:
            q = FACTOR_QUERIES[f.feature]
            if q in seen:
                continue
            seen.add(q)
            try:
                ans = self.pipeline.ask(q)
                result.explanations.append(FactorExplanation(f, ans))
            except HcaiError as exc:  # RAG down: still return the risk estimate
                result.explanations.append(
                    FactorExplanation(f, None, f"Explanation unavailable: {exc}")
                )
                result.notes.append(f"Document explanations unavailable: {exc}")
                break
        return result
