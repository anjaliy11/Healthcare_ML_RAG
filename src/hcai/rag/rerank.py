"""Optional cross-encoder reranking of the fused candidates."""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-base), CPU, lazy-loaded."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model: Any = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import, keep lazy

            self._model = CrossEncoder(self.name, device="cpu")
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]
