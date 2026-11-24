"""FAISS exact inner-product index + JSON metadata sidecar, behind a VectorStore protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from hcai.errors import IndexCorruptError, IndexMissingError
from hcai.rag.chunking import Chunk

INDEX_FILE, CHUNKS_FILE, META_FILE = "index.faiss", "chunks.json", "index_meta.json"


class VectorStore(Protocol):
    embedding_model: str

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...
    def save(self, directory: Path) -> None: ...
    def __len__(self) -> int: ...


def _sha(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


class FaissStore:
    """IndexFlatIP on normalised vectors = exact cosine. Fine for < 10k chunks."""

    def __init__(self, dim: int, embedding_model: str) -> None:
        self.dim = dim
        self.embedding_model = embedding_model
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(chunks):
            raise ValueError("Vectors must be (n, dim) and match the number of chunks.")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [
            (self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8"
        )
        meta = {
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "n_chunks": len(self.chunks),
            "doc_ids": sorted({c.doc_id for c in self.chunks}),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "index_sha256": _sha(directory / INDEX_FILE, directory / CHUNKS_FILE),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        needed = [directory / f for f in (INDEX_FILE, CHUNKS_FILE, META_FILE)]
        if not all(p.is_file() for p in needed):
            raise IndexMissingError(
                f"No search index in {directory}. Run `python -m hcai index` first."
            )
        try:
            meta = json.loads(needed[2].read_text(encoding="utf-8"))
            if meta["index_sha256"] != _sha(needed[0], needed[1]):
                raise IndexCorruptError("Index files do not match their checksum.")
            chunks = [Chunk.from_dict(d) for d in json.loads(needed[1].read_text(encoding="utf-8"))]
            index = faiss.read_index(str(needed[0]))
            if index.ntotal != len(chunks) or index.d != meta["dim"]:
                raise IndexCorruptError("Index size does not match chunk metadata.")
        except IndexCorruptError as exc:
            raise IndexCorruptError(f"{exc} Rebuild with `python -m hcai index`.") from exc
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise IndexCorruptError(
                f"Search index is unreadable ({type(exc).__name__}). Rebuild with "
                "`python -m hcai index`."
            ) from exc
        store = cls(meta["dim"], meta["embedding_model"])
        store.index, store.chunks = index, chunks
        store.meta = meta  # type: ignore[attr-defined]
        return store
