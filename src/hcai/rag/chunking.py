"""Section-aware chunking. Sizes are in words (a cheap, tokenizer-free proxy: 1 word ~ 1.3 tokens)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from hcai.rag.ingest import Document

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    chunk_idx: int
    license: str
    text: str

    @property
    def embed_text(self) -> str:
        """Title and section give the embedder context that the bare passage may lack."""
        return f"{self.title}. {self.section}.\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


def _words(s: str) -> int:
    return len(s.split())


def split_sections(text: str, default: str) -> list[tuple[str, list[str]]]:
    """Return (section heading, paragraphs). Text with no headings is one section."""
    sections: list[tuple[str, list[str]]] = []
    heading = default
    buf: list[str] = []

    def flush() -> None:
        paras = [p.strip() for p in "\n".join(buf).split("\n\n") if p.strip()]
        if paras:
            sections.append((heading, paras))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            heading, buf = m.group(1), []
        else:
            buf.append(line)
    flush()
    return sections


def _split_unit(unit: str, limit: int) -> list[str]:
    """Break an oversized paragraph by sentence, then by words (e.g. one very long line)."""
    if _words(unit) <= limit:
        return [unit]
    out: list[str] = []
    for sent in _SENTENCE_RE.split(unit):
        w = sent.split()
        if len(w) <= limit:
            out.append(sent)
        else:
            out.extend(" ".join(w[i : i + limit]) for i in range(0, len(w), limit))
    return out


def chunk_document(
    doc: Document, target_words: int, overlap_words: int, max_words: int
) -> list[Chunk]:
    if not (0 <= overlap_words < target_words <= max_words):
        raise ValueError("Require 0 <= overlap < target <= max chunk words.")
    unit_limit = max_words - overlap_words
    chunks: list[Chunk] = []
    for section, paragraphs in split_sections(doc.text, doc.title):
        units = [u for p in paragraphs for u in _split_unit(p, unit_limit)]
        bodies: list[str] = []
        cur: list[str] = []
        for u in units:
            if cur and sum(map(_words, cur)) + _words(u) > target_words:
                bodies.append("\n\n".join(cur))
                cur = []
            cur.append(u)
        if cur:
            bodies.append("\n\n".join(cur))
        prev_tail = ""
        for body in bodies:
            text = f"{prev_tail} {body}".strip() if prev_tail else body
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    url=doc.url,
                    section=section,
                    chunk_idx=len(chunks),
                    license=doc.license,
                    text=text,
                )
            )
            prev_tail = " ".join(body.split()[-overlap_words:]) if overlap_words else ""
    return chunks
