"""Shared helpers: fixture corpus builder and scripted fake LLM (no network anywhere)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from hcai.errors import LLMError
from hcai.rag.generator import LLMReply

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "docs"


def build_corpus(dest: Path, names: list[str] | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(FIXTURE_DIR.glob("*.txt")):
        if names and f.stem not in names:
            continue
        shutil.copy(f, dest / f.name)
        entries.append(
            {
                "doc_id": f.stem,
                "title": f.read_text(encoding="utf-8").splitlines()[0].lstrip("# "),
                "url": f"https://example.org/{f.stem}",
                "license": "test-fixture",
                "retrieved_at": "2026-01-01",
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "file": f.name,
            }
        )
    (dest / "manifest.json").write_text(json.dumps({"documents": entries}), encoding="utf-8")
    return dest


class ScriptedLLM:
    """Returns queued replies (str) or raises queued LLMError; records the prompts it saw."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMReply:
        self.calls.append((system, user))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMReply(r, 100, 20)


def llm_error() -> LLMError:
    return LLMError("boom")
