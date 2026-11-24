"""Load the local document snapshots described by `manifest.json`, verify and clean them."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from hcai.errors import IngestError
from hcai.logging import get_logger, log_event
from hcai.textio import decode_bytes

log = get_logger("rag.ingest")
MANIFEST = "manifest.json"

_HTML_RE = re.compile(r"<\s*(html|body|div|p|script|nav|footer|span|a)\b", re.I)
# Short lines that are page furniture, not content.
_BOILERPLATE_RE = re.compile(
    r"^(skip to (main )?content|share( on .*)?|print|cookies?( settings| policy)?|accept all"
    r"|subscribe.*|sign up.*|back to top|related (links|topics)|menu|search|home)$",
    re.I,
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    license: str
    retrieved_at: str
    sha256: str
    text: str


def clean_text(raw: str) -> str:
    """Strip HTML residue and boilerplate, normalise unicode and whitespace; keep '#' headings."""
    if _HTML_RE.search(raw):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
            tag.decompose()
        raw = soup.get_text("\n")
    text = unicodedata.normalize("NFKC", raw).replace(" ", " ").replace("�", "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    lines = []
    for line in text.splitlines():
        s = re.sub(r"[ \t]+", " ", line).strip()
        if len(s) < 40 and _BOILERPLATE_RE.match(s):
            continue
        lines.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_corpus(docs_dir: Path) -> list[Document]:
    manifest_path = docs_dir / MANIFEST
    if not manifest_path.is_file():
        raise IngestError(
            f"No {MANIFEST} in {docs_dir}. Run `python scripts/fetch_docs.py` to download the "
            "document snapshots."
        )
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise IngestError(f"{manifest_path} is not a valid manifest: {exc}") from exc

    docs: list[Document] = []
    seen: dict[str, str] = {}
    for e in entries:
        path = docs_dir / e["file"]
        if not path.is_file():
            raise IngestError(f"Manifest lists {e['file']} but the file is missing.")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            raise IngestError(
                f"{e['file']} does not match its manifest sha256 (edited or corrupted). "
                "Re-run scripts/fetch_docs.py."
            )
        text = clean_text(decode_bytes(raw))
        if not text:
            log_event(log, "doc_skipped_empty", doc_id=e["doc_id"])
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            log_event(log, "doc_skipped_duplicate", doc_id=e["doc_id"], same_as=seen[digest])
            continue
        seen[digest] = e["doc_id"]
        docs.append(
            Document(
                doc_id=e["doc_id"],
                title=e["title"],
                url=e["url"],
                license=e["license"],
                retrieved_at=e["retrieved_at"],
                sha256=e["sha256"],
                text=text,
            )
        )
    if not docs:
        raise IngestError("The corpus is empty after cleaning.")
    log_event(log, "corpus_loaded", documents=len(docs))
    return docs
