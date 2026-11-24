"""One-time corpus snapshotting (run on your machine; never at query time).

Fetches the public pages, converts them to heading-structured text, and records url, license,
retrieval time and sha256 in `manifest.json`. A manual fallback (`add_local_document`) covers
pages that fail to download or parse."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from hcai.errors import IngestError
from hcai.textio import read_text

WHO_LICENSE = "CC BY-NC-SA 3.0 IGO (verify on page)"
GOV_LICENSE = "US government work, generally public domain (verify on page)"

SOURCES: list[dict[str, str]] = [
    {"doc_id": "who-hypertension", "title": "WHO fact sheet: Hypertension",
     "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension", "license": WHO_LICENSE},
    {"doc_id": "who-cvd", "title": "WHO fact sheet: Cardiovascular diseases (CVDs)",
     "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)", "license": WHO_LICENSE},
    {"doc_id": "who-diabetes", "title": "WHO fact sheet: Diabetes",
     "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes", "license": WHO_LICENSE},
    {"doc_id": "who-healthy-diet", "title": "WHO fact sheet: Healthy diet",
     "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet", "license": WHO_LICENSE},
    {"doc_id": "who-physical-activity", "title": "WHO fact sheet: Physical activity",
     "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity", "license": WHO_LICENSE},
    {"doc_id": "who-tobacco", "title": "WHO fact sheet: Tobacco",
     "url": "https://www.who.int/news-room/fact-sheets/detail/tobacco", "license": WHO_LICENSE},
    {"doc_id": "niddk-diabetes-symptoms", "title": "NIDDK: Symptoms & causes of diabetes",
     "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes", "license": GOV_LICENSE},
    {"doc_id": "cdc-hbp-about", "title": "CDC: About high blood pressure",
     "url": "https://www.cdc.gov/high-blood-pressure/about/index.html", "license": GOV_LICENSE},
    {"doc_id": "cdc-hbp-risk-factors", "title": "CDC: High blood pressure risk factors",
     "url": "https://www.cdc.gov/high-blood-pressure/risk-factors/index.html", "license": GOV_LICENSE},
]  # fmt: skip

# cdc.gov answers 403 to non-browser clients (bot protection), so a browser-like header set is
# used. This is one polite request per page, run once by the user, not crawling.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 hcai-assessment/0.3 (educational; one-time snapshot)"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Used only when the primary URL fails. The manifest then records the URL/title actually used.
GOV = GOV_LICENSE
ALTERNATES: dict[str, list[dict[str, str]]] = {
    "cdc-hbp-about": [
        {"title": "NHLBI: What is high blood pressure?",
         "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure", "license": GOV},
    ],
    "cdc-hbp-risk-factors": [
        {"title": "NHLBI: High blood pressure causes and risk factors",
         "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure/causes", "license": GOV},
    ],
}  # fmt: skip
_NOISE = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "form",
    "noscript",
    "aside",
    "svg",
    "button",
]


def html_to_text(html: str) -> str:
    """Main content -> '# title', '## headings', paragraphs, '- bullets' (contiguous lists)."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(_NOISE):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    out: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
        if not isinstance(el, Tag):
            continue
        if el.name in {"ul", "ol"}:
            if el.find_parent(["ul", "ol"]):  # nested list: parent's text already includes it
                continue
            items = [re.sub(r"\s+", " ", li.get_text(" ", strip=True)) for li in el.find_all("li")]
            block = "\n".join(f"- {i}" for i in items if i)
        elif el.name == "p":
            if el.find_parent(["li"]):
                continue
            block = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        else:
            level = "#" if el.name == "h1" else "##"
            txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            block = f"{level} {txt}" if txt else ""
        if block:
            out.append(block)
    return "\n\n".join(out).strip() + "\n"


def _load_manifest(docs_dir: Path) -> dict[str, dict[str, str]]:
    path = docs_dir / "manifest.json"
    if not path.is_file():
        return {}
    return {e["doc_id"]: e for e in json.loads(read_text(path))["documents"]}


def _write_manifest(docs_dir: Path, entries: dict[str, dict[str, str]]) -> None:
    docs = sorted(entries.values(), key=lambda e: e["doc_id"])
    (docs_dir / "manifest.json").write_text(
        json.dumps({"documents": docs}, indent=2), encoding="utf-8"
    )


def save_document(docs_dir: Path, meta: dict[str, str], text: str) -> dict[str, str]:
    if len(text.split()) < 50:
        raise IngestError(
            f"{meta['doc_id']}: extracted text is too short ({len(text.split())} words)."
        )
    docs_dir.mkdir(parents=True, exist_ok=True)
    raw = text.encode("utf-8")
    (docs_dir / f"{meta['doc_id']}.txt").write_bytes(raw)
    entry = {
        **meta,
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file": f"{meta['doc_id']}.txt",
    }
    entries = _load_manifest(docs_dir)
    entries[meta["doc_id"]] = entry
    _write_manifest(docs_dir, entries)
    return entry


def add_local_document(
    docs_dir: Path, path: Path, doc_id: str, title: str, url: str, license_: str
) -> dict[str, str]:
    """Manual fallback: you saved the page text by hand (use '## ' for headings)."""
    return save_document(
        docs_dir,
        {"doc_id": doc_id, "title": title, "url": url, "license": license_},
        read_text(path),
    )


def failed_ids(status: dict[str, str]) -> list[str]:
    """Documents with no usable text. 'ok (alternate: ...)' is a success, not a failure."""
    return [d for d, st in status.items() if st.startswith("FAILED")]


def _download(url: str) -> str:
    import requests

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return html_to_text(resp.text)


def fetch_all(
    docs_dir: Path, only: list[str] | None = None, delay_s: float = 1.0
) -> dict[str, str]:
    """Download every source (primary URL, then alternates). Never aborts the batch.

    Returns {doc_id: 'ok' | 'ok (alternate: <url>)' | 'FAILED: ...'}."""
    import requests

    status: dict[str, str] = {}
    for meta in SOURCES:
        if only and meta["doc_id"] not in only:
            continue
        candidates = [meta, *({**meta, **alt} for alt in ALTERNATES.get(meta["doc_id"], []))]
        errors: list[str] = []
        for i, cand in enumerate(candidates):
            try:
                save_document(docs_dir, cand, _download(cand["url"]))
                status[meta["doc_id"]] = "ok" if i == 0 else f"ok (alternate: {cand['url']})"
                break
            except (requests.RequestException, IngestError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(delay_s)
        else:
            status[meta["doc_id"]] = "FAILED: " + " | ".join(errors)
        time.sleep(delay_s)
    return status
