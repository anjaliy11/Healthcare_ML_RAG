"""Per-request metrics as JSONL. Whitelisted fields only: no patient input, no question text."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ALLOWED_FIELDS = frozenset(
    {
        "request_id", "kind", "latency_ms", "retrieve_ms", "generate_ms", "top_score", "refused",
        "mode", "fallback", "input_tokens", "output_tokens", "error_type", "n_sources",
        "range_flags", "risk_band", "emergency",
    }
)  # fmt: skip


class MetricsSink(Protocol):
    def record(self, **fields: Any) -> None: ...


class MetricsLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, **fields: Any) -> None:
        row = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
        row["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")


class NullMetrics:
    def record(self, **fields: Any) -> None:
        return None


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn last line
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qa = [r for r in rows if r.get("kind") == "ask"]
    out: dict[str, Any] = {"requests_total": len(rows), "ask_requests": len(qa)}
    if qa:
        lat = np.array([r["latency_ms"] for r in qa if "latency_ms" in r], dtype=float)
        scores = np.array([r["top_score"] for r in qa if "top_score" in r], dtype=float)
        n = len(qa)
        out.update(
            refusal_rate=sum(bool(r.get("refused")) for r in qa) / n,
            emergency_rate=sum(bool(r.get("emergency")) for r in qa) / n,
            fallback_rate=sum(bool(r.get("fallback")) for r in qa) / n,
            error_rate=sum(bool(r.get("error_type")) for r in qa) / n,
            input_tokens=int(sum(r.get("input_tokens", 0) for r in qa)),
            output_tokens=int(sum(r.get("output_tokens", 0) for r in qa)),
        )
        if lat.size:
            out["latency_ms"] = {
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
            }
        if scores.size:
            out["top_score"] = {
                "min": float(scores.min()),
                "p50": float(np.percentile(scores, 50)),
                "max": float(scores.max()),
            }
    risk = [r for r in rows if r.get("kind") == "assess"]
    if risk:
        flags: dict[str, int] = {}
        for r in risk:
            for f in r.get("range_flags", []):
                flags[f] = flags.get(f, 0) + 1
        out["assess_requests"] = len(risk)
        out["out_of_range_rate"] = sum(bool(r.get("range_flags")) for r in risk) / len(risk)
        out["range_flag_counts"] = flags
    return out
