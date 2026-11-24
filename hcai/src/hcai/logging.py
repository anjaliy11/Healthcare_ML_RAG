"""Structured JSON logging with a request id. Raw patient/question text can never be logged."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

# Privacy rule: these keys are dropped from any log call (patient fields and free text).
FORBIDDEN_LOG_KEYS = frozenset(
    {
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "patient", "question", "query", "answer", "text",
    }
)  # fmt: skip


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": current_request_id(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(level: int | str = "INFO") -> None:
    root = logging.getLogger("hcai")
    if any(getattr(h, "_hcai", False) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._hcai = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"hcai.{name}")


def log_event(
    logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any
) -> dict[str, Any]:
    """Log `event` with safe fields only; returns the fields actually logged (for tests)."""
    safe = {k: v for k, v in fields.items() if k not in FORBIDDEN_LOG_KEYS}
    dropped = sorted(set(fields) - set(safe))
    if dropped:
        safe["dropped_keys"] = dropped
    logger.log(level, event, extra={"fields": safe})
    return safe
