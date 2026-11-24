from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcai.config import load_config, load_dotenv_file
from hcai.errors import ConfigError, IngestError
from hcai.eval.rag_eval import (
    doc_rank,
    faithfulness_proxy,
    load_golden,
    recommend_threshold,
    threshold_sweep,
)
from hcai.logging import JsonFormatter, log_event, new_request_id
from hcai.ml.schema import FEATURES
from hcai.monitoring.drift import check_drift
from hcai.monitoring.metrics import MetricsLog, read_records, summarize
from hcai.rag.fetch import SOURCES, add_local_document, html_to_text, save_document
from hcai.rag.ingest import load_corpus
from tests.fixtures.synth import make_synthetic

# ------------------------------------------------------------------ config / logging


def test_default_config_loads_and_paths_are_absolute():
    cfg = load_config()
    assert cfg.seed == 42 and cfg.paths.model.is_absolute()
    assert (
        cfg.llm.model == "gemini-3.5-flash-lite"
        and cfg.rag.embedding_model == "BAAI/bge-small-en-v1.5"
    )


def test_bad_config_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nunknown_key: 2\n")
    with pytest.raises(ConfigError, match="Invalid config"):
        load_config(bad)
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nFOO_TEST=from_file\nBAR_TEST='quoted'\nbad line\n")
    monkeypatch.setenv("FOO_TEST", "from_env")
    monkeypatch.delenv("BAR_TEST", raising=False)
    load_dotenv_file(f)
    import os

    assert os.environ["FOO_TEST"] == "from_env" and os.environ["BAR_TEST"] == "quoted"
    load_dotenv_file(tmp_path / "nonexistent")  # no error
    monkeypatch.delenv("BAR_TEST")


def test_log_event_drops_patient_fields_and_free_text():
    logger = logging.getLogger("hcai.test")
    safe = log_event(
        logger, "x", age=55, chol=200, question="my chest hurts", latency_ms=3, band="low"
    )
    assert safe == {"latency_ms": 3, "band": "low", "dropped_keys": ["age", "chol", "question"]}


def test_json_formatter_includes_request_id():
    rid = new_request_id()
    rec = logging.LogRecord("hcai.t", logging.INFO, "", 0, "evt", None, None)
    rec.fields = {"a": 1}  # type: ignore[attr-defined]
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["request_id"] == rid and payload["event"] == "evt" and payload["a"] == 1


# ------------------------------------------------------------------ monitoring
def test_metrics_log_whitelists_and_summarises(tmp_path: Path):
    log = MetricsLog(tmp_path / "m.jsonl")
    log.record(kind="ask", latency_ms=10, top_score=0.7, refused=False, question="secret", age=5)
    log.record(kind="ask", latency_ms=30, top_score=0.2, refused=True, fallback=True)
    log.record(kind="assess", range_flags=["chol"], risk_band="high")
    log.record(kind="assess", range_flags=[], risk_band="low")
    assert "secret" not in (tmp_path / "m.jsonl").read_text()
    s = summarize(read_records(tmp_path / "m.jsonl"))
    assert s["ask_requests"] == 2 and s["refusal_rate"] == 0.5 and s["fallback_rate"] == 0.5
    assert s["latency_ms"]["p50"] == 20 and s["range_flag_counts"] == {"chol": 1}
    assert s["out_of_range_rate"] == 0.5


def test_metrics_tolerates_torn_line_and_missing_file(tmp_path: Path):
    assert read_records(tmp_path / "none.jsonl") == [] and summarize([]) == {
        "requests_total": 0,
        "ask_requests": 0,
    }
    p = tmp_path / "m.jsonl"
    p.write_text('{"kind": "ask", "latency_ms": 1}\n{"kind": "as')
    assert len(read_records(p)) == 1


def test_drift_detects_shift_and_ignores_same_distribution():
    ref = make_synthetic(303, seed=1).drop(columns=["num"])
    same = make_synthetic(200, seed=2).drop(columns=["num"])
    assert check_drift(ref, same)["status"] == "ok"
    shifted = same.copy()
    shifted["age"] += 25
    shifted["cp"] = 4.0
    res = check_drift(ref, shifted)
    assert res["status"] == "drift" and {"age", "cp"} <= set(res["drifted"])
    assert res["out_of_training_range_rate"]["age"] > 0.2


def test_drift_needs_enough_rows():
    ref = make_synthetic(303, seed=1).drop(columns=["num"])
    assert check_drift(ref, ref.head(5))["status"] == "insufficient_data"
    assert set(FEATURES) >= set(check_drift(ref, ref)["p_values"])


# ------------------------------------------------------------------ fetch
HTML = (
    """<html><body><nav>Menu stuff</nav><main>
<h1>Hypertension</h1><p>Key facts about blood pressure. """
    + "word " * 60
    + """</p>
<h2>Risk factors</h2><ul><li>Too much salt</li><li>Low <b>activity</b><ul><li>nested</li></ul></li></ul>
<h3>Symptoms</h3><p>Most people feel nothing.</p></main><footer>Copyright</footer><script>x()</script></body></html>"""
)


def test_html_to_text_structure():
    t = html_to_text(HTML)
    assert t.startswith("# Hypertension") and "## Risk factors" in t and "## Symptoms" in t
    assert "- Too much salt\n- Low activity nested" in t
    assert "Menu stuff" not in t and "Copyright" not in t and "x()" not in t


def test_save_document_writes_manifest_and_roundtrips(tmp_path: Path):
    meta = {"doc_id": "t1", "title": "T", "url": "https://u", "license": "L"}
    entry = save_document(tmp_path, meta, html_to_text(HTML))
    assert len(entry["sha256"]) == 64 and entry["file"] == "t1.txt"
    docs = load_corpus(tmp_path)  # manifest sha matches file on disk
    assert docs[0].doc_id == "t1" and "Risk factors" in docs[0].text
    save_document(tmp_path, {**meta, "title": "T2"}, html_to_text(HTML))  # replace, not duplicate
    assert len(json.loads((tmp_path / "manifest.json").read_text())["documents"]) == 1


def test_save_document_rejects_stub_pages(tmp_path: Path):
    with pytest.raises(IngestError, match="too short"):
        save_document(
            tmp_path, {"doc_id": "x", "title": "t", "url": "u", "license": "l"}, "# Access denied"
        )


def test_add_local_document(tmp_path: Path):
    f = tmp_path / "page.txt"
    f.write_text("## Heading\n" + "word " * 80)
    e = add_local_document(tmp_path / "docs", f, "manual", "Manual", "https://u", "L")
    assert e["doc_id"] == "manual" and (tmp_path / "docs" / "manual.txt").is_file()


def test_sources_are_8_to_10_unique_with_licenses():
    assert 5 <= len(SOURCES) <= 10 and len({s["doc_id"] for s in SOURCES}) == len(SOURCES)
    assert all(s["license"] and s["url"].startswith("https://") for s in SOURCES)


# ------------------------------------------------------------------ eval helpers
def test_golden_set_shape_and_ids_match_sources():
    items = load_golden()
    assert len(items) >= 30 and len({i.id for i in items}) == len(items)
    assert {i.category for i in items} == {"answerable", "unanswerable", "adversarial"}
    known = {s["doc_id"] for s in SOURCES}
    assert all(set(i.expected_doc_ids) <= known for i in items)
    assert sum(i.category == "answerable" for i in items) >= 20
    assert any(i.expect == "emergency" for i in items)


def test_eval_helpers():
    assert doc_rank(["a", "b", "c"], ["c", "z"]) == 3 and doc_rank(["a"], ["x"]) is None
    assert (
        faithfulness_proxy(
            "Salt raises blood pressure in adults [C1].",
            ["salt raises blood pressure in many adults"],
        )
        == 1.0
    )
    assert (
        faithfulness_proxy(
            "Bananas cure every disease quickly [C1].", ["salt raises blood pressure"]
        )
        == 0.0
    )
    sweep = threshold_sweep([0.8, 0.7, 0.9], [0.2, 0.3])
    rec = recommend_threshold(sweep)
    assert rec is not None and 0.3 < rec <= 0.7
    assert sweep[0]["false_refusal_rate"] == 0 and sweep[-1]["false_answer_rate"] <= 1
    assert recommend_threshold([]) is None and threshold_sweep([], []) == []
    assert np.isfinite([r["threshold"] for r in sweep]).all()
    _ = pd  # keep import used
