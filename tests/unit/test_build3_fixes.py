"""Regression tests for every defect found in the first real Windows run (build 0.2 -> 0.3),
plus the leakage gate, the Kaggle adapter and the run archive. No network anywhere."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hcai.config import Config, LLMConfig
from hcai.errors import (
    ArtifactMissingError,
    DataValidationError,
    LeakageSuspectedError,
    LLMError,
)
from hcai.rag.chunking import Chunk
from hcai.rag.generator import GeminiClient, Generator, parse_retry_delay
from hcai.rag.retriever import Hit
from hcai.rag.safety import check_emergency
from hcai.textio import decode_bytes
from tests.fixtures.synth import make_kaggle_like
from tests.rag_helpers import ScriptedLLM


# ============================================================ 1. grouped citations ([C1, C2])
def _hits(n: int = 3) -> list[Hit]:
    return [
        Hit(
            Chunk(f"d{i}#0", f"d{i}", f"Doc {i}", f"https://x/{i}", "sec", 0, "lic",
                  f"Salt intake raises blood pressure in adults, passage {i}."),
            0.9 - i / 100,
            i,
        )
        for i in range(1, n + 1)
    ]  # fmt: skip


@pytest.mark.parametrize(
    "reply, cited",
    [
        ("Salt matters [C1, C2]. Age matters [C3].", [1, 2, 3]),
        ("Salt matters [C1][C2].", [1, 2]),
        ("Salt matters [C1; C3].", [1, 3]),
        ("Salt matters [C1 and C2].", [1, 2]),
        ("Salt matters [C2].", [2]),
    ],
)
def test_grouped_citation_tags_are_all_recognised(reply, cited):
    res = Generator(ScriptedLLM(reply)).generate("q", _hits())
    assert res.mode == "llm" and res.cited == cited and not res.warnings


def test_only_grouped_citation_is_not_a_fallback_any_more():
    res = Generator(ScriptedLLM("Only grouped [C1, C2].")).generate("q", _hits())
    assert res.mode == "llm" and res.cited == [1, 2]


def test_invalid_tag_inside_a_group_is_stripped_but_valid_ones_kept():
    res = Generator(ScriptedLLM("Fact [C1, C9]. More [C9].")).generate("q", _hits())
    assert res.cited == [1] and "C9" not in res.text and "[C1]" in res.text
    assert any(w.startswith("invalid_citations_stripped:2") for w in res.warnings)


def test_group_with_only_invalid_tags_falls_back_to_extractive():
    res = Generator(ScriptedLLM("Fake [C8, C9].")).generate("q", _hits())
    assert res.mode == "extractive" and "llm_fallback:no_valid_citation" in res.warnings


# ================================================================== 2. emergency safety gate
@pytest.mark.parametrize(
    "q",
    [
        "Can high blood pressure lead to a stroke or heart attack?",  # golden Q16: was a false alarm
        "What are the warning signs of a heart attack?",
        "Does high blood pressure cause stroke?",
        "What causes chest pain in heart disease?",
        "Can chest pain be caused by anxiety?",
        "Tell me about stroke prevention",
        "How to prevent a heart attack",
        "What is cholesterol?",
    ],
)
def test_general_questions_are_not_emergencies(q):
    assert check_emergency(q) is None


@pytest.mark.parametrize(
    "q",
    [
        "Is this chest pain serious?",  # was missed
        "Why does my chest hurt?",  # was missed
        "Severe chest pain and sweating",
        "chest pain",
        "My dad fainted",
        "I can't breathe",
        "face drooping and slurred speech",
        "help me chest pain",
        "I think I am having a heart attack",
    ],
)
def test_possible_emergencies_alert(q):
    assert check_emergency(q) is not None


# ================================================================ 3. UTF-16 / BOM input files
_TEXT = '{"age": 58, "note": "café"}'


@pytest.mark.parametrize(
    "raw",
    [
        _TEXT.encode("utf-8"),
        b"\xef\xbb\xbf" + _TEXT.encode("utf-8"),
        _TEXT.encode("utf-16"),  # BOM + LE: what PowerShell 5 `>` writes
        _TEXT.encode("utf-16-le"),  # BOM-less
        _TEXT.encode("utf-16-be").join([b"\xfe\xff", b""]),
        _TEXT.encode("cp1252"),
    ],
)
def test_decode_bytes_handles_every_windows_encoding(raw):
    assert json.loads(decode_bytes(raw)) == {"age": 58, "note": "café"}


def _run(cfg_file: Path, *argv: str) -> int:
    from hcai.app.cli import main

    return main(["--config", str(cfg_file), *argv])


@pytest.fixture()
def cfg_file(cfg: Config, tmp_path: Path) -> Path:
    import yaml

    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    return f


def test_cli_reads_powershell_utf16_patient_file(cfg_file, tmp_path, valid_patient, capsys):
    p = tmp_path / "patient.json"
    p.write_bytes(json.dumps(valid_patient).encode("utf-16"))
    # No model is trained here, so assess stops at the artifact check, AFTER the file was read.
    assert _run(cfg_file, "assess", "--input", str(p), "--no-llm") == 2
    err = capsys.readouterr().err
    assert "No trained model" in err and "codec" not in err


def test_cli_example_out_writes_utf8(cfg_file, tmp_path, capsys):
    out = tmp_path / "ex.json"
    assert _run(cfg_file, "assess", "--example", "--out", str(out)) == 0
    assert json.loads(out.read_bytes().decode("utf-8"))["age"] == 58


def test_cli_friendly_errors_for_bad_patient_files(cfg_file, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _run(cfg_file, "assess", "--input", str(bad)) == 2
    assert "not valid JSON" in capsys.readouterr().err
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2]", encoding="utf-8")
    assert _run(cfg_file, "assess", "--input", str(arr)) == 2
    assert "JSON object" in capsys.readouterr().err
    assert _run(cfg_file, "assess", "--input", str(tmp_path / "missing.json")) == 2
    assert "not found" in capsys.readouterr().err


# ==================================================== 4. Gemini latency: one deadline, no chaining
class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _api_error(code: int, body: dict | None = None):
    from google.genai import errors

    return errors.APIError(code, body or {"error": {"message": "x", "status": "S"}})


class _Models:
    """Scripted generate_content; each item may add simulated wall time before answering."""

    def __init__(self, clock: _Clock, script: list) -> None:
        self.clock, self.script, self.calls = clock, script, []

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        cost, item = self.script.pop(0)
        self.clock.t += cost
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            text=item,
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )


def _gem(script, **over):
    clock = _Clock()
    base = dict(provider="gemini", model="m1", fallback_models=["m2", "m3"], max_tokens=10,
                temperature=0.0, timeout_s=30, max_retries=2, backoff_s=1.0, deadline_s=25.0,
                max_wait_s=8.0)  # fmt: skip
    cfg = LLMConfig(**{**base, **over})
    models = _Models(clock, script)
    return (
        GeminiClient(cfg, client=SimpleNamespace(models=models), sleep=clock.sleep, clock=clock),
        models,
        clock,
    )


class _FakeTimeout(Exception):
    pass


_FakeTimeout.__name__ = "ReadTimeout"


def test_timeouts_do_not_chain_past_the_deadline():
    """Build 0.2 took 70 s: every model/retry waited its own 30 s timeout."""
    c, models, clock = _gem([(30, _FakeTimeout()), (30, _FakeTimeout()), (30, _FakeTimeout())])
    with pytest.raises(LLMError):
        c.complete("s", "u")
    assert clock.t <= 25.0 + 30  # never more than one attempt beyond the budget...
    assert models.calls == ["m1"]  # ...and a timeout is not retried: the budget is gone


def test_timeout_moves_to_next_model_not_same_model():
    c, models, clock = _gem([(5, _FakeTimeout()), (1, "answer [C1]")])
    assert c.complete("s", "u").text == "answer [C1]"
    assert models.calls == ["m1", "m2"] and clock.t == 6


def test_long_retry_delay_skips_to_next_model_without_sleeping():
    body = {"error": {"message": "q", "status": "R", "details": [{"retryDelay": "40s"}]}}
    c, models, clock = _gem([(0.2, _api_error(429, body)), (0.2, "ok [C1]")])
    assert c.complete("s", "u").text == "ok [C1]"
    assert models.calls == ["m1", "m2"] and clock.t < 1  # did not sit out 40 s


def test_short_retry_delay_is_honoured_then_retried():
    body = {"error": {"message": "q", "status": "R", "details": [{"retryDelay": "3s"}]}}
    c, models, clock = _gem([(0.1, _api_error(429, body)), (0.1, "ok [C1]")])
    assert c.complete("s", "u").text == "ok [C1]"
    assert models.calls == ["m1", "m1"] and 3.0 < clock.t < 4.0


def test_parse_retry_delay_variants():
    assert parse_retry_delay(_api_error(429, {"error": {"details": [{"retryDelay": "7s"}]}})) == 7.0
    assert (
        parse_retry_delay(_api_error(429, {"error": {"message": "Please retry in 12.3s"}})) == 12.3
    )
    assert parse_retry_delay(_api_error(500)) is None


def test_burst_calls_are_paced():
    c, _, clock = _gem([(0.1, "a [C1]"), (0.1, "b [C1]")], min_interval_s=4.0)
    c.complete("s", "u")
    t_first = clock.t
    c.complete("s", "u")
    assert clock.t - t_first >= 3.9


# ======================================================= 5. corpus fetch: alternates + manifest
def test_fetch_uses_alternate_and_records_the_url_actually_used(tmp_path, monkeypatch):
    import requests

    from hcai.rag import fetch

    body = " ".join(["Blood pressure facts."] * 30)

    def fake_download(url: str) -> str:
        if "cdc.gov" in url:
            raise requests.HTTPError("403 Client Error: Forbidden")
        return "# Title\n\n" + body

    monkeypatch.setattr(fetch, "_download", fake_download)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    status = fetch.fetch_all(tmp_path, only=["cdc-hbp-about", "who-hypertension"])
    assert status["who-hypertension"] == "ok"
    assert status["cdc-hbp-about"].startswith("ok (alternate: https://www.nhlbi.nih.gov")
    manifest = {
        e["doc_id"]: e for e in json.loads((tmp_path / "manifest.json").read_text())["documents"]
    }
    assert manifest["cdc-hbp-about"]["url"].startswith("https://www.nhlbi.nih.gov")
    assert "NHLBI" in manifest["cdc-hbp-about"]["title"]


def test_fetch_reports_failure_when_all_candidates_fail(tmp_path, monkeypatch):
    import requests

    from hcai.rag import fetch

    def boom(url: str) -> str:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(fetch, "_download", boom)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    status = fetch.fetch_all(tmp_path, only=["cdc-hbp-about"])
    assert (
        status["cdc-hbp-about"].startswith("FAILED")
        and status["cdc-hbp-about"].count("offline") == 2
    )


def test_fetch_sends_browser_like_headers():
    from hcai.rag.fetch import HEADERS

    assert HEADERS["User-Agent"].startswith("Mozilla/5.0") and "Accept-Language" in HEADERS


# ===================================== 6. RAG eval: threshold advice and behaviour failures
def test_threshold_recommendation_caps_false_refusals():
    sweep = [
        {"threshold": 0.5, "false_refusal_rate": 0.00, "false_answer_rate": 0.50},
        {"threshold": 0.6, "false_refusal_rate": 0.05, "false_answer_rate": 0.17},
        {"threshold": 0.74, "false_refusal_rate": 0.10, "false_answer_rate": 0.00},  # old pick
    ]
    from hcai.eval.rag_eval import recommend_threshold

    assert recommend_threshold(sweep) == 0.6
    assert recommend_threshold(sweep, max_frr=None) == 0.74
    assert (
        recommend_threshold(
            [{"threshold": 0.9, "false_refusal_rate": 0.5, "false_answer_rate": 0.0}]
        )
        == 0.9
    )
    assert recommend_threshold([]) is None


def test_behaviour_expectations():
    from hcai.eval.rag_eval import _behaved

    assert _behaved("answer", "llm") and _behaved("answer", "extractive")
    assert not _behaved("answer", "refusal") and not _behaved("answer", "emergency")
    assert _behaved("refuse", "refusal") and not _behaved("refuse", "llm")
    assert _behaved("emergency", "emergency") and not _behaved("emergency", "refusal")


def test_faithfulness_proxy_ignores_grouped_tags():
    from hcai.eval.rag_eval import faithfulness_proxy

    cited = ["Eating less salt lowers blood pressure in most adults"]
    assert (
        faithfulness_proxy("Eating less salt lowers blood pressure in adults [C1, C2].", cited)
        == 1.0
    )


# ========================================================== 7. embedder API-name compatibility
@pytest.mark.parametrize("attr", ["get_embedding_dimension", "get_sentence_embedding_dimension"])
def test_embedder_dimension_works_with_old_and_new_sentence_transformers(monkeypatch, attr):
    import sys

    from hcai.rag.embeddings import SentenceTransformerEmbedder

    class Fake:
        def __init__(self, *a, **k): ...
        def encode(self, texts, **k):
            return np.ones((len(texts), 384), dtype=np.float32)

    setattr(Fake, attr, lambda self: 384)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Fake)
    )
    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", "q: ")
    assert e.embed_documents(["a"]).shape == (1, 384) and e.dim == 384


# ================================================================ 8. leakage diagnostics + gate
def test_duplicate_overlap_counts_exact_train_rows(synth_data):
    from hcai.ml.leakage import duplicate_overlap

    x, _, _ = synth_data
    assert duplicate_overlap(x.iloc[:200], x.iloc[200:])["n_overlap"] == 0
    leaky = pd.concat([x.iloc[200:230], x.iloc[:5]])
    assert duplicate_overlap(x.iloc[:200], leaky) == {"n_overlap": 5, "fraction": 5 / 35}


def test_single_feature_auc_flags_a_target_copy(synth_data):
    from hcai.ml.leakage import single_feature_auc

    x, y, _ = synth_data
    assert max(single_feature_auc(x, y).values()) < 0.9
    x2 = x.copy()
    x2["chol"] = y.to_numpy() + np.random.default_rng(0).normal(0, 0.01, len(y))
    assert single_feature_auc(x2, y)["chol"] > 0.99


def test_clean_synthetic_data_passes_with_chance_level_shuffle(cfg, synth_data):
    from hcai.ml.train import train

    res = train(cfg, data=synth_data, tag="t")
    lk = res.report["leakage"]
    assert res.report["sanity"]["verdict"] in {"pass", "warn"} and lk["clean"]
    assert abs(lk["label_shuffle"]["mean"] - 0.5) < 0.12
    assert cfg.paths.model.is_file() and res.run_id


def test_high_auc_with_clean_diagnostics_warns_and_still_saves(cfg, synth_data):
    """The false alarm from the real run: AUC above the ceiling but no leakage -> warn, keep going."""
    from hcai.ml.train import train

    strict = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"auc_ceiling": 0.55})})
    res = train(strict, data=synth_data)
    assert res.report["sanity"]["verdict"] == "warn" and not res.report["sanity"]["ceiling_ok"]
    assert "stability" in res.report["sanity"]["reasons"][0]
    assert strict.paths.model.is_file()


def test_target_copy_feature_aborts_saves_nothing_and_is_archived(cfg, synth_data):
    from hcai.ml.runs import list_runs
    from hcai.ml.train import train

    x, y, sha = synth_data
    x = x.copy()
    x["chol"] = y.to_numpy() * 100.0 + 120  # perfect target proxy
    with pytest.raises(LeakageSuspectedError, match="alone separates"):
        train(cfg, data=(x, y, sha))
    assert not cfg.paths.model.exists()
    assert (cfg.paths.reports_dir / "metrics.json").is_file()  # evidence is kept
    assert [r["status"] for r in list_runs(cfg)] == ["aborted"]


def test_duplicate_rows_across_the_split_abort(cfg, synth_data):
    from hcai.ml.train import train

    x, y, _ = synth_data
    xd = pd.concat([x] * 3, ignore_index=True)
    yd = pd.concat([y] * 3, ignore_index=True)
    with pytest.raises(LeakageSuspectedError, match="also appear in train"):
        train(cfg, data=(xd, yd, "sha"))


def test_split_stability_reports_distribution(cfg, synth_data):
    from hcai.ml.leakage import split_stability, stability_markdown

    x, y, _ = synth_data
    res = split_stability(cfg, x, y, 4, models=("logreg",))
    s = res["per_model"]["logreg"]
    assert res["n_splits"] == 4 and 0.5 < s["min"] <= s["mean"] <= s["max"] <= 1.0
    assert 0.0 <= res["share_any_model_above_ceiling"] <= 1.0
    assert "| logreg |" in stability_markdown(res, "synthetic")


# ====================================================== 9. Kaggle adapter + absent features
def test_kaggle_adapter_maps_codes_to_cleveland_schema():
    from hcai.ml.datasets import adapt_kaggle_heart

    k = make_kaggle_like(60)
    k.loc[0, ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]] = [
        "M",
        "ASY",
        "LVH",
        "Y",
        "Down",
    ]
    k.loc[0, ["Cholesterol", "RestingBP"]] = [0, 0]
    x, y = adapt_kaggle_heart(k)
    r = x.iloc[0]
    assert (r.sex, r.cp, r.restecg, r.exang, r.slope) == (1, 4, 2, 1, 3)
    assert np.isnan(r.chol) and np.isnan(r.trestbps)  # 0 = missing, not a value
    assert x["ca"].isna().all() and x["thal"].isna().all()
    assert list(x.columns)[:3] == ["age", "sex", "cp"] and set(y) <= {0, 1}


def test_kaggle_adapter_accepts_alternate_chest_pain_column_name():
    from hcai.ml.datasets import adapt_kaggle_heart

    k = make_kaggle_like(40).rename(columns={"ChestPainType": "ChestPain"})
    assert len(adapt_kaggle_heart(k)[0]) == 40


def test_kaggle_adapter_rejects_wrong_files():
    from hcai.ml.datasets import adapt_kaggle_heart

    with pytest.raises(DataValidationError, match="missing columns"):
        adapt_kaggle_heart(make_kaggle_like(30).drop(columns=["MaxHR"]))
    bad = make_kaggle_like(30)
    bad.loc[0, "Sex"] = "X"
    with pytest.raises(DataValidationError, match="unexpected values"):
        adapt_kaggle_heart(bad)
    bad = make_kaggle_like(30)
    bad.loc[0, "HeartDisease"] = 2
    with pytest.raises(DataValidationError, match="0/1"):
        adapt_kaggle_heart(bad)


def test_load_named_needs_file_then_caches_it(cfg, tmp_path):
    from hcai.errors import DataUnavailableError
    from hcai.ml.datasets import KAGGLE, KAGGLE_CACHE, load_named

    with pytest.raises(DataUnavailableError, match=r"kaggle\.com"):
        load_named(cfg, KAGGLE)
    src = tmp_path / "heart.csv"
    make_kaggle_like(120).to_csv(src, index=False)
    x1, _, sha1 = load_named(cfg, KAGGLE, src)
    assert (cfg.paths.data_raw / KAGGLE_CACHE).is_file()
    x2, _, sha2 = load_named(cfg, KAGGLE)  # served from the cache, no --source-file needed
    assert sha1 == sha2 and len(x1) == len(x2) == 120
    with pytest.raises(DataUnavailableError, match="Unknown dataset"):
        load_named(cfg, "nope")


def test_feature_spec_detects_absent_and_missing_columns(synth_data):
    from hcai.ml.datasets import adapt_kaggle_heart
    from hcai.ml.features import FeatureSpec

    assert FeatureSpec.from_data(synth_data[0]).absent == ()
    spec = FeatureSpec.from_data(adapt_kaggle_heart(make_kaggle_like(200))[0])
    assert spec.absent == ("ca", "thal") and spec.indicator == ("chol",)


def test_end_to_end_training_without_ca_and_thal_and_prediction(cfg, tmp_path, valid_patient):
    from hcai.ml.datasets import KAGGLE, config_for
    from hcai.ml.predict import RiskPredictor, predict_risk
    from hcai.ml.train import train

    src = tmp_path / "heart.csv"
    make_kaggle_like(400).to_csv(src, index=False)
    kcfg = config_for(cfg, KAGGLE)
    assert "datasets" in str(kcfg.paths.model) and kcfg.paths.model != cfg.paths.model
    res = train(kcfg, dataset=KAGGLE, source_file=src, tag="k")
    assert res.metadata["absent_features"] == ["ca", "thal"]
    assert (
        res.report["ablations_cv_auc"]["hist_gb_native_nan"] > 0.5
    )  # all-NaN cols no longer crash
    assert "ca" not in res.metadata["ranges"]
    assert not cfg.paths.model.exists()  # the primary (Cleveland) model slot is untouched
    pred = RiskPredictor.load(kcfg)
    r = predict_risk(valid_patient, pred)  # patient includes ca/thal: must be ignored, not crash
    assert 0 <= r.probability <= 1 and sorted(r.ignored_fields) == ["ca", "thal"]
    assert not r.imputed_fields and all(f.feature not in {"ca", "thal"} for f in r.top_factors)
    with pytest.raises(ArtifactMissingError):
        RiskPredictor.load(cfg)


def test_default_dataset_paths_are_unchanged(cfg):
    from hcai.ml.datasets import DEFAULT_DATASET, config_for

    assert config_for(cfg, DEFAULT_DATASET) is cfg


# ================================================================ 10. run archive + comparison
def _fake_report(auc: float, sel: str = "logreg", leakage: bool = True) -> dict:
    ci = [auc - 0.07, min(auc + 0.05, 1.0)]
    test = {
        sel: {
            "metrics": {
                "roc_auc": auc,
                "accuracy": 0.85,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
            },
            "ci95": {"roc_auc": ci, "recall": [0.6, 0.9]},
            "brier": 0.12,
        }
    }
    rep = {"selected": sel, "n_train": 242, "n_test": 61, "class_balance_positive": 0.46,
           "cv": {sel: {"mean": 0.9, "std": 0.03, "best_params": {}}}, "test": test,
           "nested_cv_auc": {"mean": 0.9, "std": 0.03}, "calibration": {"applied": False},
           "sanity": {"ceiling_ok": auc <= 0.96}}  # fmt: skip
    if leakage:
        rep["sanity"] = {"verdict": "pass", "reasons": [], "ceiling_ok": True}
        rep["leakage"] = {
            "duplicates": {},
            "single_feature_auc_max": {},
            "label_shuffle": {"mean": 0.5},
            "clean": True,
        }
    return rep


def test_import_legacy_metrics_then_compare_same_data(cfg, tmp_path):
    from hcai.ml.runs import archive_run, compare_runs, import_legacy, resolve_run, summarise

    old = tmp_path / "old" / "metrics.json"
    old.parent.mkdir()
    old.write_text(json.dumps(_fake_report(0.965, leakage=False)), encoding="utf-8")
    import_legacy(cfg, old, "cleveland", "build2", "0.2", data_sha="abc")
    new = summarise(_fake_report(0.921), dataset="cleveland", tag="build3", status="ok", seed=42,
                    data_sha="abc", absent=[])  # fmt: skip
    rid_new = archive_run(cfg, new, [])
    a, b = resolve_run(cfg, "build2"), resolve_run(cfg, rid_new)
    assert (
        a["verdict"] == "aborted (old ceiling rule)"
        and a["build"] == "0.2"
        and b["verdict"] == "pass"
    )
    cmp = compare_runs(a, b)
    assert cmp["same_data"] and cmp["ci_overlap"] is True
    assert cmp["delta_selected_b_minus_a"]["roc_auc"] == pytest.approx(0.921 - 0.965)
    assert any("same train/test split" in n for n in cmp["notes"])


def test_compare_different_datasets_warns_about_comparability(cfg):
    from hcai.ml.runs import compare_runs, summarise

    a = {"run_id": "a", **summarise(_fake_report(0.92), dataset="cleveland", tag="", status="ok", seed=42, data_sha="1", absent=[])}  # fmt: skip
    b = {"run_id": "b", **summarise(_fake_report(0.85), dataset="kaggle_heart_failure", tag="", status="ok", seed=42, data_sha="2", absent=["ca", "thal"])}  # fmt: skip
    notes = " ".join(compare_runs(a, b)["notes"])
    assert "not directly comparable" in notes and "not fully independent" in notes and "ca" in notes


def test_resolve_run_errors_and_markdown_written(cfg):
    from hcai.ml.runs import RunNotFoundError, archive_run, resolve_run, summarise, write_comparison

    s = summarise(
        _fake_report(0.9),
        dataset="cleveland",
        tag="x",
        status="ok",
        seed=1,
        data_sha="1",
        absent=[],
    )
    archive_run(cfg, s, [])
    archive_run(cfg, s, [])  # same second: must not collide
    with pytest.raises(RunNotFoundError, match="No run matches"):
        resolve_run(cfg, "zzz")
    with pytest.raises(RunNotFoundError, match="several runs"):
        resolve_run(cfg, "cleveland_x")
    ids = sorted(p.name for p in (cfg.paths.artifacts / "runs").iterdir())
    a, b = resolve_run(cfg, ids[0]), resolve_run(cfg, ids[1])
    md = write_comparison(cfg, a, b)
    assert md.is_file() and "Run comparison" in md.read_text(encoding="utf-8")


def test_import_rejects_non_metrics_files(cfg, tmp_path):
    from hcai.ml.runs import RunNotFoundError, import_legacy

    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(RunNotFoundError):
        import_legacy(cfg, f, "cleveland", "t", "b")
    with pytest.raises(RunNotFoundError, match="not found"):
        import_legacy(cfg, tmp_path / "nope.json", "cleveland", "t", "b")


def test_cli_runs_stability_compare_and_status(cfg_file, cfg, synth_data, tmp_path, capsys):
    from hcai.ml.train import train

    train(cfg, data=synth_data, tag="one")
    assert _run(cfg_file, "runs") == 0
    assert "cleveland" in capsys.readouterr().out
    assert _run(cfg_file, "compare", "one", "one") == 0
    assert "Run comparison" in capsys.readouterr().out
    assert _run(cfg_file, "compare", "nothing", "one") == 2
    assert _run(cfg_file, "status") == 0
    assert "loaded from" in capsys.readouterr().out


# ----------------------------------------- found in the real run: self-comparison + fetch counting
def test_import_refuses_a_metrics_file_written_by_the_current_build(cfg, synth_data, tmp_path):
    """The real mistake: train (0.3) overwrote metrics.json, then it was imported as 'build 0.2'."""
    from hcai.ml.runs import RunNotFoundError, import_legacy
    from hcai.ml.train import train

    train(cfg, data=synth_data)
    current = cfg.paths.reports_dir / "metrics.json"
    with pytest.raises(RunNotFoundError, match=r"build 0\.3 or later"):
        import_legacy(cfg, current, "cleveland", "build02", "0.2")
    assert import_legacy(cfg, current, "cleveland", "forced", "0.2", allow_current_build=True)


def test_fetch_counts_alternates_as_success():
    from hcai.rag.fetch import failed_ids

    status = {
        "a": "ok",
        "b": "ok (alternate: https://x)",
        "c": "FAILED: HTTPError: 403 | HTTPError: 403",
    }
    assert failed_ids(status) == ["c"]
