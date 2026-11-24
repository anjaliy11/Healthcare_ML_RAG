"""Combined flow, degradation, CLI and Streamlit smoke tests (offline: hashing embedder, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hcai.app import cli
from hcai.app.service import FACTOR_QUERIES, Services
from hcai.config import Config
from hcai.errors import ArtifactMissingError, IndexMissingError, PatientInputError
from hcai.ml.data import hash_frame, validate_and_split_target
from hcai.ml.schema import FEATURES, TARGET
from hcai.ml.train import train
from hcai.rag.embeddings import HashingEmbedder
from hcai.rag.pipeline import build_index
from hcai.rag.safety import check_emergency
from tests.fixtures.synth import make_synthetic
from tests.rag_helpers import build_corpus

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stack(tmp_path_factory):  # type: ignore[no-untyped-def]
    """A fully built stack in a temp dir: model + index, written config file."""
    import pandas as pd

    from hcai.config import load_config, project_root

    tmp = tmp_path_factory.mktemp("stack")
    base = load_config()
    paths = {k: tmp / v.relative_to(project_root()) for k, v in base.paths.model_dump().items()}
    ml = base.ml.model_copy(
        update={"cv_repeats": 1, "bootstrap_resamples": 50, "rf_trees": 30, "hgb_max_iter": 30,
                "permutation_repeats": 2,
                "grids": {"logreg": {"C": [1.0]}, "random_forest": {"max_depth": [3]}, "hist_gb": {"max_depth": [2]}}}
    )  # fmt: skip
    cfg = base.model_copy(update={"paths": type(base.paths)(**paths), "ml": ml})
    x, y = validate_and_split_target(make_synthetic(), 303)
    train(cfg, data=(x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1))))
    build_corpus(cfg.paths.docs_dir)
    build_index(cfg, HashingEmbedder())
    cfg_file = tmp / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
    return cfg, cfg_file


def test_all_factor_queries_pass_the_safety_gate():
    """Canned explanation queries must never be mistaken for emergencies."""
    assert set(FACTOR_QUERIES) == set(FEATURES)
    assert all(check_emergency(q) is None for q in FACTOR_QUERIES.values())


def test_assess_combines_risk_and_cited_explanations(stack, valid_patient):
    cfg, _ = stack
    svc = Services(cfg, use_llm=False)
    res = svc.assess(valid_patient)
    assert res.risk.top_factors and res.disclaimer
    assert res.explanations
    answered = [e.answer for e in res.explanations if e.answer and not e.answer.refused]
    assert all(a.sources for a in answered)  # no uncited explanation


def test_assess_rejects_invalid_patient(stack, valid_patient):
    cfg, _ = stack
    valid_patient["chol"] = 5000
    with pytest.raises(PatientInputError):
        Services(cfg, use_llm=False).assess(valid_patient)


def test_assess_degrades_when_index_missing(stack, valid_patient, tmp_path: Path):
    cfg, _ = stack
    broken = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"index_dir": tmp_path / "nope"})}
    )
    res = Services(broken, use_llm=False).assess(valid_patient)
    assert res.risk.probability >= 0 and any("unavailable" in n for n in res.notes)
    with pytest.raises(IndexMissingError):
        Services(broken, use_llm=False).ask("salt")


def test_ask_works_when_model_missing(stack, tmp_path: Path):
    cfg, _ = stack
    broken = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"model": tmp_path / "m.joblib"})}
    )
    svc = Services(broken, use_llm=False)
    assert svc.ask("What are the common symptoms of diabetes?").sources
    with pytest.raises(ArtifactMissingError):
        svc.predictor  # noqa: B018
    assert "unavailable" in svc.provenance()["model_version"]


def test_assess_logs_no_patient_values(stack, valid_patient):
    cfg, _ = stack
    cfg.paths.metrics_log.unlink(missing_ok=True)
    Services(cfg, use_llm=False).assess({**valid_patient, "chol": 199})
    raw = cfg.paths.metrics_log.read_text()
    assert "199" not in raw and "chol" not in raw.replace("range_flags", "")


# ----------------------------------------------------------------- CLI
def run_cli(cfg_file: Path, *argv: str) -> int:
    return cli.main(["--config", str(cfg_file), *argv])


def test_cli_status_ask_and_assess(stack, capsys, tmp_path: Path, valid_patient, monkeypatch):
    _, cfg_file = stack
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run_cli(cfg_file, "status") == 0
    assert "[x] trained model" in capsys.readouterr().out

    assert run_cli(cfg_file, "ask", "What are the common symptoms of diabetes?") == 0
    out = capsys.readouterr().out
    assert "Sources:" in out and "not a medical device" in out

    assert run_cli(cfg_file, "ask", "I have chest pain") == 0
    assert "emergency" in capsys.readouterr().out.lower()

    p = tmp_path / "patient.json"
    p.write_text(json.dumps(valid_patient))
    assert run_cli(cfg_file, "assess", "--input", str(p)) == 0
    out = capsys.readouterr().out
    assert "Estimated risk" in out and "not a medical device" in out

    assert run_cli(cfg_file, "assess", "--example") == 0
    assert json.loads(capsys.readouterr().out)["age"] == 58


def test_cli_flags_and_friendly_errors(stack, capsys):
    _, cfg_file = stack
    rc = run_cli(cfg_file, "assess", "--age", "200", "--sex", "1")
    assert rc == 2
    assert "Invalid patient input" in capsys.readouterr().err
    assert run_cli(cfg_file, "ask", "   ") == 2


def test_cli_monitor_and_drift(stack, capsys, tmp_path: Path):
    _, cfg_file = stack
    assert run_cli(cfg_file, "monitor") == 0
    assert "requests_total" in capsys.readouterr().out
    make_synthetic(120, seed=5).drop(columns=[TARGET]).to_csv(tmp_path / "in.csv", index=False)
    assert run_cli(cfg_file, "monitor", "--drift-csv", str(tmp_path / "in.csv")) == 0
    assert '"status"' in capsys.readouterr().out


def test_cli_missing_artifacts_message(tmp_path: Path, capsys, cfg: Config):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
    assert run_cli(f, "ask", "salt") == 2
    assert "python -m hcai index" in capsys.readouterr().err
    assert run_cli(f, "assess", "--example") == 0


def test_cli_index_offline_and_eval(stack, capsys):
    cfg, cfg_file = stack
    assert run_cli(cfg_file, "index", "--offline") == 0
    assert "OFFLINE" in capsys.readouterr().out
    assert run_cli(cfg_file, "eval", "--no-llm") == 0
    out = capsys.readouterr().out
    assert "Threshold trade-off" in out
    assert (cfg.paths.reports_dir / "rag_eval.json").is_file()


# ----------------------------------------------------------------- Streamlit
def test_streamlit_app_renders_and_answers(stack, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _, cfg_file = stack
    monkeypatch.setenv("HCAI_CONFIG", str(cfg_file))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[2]
    at = AppTest.from_file(str(root / "src/hcai/app/streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not a medical device" in w.value for w in at.warning)
    at.tabs[0].button[0].click().run()  # submit the form with defaults
    assert not at.exception
    assert any("Model risk estimate" in m.label for m in at.metric)
    at.tabs[1].text_input[0].set_value("What are the common symptoms of diabetes?").run()
    at.tabs[1].button[0].click().run()
    assert not at.exception


# ----------------------------------------------------------------- data loading + real-embedder wiring
def test_cli_data_and_train_from_source_file(stack, capsys, tmp_path: Path):
    cfg, cfg_file = stack
    src = tmp_path / "heart.csv"
    make_synthetic(seed=3).to_csv(src, index=False)
    assert run_cli(cfg_file, "data", "--source-file", str(src)) == 0
    out = capsys.readouterr().out
    assert "rows=303" in out and '"ca": 4' in out and '"thal": 2' in out
    assert (cfg.paths.data_raw / "heart_cleveland.csv").is_file()  # cached for next time
    assert run_cli(cfg_file, "data") == 0  # now served from the cache, no network
    capsys.readouterr()
    assert run_cli(cfg_file, "train") == 0
    out = capsys.readouterr().out
    assert "selected:" in out and "auc" in out


def test_load_raw_from_uci_processed_format_and_failure_message(
    cfg: Config, tmp_path: Path, monkeypatch
):
    import sys
    import types

    from hcai.errors import DataUnavailableError
    from hcai.ml.data import load_dataset

    with pytest.raises(DataUnavailableError, match="not found"):
        load_dataset(cfg, tmp_path / "does_not_exist.csv")

    fake = types.ModuleType("ucimlrepo")

    def boom(id: int):
        raise ConnectionError("offline")

    fake.fetch_ucirepo = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ucimlrepo", fake)
    with pytest.raises(DataUnavailableError, match=r"Could not download.*--source-file"):
        load_dataset(cfg)

    rows = make_synthetic(seed=4)
    raw = tmp_path / "processed.cleveland.data"
    rows.to_csv(raw, header=False, index=False, na_rep="?")
    x, y, sha = load_dataset(cfg, raw)
    assert len(x) == 303 and len(sha) == 64 and set(y.unique()) == {0, 1}


def test_sentence_transformer_embedder_applies_query_prefix_without_downloading():
    import numpy as np

    from hcai.rag.embeddings import SentenceTransformerEmbedder

    seen: list[list[str]] = []

    class FakeST:
        def encode(self, texts, **kw):
            seen.append(list(texts))
            assert kw["normalize_embeddings"] is True
            return np.ones((len(texts), 384), dtype=np.float64)

        def get_sentence_embedding_dimension(self):
            return 384

    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", "QUERY: ")
    e._model = FakeST()
    assert e.embed_queries(["hello"]).dtype == np.float32
    assert seen[-1] == ["QUERY: hello"]  # instruction only on the query side
    e.embed_documents(["doc"])
    assert seen[-1] == ["doc"] and e.dim == 384
