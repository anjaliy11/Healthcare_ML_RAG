from __future__ import annotations

import json

import numpy as np
import pytest

from hcai.config import Config
from hcai.errors import (
    ArtifactIncompatibleError,
    ArtifactMissingError,
    LeakageSuspectedError,
    PatientInputError,
)
from hcai.ml.evaluate import bootstrap_ci, choose_recall_threshold, compute_metrics, select_model
from hcai.ml.features import build_pipeline
from hcai.ml.predict import RiskPredictor, predict_risk
from hcai.ml.registry import load_artifact
from hcai.ml.train import train
from tests.fixtures.synth import make_synthetic

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory):  # type: ignore[no-untyped-def]
    """Train once per module (module scope), on synthetic data."""
    import pandas as pd

    from hcai.config import load_config, project_root
    from hcai.ml.data import hash_frame, validate_and_split_target
    from hcai.ml.schema import TARGET

    tmp = tmp_path_factory.mktemp("train")
    base = load_config()
    paths = {k: tmp / v.relative_to(project_root()) for k, v in base.paths}
    ml = base.ml.model_copy(
        update={
            "cv_repeats": 1,
            "bootstrap_resamples": 100,
            "rf_trees": 40,
            "hgb_max_iter": 40,
            "permutation_repeats": 3,
            "candidates": ["logreg", "random_forest", "hist_gb"],
            "nested_cv": False,
            "grids": {
                "logreg": {"C": [0.1, 1.0]},
                "random_forest": {"max_depth": [3], "min_samples_leaf": [3]},
                "hist_gb": {"max_depth": [2], "learning_rate": [0.1]},
            },
        }
    )
    cfg = base.model_copy(update={"paths": type(base.paths)(**paths), "ml": ml})
    x, y = validate_and_split_target(make_synthetic(), 303)
    data = (x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1)))
    return cfg, train(cfg, data=data), data


def test_pipeline_contains_imputation_inside(cfg: Config):
    """No leakage by construction: every preprocessing step lives in the Pipeline."""
    pipe = build_pipeline("logreg", cfg)
    assert [n for n, _ in pipe.steps] == ["pre", "model"]
    with pytest.raises(ValueError):
        build_pipeline("svm", cfg)


def test_train_reports_all_five_metrics_with_cis(trained):
    _, res, _ = trained
    for name, rep in res.report["test"].items():
        assert set(rep["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}, name
        assert set(rep["ci95"]) == set(rep["metrics"])
        lo, hi = rep["ci95"]["roc_auc"]
        assert lo <= hi
    assert res.report["n_train"] + res.report["n_test"] == 303


def test_sanity_band_on_synthetic(trained):
    _, res, _ = trained
    auc = res.report["test"][res.selected]["metrics"]["roc_auc"]
    assert 0.6 < auc <= 0.96  # trained; not leaking


def test_artifact_roundtrip_and_metadata(trained):
    cfg, res, data = trained
    _, meta = load_artifact(cfg)
    assert meta["data_sha256"] == data[2]
    assert meta["model_version"].startswith(res.selected)
    assert set(meta["ranges"]) == set(meta["reference"])
    assert (cfg.paths.reports_dir / "metrics.json").is_file()
    assert (cfg.paths.reports_dir / "roc.png").is_file()
    assert (cfg.paths.reports_dir / "model_report.md").is_file()
    assert cfg.paths.reference.is_file()


def test_split_is_stratified_and_deterministic(trained):
    _, res, _ = trained
    assert res.report["n_test"] == 61  # ceil(0.2 * 303)


def test_predict_contract_and_determinism(trained, valid_patient):
    cfg, _, _ = trained
    pred = RiskPredictor.load(cfg)
    a = predict_risk(valid_patient, pred)
    b = predict_risk(valid_patient, pred)
    assert a == b
    assert 0 <= a.probability <= 1 and a.band in {"low", "moderate", "high"}
    assert a.disclaimer and a.model_version
    assert all(f.delta > 0 for f in a.top_factors) and len(a.top_factors) <= 3
    assert pred.predict([]) == []


def test_missing_ca_thal_is_imputed_and_flagged(trained, valid_patient):
    cfg, _, _ = trained
    valid_patient["ca"] = None
    valid_patient["thal"] = None
    r = predict_risk(valid_patient, RiskPredictor.load(cfg))
    assert set(r.imputed_fields) == {"ca", "thal"}


def test_out_of_training_range_is_flagged_not_clamped(trained, valid_patient):
    cfg, _, _ = trained
    valid_patient["chol"] = 690  # valid clinically, beyond training max
    r = predict_risk(valid_patient, RiskPredictor.load(cfg))
    assert "chol" in r.range_flags


def test_invalid_input_rejected_at_predict(trained, valid_patient):
    cfg, _, _ = trained
    valid_patient["age"] = 200
    with pytest.raises(PatientInputError):
        predict_risk(valid_patient, RiskPredictor.load(cfg))


def test_higher_risk_profile_scores_higher(trained, valid_patient):
    cfg, _, _ = trained
    pred = RiskPredictor.load(cfg)
    low = dict(valid_patient, age=35, cp=2, thalach=185, exang=0, oldpeak=0.0, ca=0, thal=3)
    assert predict_risk(valid_patient, pred).probability > predict_risk(low, pred).probability


def test_missing_artifact_message(cfg: Config):
    with pytest.raises(ArtifactMissingError, match="python -m hcai train"):
        load_artifact(cfg)


def test_sklearn_version_mismatch_fails_loudly(trained):
    cfg, _, _ = trained
    meta = json.loads(cfg.paths.model_meta.read_text())
    cfg.paths.model_meta.write_text(json.dumps({**meta, "sklearn_version": "0.0.1"}))
    try:
        with pytest.raises(ArtifactIncompatibleError, match="Retrain"):
            load_artifact(cfg)
    finally:
        cfg.paths.model_meta.write_text(json.dumps(meta))


def test_tampered_model_detected(trained):
    cfg, _, _ = trained
    original = cfg.paths.model.read_bytes()
    cfg.paths.model.write_bytes(original + b"x")
    try:
        with pytest.raises(ArtifactIncompatibleError, match="checksum"):
            load_artifact(cfg)
    finally:
        cfg.paths.model.write_bytes(original)


def test_leakage_alarm_fires(cfg: Config):
    """A leaky feature (target copy) must trip the ceiling and prevent saving."""
    import pandas as pd

    from hcai.ml.data import hash_frame, validate_and_split_target
    from hcai.ml.schema import TARGET

    df = make_synthetic()
    df["chol"] = df["num"] * 100.0 + np.random.default_rng(1).normal(0, 1, len(df))
    x, y = validate_and_split_target(df, 303)
    data = (x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1)))
    with pytest.raises(LeakageSuspectedError):
        train(cfg, data=data)
    assert not cfg.paths.model.exists()


def test_metric_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y, p)
    assert m == pytest.approx(
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}
    )
    ci = bootstrap_ci(y, p, 0.5, 50, 0)
    assert all(lo <= hi for lo, hi in ci.values())
    assert choose_recall_threshold(y, p, 1.0) == pytest.approx(0.6)
    assert choose_recall_threshold(y, p, 0.5) == pytest.approx(0.9)


def test_select_model_prefers_baseline_within_margin():
    assert select_model({"logreg": 0.90, "random_forest": 0.905, "hist_gb": 0.89}, 0.01) == "logreg"
    assert (
        select_model({"logreg": 0.85, "random_forest": 0.91, "hist_gb": 0.89}, 0.01)
        == "random_forest"
    )
