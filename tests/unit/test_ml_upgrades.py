"""Engineered features, extra candidates, voting, calibration, paired bootstrap, nested CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from hcai.config import Config
from hcai.ml.evaluate import explain_selection, paired_auc_diff_ci
from hcai.ml.features import ENGINEERED, AddClinicalFeatures, build_pipeline, build_voting
from hcai.ml.predict import RiskPredictor
from hcai.ml.registry import load_artifact
from hcai.ml.schema import FEATURES
from hcai.ml.train import train


def _with(cfg: Config, **ml: object) -> Config:
    return cfg.model_copy(update={"ml": cfg.ml.model_copy(update=ml)})


# ------------------------------------------------------------------ feature engineering
def test_clinical_features_values_nan_safety_and_no_mutation():
    x = pd.DataFrame(
        {"age": [60.0, 50.0, 230.0, np.nan], "thalach": [128.0, 170.0, 100.0, 150.0],
         "oldpeak": [2.0, 0.0, 1.0, 1.0], "exang": [1.0, 1.0, 0.0, np.nan]}
    )  # fmt: skip
    before = x.copy()
    out = AddClinicalFeatures().fit(x).transform(x)
    pd.testing.assert_frame_equal(x, before)  # input untouched
    assert out.loc[0, "hr_pct_max"] == pytest.approx(128 / 160)
    assert out.loc[1, "oldpeak_x_exang"] == 0.0 and out.loc[0, "oldpeak_x_exang"] == 2.0
    assert np.isfinite(out.loc[2, "hr_pct_max"])  # age >= 220 clipped, no division blow-up
    assert np.isnan(out.loc[3, "hr_pct_max"]) and np.isnan(out.loc[3, "oldpeak_x_exang"])
    assert set(ENGINEERED) <= set(out.columns)


def test_logreg_fe_pipeline_has_fe_step_and_handles_missing(cfg: Config, synth_data):
    x, y, _ = synth_data
    pipe = build_pipeline("logreg_fe", cfg).fit(x, y)
    assert [n for n, _ in pipe.steps] == ["fe", "pre", "model"]
    row = x.iloc[[0]].copy()
    row["ca"] = np.nan
    p = pipe.predict_proba(row)[0, 1]
    assert 0 <= p <= 1
    assert list(x.columns) == FEATURES  # the 13 raw inputs are all the caller ever supplies


def test_svm_and_voting_produce_probabilities(cfg: Config, synth_data):
    x, y, _ = synth_data
    svm = build_pipeline("svm_rbf", cfg).fit(x, y)
    vote = build_voting({k: build_pipeline(k, cfg) for k in ("logreg", "svm_rbf")}).fit(x, y)
    for m in (svm, vote):
        p = m.predict_proba(x.head(5))
        assert p.shape == (5, 2) and np.allclose(p.sum(axis=1), 1)


# ------------------------------------------------------------------ paired bootstrap
def test_paired_auc_difference_zero_for_identical_and_positive_for_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good = y + rng.normal(0, 0.6, 200)
    bad = rng.normal(size=200)
    same = paired_auc_diff_ci(y, good, good, 200, 1)
    assert same == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    better = paired_auc_diff_ci(y, good, bad, 200, 1)
    assert (
        better["diff"] > 0.2 and better["lo"] > 0 and better["lo"] <= better["diff"] <= better["hi"]
    )
    worse = paired_auc_diff_ci(y, bad, good, 200, 1)
    assert worse["hi"] < 0


def test_explanation_states_significance_honestly():
    cv = {"logreg": {"mean": 0.90, "std": 0.02}, "svm_rbf": {"mean": 0.905, "std": 0.02}}
    test = {k: {"ci95": {"roc_auc": [0.8, 0.95]}} for k in cv}
    none = explain_selection(
        "logreg", cv, test, 0.01, {"svm_rbf": {"diff": 0.01, "lo": -0.05, "hi": 0.06}}
    )
    assert "none is demonstrably better" in none
    some = explain_selection(
        "svm_rbf", cv, test, 0.01, {"svm_rbf": {"diff": 0.08, "lo": 0.02, "hi": 0.14}}
    )
    assert "interval excludes 0" in some and "+0.080" in some


# ------------------------------------------------------------------ full training with all candidates
@pytest.mark.slow
def test_train_with_all_candidates_reports_everything(cfg: Config, synth_data):
    cfg = _with(
        cfg,
        candidates=["logreg", "logreg_fe", "svm_rbf", "random_forest", "hist_gb", "voting"],
        voting_members=["logreg", "svm_rbf", "random_forest"],
        nested_cv=True,
        grids={"logreg": {"C": [0.1, 1.0]}, "logreg_fe": {"C": [0.1, 1.0]},
               "svm_rbf": {"estimator__C": [1.0], "estimator__gamma": [0.02]}, "random_forest": {"max_depth": [3]},
               "hist_gb": {"max_depth": [2]}},
    )  # fmt: skip
    res = train(cfg, data=synth_data)
    r = res.report
    assert set(r["test"]) == set(r["cv"]) == set(cfg.ml.candidates)
    assert r["cv"]["voting"]["best_params"]["members"] == ["logreg", "svm_rbf", "random_forest"]
    assert set(r["auc_diff_vs_baseline"]) == set(cfg.ml.candidates) - {"logreg"}
    assert r["nested_cv_auc"] and 0.5 < r["nested_cv_auc"]["mean"] <= 1
    assert "applied" in r["calibration"] and "brier_oof_raw" in r["calibration"]
    assert res.selected in cfg.ml.candidates
    md = (cfg.paths.reports_dir / "model_report.md").read_text()
    for section in ("nested CV", "paired bootstrap", "Calibration", "Ablations"):
        assert section in md
    RiskPredictor.load(cfg)  # whatever won (incl. voting) loads and is usable


@pytest.mark.slow
def test_forced_calibration_is_saved_and_usable(cfg: Config, synth_data, valid_patient):
    cfg = _with(cfg, calibration_margin=-1.0)  # always "improves": exercises the applied branch
    res = train(cfg, data=synth_data)
    assert res.report["calibration"]["applied"] is True
    pipe, _ = load_artifact(cfg)
    assert isinstance(pipe, CalibratedClassifierCV)
    pred = RiskPredictor.load(cfg)
    out = pred.predict(
        [__import__("hcai.ml.schema", fromlist=["x"]).PatientInput(**valid_patient)]
    )[0]
    assert 0 <= out.probability <= 1 and out.top_factors is not None


@pytest.mark.slow
def test_calibration_disabled_and_voting_needs_two_members(cfg: Config, synth_data):
    cfg = _with(
        cfg, calibration_margin=None, candidates=["logreg", "voting"], voting_members=["logreg"]
    )
    res = train(cfg, data=synth_data)
    assert res.report["calibration"] == {
        "brier_oof_raw": res.report["calibration"]["brier_oof_raw"],
        "applied": False,
    }
    assert "voting" not in res.report["test"]  # one member: voting skipped, not crashed
    assert res.selected == "logreg"
