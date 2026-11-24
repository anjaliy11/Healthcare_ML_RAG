"""Train, select and evaluate. Test data is touched once, after selection and thresholding."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from hcai.config import Config
from hcai.errors import LeakageSuspectedError
from hcai.logging import get_logger, log_event
from hcai.ml.datasets import DEFAULT_DATASET, load_named
from hcai.ml.evaluate import (
    choose_recall_threshold,
    explain_selection,
    full_report,
    paired_auc_diff_ci,
    save_plots,
    select_model,
)
from hcai.ml.features import FeatureSpec, build_pipeline, build_voting
from hcai.ml.leakage import judge, run_diagnostics
from hcai.ml.registry import save_artifact
from hcai.ml.runs import archive_run, summarise
from hcai.ml.schema import BINARY, CATEGORICAL, FEATURES, NUMERIC

log = get_logger("ml.train")


@dataclass
class TrainResult:
    selected: str
    metadata: dict[str, Any]
    report: dict[str, Any]
    run_id: str = ""


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def training_ranges(x: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed range per feature that the dataset provides (absent features are skipped)."""
    return {
        c: {"min": float(x[c].min()), "max": float(x[c].max())}
        for c in FEATURES
        if x[c].notna().any()
    }


def reference_values(x: pd.DataFrame) -> dict[str, float]:
    """Typical value per feature (median / mode), used for per-patient factor attribution."""
    ref: dict[str, float] = {}
    for c in NUMERIC:
        if x[c].notna().any():
            ref[c] = float(x[c].median())
    for c in BINARY + CATEGORICAL:
        if x[c].notna().any():
            ref[c] = float(x[c].mode(dropna=True).iloc[0])
    return ref


def _nested_cv(
    name: str, x: pd.DataFrame, y: pd.Series, cfg: Config, spec: FeatureSpec
) -> dict[str, float] | None:
    """Unbiased CV estimate for 'tune the winner, then score it' (grid search inside each fold)."""
    if not cfg.ml.nested_cv or name == "voting":
        return None
    grid = {f"model__{k}": v for k, v in cfg.ml.grids.get(name, {}).items()}
    inner = StratifiedKFold(cfg.ml.cv_splits, shuffle=True, random_state=cfg.seed)
    outer = RepeatedStratifiedKFold(
        n_splits=cfg.ml.cv_splits, n_repeats=cfg.ml.cv_repeats, random_state=cfg.seed + 1
    )
    gs = GridSearchCV(build_pipeline(name, cfg, spec=spec), grid, scoring="roc_auc", cv=inner)
    s = cross_val_score(gs, x, y, cv=outer, scoring="roc_auc")
    return {"mean": float(s.mean()), "std": float(s.std())}


def _ablations(
    x: pd.DataFrame, y: pd.Series, cfg: Config, cv: Any, spec: FeatureSpec
) -> dict[str, float]:
    """CV-only comparison on the training split: does the missing-indicator / imputer matter?"""

    def score(pipe: Any) -> float:
        return float(cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc").mean())

    return {
        "logreg_with_indicator": score(build_pipeline("logreg", cfg, True, spec)),
        "logreg_without_indicator": score(build_pipeline("logreg", cfg, False, spec)),
        "hist_gb_imputed": score(build_pipeline("hist_gb", cfg, spec=spec)),
        "hist_gb_native_nan": score(build_pipeline("hist_gb_native_nan", cfg, spec=spec)),
    }


def train(
    cfg: Config,
    source_file: Path | None = None,
    data: tuple[pd.DataFrame, pd.Series, str] | None = None,
    dataset: str = DEFAULT_DATASET,
    tag: str = "",
) -> TrainResult:
    x, y, data_sha = data if data is not None else load_named(cfg, dataset, source_file)
    seed = cfg.seed
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.ml.test_size, stratify=y, random_state=seed
    )
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.ml.cv_splits, n_repeats=cfg.ml.cv_repeats, random_state=seed
    )
    spec = FeatureSpec.from_data(x_tr)  # structure only (which columns exist / have NaN)

    fitted: dict[str, Any] = {}
    cv_summary: dict[str, dict[str, Any]] = {}
    for name in (n for n in cfg.ml.candidates if n != "voting"):
        grid = {f"model__{k}": v for k, v in cfg.ml.grids.get(name, {}).items()}
        gs = GridSearchCV(
            build_pipeline(name, cfg, spec=spec), grid, scoring="roc_auc", cv=cv, refit=True
        )
        gs.fit(x_tr, y_tr)
        i = int(gs.best_index_)
        fitted[name] = gs.best_estimator_
        cv_summary[name] = {
            "mean": float(gs.cv_results_["mean_test_score"][i]),
            "std": float(gs.cv_results_["std_test_score"][i]),
            "best_params": gs.best_params_,
        }
        log_event(log, "cv_done", model=name, cv_auc=round(cv_summary[name]["mean"], 4))

    members = [m for m in cfg.ml.voting_members if m in fitted]
    if "voting" in cfg.ml.candidates and len(members) >= 2:
        vote = build_voting({m: clone(fitted[m]) for m in members})
        scores = cross_val_score(vote, x_tr, y_tr, cv=cv, scoring="roc_auc")
        fitted["voting"] = vote.fit(x_tr, y_tr)
        cv_summary["voting"] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "best_params": {"members": members},
        }

    selected = select_model({k: v["mean"] for k, v in cv_summary.items()}, cfg.ml.selection_margin)

    # Operating point from out-of-fold predictions on TRAIN only.
    oof_cv = StratifiedKFold(cfg.ml.cv_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(fitted[selected]), x_tr, y_tr, cv=oof_cv, method="predict_proba")[
        :, 1
    ]
    # Calibration: a shown "risk %" should mean something. Applied only if it measurably helps
    # out-of-fold (tree models usually need it; logistic regression usually does not).
    brier_raw = float(brier_score_loss(y_tr, oof))
    calibration: dict[str, Any] = {"brier_oof_raw": brier_raw, "applied": False}
    if cfg.ml.calibration_margin is not None:
        calibrated = CalibratedClassifierCV(clone(fitted[selected]), method="sigmoid", cv=5)
        oof_cal = cross_val_predict(calibrated, x_tr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
        brier_cal = float(brier_score_loss(y_tr, oof_cal))
        calibration["brier_oof_calibrated"] = brier_cal
        if brier_raw - brier_cal > cfg.ml.calibration_margin:
            fitted[selected] = clone(calibrated).fit(x_tr, y_tr)
            oof, calibration["applied"] = oof_cal, True
    recall_threshold = choose_recall_threshold(y_tr.to_numpy(), oof, cfg.ml.target_recall)

    # Single test evaluation.
    y_te_arr = y_te.to_numpy()
    test_proba = {n: m.predict_proba(x_te)[:, 1] for n, m in fitted.items()}
    n_boot, boot_seed = cfg.ml.bootstrap_resamples, seed
    test_reports = {
        n: full_report(y_te_arr, p, 0.5, n_boot, boot_seed) for n, p in test_proba.items()
    }
    baseline = "logreg" if "logreg" in test_proba else selected
    diffs = {
        n: paired_auc_diff_ci(y_te_arr, p, test_proba[baseline], n_boot, boot_seed)
        for n, p in test_proba.items()
        if n != baseline
    }
    selected_at_recall = full_report(
        y_te_arr, test_proba[selected], recall_threshold, n_boot, boot_seed
    )

    imp = permutation_importance(
        fitted[selected],
        x_te,
        y_te,
        scoring="roc_auc",
        n_repeats=cfg.ml.permutation_repeats,
        random_state=seed,
    )
    importance = {
        f: {"mean": float(m), "std": float(s)}
        for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std, strict=True)
    }

    report: dict[str, Any] = {
        "selected": selected,
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "class_balance_positive": float(y.mean()),
        "cv": cv_summary,
        "test": test_reports,
        "selected_at_recall_threshold": {
            **selected_at_recall,
            "target_recall": cfg.ml.target_recall,
        },
        "calibration": calibration,
        "auc_diff_vs_baseline": diffs,
        "nested_cv_auc": _nested_cv(selected, x_tr, y_tr, cfg, spec),
        "ablations_cv_auc": _ablations(x_tr, y_tr, cfg, cv, spec),
        "permutation_importance_test": importance,
        "explanation": explain_selection(
            selected, cv_summary, test_reports, cfg.ml.selection_margin, diffs
        ),
    }

    # Leakage gate: direct diagnostics decide; the AUC level alone only warns (see leakage.py).
    sel_auc = test_reports[selected]["metrics"]["roc_auc"]
    max_auc = max(r["metrics"]["roc_auc"] for r in test_reports.values())
    sel_ci = test_reports[selected]["ci95"]["roc_auc"]
    diag = run_diagnostics(x_tr, x_te, y_tr, y_te, cfg, spec)
    report["leakage"] = diag
    report["dataset"] = {"name": dataset, "absent_features": list(spec.absent), "rows": len(x)}
    report["sanity"] = judge(diag, sel_auc, max_auc, (sel_ci[0], sel_ci[1]), cfg)
    verdict = report["sanity"]["verdict"]
    _write_reports(cfg, report, y_te_arr, test_proba, selected)

    def archive(status: str) -> str:
        summary = summarise(
            report, dataset=dataset, tag=tag, status=status, seed=seed, data_sha=data_sha,
            absent=list(spec.absent), git_sha=_git_sha(),
        )  # fmt: skip
        files = [cfg.paths.reports_dir / "metrics.json", cfg.paths.reports_dir / "model_report.md"]
        return archive_run(cfg, summary, files)

    if verdict == "fail":
        run_id = archive("aborted")
        raise LeakageSuspectedError(
            "Model NOT saved: " + "; ".join(report["sanity"]["reasons"])
            + f". Evidence: {cfg.paths.reports_dir / 'metrics.json'} (run {run_id})."
        )  # fmt: skip
    if verdict == "warn":
        log_event(log, "auc_above_ceiling_but_clean", selected_auc=round(sel_auc, 4))

    metadata = {
        "model_name": selected,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "seed": seed,
        "data_sha256": data_sha,
        "threshold_default": 0.5,
        "threshold_recall": recall_threshold,
        "band_low": cfg.ml.band_low,
        "band_high": cfg.ml.band_high,
        "dataset": dataset,
        "absent_features": list(spec.absent),
        "indicator_cols": list(spec.indicator),
        "ranges": training_ranges(x_tr),
        "reference": reference_values(x_tr),
        "metrics_test": test_reports[selected],
        "cv": cv_summary[selected],
        "sanity": report["sanity"],
    }
    metadata = save_artifact(fitted[selected], metadata, cfg)
    cfg.paths.reference.parent.mkdir(parents=True, exist_ok=True)
    x_tr.to_csv(cfg.paths.reference, index=False)  # public data; used for drift checks
    log_event(log, "model_saved", model=selected, version=metadata["model_version"])
    run_id = archive("ok")
    return TrainResult(selected=selected, metadata=metadata, report=report, run_id=run_id)


def _leak_md(report: dict[str, Any]) -> str:
    s, lk = report["sanity"], report["leakage"]
    sf = lk["single_feature_auc_max"]
    lines = [
        f"- selected-model test ROC-AUC {s['selected_test_auc']:.3f}; best candidate "
        f"{s['max_test_auc']:.3f}; warn level {s['ceiling']}, hard limit {s['abort_limit']}",
        f"- train/test duplicate rows: {lk['duplicates']['n_overlap']} "
        f"({lk['duplicates']['fraction']:.1%} of test)",
        f"- strongest single feature: `{sf['feature']}` alone AUC {sf['auc']:.3f}",
        f"- label-shuffle test AUC (expect ~0.5): mean {lk['label_shuffle']['mean']:.3f}, "
        f"max {lk['label_shuffle']['max']:.3f} over {lk['label_shuffle']['repeats']} shuffles",
    ]
    lines += [f"- {r}" for r in s["reasons"]]
    return "\n".join(lines)


def _cal(report: dict[str, Any]) -> str:
    v = report["calibration"].get("brier_oof_calibrated")
    return "n/a" if v is None else f"{v:.3f}"


def _nested(report: dict[str, Any]) -> str:
    n = report.get("nested_cv_auc")
    return "not computed" if not n else f"ROC-AUC {n['mean']:.3f} ± {n['std']:.3f}"


def _diffs(report: dict[str, Any]) -> str:
    d = report["auc_diff_vs_baseline"]
    if not d:
        return "n/a"
    return "\n".join(
        f"- {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in d.items()
    )


def _fmt(report: dict[str, Any], name: str) -> str:
    t = report["test"][name]
    cells = []
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lo, hi = t["ci95"][m]
        cells.append(f"{t['metrics'][m]:.3f} [{lo:.2f}, {hi:.2f}]")
    cv = report["cv"][name]
    return f"| {name} | " + " | ".join(cells) + f" | {cv['mean']:.3f}±{cv['std']:.3f} |"


def _write_reports(
    cfg: Config,
    report: dict[str, Any],
    y_te: np.ndarray,
    test_proba: dict[str, np.ndarray],
    selected: str,
) -> None:
    out = cfg.paths.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    header = (
        "| model | accuracy | precision | recall | F1 | ROC-AUC | CV ROC-AUC (train) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table = header + "\n".join(_fmt(report, n) for n in report["test"])
    rec = report["selected_at_recall_threshold"]
    md = f"""# Model report (generated by `python -m hcai train`; do not edit by hand)

dataset `{report["dataset"]["name"]}` ({report["dataset"]["rows"]} rows; absent features: {", ".join(report["dataset"]["absent_features"]) or "none"})
sklearn {sklearn.__version__} · seed {cfg.seed} · train n={report["n_train"]} · test n={report["n_test"]}
· positive rate {report["class_balance_positive"]:.2f}

## Held-out test metrics at threshold 0.5 (point [95% bootstrap CI])
{table}

**Selected:** `{selected}`

{report["explanation"]}

## Recall-oriented operating point (threshold chosen on out-of-fold train predictions)
threshold {rec["threshold"]:.3f} (target recall {rec["target_recall"]:.2f}) →
recall {rec["metrics"]["recall"]:.3f}, precision {rec["metrics"]["precision"]:.3f},
confusion {rec["confusion"]}

## Calibration
Test Brier score ({selected}): {report["test"][selected]["brier"]:.3f}. See `calibration.png`.
Out-of-fold Brier on train: raw {report["calibration"]["brier_oof_raw"]:.3f}, sigmoid-calibrated {_cal(report)};
calibration applied to the saved model: **{report["calibration"]["applied"]}**.

## Honest performance of "tune the winner, then score it" (nested CV, train split)
{_nested(report)}

## ROC-AUC difference vs the logistic-regression baseline (test, paired bootstrap 95% CI)
{_diffs(report)}

## Ablations (CV ROC-AUC on train split, default hyper-parameters)
{json.dumps(report["ablations_cv_auc"], indent=2)}

## Sanity and leakage checks: **{report["sanity"]["verdict"].upper()}**
{_leak_md(report)}

## Permutation importance (test, ROC-AUC drop)
""" + "\n".join(
        f"- {f}: {v['mean']:.3f} ± {v['std']:.3f}"
        for f, v in sorted(
            report["permutation_importance_test"].items(), key=lambda kv: -kv[1]["mean"]
        )
    )
    (out / "model_report.md").write_text(md + "\n", encoding="utf-8")
    save_plots(out, y_te, test_proba, selected)
