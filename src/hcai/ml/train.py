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
from sklearn.inspection import permutation_importance
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
from hcai.ml.data import load_dataset
from hcai.ml.evaluate import (
    choose_recall_threshold,
    explain_selection,
    full_report,
    save_plots,
    select_model,
)
from hcai.ml.features import MODEL_NAMES, build_pipeline
from hcai.ml.registry import save_artifact
from hcai.ml.schema import BINARY, CATEGORICAL, FEATURES, NUMERIC

log = get_logger("ml.train")


@dataclass
class TrainResult:
    selected: str
    metadata: dict[str, Any]
    report: dict[str, Any]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def training_ranges(x: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {c: {"min": float(x[c].min()), "max": float(x[c].max())} for c in FEATURES}


def reference_values(x: pd.DataFrame) -> dict[str, float]:
    """Typical value per feature (median / mode), used for per-patient factor attribution."""
    ref: dict[str, float] = {}
    for c in NUMERIC:
        ref[c] = float(x[c].median())
    for c in BINARY + CATEGORICAL:
        ref[c] = float(x[c].mode(dropna=True).iloc[0])
    return ref


def _ablations(x: pd.DataFrame, y: pd.Series, cfg: Config, cv: Any) -> dict[str, float]:
    """CV-only comparison on the training split: does the missing-indicator / imputer matter?"""

    def score(pipe: Any) -> float:
        return float(cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc").mean())

    return {
        "logreg_with_indicator": score(build_pipeline("logreg", cfg, True)),
        "logreg_without_indicator": score(build_pipeline("logreg", cfg, False)),
        "hist_gb_imputed": score(build_pipeline("hist_gb", cfg)),
        "hist_gb_native_nan": score(build_pipeline("hist_gb_native_nan", cfg)),
    }


def train(
    cfg: Config,
    source_file: Path | None = None,
    data: tuple[pd.DataFrame, pd.Series, str] | None = None,
) -> TrainResult:
    x, y, data_sha = data if data is not None else load_dataset(cfg, source_file)
    seed = cfg.seed
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.ml.test_size, stratify=y, random_state=seed
    )
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.ml.cv_splits, n_repeats=cfg.ml.cv_repeats, random_state=seed
    )

    fitted: dict[str, Any] = {}
    cv_summary: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        grid = {f"model__{k}": v for k, v in cfg.ml.grids[name].items()}
        gs = GridSearchCV(build_pipeline(name, cfg), grid, scoring="roc_auc", cv=cv, refit=True)
        gs.fit(x_tr, y_tr)
        i = int(gs.best_index_)
        fitted[name] = gs.best_estimator_
        cv_summary[name] = {
            "mean": float(gs.cv_results_["mean_test_score"][i]),
            "std": float(gs.cv_results_["std_test_score"][i]),
            "best_params": gs.best_params_,
        }
        log_event(log, "cv_done", model=name, cv_auc=round(cv_summary[name]["mean"], 4))

    selected = select_model({k: v["mean"] for k, v in cv_summary.items()}, cfg.ml.selection_margin)

    # Operating point from out-of-fold predictions on TRAIN only.
    oof = cross_val_predict(
        clone(fitted[selected]),
        x_tr,
        y_tr,
        cv=StratifiedKFold(cfg.ml.cv_splits, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    recall_threshold = choose_recall_threshold(y_tr.to_numpy(), oof, cfg.ml.target_recall)

    # Single test evaluation.
    y_te_arr = y_te.to_numpy()
    test_proba = {n: m.predict_proba(x_te)[:, 1] for n, m in fitted.items()}
    n_boot, boot_seed = cfg.ml.bootstrap_resamples, seed
    test_reports = {
        n: full_report(y_te_arr, p, 0.5, n_boot, boot_seed) for n, p in test_proba.items()
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
        "ablations_cv_auc": _ablations(x_tr, y_tr, cfg, cv),
        "permutation_importance_test": importance,
        "explanation": explain_selection(
            selected, cv_summary, test_reports, cfg.ml.selection_margin
        ),
    }

    # Sanity gate: write evidence first, then refuse to ship a suspicious model.
    best_auc = max(r["metrics"]["roc_auc"] for r in test_reports.values())
    report["sanity"] = {
        "max_test_auc": best_auc,
        "ceiling": cfg.ml.auc_ceiling,
        "floor": cfg.ml.auc_floor,
        "ceiling_ok": best_auc <= cfg.ml.auc_ceiling,
        "floor_ok": test_reports[selected]["metrics"]["roc_auc"] >= cfg.ml.auc_floor,
    }
    _write_reports(cfg, report, y_te_arr, test_proba, selected)
    if not report["sanity"]["ceiling_ok"]:
        raise LeakageSuspectedError(
            f"Test ROC-AUC {best_auc:.3f} exceeds the {cfg.ml.auc_ceiling} sanity ceiling "
            "(Cleveland is typically 0.88-0.93). Suspect leakage; model NOT saved. "
            f"See {cfg.paths.reports_dir / 'metrics.json'}."
        )

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
    return TrainResult(selected=selected, metadata=metadata, report=report)


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
Brier score ({selected}): {report["test"][selected]["brier"]:.3f}. See `calibration.png`.

## Ablations (CV ROC-AUC on train split, default hyper-parameters)
{json.dumps(report["ablations_cv_auc"], indent=2)}

## Sanity
{json.dumps(report["sanity"], indent=2)}

## Permutation importance (test, ROC-AUC drop)
""" + "\n".join(
        f"- {f}: {v['mean']:.3f} ± {v['std']:.3f}"
        for f, v in sorted(
            report["permutation_importance_test"].items(), key=lambda kv: -kv[1]["mean"]
        )
    )
    (out / "model_report.md").write_text(md + "\n", encoding="utf-8")
    save_plots(out, y_te, test_proba, selected)
