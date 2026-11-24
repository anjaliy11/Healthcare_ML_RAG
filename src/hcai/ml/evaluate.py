"""Metrics, bootstrap confidence intervals, threshold choice, plots and the plain-English
model-selection explanation. Everything here is pure except `save_plots`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs. Resamples with a single class are skipped (AUC undefined)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], proba[idx], threshold)
        for k in METRIC_NAMES:
            draws[k].append(m[k])
    return {
        k: (
            float(np.percentile(v, 100 * alpha / 2)),
            float(np.percentile(v, 100 * (1 - alpha / 2))),
        )
        for k, v in draws.items()
        if v
    }


def full_report(
    y: np.ndarray, proba: np.ndarray, threshold: float, n_resamples: int, seed: int
) -> dict[str, Any]:
    point = compute_metrics(y, proba, threshold)
    ci = bootstrap_ci(y, proba, threshold, n_resamples, seed)
    tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "metrics": point,
        "ci95": {k: list(v) for k, v in ci.items()},
        "brier": float(brier_score_loss(y, proba)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": len(y),
    }


def choose_recall_threshold(y: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Largest threshold whose recall >= target. Use out-of-fold predictions, never test."""
    positives = max(int(y.sum()), 1)
    for thr in np.unique(proba)[::-1]:
        recall = float(((proba >= thr) & (y == 1)).sum()) / positives
        if recall >= target_recall:
            return float(thr)
    return float(proba.min())


def select_model(cv_means: dict[str, float], margin: float, baseline: str = "logreg") -> str:
    """Highest CV ROC-AUC, but keep the interpretable baseline unless beaten by > margin."""
    best = max(cv_means, key=lambda k: cv_means[k])
    if baseline in cv_means and cv_means[best] - cv_means[baseline] <= margin:
        return baseline
    return best


def _overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def explain_selection(
    selected: str,
    cv: dict[str, dict[str, float]],
    test: dict[str, dict[str, Any]],
    margin: float,
) -> str:
    """Generate the 'which model is best and why' text from the numbers (no hand-typed claims)."""
    ranked = sorted(cv, key=lambda k: -cv[k]["mean"])
    lines = [
        "Cross-validated ROC-AUC (train split only): "
        + ", ".join(f"{k} {cv[k]['mean']:.3f}±{cv[k]['std']:.3f}" for k in ranked)
        + ".",
    ]
    top = ranked[0]
    if selected == "logreg" and top != "logreg":
        lines.append(
            f"{top} scored highest in CV but by no more than the {margin:.2f} selection margin over "
            "logistic regression, so the interpretable, well-calibrated baseline was kept."
        )
    elif selected == top:
        lines.append(f"{selected} had the highest CV ROC-AUC and beat the baseline by the margin.")
    sel_auc_ci = test[selected]["ci95"]["roc_auc"]
    indistinct = [
        k for k in test if k != selected and _overlap(sel_auc_ci, test[k]["ci95"]["roc_auc"])
    ]
    if indistinct:
        lines.append(
            "On the held-out test set the 95% bootstrap CIs for ROC-AUC of "
            + ", ".join(indistinct)
            + f" overlap with {selected}: the test set is too small to call a real difference."
        )
    else:
        lines.append("Test ROC-AUC CIs do not overlap with the other models.")
    return " ".join(lines)


def save_plots(
    reports_dir: Path,
    y_test: np.ndarray,
    test_proba: dict[str, np.ndarray],
    selected: str,
) -> list[Path]:
    """ROC curves (all models) and a reliability diagram (selected model)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in test_proba.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    paths.append(reports_dir / "roc.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(
        y_test, test_proba[selected], n_bins=5, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label=selected)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(
        xlabel="Mean predicted probability", ylabel="Observed fraction", title="Calibration (test)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(reports_dir / "calibration.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths
