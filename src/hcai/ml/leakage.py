"""Direct leakage diagnostics and a split-stability study.

Why this exists: a fixed "AUC > 0.96 means leakage" rule is a symptom check, not a leakage check.
On a ~61-row test set the AUC of a perfectly clean pipeline varies by several points from split
to split (measured, see `hcai stability`), so the fixed rule produced false alarms and aborted a
valid run. Leakage is tested for directly instead:

1. exact train/test duplicate rows (the same patient on both sides of the split);
2. a single feature that alone almost perfectly separates the classes (a target proxy);
3. label-shuffle: fit the baseline on randomly permuted training labels; its test AUC must be
   near 0.5. If it is not, information about the true label reaches the model by a side channel.

A high-but-explained test AUC is a WARNING (kept visible); proven leakage, or an AUC beyond any
plausible range (`auc_abort`), stops the run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from hcai.config import Config
from hcai.ml.features import FeatureSpec, build_pipeline
from hcai.ml.schema import FEATURES


def duplicate_overlap(x_tr: pd.DataFrame, x_te: pd.DataFrame) -> dict[str, Any]:
    """Share of test rows whose full feature vector (NaN == NaN) also occurs in train."""
    tr = set(map(tuple, x_tr.fillna(-9999.0).round(6).to_numpy()))
    hits = sum(tuple(r) in tr for r in x_te.fillna(-9999.0).round(6).to_numpy())
    return {"n_overlap": int(hits), "fraction": hits / max(len(x_te), 1)}


def single_feature_auc(x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Per-feature univariate AUC (orientation-free: max(auc, 1-auc)); NaN rows are skipped."""
    out: dict[str, float] = {}
    for c in FEATURES:
        col = x[c]
        ok = col.notna()
        if ok.sum() < 10 or y[ok].nunique() < 2 or col[ok].nunique() < 2:
            continue
        a = roc_auc_score(y[ok], col[ok])
        out[c] = float(max(a, 1 - a))
    return out


def label_shuffle_auc(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_te: pd.DataFrame,
    y_te: pd.Series,
    cfg: Config,
    spec: FeatureSpec,
    repeats: int,
) -> dict[str, Any]:
    """Baseline fit on permuted TRAIN labels, scored on the real TEST labels: expect ~0.5."""
    rng = np.random.default_rng(cfg.seed + 7)
    aucs = []
    for _ in range(repeats):
        y_perm = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
        pipe = build_pipeline("logreg", cfg, spec=spec).fit(x_tr, y_perm)
        aucs.append(float(roc_auc_score(y_te, pipe.predict_proba(x_te)[:, 1])))
    return {"mean": float(np.mean(aucs)), "max": float(np.max(aucs)), "repeats": repeats}


def run_diagnostics(
    x_tr: pd.DataFrame,
    x_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    cfg: Config,
    spec: FeatureSpec,
) -> dict[str, Any]:
    ml = cfg.ml
    dup = duplicate_overlap(x_tr, x_te)
    sf = single_feature_auc(x_tr, y_tr)
    top = max(sf.items(), key=lambda kv: kv[1]) if sf else ("n/a", 0.5)
    shuffle = label_shuffle_auc(x_tr, y_tr, x_te, y_te, cfg, spec, ml.leak_shuffle_repeats)
    problems: list[str] = []
    if dup["fraction"] > ml.leak_duplicate_frac_max:
        problems.append(
            f"{dup['n_overlap']} test rows ({dup['fraction']:.0%}) also appear in train "
            f"(limit {ml.leak_duplicate_frac_max:.0%})"
        )
    if top[1] >= ml.leak_single_feature_max:
        problems.append(f"feature '{top[0]}' alone separates the classes (AUC {top[1]:.3f})")
    if shuffle["mean"] > ml.leak_shuffle_max:
        problems.append(
            f"label-shuffle test AUC {shuffle['mean']:.3f} > {ml.leak_shuffle_max} "
            "(a model trained on random labels should score ~0.5)"
        )
    return {
        "duplicates": dup,
        "single_feature_auc_max": {"feature": top[0], "auc": top[1]},
        "single_feature_auc": sf,
        "label_shuffle": shuffle,
        "problems": problems,
        "clean": not problems,
    }


def judge(
    diag: dict[str, Any], selected_auc: float, max_auc: float, ci: tuple[float, float], cfg: Config
) -> dict[str, Any]:
    """Combine diagnostics and the AUC level into pass / warn / fail with human reasons."""
    ml = cfg.ml
    reasons: list[str] = list(diag["problems"])
    verdict = "pass"
    if selected_auc > ml.auc_abort:
        reasons.append(f"selected test ROC-AUC {selected_auc:.3f} > hard limit {ml.auc_abort}")
    if reasons:
        verdict = "fail"
    elif max(selected_auc, max_auc) > ml.auc_ceiling:
        verdict = "warn"
        reasons.append(
            f"test ROC-AUC {max(selected_auc, max_auc):.3f} is above the usual ceiling "
            f"{ml.auc_ceiling}, but every direct leakage check is clean "
            f"(selected-model 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]). With a small test set this is "
            "within split noise; run `python -m hcai stability` to measure how often that happens."
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "ceiling": ml.auc_ceiling,
        "abort_limit": ml.auc_abort,
        "floor": ml.auc_floor,
        "max_test_auc": max_auc,
        "selected_test_auc": selected_auc,
        "ceiling_ok": max(selected_auc, max_auc) <= ml.auc_ceiling,
        "floor_ok": selected_auc >= ml.auc_floor,
    }


def split_stability(
    cfg: Config,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    models: tuple[str, ...] = ("logreg", "random_forest", "hist_gb"),
) -> dict[str, Any]:
    """Distribution of held-out ROC-AUC over many random stratified splits (default parameters).

    This answers "is 0.96 on this split unusual?" with data instead of a rule of thumb."""
    spec = FeatureSpec.from_data(x)
    rows = []
    for i in range(n_splits):
        xt, xe, yt, ye = train_test_split(
            x, y, test_size=cfg.ml.test_size, stratify=y, random_state=cfg.seed + 1000 + i
        )
        rows.append(
            {
                m: float(
                    roc_auc_score(
                        ye, build_pipeline(m, cfg, spec=spec).fit(xt, yt).predict_proba(xe)[:, 1]
                    )
                )
                for m in models
            }
        )
    df = pd.DataFrame(rows)
    ceiling = cfg.ml.auc_ceiling
    per_model = {
        m: {
            "mean": float(df[m].mean()),
            "std": float(df[m].std()),
            "p05": float(df[m].quantile(0.05)),
            "p95": float(df[m].quantile(0.95)),
            "min": float(df[m].min()),
            "max": float(df[m].max()),
            "share_above_ceiling": float((df[m] > ceiling).mean()),
        }
        for m in models
    }
    return {
        "n_splits": n_splits,
        "n_test": round(len(x) * cfg.ml.test_size),
        "ceiling": ceiling,
        "per_model": per_model,
        "share_any_model_above_ceiling": float((df.max(axis=1) > ceiling).mean()),
        "share_any_model_above_abort": float((df.max(axis=1) > cfg.ml.auc_abort).mean()),
    }


def stability_markdown(res: dict[str, Any], dataset: str) -> str:
    lines = [
        f"# Split-stability study ({dataset})\n",
        f"{res['n_splits']} random stratified splits, test n≈{res['n_test']}, default "
        "hyper-parameters, same leak-free pipeline.\n",
        "| model | mean AUC | std | 5th pct | 95th pct | max | share of splits > ceiling |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {m} | {s['mean']:.3f} | {s['std']:.3f} | {s['p05']:.3f} | {s['p95']:.3f} "
        f"| {s['max']:.3f} | {s['share_above_ceiling']:.0%} |"
        for m, s in res["per_model"].items()
    ]
    lines += [
        "",
        f"Share of splits where the best of these models exceeds the {res['ceiling']} ceiling: "
        f"**{res['share_any_model_above_ceiling']:.0%}**; above the hard limit: "
        f"{res['share_any_model_above_abort']:.0%}.",
        "",
        "Reading: if clean splits regularly exceed the ceiling, a single test AUC above it is not "
        "evidence of leakage by itself. That is why the gate relies on direct diagnostics.",
    ]
    return "\n".join(lines) + "\n"
