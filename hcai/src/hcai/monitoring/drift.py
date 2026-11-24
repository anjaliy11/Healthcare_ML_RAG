"""Input drift vs. the training distribution (KS for numerics, chi-square for categoricals).

Operates on a batch the operator supplies (e.g. a CSV export). Raw patient inputs are never
persisted by the app, so drift is computed on demand, not from logs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from hcai.ml.schema import BINARY, CATEGORICAL, FEATURES, NUMERIC

MIN_SAMPLES = 30


def check_drift(
    reference: pd.DataFrame, incoming: pd.DataFrame, alpha: float = 0.01
) -> dict[str, Any]:
    """Return per-feature p-values and the list of drifted features (p < alpha)."""
    if len(incoming) < MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "n": len(incoming),
            "message": f"Need at least {MIN_SAMPLES} rows for a meaningful test.",
            "drifted": [],
            "p_values": {},
        }
    p_values: dict[str, float] = {}
    for col in FEATURES:
        ref, inc = reference[col].dropna(), incoming[col].dropna()
        if ref.empty or inc.empty:
            continue
        if col in NUMERIC:
            p_values[col] = float(stats.ks_2samp(ref, inc).pvalue)
        elif col in BINARY + CATEGORICAL:
            cats = sorted(set(ref) | set(inc))
            table = np.array([[(ref == c).sum() for c in cats], [(inc == c).sum() for c in cats]])
            table = table[:, table.sum(axis=0) > 0]
            p_values[col] = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
    drifted = sorted(c for c, p in p_values.items() if p < alpha)
    out_of_range = {
        c: float(((incoming[c] < reference[c].min()) | (incoming[c] > reference[c].max())).mean())
        for c in NUMERIC
        if c in incoming
    }
    return {
        "status": "drift" if drifted else "ok",
        "n": len(incoming),
        "alpha": alpha,
        "drifted": drifted,
        "p_values": p_values,
        "out_of_training_range_rate": out_of_range,
    }
