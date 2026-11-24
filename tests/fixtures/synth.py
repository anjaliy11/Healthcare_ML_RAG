"""SYNTHETIC patient data for tests only. NOT real data; never use it to report metrics.

It mimics the Cleveland schema (13 features, 303 rows, ~46% positive, 4 NaN `ca`, 2 NaN `thal`)
so tests can exercise the pipeline offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcai.ml.schema import FEATURES, TARGET


def make_synthetic(n: int = 303, seed: int = 0, signal: float = 1.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n).astype(float)
    sex = rng.integers(0, 2, n).astype(float)
    cp = rng.choice([1, 2, 3, 4], n, p=[0.08, 0.17, 0.28, 0.47]).astype(float)
    trestbps = rng.normal(131, 17, n).clip(94, 200).round()
    chol = rng.normal(246, 52, n).clip(126, 564).round()
    fbs = (rng.random(n) < 0.15).astype(float)
    restecg = rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49]).astype(float)
    thalach = (200 - 0.7 * age + rng.normal(0, 18, n)).clip(71, 202).round()
    exang = (rng.random(n) < 0.33).astype(float)
    oldpeak = rng.gamma(1.2, 0.9, n).clip(0, 6.2).round(1)
    slope = rng.choice([1, 2, 3], n, p=[0.47, 0.46, 0.07]).astype(float)
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]).astype(float)
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39]).astype(float)
    z = (
        0.04 * (age - 54) + 0.7 * (cp == 4) + 0.5 * sex - 0.03 * (thalach - 150)
        + 0.5 * oldpeak + 0.8 * ca + 0.9 * (thal == 7) + 0.6 * exang
    )  # fmt: skip
    z = signal * (z - z.mean()) / z.std() + rng.normal(0, 1.0, n)
    num = (z > np.quantile(z, 0.54)).astype(int) * rng.integers(1, 5, n)
    df = pd.DataFrame(
        dict(
            zip(
                FEATURES,
                [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ],
                strict=True,
            )
        )
    )
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    df[TARGET] = num
    return df
