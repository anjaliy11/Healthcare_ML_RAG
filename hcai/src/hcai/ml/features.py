"""One sklearn Pipeline per model: imputation, encoding, scaling and the model are fit together
on train only, so leakage is impossible by construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from hcai.config import Config
from hcai.ml.schema import BINARY, CATEGORICAL, FEATURES, MISSING_INDICATOR_COLS, NUMERIC

MODEL_NAMES = ("logreg", "logreg_fe", "svm_rbf", "random_forest", "hist_gb", "voting")
ENGINEERED = ["hr_pct_max", "oldpeak_x_exang"]


@dataclass(frozen=True)
class FeatureSpec:
    """Which of the 13 canonical columns a dataset really provides.

    `absent`: columns that are entirely missing (e.g. `ca`, `thal` in the Kaggle Heart Failure
    data). They are dropped inside the pipeline, so inference still takes the same 13-column
    frame. `indicator`: columns that get a was-missing flag (ca/thal by default, plus any other
    column that has NaN in the training data). Derived from TRAIN data structure only."""

    absent: tuple[str, ...] = ()
    indicator: tuple[str, ...] = tuple(MISSING_INDICATOR_COLS)

    @classmethod
    def from_data(cls, x: pd.DataFrame) -> FeatureSpec:
        absent = tuple(c for c in FEATURES if x[c].isna().all())
        with_nan = [c for c in FEATURES if x[c].isna().any()]
        wanted = [c for c in FEATURES if c in MISSING_INDICATOR_COLS or c in with_nan]
        return cls(absent, tuple(c for c in wanted if c not in absent))

    def keep(self, cols: list[str]) -> list[str]:
        return [c for c in cols if c not in self.absent]


class AddClinicalFeatures(BaseEstimator, TransformerMixin):
    """Stateless, NaN-safe domain features (nothing is learned, so nothing can leak).

    hr_pct_max: achieved max heart rate as a share of the age-predicted maximum (220 - age);
    chronotropic response matters more than the raw rate. oldpeak_x_exang: ST depression that
    occurs together with exercise-induced angina, a classic joint warning sign."""

    def fit(self, X: pd.DataFrame, y: object = None) -> AddClinicalFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["hr_pct_max"] = out["thalach"] / (220.0 - out["age"]).clip(lower=1.0)
        out["oldpeak_x_exang"] = out["oldpeak"] * out["exang"]
        return out


def build_preprocessor(
    scale: bool,
    use_indicator: bool = True,
    extra_numeric: list[str] | None = None,
    spec: FeatureSpec | None = None,
) -> ColumnTransformer:
    """Median-impute numerics, mode-impute + one-hot categoricals, flag imputed columns."""
    spec = spec or FeatureSpec()
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", Pipeline(numeric_steps), spec.keep(NUMERIC) + (extra_numeric or [])),
        ("bin", SimpleImputer(strategy="most_frequent"), spec.keep(BINARY)),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            spec.keep(CATEGORICAL),
        ),
    ]
    if use_indicator and spec.indicator:
        transformers.append(("missing", MissingIndicator(features="all"), list(spec.indicator)))
    return ColumnTransformer(transformers)


def build_pipeline(
    name: str, cfg: Config, use_indicator: bool = True, spec: FeatureSpec | None = None
) -> Pipeline:
    """Factory: scaling only for the linear model (trees do not need it)."""
    seed = cfg.seed
    if name == "logreg":
        model = LogisticRegression(max_iter=2000, random_state=seed)
        pre = build_preprocessor(scale=True, use_indicator=use_indicator, spec=spec)
    elif name == "logreg_fe":
        model = LogisticRegression(max_iter=2000, random_state=seed)
        pre = build_preprocessor(
            scale=True, use_indicator=use_indicator, extra_numeric=ENGINEERED, spec=spec
        )
        return Pipeline([("fe", AddClinicalFeatures()), ("pre", pre), ("model", model)])
    elif name == "svm_rbf":
        # sklearn 1.9 deprecates SVC(probability=True); this is the recommended replacement.
        model = CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), ensemble=False, cv=5)
        pre = build_preprocessor(scale=True, use_indicator=use_indicator, spec=spec)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=cfg.ml.rf_trees, random_state=seed, n_jobs=1)
        pre = build_preprocessor(scale=False, use_indicator=use_indicator, spec=spec)
    elif name == "hist_gb":
        model = HistGradientBoostingClassifier(max_iter=cfg.ml.hgb_max_iter, random_state=seed)
        pre = build_preprocessor(scale=False, use_indicator=use_indicator, spec=spec)
    elif name == "hist_gb_native_nan":
        # Cross-check for imputation: HGB consumes NaN directly, no imputer.
        model = HistGradientBoostingClassifier(max_iter=cfg.ml.hgb_max_iter, random_state=seed)
        # HGB rejects all-NaN columns, so only the columns the dataset really has are passed on.
        keep = ColumnTransformer([("keep", "passthrough", (spec or FeatureSpec()).keep(FEATURES))])
        return Pipeline([("pre", keep), ("model", model)])
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    return Pipeline([("pre", pre), ("model", model)])


def build_voting(members: dict[str, Pipeline]) -> VotingClassifier:
    """Soft-vote over already-tuned member pipelines (each keeps its own preprocessing)."""
    return VotingClassifier(list(members.items()), voting="soft")
