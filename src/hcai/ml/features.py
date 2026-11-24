"""One sklearn Pipeline per model: imputation, encoding, scaling and the model are fit together
on train only, so leakage is impossible by construction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hcai.config import Config
from hcai.ml.schema import BINARY, CATEGORICAL, MISSING_INDICATOR_COLS, NUMERIC

MODEL_NAMES = ("logreg", "random_forest", "hist_gb")


def build_preprocessor(scale: bool, use_indicator: bool = True) -> ColumnTransformer:
    """Median-impute numerics, mode-impute + one-hot categoricals, flag imputed ca/thal."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            CATEGORICAL,
        ),
    ]
    if use_indicator:
        transformers.append(("missing", MissingIndicator(features="all"), MISSING_INDICATOR_COLS))
    return ColumnTransformer(transformers)


def build_pipeline(name: str, cfg: Config, use_indicator: bool = True) -> Pipeline:
    """Factory: scaling only for the linear model (trees do not need it)."""
    seed = cfg.seed
    if name == "logreg":
        model = LogisticRegression(max_iter=2000, random_state=seed)
        pre = build_preprocessor(scale=True, use_indicator=use_indicator)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=cfg.ml.rf_trees, random_state=seed, n_jobs=1)
        pre = build_preprocessor(scale=False, use_indicator=use_indicator)
    elif name == "hist_gb":
        model = HistGradientBoostingClassifier(max_iter=cfg.ml.hgb_max_iter, random_state=seed)
        pre = build_preprocessor(scale=False, use_indicator=use_indicator)
    elif name == "hist_gb_native_nan":
        # Cross-check for imputation: HGB consumes NaN directly, no imputer.
        model = HistGradientBoostingClassifier(max_iter=cfg.ml.hgb_max_iter, random_state=seed)
        return Pipeline([("model", model)])
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    return Pipeline([("pre", pre), ("model", model)])
