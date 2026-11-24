"""Dataset registry. Cleveland stays the primary, default dataset (settled decision); the Kaggle
"Heart Failure Prediction Dataset" is a SECONDARY dataset used for a cross-dataset comparison.

Every adapter maps its source onto the canonical 13-column Cleveland schema (same names, same
codes) so one pipeline, one predictor and one report format serve both. Columns a source does not
have are left entirely NaN and dropped inside the pipeline (`FeatureSpec.absent`)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hcai.config import Config, Paths
from hcai.errors import DataUnavailableError, DataValidationError
from hcai.logging import get_logger, log_event
from hcai.ml.data import hash_frame, load_dataset
from hcai.ml.schema import FEATURES, TARGET
from hcai.textio import read_text

log = get_logger("ml.datasets")

DEFAULT_DATASET = "cleveland"
KAGGLE = "kaggle_heart_failure"
KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
KAGGLE_CACHE = "kaggle_heart_failure.csv"
KAGGLE_ROWS = 918


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    title: str
    source: str
    notes: str


INFO = {
    DEFAULT_DATASET: DatasetInfo(
        DEFAULT_DATASET,
        "UCI Heart Disease, Cleveland subset (303 rows)",
        "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "Primary dataset. Target is num>0. NaN in ca (4) and thal (2).",
    ),
    KAGGLE: DatasetInfo(
        KAGGLE,
        "Kaggle Heart Failure Prediction Dataset (918 rows; combines 5 heart-disease datasets)",
        KAGGLE_URL,
        "Secondary dataset. Has no ca/thal (absent, dropped in the pipeline). Cholesterol and "
        "resting BP recorded as 0 are treated as missing. Cleveland rows are part of this "
        "combined data, so the two results are not independent evidence.",
    ),
}
DATASETS = tuple(INFO)

_KAGGLE_REQUIRED = {
    "Age": "age", "Sex": "sex", "RestingBP": "trestbps", "Cholesterol": "chol",
    "FastingBS": "fbs", "RestingECG": "restecg", "MaxHR": "thalach",
    "ExerciseAngina": "exang", "Oldpeak": "oldpeak", "ST_Slope": "slope",
    "HeartDisease": TARGET,
}  # fmt: skip
_CP_COLUMN_NAMES = ("ChestPainType", "ChestPain")  # the column is named differently in mirrors
_SEX = {"M": 1, "F": 0}
_CP = {"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4}  # same 1-4 coding as Cleveland cp
_ECG = {"Normal": 0, "ST": 1, "LVH": 2}  # same 0-2 coding as Cleveland restecg
_ANGINA = {"N": 0, "Y": 1}
_SLOPE = {"Up": 1, "Flat": 2, "Down": 3}  # same 1-3 coding as Cleveland slope


def _map(df: pd.DataFrame, col: str, mapping: dict[str, int]) -> pd.Series:
    raw = df[col].astype(str).str.strip()
    bad = sorted(set(raw) - set(mapping))
    if bad:
        raise DataValidationError(
            f"Column '{col}' has unexpected values {bad[:5]}; expected {sorted(mapping)}. "
            "This does not look like the Kaggle Heart Failure Prediction file."
        )
    return raw.map(mapping).astype(float)


def adapt_kaggle_heart(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Kaggle schema -> canonical features (+ binary target). Validates, never guesses."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cp_col = next((c for c in _CP_COLUMN_NAMES if c in df.columns), None)
    missing = [c for c in _KAGGLE_REQUIRED if c not in df.columns] + (
        [] if cp_col else ["ChestPainType"]
    )
    if missing:
        raise DataValidationError(
            f"Not the Kaggle Heart Failure file: missing columns {missing}. "
            f"Expected Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, "
            f"MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease (from {KAGGLE_URL})."
        )
    y = pd.to_numeric(df["HeartDisease"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise DataValidationError("HeartDisease must be 0/1 with no missing values.")
    num = {k: pd.to_numeric(df[k], errors="coerce") for k in ("Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak")}  # fmt: skip
    if any(v.isna().any() for v in num.values()):
        bad = [k for k, v in num.items() if v.isna().any()]
        raise DataValidationError(f"Non-numeric or missing values in: {bad}.")
    if not num["FastingBS"].isin([0, 1]).all():
        raise DataValidationError("FastingBS must be 0/1.")
    x = pd.DataFrame(
        {
            "age": num["Age"],
            "sex": _map(df, "Sex", _SEX),
            "cp": _map(df, cp_col or "", _CP),
            # 0 is a data-entry placeholder here (a living patient has neither 0 mmHg nor 0 mg/dl)
            "trestbps": num["RestingBP"].replace(0, np.nan),
            "chol": num["Cholesterol"].replace(0, np.nan),
            "fbs": num["FastingBS"],
            "restecg": _map(df, "RestingECG", _ECG),
            "thalach": num["MaxHR"],
            "exang": _map(df, "ExerciseAngina", _ANGINA),
            "oldpeak": num["Oldpeak"],
            "slope": _map(df, "ST_Slope", _SLOPE),
            "ca": np.nan,
            "thal": np.nan,
        }
    )[FEATURES].astype(float)
    return x, y.astype(int)


def load_named(
    cfg: Config, dataset: str = DEFAULT_DATASET, source_file: Path | None = None
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Features, binary target and data hash for a registered dataset."""
    if dataset == DEFAULT_DATASET:
        return load_dataset(cfg, source_file)
    if dataset != KAGGLE:
        raise DataUnavailableError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}.")
    cache = cfg.paths.data_raw / KAGGLE_CACHE
    src = source_file or (cache if cache.is_file() else None)
    if src is None:
        raise DataUnavailableError(
            "Kaggle dataset file not found. Download `heart.csv` from "
            f"{KAGGLE_URL} (free Kaggle account, 'Download'), then run:\n"
            f"  python -m hcai train --dataset {KAGGLE} --source-file <path>\\heart.csv"
        )
    if not Path(src).is_file():
        raise DataUnavailableError(f"Source file not found: {src}")
    from io import StringIO

    raw = pd.read_csv(StringIO(read_text(Path(src))))
    x, y = adapt_kaggle_heart(raw)
    if len(x) != KAGGLE_ROWS:
        log_event(log, "kaggle_row_count_differs", expected=KAGGLE_ROWS, found=len(x))
    if source_file is not None and Path(src).resolve() != cache.resolve():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cache)
    log_event(log, "dataset_loaded", dataset=dataset, rows=len(x), positives=int(y.sum()))
    return x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1))


def config_for(cfg: Config, dataset: str) -> Config:
    """Non-default datasets write their model/reports under artifacts/datasets/<name>/ so they
    never overwrite the primary (Cleveland) model that the app serves."""
    if dataset == DEFAULT_DATASET:
        return cfg
    base = cfg.paths.artifacts / "datasets" / dataset
    paths: Paths = cfg.paths.model_copy(
        update={
            "model": base / "model.joblib",
            "model_meta": base / "model_metadata.json",
            "reference": base / "train_reference.csv",
            "reports_dir": base / "reports",
        }
    )
    return cfg.model_copy(update={"paths": paths})
