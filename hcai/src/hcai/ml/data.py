"""Load, validate and binarise the UCI Cleveland data. Cached locally after the first fetch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from hcai.config import Config
from hcai.errors import DataUnavailableError, DataValidationError
from hcai.logging import get_logger, log_event
from hcai.ml.schema import FEATURES, TARGET

log = get_logger("ml.data")
CACHE_NAME = "heart_cleveland.csv"
UCI_COLUMNS = [*FEATURES, TARGET]


def read_processed_cleveland(path: Path) -> pd.DataFrame:
    """Parse UCI's raw `processed.cleveland.data` ('?' marks missing, no header)."""
    return pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?")


def _fetch_ucimlrepo(ucirepo_id: int) -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=ucirepo_id)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    except Exception as exc:  # network, schema, library errors all mean the same to the user
        raise DataUnavailableError(
            f"Could not download the UCI dataset ({type(exc).__name__}). Check your internet "
            "connection, or download `processed.cleveland.data` from "
            "https://archive.ics.uci.edu/dataset/45/heart+disease and run: "
            "python -m hcai data --source-file <path>"
        ) from exc
    return df


def hash_frame(df: pd.DataFrame) -> str:
    """Deterministic sha256 of the canonical CSV bytes (stored in the model artifact)."""
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def load_raw(cfg: Config, source_file: Path | None = None) -> pd.DataFrame:
    """Return the raw frame (features + `num`), using the cache when present."""
    cache = cfg.paths.data_raw / CACHE_NAME
    if source_file is not None:
        if not source_file.is_file():
            raise DataUnavailableError(f"Source file not found: {source_file}")
        df = (
            read_processed_cleveland(source_file)
            if source_file.suffix == ".data"
            else pd.read_csv(source_file, na_values="?")
        )
    elif cache.is_file():
        df = pd.read_csv(cache)
        return df
    else:
        df = _fetch_ucimlrepo(cfg.ml.ucirepo_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def validate_and_split_target(
    df: pd.DataFrame, expected_rows: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Check schema, coerce to float, collapse `num` (0-4) to presence (num > 0)."""
    missing_cols = [c for c in UCI_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Dataset is missing columns: {missing_cols}")
    df = df[UCI_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df[TARGET].isna().any():
        raise DataValidationError("Target column `num` has missing/non-numeric values.")
    if not df[TARGET].isin([0, 1, 2, 3, 4]).all():
        raise DataValidationError("Target `num` must be in 0..4.")
    if expected_rows is not None and len(df) != expected_rows:
        raise DataValidationError(
            f"Expected {expected_rows} rows (Cleveland subset) but found {len(df)}. "
            "You may have the combined 4-site file; use processed.cleveland.data."
        )
    x = df[FEATURES].astype(float)
    y = (df[TARGET] > 0).astype(int)
    log_event(
        log,
        "dataset_loaded",
        rows=len(df),
        positives=int(y.sum()),
        missing_cells={c: int(n) for c, n in x.isna().sum().items() if n},
    )
    return x, y


def load_dataset(
    cfg: Config, source_file: Path | None = None
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Features, binary target and data hash."""
    raw = load_raw(cfg, source_file)
    x, y = validate_and_split_target(raw, cfg.ml.expected_rows)
    return x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1))
