"""Typed config loaded from one YAML file; paths resolved against the project root."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from hcai.errors import ConfigError

DEFAULT_CONFIG_NAME = "configs/default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Strict):
    data_raw: Path
    docs_dir: Path
    artifacts: Path
    model: Path
    model_meta: Path
    reference: Path
    index_dir: Path
    reports_dir: Path
    metrics_log: Path


class MLConfig(_Strict):
    ucirepo_id: int
    expected_rows: int | None
    test_size: float
    cv_splits: int
    cv_repeats: int
    bootstrap_resamples: int
    selection_margin: float
    target_recall: float
    band_low: float
    band_high: float
    auc_floor: float
    auc_ceiling: float
    rf_trees: int
    hgb_max_iter: int
    permutation_repeats: int
    top_factors: int
    grids: dict[str, dict[str, list[float | int | None]]]


class RagConfig(_Strict):
    embedding_model: str
    query_prefix: str
    chunk_target_words: int
    chunk_overlap_words: int
    chunk_max_words: int
    top_k: int
    score_threshold: float
    offline_score_threshold: float
    max_query_chars: int
    extractive_sentences: int


class LLMConfig(_Strict):
    provider: str
    model: str
    fallback_models: list[str]
    max_tokens: int
    temperature: float
    timeout_s: float
    max_retries: int
    backoff_s: float


class Config(_Strict):
    seed: int
    paths: Paths
    ml: MLConfig
    rag: RagConfig
    llm: LLMConfig


def project_root() -> Path:
    """Repo root = parent of `src/` (works for editable installs)."""
    return Path(__file__).resolve().parents[2]


def load_dotenv_file(path: Path) -> None:
    """Minimal .env reader (no extra dependency). Existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    """Load YAML config; relative paths are resolved against `root`."""
    root = root or project_root()
    cfg_path = Path(path or os.environ.get("HCAI_CONFIG") or root / DEFAULT_CONFIG_NAME)
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read config {cfg_path}: {exc}") from exc
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {cfg_path}: {exc}") from exc
    resolved = {k: (root / v if not v.is_absolute() else v) for k, v in cfg.paths}
    cfg = cfg.model_copy(update={"paths": Paths(**resolved)})
    if not os.environ.get("HCAI_NO_DOTENV"):  # tests set this so a real .env never leaks in
        load_dotenv_file(root / ".env")
    return cfg
