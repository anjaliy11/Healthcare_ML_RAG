"""Save/load the model artifact with version, schema and integrity checks.
joblib is pickle: only load artifacts you trained yourself (see README)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from hcai.config import Config
from hcai.errors import ArtifactIncompatibleError, ArtifactMissingError
from hcai.ml.schema import FEATURES


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_artifact(pipeline: Pipeline, metadata: dict[str, Any], cfg: Config) -> dict[str, Any]:
    cfg.paths.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.paths.model)
    sha = file_sha256(cfg.paths.model)
    metadata = {
        **metadata,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "model_sha256": sha,
        "model_version": f"{metadata['model_name']}-{sha[:8]}",
    }
    cfg.paths.model_meta.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def load_artifact(cfg: Config) -> tuple[Pipeline, dict[str, Any]]:
    """Load and verify. Fails loudly, with a remediation message, on any mismatch."""
    if not cfg.paths.model.is_file() or not cfg.paths.model_meta.is_file():
        raise ArtifactMissingError(
            "No trained model found. Run `python -m hcai train` first "
            f"(expected {cfg.paths.model})."
        )
    meta: dict[str, Any] = json.loads(cfg.paths.model_meta.read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        raise ArtifactIncompatibleError(
            f"Model was trained with scikit-learn {meta.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed. Retrain with `python -m hcai train` "
            "or install the matching version."
        )
    if meta.get("features") != FEATURES:
        raise ArtifactIncompatibleError(
            "Model feature schema differs from the code. Retrain with `python -m hcai train`."
        )
    if file_sha256(cfg.paths.model) != meta.get("model_sha256"):
        raise ArtifactIncompatibleError(
            "Model file does not match its metadata checksum (modified or corrupted). Retrain."
        )
    return joblib.load(cfg.paths.model), meta
