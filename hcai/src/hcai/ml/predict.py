"""Prediction contract: predict_risk(PatientInput) -> RiskResult."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from hcai import DISCLAIMER
from hcai.config import Config
from hcai.ml.registry import load_artifact
from hcai.ml.schema import FEATURES, LABELS, PatientInput, parse_patient, to_frame


class Factor(BaseModel):
    feature: str
    label: str
    patient_value: float
    delta: float  # probability change vs. replacing this feature with a typical (reference) value


class RiskResult(BaseModel):
    probability: float
    band: str
    top_factors: list[Factor]
    range_flags: list[str]  # features outside the training range (the model extrapolates)
    imputed_fields: list[str]
    ignored_fields: list[str] = []  # given by the user but unused: this model's data lacks them
    model_version: str
    disclaimer: str = DISCLAIMER


class RiskPredictor:
    def __init__(self, pipeline: Pipeline, meta: dict[str, Any], top_factors: int = 3) -> None:
        self.pipeline = pipeline
        self.meta = meta
        self.top_n = top_factors

    @classmethod
    def load(cls, cfg: Config) -> RiskPredictor:
        pipeline, meta = load_artifact(cfg)
        return cls(pipeline, meta, cfg.ml.top_factors)

    @property
    def absent(self) -> list[str]:
        return list(self.meta.get("absent_features", []))

    @property
    def used(self) -> list[str]:
        return [f for f in FEATURES if f not in self.absent]

    def band(self, p: float) -> str:
        if p < self.meta["band_low"]:
            return "low"
        return "moderate" if p < self.meta["band_high"] else "high"

    def _factors(self, row: pd.DataFrame, base: float) -> list[Factor]:
        """Model-agnostic occlusion: set one known feature at a time to its typical value."""
        ref = self.meta["reference"]
        known = [f for f in self.used if not np.isnan(row.iloc[0][f])]
        variants = []
        for f in known:
            v = row.copy()
            v[f] = ref[f]
            variants.append(v)
        probs = self.pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        out = [
            Factor(
                feature=f,
                label=LABELS[f],
                patient_value=float(row.iloc[0][f]),
                delta=float(base - p),
            )
            for f, p in zip(known, probs, strict=True)
        ]
        out = [f for f in out if f.delta > 0]  # only risk-increasing contributors
        return sorted(out, key=lambda f: -f.delta)[: self.top_n]

    def predict(self, patients: list[PatientInput]) -> list[RiskResult]:
        if not patients:
            return []
        frame = to_frame(patients)
        probs = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for i, p in enumerate(probs):
            row = frame.iloc[[i]].reset_index(drop=True)
            flags = [
                f
                for f in self.used
                if not np.isnan(row.iloc[0][f])
                and not (
                    self.meta["ranges"][f]["min"] <= row.iloc[0][f] <= self.meta["ranges"][f]["max"]
                )
            ]
            results.append(
                RiskResult(
                    probability=float(p),
                    band=self.band(float(p)),
                    top_factors=self._factors(row, float(p)),
                    range_flags=flags,
                    imputed_fields=[f for f in self.used if np.isnan(row.iloc[0][f])],
                    ignored_fields=[f for f in self.absent if not np.isnan(row.iloc[0][f])],
                    model_version=self.meta["model_version"],
                )
            )
        return results


def predict_risk(patient: dict[str, object] | PatientInput, predictor: RiskPredictor) -> RiskResult:
    """Validate (reject, never clamp) then predict."""
    return predictor.predict([parse_patient(patient)])[0]
