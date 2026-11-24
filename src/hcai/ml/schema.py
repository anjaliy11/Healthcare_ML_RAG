"""Feature schema and boundary validation. Out-of-range input is rejected, never clamped."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hcai.errors import PatientInputError

FEATURES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]  # fmt: skip
NUMERIC: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY: list[str] = ["sex", "fbs", "exang"]
CATEGORICAL: list[str] = ["cp", "restecg", "slope", "thal"]
MISSING_INDICATOR_COLS: list[str] = ["ca", "thal"]  # the only columns with real NaNs in Cleveland
CLINICAL_TEST_FIELDS: list[str] = ["ca", "thal", "oldpeak", "restecg", "thalach"]
TARGET = "num"

LABELS: dict[str, str] = {
    "age": "age",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "serum cholesterol",
    "fbs": "fasting blood sugar > 120 mg/dl",
    "restecg": "resting ECG result",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "slope of peak exercise ST segment",
    "ca": "major vessels coloured by fluoroscopy",
    "thal": "thalassemia test result",
}


class PatientInput(BaseModel):
    """UCI Cleveland coding. `ca` and `thal` may be None (imputed and flagged)."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120, description="years")
    sex: Literal[0, 1] = Field(description="0=female, 1=male")
    cp: Literal[1, 2, 3, 4] = Field(
        description="chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(ge=70, le=250, description="resting blood pressure, mm Hg")
    chol: float = Field(ge=100, le=700, description="serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(ge=60, le=250, description="max heart rate achieved, bpm")
    exang: Literal[0, 1] = Field(description="exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(description="1 upsloping, 2 flat, 3 downsloping")
    ca: Literal[0, 1, 2, 3] | None = Field(default=None, description="vessels coloured (0-3)")
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="3 normal, 6 fixed defect, 7 reversible defect"
    )


def parse_patient(data: dict[str, object] | PatientInput) -> PatientInput:
    """Validate raw input; raise PatientInputError with a readable list of problems."""
    if isinstance(data, PatientInput):
        return data
    try:
        return PatientInput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        raise PatientInputError(
            f"Invalid patient input ({problems}). Nothing was clamped."
        ) from exc


def to_frame(patients: list[PatientInput]) -> pd.DataFrame:
    """Rows in canonical column order, all float64 (None -> NaN) to match training dtypes."""
    rows = [{f: getattr(p, f) for f in FEATURES} for p in patients]
    df = pd.DataFrame(rows, columns=FEATURES)
    return df.astype(float).replace({None: np.nan})
