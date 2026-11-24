from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcai.errors import DataValidationError, PatientInputError
from hcai.ml.data import hash_frame, read_processed_cleveland, validate_and_split_target
from hcai.ml.schema import FEATURES, PatientInput, parse_patient, to_frame


def test_valid_patient_parses(valid_patient):
    p = parse_patient(valid_patient)
    assert isinstance(p, PatientInput)
    assert to_frame([p]).shape == (1, 13)


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 0), ("age", 200), ("trestbps", 0), ("chol", 5000), ("chol", float("nan")),
        ("trestbps", float("inf")), ("sex", 2), ("cp", 0), ("cp", 5), ("restecg", 3),
        ("thal", 5), ("ca", 4), ("oldpeak", -0.1), ("age", "abc"), ("age", 45.5),
    ],
)  # fmt: skip
def test_out_of_range_or_wrong_type_is_rejected_not_clamped(valid_patient, field, value):
    valid_patient[field] = value
    with pytest.raises(PatientInputError):
        parse_patient(valid_patient)


def test_missing_required_field_rejected(valid_patient):
    del valid_patient["age"]
    with pytest.raises(PatientInputError, match="age"):
        parse_patient(valid_patient)


def test_extra_field_rejected(valid_patient):
    valid_patient["name"] = "x"
    with pytest.raises(PatientInputError):
        parse_patient(valid_patient)


def test_ca_thal_may_be_none_and_become_nan(valid_patient):
    valid_patient["ca"] = None
    valid_patient["thal"] = None
    frame = to_frame([parse_patient(valid_patient)])
    assert math.isnan(frame.loc[0, "ca"]) and math.isnan(frame.loc[0, "thal"])


def test_boundary_values_accepted(valid_patient):
    valid_patient.update(age=18, trestbps=70, chol=700, thalach=250, oldpeak=10)
    parse_patient(valid_patient)


def test_target_collapse_and_stratification_inputs(synth_df):
    x, y = validate_and_split_target(synth_df, 303)
    assert set(y.unique()) == {0, 1}
    assert (y == (synth_df["num"] > 0).astype(int)).all()
    assert list(x.columns) == FEATURES
    assert x["ca"].isna().sum() == 4 and x["thal"].isna().sum() == 2


def test_wrong_row_count_rejected(synth_df):
    with pytest.raises(DataValidationError, match=r"303"):
        validate_and_split_target(synth_df.head(100), 303)


def test_missing_column_and_bad_target_rejected(synth_df):
    with pytest.raises(DataValidationError, match=r"missing columns"):
        validate_and_split_target(synth_df.drop(columns=["chol"]), None)
    bad = synth_df.copy()
    bad.loc[0, "num"] = 9
    with pytest.raises(DataValidationError, match=r"0\.\.4"):
        validate_and_split_target(bad, None)
    bad.loc[0, "num"] = np.nan
    with pytest.raises(DataValidationError, match="Target"):
        validate_and_split_target(bad, None)


def test_hash_is_deterministic(synth_df):
    assert hash_frame(synth_df) == hash_frame(synth_df.copy())
    changed = synth_df.copy()
    changed.loc[0, "age"] += 1
    assert hash_frame(changed) != hash_frame(synth_df)


def test_read_processed_cleveland_handles_question_marks(tmp_path: Path):
    p = tmp_path / "processed.cleveland.data"
    p.write_text(
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,3.0,2\n",
        encoding="utf-8",
    )
    df = read_processed_cleveland(p)
    assert df.shape == (2, 14) and pd.isna(df.loc[1, "ca"])
