from __future__ import annotations

from pathlib import Path

import pytest

from hcai.config import Config, load_config, project_root
from hcai.ml.data import hash_frame, validate_and_split_target
from hcai.ml.schema import TARGET
from tests.fixtures.synth import make_synthetic

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_blas():  # type: ignore[no-untyped-def]
    """Cap BLAS/OpenMP threads: tiny matrices gain nothing from many threads, and on machines
    with many cores the oversubscription can make model-fitting tests several times slower."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may ever pick up a real API key (and so never touch the network)."""
    monkeypatch.setenv("HCAI_NO_DOTENV", "1")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Real default config, but every path redirected into tmp and training made cheap."""
    base = load_config()
    paths = {k: tmp_path / Path(v).relative_to(project_root()) for k, v in base.paths}
    ml = base.ml.model_copy(
        update={
            "cv_repeats": 1,
            "bootstrap_resamples": 100,
            "rf_trees": 40,
            "hgb_max_iter": 40,
            "permutation_repeats": 3,
            "candidates": ["logreg", "random_forest", "hist_gb"],
            "nested_cv": False,
            "grids": {
                "logreg": {"C": [0.1, 1.0]},
                "random_forest": {"max_depth": [3], "min_samples_leaf": [3]},
                "hist_gb": {"max_depth": [2], "learning_rate": [0.1]},
            },
        }
    )
    return base.model_copy(update={"paths": type(base.paths)(**paths), "ml": ml})


@pytest.fixture(scope="session")
def synth_df():  # type: ignore[no-untyped-def]
    return make_synthetic()


@pytest.fixture()
def synth_data(synth_df):  # type: ignore[no-untyped-def]
    x, y = validate_and_split_target(synth_df, 303)
    import pandas as pd

    return x, y, hash_frame(pd.concat([x, y.rename(TARGET)], axis=1))


@pytest.fixture()
def valid_patient() -> dict[str, object]:
    return {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 250, "fbs": 0, "restecg": 0,
        "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7,
    }  # fmt: skip
