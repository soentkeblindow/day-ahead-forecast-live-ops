"""Golden-fixture regression anchor for the default LGBMForecaster path.

Sprint 6 / Step 6.1, Beleg C: proves that copying the model code did not
silently shift its default behaviour, and is the regression anchor for every
later change (Sprint 6.3 onward). The dataset is fixed and synthetic --
constructed here in code, never loaded from a file -- so the test verifies
the model, not a data artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_price_forecast.models.lgbm import LGBMForecaster

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_lgbm_predictions.parquet"

_N_DAYS = 60
_N_FEATURES = 6
_SEED = 1234567
_TRAIN_DAYS = 45


def make_golden_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Fixed synthetic hourly feature matrix and price series, pre-split into train/test."""
    idx = pd.date_range("2021-01-01", periods=_N_DAYS * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(_SEED)
    x = pd.DataFrame(
        rng.normal(0, 1, (len(idx), _N_FEATURES)),
        index=idx,
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    y = pd.Series(50.0 + rng.normal(0, 10, len(idx)), index=idx, name="price")

    n_train = _TRAIN_DAYS * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_index = pd.DatetimeIndex(x_test.index)
    return x_train, x_test, y_train, test_index


def test_default_forecaster_matches_golden_fixture() -> None:
    x_train, x_test, y_train, test_index = make_golden_dataset()

    model = LGBMForecaster()
    model.fit(y_train, x_train)
    preds = model.predict(test_index, history=y_train, x_test=x_test)

    golden = pd.read_parquet(FIXTURE_PATH)["y_pred"]
    pd.testing.assert_series_equal(preds, golden, check_exact=True, check_freq=False)
