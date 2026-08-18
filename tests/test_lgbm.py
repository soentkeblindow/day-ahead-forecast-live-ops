from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.models.lgbm import LGBMForecaster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xy(
    n_days: int = 60, n_features: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic hourly feature matrix and price series."""
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(0, 1, (len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(50.0 + rng.normal(0, 10, len(idx)), index=idx, name="price")
    return x, y


def _split(
    x: pd.DataFrame, y: pd.Series, train_days: int = 40
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    n = train_days * 24
    x_train, x_test = x.iloc[:n], x.iloc[n:]
    y_train = y.iloc[:n]
    return x_train, x_test, y_train, pd.DatetimeIndex(x_test.index)


# ---------------------------------------------------------------------------
# Protocol conformity
# ---------------------------------------------------------------------------


def test_protocol_via_run_backtest() -> None:
    x, y = _make_xy(n_days=10)
    folds = list(walk_forward_splits(pd.DatetimeIndex(x.index), test_start="2021-01-06"))
    model = LGBMForecaster()
    predictions = run_backtest(y, model, folds, refit_every=3, x=x)
    assert "y_pred" in predictions.columns
    assert predictions["y_pred"].notna().all()


# ---------------------------------------------------------------------------
# Smoke: output shape and name
# ---------------------------------------------------------------------------


def test_fit_predict_smoke() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)
    model = LGBMForecaster()
    model.fit(y_train, x_train)
    preds = model.predict(test_index, history=y_train, x_test=x_test)

    assert isinstance(preds, pd.Series)
    assert preds.name == "y_pred"
    assert list(preds.index) == list(test_index)
    assert preds.notna().all()
    # Plausible EUR/MWh range: model was trained on data centred at 50
    assert preds.between(-500, 1000).all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)

    m1 = LGBMForecaster(random_state=7)
    m1.fit(y_train, x_train)
    p1 = m1.predict(test_index, history=y_train, x_test=x_test)

    m2 = LGBMForecaster(random_state=7)
    m2.fit(y_train, x_train)
    p2 = m2.predict(test_index, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# NaN feature column
# ---------------------------------------------------------------------------


def test_all_nan_feature_column_no_crash() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["all_nan"] = float("nan")
    x_test["all_nan"] = float("nan")

    model = LGBMForecaster()
    model.fit(y_train, x_train)  # must not raise
    preds = model.predict(test_index, history=y_train, x_test=x_test)
    assert preds.notna().all()


# ---------------------------------------------------------------------------
# Error contracts
# ---------------------------------------------------------------------------


def test_fit_raises_without_x_train() -> None:
    x, y = _make_xy()
    _, _, y_train, _ = _split(x, y)
    with pytest.raises(ValueError, match="feature matrix"):
        LGBMForecaster().fit(y_train, x_train=None)


def test_predict_raises_before_fit() -> None:
    x, y = _make_xy()
    x_train, x_test, _, test_index = _split(x, y)
    with pytest.raises(RuntimeError, match="before fit"):
        LGBMForecaster().predict(test_index, history=y.iloc[:0], x_test=x_test)


def test_predict_raises_without_x_test() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)
    model = LGBMForecaster()
    model.fit(y_train, x_train)
    with pytest.raises(ValueError, match="feature matrix"):
        model.predict(test_index, history=y_train, x_test=None)


# ---------------------------------------------------------------------------
# Alpha wiring and directional check
# ---------------------------------------------------------------------------


def test_alpha_stored_on_internal_model() -> None:
    x, y = _make_xy()
    x_train, _, y_train, _ = _split(x, y)
    model = LGBMForecaster(alpha=0.7)
    model.fit(y_train, x_train)
    assert model._model is not None
    params = model._model.get_params()
    assert params["objective"] == "quantile"
    assert params["alpha"] == pytest.approx(0.7)


def test_alpha_direction() -> None:
    # On spread-out data, q0.95 predictions must exceed q0.05 predictions on average.
    x, y = _make_xy(n_days=80, seed=1)
    x_train, x_test, y_train, test_index = _split(x, y, train_days=60)

    m05 = LGBMForecaster(alpha=0.05)
    m05.fit(y_train, x_train)
    p05 = m05.predict(test_index, history=y_train, x_test=x_test)

    m95 = LGBMForecaster(alpha=0.95)
    m95.fit(y_train, x_train)
    p95 = m95.predict(test_index, history=y_train, x_test=x_test)

    assert p95.mean() >= p05.mean()


# ---------------------------------------------------------------------------
# fitted_estimator accessor (Sprint 3.5)
# ---------------------------------------------------------------------------


def test_fitted_estimator_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = LGBMForecaster().fitted_estimator


def test_fitted_estimator_after_fit_returns_lgbm_regressor() -> None:
    from lightgbm import LGBMRegressor

    x, y = _make_xy()
    x_train, _, y_train, _ = _split(x, y)
    model = LGBMForecaster()
    model.fit(y_train, x_train)
    est = model.fitted_estimator
    assert isinstance(est, LGBMRegressor)
    # A fitted estimator must have booster_ and know the feature count.
    assert hasattr(est, "booster_")
    assert est.n_features_in_ == x_train.shape[1]
