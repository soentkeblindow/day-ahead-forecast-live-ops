from __future__ import annotations

from typing import Any

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


def _make_xy_right_skewed(
    n_days: int = 60, n_features: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data whose target noise is right-skewed (lognormal-additive).

    The noise is independent of the features, so a fitted model's predictions
    converge to the (conditional-on-nothing) mean or median of this noise
    depending on objective -- exactly the asymmetry Step 6.3 exists to detect.
    """
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(0, 1, (len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    noise = rng.lognormal(mean=0.0, sigma=1.0, size=len(idx))  # mean > median
    y = pd.Series(50.0 + noise, index=idx, name="price")
    return x, y


class _RecordingLGBMRegressor:
    """Stand-in for LGBMRegressor that records constructor kwargs, fits nothing.

    Used to inspect exactly which keyword arguments LGBMForecaster.fit passes
    through, without paying for (or depending on the output of) a real fit.
    """

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _RecordingLGBMRegressor:
        return self


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


# ---------------------------------------------------------------------------
# objective wiring (Step 6.3)
# ---------------------------------------------------------------------------


def test_l2_objective_maps_to_lightgbm_regression_and_omits_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _make_xy()
    x_train, _, y_train, _ = _split(x, y)
    monkeypatch.setattr("energy_price_forecast.models.lgbm.LGBMRegressor", _RecordingLGBMRegressor)

    LGBMForecaster(objective="l2").fit(y_train, x_train)

    kwargs = _RecordingLGBMRegressor.last_kwargs
    assert kwargs is not None
    assert kwargs["objective"] == "regression"
    # Must be entirely absent, not just None -- LightGBM reads a present
    # `alpha` under objective="regression" as a Huber scale parameter.
    assert "alpha" not in kwargs


def test_quantile_objective_passes_expected_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _make_xy()
    x_train, _, y_train, _ = _split(x, y)
    monkeypatch.setattr("energy_price_forecast.models.lgbm.LGBMRegressor", _RecordingLGBMRegressor)

    LGBMForecaster(objective="quantile", alpha=0.7).fit(y_train, x_train)

    kwargs = _RecordingLGBMRegressor.last_kwargs
    assert kwargs is not None
    assert kwargs["objective"] == "quantile"
    assert kwargs["alpha"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# alpha / objective error contracts and default resolution (Step 6.3, D2)
# ---------------------------------------------------------------------------


def test_l2_with_explicit_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        LGBMForecaster(objective="l2", alpha=0.5)


def test_l2_with_non_median_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        LGBMForecaster(objective="l2", alpha=0.95)


def test_unknown_objective_raises() -> None:
    with pytest.raises(ValueError, match="objective"):
        LGBMForecaster(objective="unsinn")  # type: ignore[arg-type]


def test_default_objective_and_alpha() -> None:
    model = LGBMForecaster()
    assert model.objective == "quantile"
    assert model.alpha == pytest.approx(0.5)


def test_l2_default_alpha_is_none() -> None:
    model = LGBMForecaster(objective="l2")
    assert model.alpha is None


# ---------------------------------------------------------------------------
# reproducibility and NaN handling under the l2 objective (Step 6.3)
# ---------------------------------------------------------------------------


def test_l2_reproducibility() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)

    m1 = LGBMForecaster(objective="l2", random_state=7)
    m1.fit(y_train, x_train)
    p1 = m1.predict(test_index, history=y_train, x_test=x_test)

    m2 = LGBMForecaster(objective="l2", random_state=7)
    m2.fit(y_train, x_train)
    p2 = m2.predict(test_index, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p1, p2)


def test_l2_all_nan_feature_column_no_crash() -> None:
    x, y = _make_xy()
    x_train, x_test, y_train, test_index = _split(x, y)
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["all_nan"] = float("nan")
    x_test["all_nan"] = float("nan")

    model = LGBMForecaster(objective="l2")
    model.fit(y_train, x_train)  # must not raise
    preds = model.predict(test_index, history=y_train, x_test=x_test)
    assert preds.notna().all()


# ---------------------------------------------------------------------------
# mean-above-median direction under right-skewed noise (Step 6.3)
# ---------------------------------------------------------------------------


def test_l2_mean_exceeds_quantile_median_under_right_skew() -> None:
    x, y = _make_xy_right_skewed(n_days=80, seed=1)
    x_train, x_test, y_train, test_index = _split(x, y, train_days=60)

    mean_model = LGBMForecaster(objective="l2")
    mean_model.fit(y_train, x_train)
    mean_preds = mean_model.predict(test_index, history=y_train, x_test=x_test)

    median_model = LGBMForecaster(objective="quantile", alpha=0.5)
    median_model.fit(y_train, x_train)
    median_preds = median_model.predict(test_index, history=y_train, x_test=x_test)

    assert mean_preds.mean() > median_preds.mean()
