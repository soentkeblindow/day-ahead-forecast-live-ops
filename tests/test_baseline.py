"""Tests for LassoForecaster in models/baseline.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.models.baseline import LassoForecaster, OLSForecaster, RidgeForecaster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _make_xy(
    n_train: int = 24 * 60,
    n_test: int = 24,
    n_features: int = 4,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx = _hourly_utc("2021-01-01", n_train)
    test_idx = _hourly_utc("2021-03-02", n_test)
    x_train = pd.DataFrame(
        rng.normal(size=(n_train, n_features)),
        index=train_idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_train = pd.Series(rng.normal(loc=50, scale=20, size=n_train), index=train_idx)
    x_test = pd.DataFrame(
        rng.normal(size=(n_test, n_features)),
        index=test_idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    return y_train, x_train, test_idx, x_test


# ---------------------------------------------------------------------------
# asinh / sinh round-trip
# ---------------------------------------------------------------------------


def test_asinh_sinh_roundtrip() -> None:
    values = np.array([0.0, -50.0, 100.0, 800.0, -200.0])
    np.testing.assert_allclose(np.sinh(np.arcsinh(values)), values, rtol=1e-12)


# ---------------------------------------------------------------------------
# Smoke: fit then predict returns a well-formed Series
# ---------------------------------------------------------------------------


def test_fit_predict_returns_valid_series() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    model = LassoForecaster(cv_splits=3)
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)

    assert isinstance(result, pd.Series)
    assert result.name == "y_pred"
    assert len(result) == len(test_idx)
    assert result.index.equals(test_idx)
    assert result.notna().all()
    # Predictions must be on EUR/MWh scale (roughly −500 to 3000), not asinh scale (~0–8).
    assert result.abs().max() > 1.0


# ---------------------------------------------------------------------------
# Predictions are on EUR/MWh scale, not asinh scale
# ---------------------------------------------------------------------------


def test_predictions_on_eur_mwh_scale_not_asinh() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    model = LassoForecaster(cv_splits=3)
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)
    # asinh(50) ≈ 4.6; a real price of ~50 EUR/MWh should appear as ~50 here, not ~4.6.
    assert result.abs().mean() > 5.0


# ---------------------------------------------------------------------------
# All-NaN column in x_train must not crash; predict must return finite values
# ---------------------------------------------------------------------------


def test_all_nan_column_does_not_crash() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["f0"] = np.nan
    x_test["f0"] = np.nan

    model = LassoForecaster(cv_splits=3)
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)

    assert result.notna().all()
    assert np.isfinite(result.to_numpy()).all()


# ---------------------------------------------------------------------------
# Guard: x_train is None → ValueError; predict before fit → RuntimeError
# ---------------------------------------------------------------------------


def test_fit_without_x_raises_value_error() -> None:
    model = LassoForecaster()
    y = pd.Series([1.0, 2.0], index=_hourly_utc("2021-01-01", 2))
    with pytest.raises(ValueError, match="x_train"):
        model.fit(y, x_train=None)


def test_predict_before_fit_raises_runtime_error() -> None:
    model = LassoForecaster()
    test_idx = _hourly_utc("2021-01-01", 24)
    history = pd.Series(50.0, index=_hourly_utc("2020-01-01", 24))
    x_test = pd.DataFrame({"f0": 1.0}, index=test_idx)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(test_idx, history=history, x_test=x_test)


def test_predict_without_x_raises_value_error() -> None:
    y_train, x_train, test_idx, _ = _make_xy()
    model = LassoForecaster(cv_splits=3)
    model.fit(y_train, x_train)
    history = y_train
    with pytest.raises(ValueError, match="x_test"):
        model.predict(test_idx, history=history, x_test=None)


# ---------------------------------------------------------------------------
# Reproducibility: same random_state → identical predictions
# ---------------------------------------------------------------------------


def test_reproducibility() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()

    m1 = LassoForecaster(cv_splits=3, random_state=42)
    m1.fit(y_train, x_train)
    p1 = m1.predict(test_idx, history=y_train, x_test=x_test)

    m2 = LassoForecaster(cv_splits=3, random_state=42)
    m2.fit(y_train, x_train)
    p2 = m2.predict(test_idx, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# Leakage contract: different training distributions → different pipeline stats
# ---------------------------------------------------------------------------


def test_leakage_contract_separate_pipelines() -> None:
    """Two models trained on different distributions must produce different predictions
    for the same x_test -- confirming that pipeline stats are train-only and not shared.
    """
    rng = np.random.default_rng(7)
    n_train, n_test, n_features = 24 * 30, 24, 3
    test_idx = _hourly_utc("2021-06-01", n_test)
    x_test = pd.DataFrame(
        rng.normal(size=(n_test, n_features)),
        index=test_idx,
        columns=["a", "b", "c"],
    )

    # Fold A: low-price regime
    idx_a = _hourly_utc("2021-01-01", n_train)
    x_a = pd.DataFrame(
        rng.normal(loc=0, scale=1, size=(n_train, n_features)), index=idx_a, columns=["a", "b", "c"]
    )
    y_a = pd.Series(rng.normal(loc=20, scale=5, size=n_train), index=idx_a)

    # Fold B: high-price regime with different scale
    idx_b = _hourly_utc("2021-02-01", n_train)
    x_b = pd.DataFrame(
        rng.normal(loc=10, scale=5, size=(n_train, n_features)),
        index=idx_b,
        columns=["a", "b", "c"],
    )
    y_b = pd.Series(rng.normal(loc=200, scale=50, size=n_train), index=idx_b)

    m_a = LassoForecaster(cv_splits=3, random_state=0)
    m_a.fit(y_a, x_a)
    pred_a = m_a.predict(test_idx, history=y_a, x_test=x_test)

    m_b = LassoForecaster(cv_splits=3, random_state=0)
    m_b.fit(y_b, x_b)
    pred_b = m_b.predict(test_idx, history=y_b, x_test=x_test)

    assert not pred_a.equals(pred_b), "predictions must differ for different training distributions"


# ---------------------------------------------------------------------------
# target_transform parameter
# ---------------------------------------------------------------------------


def test_unknown_target_transform_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown target_transform"):
        LassoForecaster(target_transform="log")


def test_identity_transform_smoke() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    model = LassoForecaster(cv_splits=3, target_transform="identity")
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)

    assert isinstance(result, pd.Series)
    assert result.notna().all()
    # identity transform: predictions live on the raw scale (not compressed ~0-8)
    assert result.abs().mean() > 5.0


def test_asinh_regression_matches_default() -> None:
    """Explicit target_transform='asinh' must produce the same predictions as the default."""
    y_train, x_train, test_idx, x_test = _make_xy()

    m_default = LassoForecaster(cv_splits=3, random_state=0)
    m_default.fit(y_train, x_train)
    p_default = m_default.predict(test_idx, history=y_train, x_test=x_test)

    m_explicit = LassoForecaster(cv_splits=3, random_state=0, target_transform="asinh")
    m_explicit.fit(y_train, x_train)
    p_explicit = m_explicit.predict(test_idx, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p_default, p_explicit)


# ---------------------------------------------------------------------------
# Smoke tests: RidgeForecaster and OLSForecaster
# ---------------------------------------------------------------------------


def test_ridge_forecaster_smoke() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    model = RidgeForecaster()
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)

    assert isinstance(result, pd.Series)
    assert result.name == "y_pred"
    assert len(result) == len(test_idx)
    assert result.notna().all()
    assert result.abs().mean() > 5.0


def test_ols_forecaster_smoke() -> None:
    y_train, x_train, test_idx, x_test = _make_xy()
    model = OLSForecaster()
    model.fit(y_train, x_train)
    result = model.predict(test_idx, history=y_train, x_test=x_test)

    assert isinstance(result, pd.Series)
    assert result.name == "y_pred"
    assert len(result) == len(test_idx)
    assert result.notna().all()
    assert result.abs().mean() > 5.0
