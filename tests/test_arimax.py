"""Tests for models/arimax.py: fourier_terms, select_arimax_exog, ARIMAXForecaster."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.metrics import quantile_crossing_rate
from energy_price_forecast.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.models.arimax import (
    _DEFAULT_ORDER,
    _FOURIER_DAILY_K,
    _FOURIER_WEEKLY_K,
    ARIMAX_EXOG_COLUMNS,
    ARIMAXForecaster,
    fourier_terms,
    select_arimax_exog,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exog(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly exog DataFrame with the 9 curated ARIMAX columns."""
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    n = len(idx)
    local = idx.tz_convert("Europe/Berlin")
    return pd.DataFrame(
        {
            "residual_load_forecast": rng.normal(30_000, 5_000, n),
            "ttf_gas_lag_48h": rng.uniform(10, 50, n),
            "eua_co2_lag_48h": rng.uniform(20, 80, n),
            "eua_missing": np.zeros(n),
            "is_weekend": (local.dayofweek >= 5).astype(float),
            "is_holiday": np.zeros(n),
            "is_regional_holiday": np.zeros(n),
            "is_crisis": np.zeros(n),
            "is_post_crisis": np.zeros(n),
        },
        index=idx,
    )


def _make_price(n_days: int = 40, seed: int = 42) -> pd.Series:
    """Synthetic hourly price series (EUR/MWh) with mild autocorrelation."""
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 10, len(idx))
    # AR(1) component so ARMA has something to model
    for i in range(1, len(noise)):
        noise[i] += 0.5 * noise[i - 1]
    return pd.Series(50.0 + noise, index=idx, name="day_ahead_price")


def _fit_model(n_train_days: int = 30, alpha: float = 0.5) -> tuple[ARIMAXForecaster, pd.Series]:
    """Fit and return a model + test prices for reuse across tests."""
    x = _make_exog(n_days=n_train_days + 2)
    y = _make_price(n_days=n_train_days + 2)
    n_train = n_train_days * 24
    x_train, y_train = x.iloc[:n_train], y.iloc[:n_train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster(alpha=alpha)
        model.fit(y_train, x_train)
    return model, y


# ---------------------------------------------------------------------------
# fourier_terms — column count
# ---------------------------------------------------------------------------


def test_fourier_daily_column_count() -> None:
    idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    result = fourier_terms(idx, period_hours=24, k=4)
    assert result.shape[1] == 8  # 2 * k


def test_fourier_daily_column_names() -> None:
    idx = pd.date_range("2021-01-01", periods=24, freq="h", tz="UTC")
    result = fourier_terms(idx, period_hours=24, k=4)
    expected = [
        "sin_24_1",
        "cos_24_1",
        "sin_24_2",
        "cos_24_2",
        "sin_24_3",
        "cos_24_3",
        "sin_24_4",
        "cos_24_4",
    ]
    assert result.columns.tolist() == expected


def test_fourier_weekly_column_count() -> None:
    idx = pd.date_range("2021-01-01", periods=168 * 2, freq="h", tz="UTC")
    result = fourier_terms(idx, period_hours=168, k=2)
    assert result.shape[1] == 4  # 2 * k


# ---------------------------------------------------------------------------
# fourier_terms — hand-check at local midnight (winter + summer DST)
# ---------------------------------------------------------------------------


def test_fourier_hand_check_local_midnight_winter() -> None:
    # 2020-12-31 23:00 UTC = 2021-01-01 00:00 Europe/Berlin (UTC+1, winter)
    idx = pd.DatetimeIndex(["2020-12-31 23:00:00+00:00"])
    result = fourier_terms(idx, period_hours=24, k=4)
    # local hour = 0 → angle = 2π*h*0/24 = 0 for all h → sin=0, cos=1
    assert result["sin_24_1"].iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert result["cos_24_1"].iloc[0] == pytest.approx(1.0, abs=1e-10)


def test_fourier_hand_check_local_midnight_summer() -> None:
    # 2021-05-31 22:00 UTC = 2021-06-01 00:00 Europe/Berlin (UTC+2, summer DST)
    idx = pd.DatetimeIndex(["2021-05-31 22:00:00+00:00"])
    result = fourier_terms(idx, period_hours=24, k=4)
    assert result["sin_24_1"].iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert result["cos_24_1"].iloc[0] == pytest.approx(1.0, abs=1e-10)


def test_fourier_values_in_unit_interval() -> None:
    idx = pd.date_range("2021-01-01", periods=24 * 7, freq="h", tz="UTC")
    for period_hours, k in [(24, _FOURIER_DAILY_K), (168, _FOURIER_WEEKLY_K)]:
        if k == 0:
            continue  # k=0 produces an empty DataFrame; nothing to range-check
        result = fourier_terms(idx, period_hours=period_hours, k=k)
        assert result.values.min() >= -1.0 - 1e-12
        assert result.values.max() <= 1.0 + 1e-12


def test_fourier_raises_on_unsupported_period() -> None:
    idx = pd.date_range("2021-01-01", periods=24, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="unsupported period_hours"):
        fourier_terms(idx, period_hours=12, k=2)


# ---------------------------------------------------------------------------
# select_arimax_exog
# ---------------------------------------------------------------------------


def test_select_arimax_exog_returns_exact_columns() -> None:
    x = _make_exog(n_days=5)
    subset = select_arimax_exog(x)
    assert list(subset.columns) == list(ARIMAX_EXOG_COLUMNS)


def test_select_arimax_exog_raises_on_missing_column() -> None:
    x = _make_exog(n_days=5).drop(columns=["residual_load_forecast"])
    with pytest.raises(ValueError, match="missing ARIMAX exog columns"):
        select_arimax_exog(x)


def test_select_arimax_exog_excludes_non_curated_columns() -> None:
    x = _make_exog(n_days=5)
    x["price_lag_24h"] = 0.0
    x["load_forecast"] = 0.0
    x["renewable_share_forecast"] = 0.0
    subset = select_arimax_exog(x)
    for col in ("price_lag_24h", "load_forecast", "renewable_share_forecast"):
        assert col not in subset.columns


# ---------------------------------------------------------------------------
# ARIMAXForecaster — structural checks
# ---------------------------------------------------------------------------


def test_arimax_default_order() -> None:
    model = ARIMAXForecaster()
    assert model.order == _DEFAULT_ORDER
    assert model.order[1] == 0  # d=0 is structural


def test_arimax_design_matrix_shape() -> None:
    """Design matrix has 2*daily_k + 2*weekly_k + n_active_exog columns."""
    x = _make_exog(n_days=5)
    model = ARIMAXForecaster()
    raw_std = x.std(ddof=0)
    model._zero_var_cols = set(raw_std[raw_std == 0.0].index.tolist())
    model._scaler_mean = x.mean()
    model._scaler_std = raw_std.replace(0.0, 1.0)
    design = model._design(pd.DatetimeIndex(x.index), x)
    n_active_exog = len(ARIMAX_EXOG_COLUMNS) - len(model._zero_var_cols)
    expected_cols = 2 * _FOURIER_DAILY_K + 2 * _FOURIER_WEEKLY_K + n_active_exog
    assert design.shape[1] == expected_cols  # 8 + 0 + 4 = 12 (5 zero-var cols dropped)


# ---------------------------------------------------------------------------
# ARIMAXForecaster — zero-variance column dropping
# ---------------------------------------------------------------------------


def test_design_drops_zero_variance_columns() -> None:
    """Zero-variance exog columns are excluded from the design matrix after fit."""
    x = _make_exog(n_days=35)
    y = _make_price(n_days=35)
    n_train = 30 * 24
    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    # The synthetic fixture has 5 all-zero columns
    expected_zero_var = {
        "eua_missing",
        "is_holiday",
        "is_regional_holiday",
        "is_crisis",
        "is_post_crisis",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster()
        model.fit(y_train, x_train)

    assert model._zero_var_cols == expected_zero_var
    design = model._design(pd.DatetimeIndex(x_train.index), x_train)
    for col in expected_zero_var:
        assert col not in design.columns


# ---------------------------------------------------------------------------
# ARIMAXForecaster — warm starting
# ---------------------------------------------------------------------------


def test_warm_start_params_set_after_fit() -> None:
    """_start_params is a numpy array with the correct length after fit."""
    x = _make_exog(n_days=35)
    y = _make_price(n_days=35)
    n_train = 30 * 24
    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    model = ARIMAXForecaster()
    params_before = model._start_params
    assert params_before is None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y_train, x_train)

    params_after = model._start_params
    assert params_after is not None
    # AR(2) + sigma2 = 3 extra; exog cols after zero-var drop
    n_active_exog = len(ARIMAX_EXOG_COLUMNS) - len(model._zero_var_cols)
    expected_len = 2 * _FOURIER_DAILY_K + 2 * _FOURIER_WEEKLY_K + n_active_exog + model.order[0] + 1
    assert len(params_after) == expected_len


def test_warm_start_skipped_on_param_count_change() -> None:
    """fit() falls back to cold start silently when param count changes between folds."""
    x = _make_exog(n_days=35)
    y = _make_price(n_days=35)
    n_train = 30 * 24
    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster()
        model.fit(y_train, x_train)

    # Inject a stale _start_params of the wrong length to simulate a param-count change
    model._start_params = np.zeros(999)

    # Second fit must not crash and must overwrite _start_params with the correct length
    x_train2 = x.iloc[24 : n_train + 24]
    y_train2 = y.iloc[24 : n_train + 24]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y_train2, x_train2)

    assert model._start_params is not None
    assert len(model._start_params) != 999


# ---------------------------------------------------------------------------
# ARIMAXForecaster — fit / predict smoke
# ---------------------------------------------------------------------------


def test_fit_predict_smoke() -> None:
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_index = pd.DatetimeIndex(x_test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster()
        model.fit(y_train, x_train)
        preds = model.predict(test_index, history=y_train, x_test=x_test)

    assert isinstance(preds, pd.Series)
    assert preds.name == "y_pred"
    assert list(preds.index) == list(test_index)
    assert preds.notna().all()
    assert preds.between(-500, 1_000).all()


# ---------------------------------------------------------------------------
# ARIMAXForecaster — protocol conformity via run_backtest
# ---------------------------------------------------------------------------


def test_protocol_via_run_backtest() -> None:
    n_days = 22
    train_span = 15
    x = _make_exog(n_days=n_days)
    y = _make_price(n_days=n_days)

    folds = list(
        walk_forward_splits(
            pd.DatetimeIndex(x.index),
            test_start="2021-01-17",
            window="rolling",
            train_span_days=train_span,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster(resid_warmup=0)
        predictions = run_backtest(y, model, folds, refit_every=10, x=x)

    assert "y_pred" in predictions.columns
    assert predictions["y_pred"].notna().all()


# ---------------------------------------------------------------------------
# ARIMAXForecaster — reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility() -> None:
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_index = pd.DatetimeIndex(x_test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = ARIMAXForecaster()
        m1.fit(y_train, x_train)
        p1 = m1.predict(test_index, history=y_train, x_test=x_test)

        m2 = ARIMAXForecaster()
        m2.fit(y_train, x_train)
        p2 = m2.predict(test_index, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# ARIMAXForecaster — quantile wiring, monotonicity, crossing ≡ 0
# ---------------------------------------------------------------------------


def test_quantile_offset_monotonicity() -> None:
    """Frozen offsets must be monotone: Q_0.05 <= Q_0.50 <= Q_0.95."""
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    offsets = {}
    for alpha in (0.05, 0.5, 0.95):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMAXForecaster(alpha=alpha)
            model.fit(y_train, x_train)
        offsets[alpha] = model._offsets[alpha]

    assert offsets[0.05] <= offsets[0.5]
    assert offsets[0.5] <= offsets[0.95]


def test_quantile_direction() -> None:
    """Mean q95 prediction >= mean q50 >= mean q05."""
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_index = pd.DatetimeIndex(x_test.index)
    preds = {}
    for alpha in (0.05, 0.5, 0.95):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMAXForecaster(alpha=alpha)
            m.fit(y_train, x_train)
            preds[alpha] = m.predict(test_index, history=y_train, x_test=x_test)

    assert preds[0.95].mean() >= preds[0.5].mean()
    assert preds[0.5].mean() >= preds[0.05].mean()


def test_crossing_rate_is_zero() -> None:
    """ARIMAX crossing_rate must be exactly 0 (monotone offsets + shared point)."""
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_index = pd.DatetimeIndex(x_test.index)
    preds = {}
    for alpha in (0.05, 0.5, 0.95):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMAXForecaster(alpha=alpha)
            m.fit(y_train, x_train)
            preds[alpha] = m.predict(test_index, history=y_train, x_test=x_test)

    rates = quantile_crossing_rate(preds[0.05], preds[0.5], preds[0.95])
    assert rates["crossing_low_above_mid"] == pytest.approx(0.0)
    assert rates["crossing_mid_above_high"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ARIMAXForecaster — NaN exog handling
# ---------------------------------------------------------------------------


def test_nan_exog_no_crash() -> None:
    """NaN in eua_co2_lag_48h (e.g. pre-Oct-2021) must not crash fit or predict."""
    x = _make_exog(n_days=32)
    y = _make_price(n_days=32)
    n_train = 30 * 24
    x_train, x_test = x.iloc[:n_train].copy(), x.iloc[n_train:].copy()
    y_train = y.iloc[:n_train]
    # Inject NaN block into the training exog (simulates missing EUA data)
    x_train.loc[x_train.index[:48], "eua_co2_lag_48h"] = float("nan")
    test_index = pd.DatetimeIndex(x_test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster()
        model.fit(y_train, x_train)
        preds = model.predict(test_index, history=y_train, x_test=x_test)

    assert preds.notna().all()


# ---------------------------------------------------------------------------
# ARIMAXForecaster — error contracts
# ---------------------------------------------------------------------------


def test_fit_raises_without_x_train() -> None:
    model = ARIMAXForecaster()
    y = _make_price(n_days=5)
    with pytest.raises(ValueError, match="exog matrix"):
        model.fit(y, x_train=None)


def test_predict_raises_before_fit() -> None:
    x = _make_exog(n_days=5)
    y = _make_price(n_days=5)
    model = ARIMAXForecaster()
    test_index = pd.DatetimeIndex(x.index[:24])
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(test_index, history=y.iloc[:0], x_test=x.iloc[:24])


def test_predict_raises_without_x_test() -> None:
    model, y = _fit_model()
    x = _make_exog(n_days=32)
    n_train = 30 * 24
    test_index = pd.DatetimeIndex(x.index[n_train:])
    y_train = y.iloc[:n_train]
    with pytest.raises(ValueError, match="exog matrix"):
        model.predict(test_index, history=y_train, x_test=None)


# ---------------------------------------------------------------------------
# ARIMAXForecaster — exog_coefficients accessor (Sprint 3.5)
# ---------------------------------------------------------------------------


def test_exog_coefficients_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = ARIMAXForecaster().exog_coefficients


def test_exog_coefficients_after_fit_is_named_series() -> None:
    """exog_coefficients returns a named pd.Series subset of ARIMAX_EXOG_COLUMNS."""
    # Use a fixture with variance in all columns so zero-var drop is minimal.
    x = _make_exog(n_days=35)
    y = _make_price(n_days=35)
    n_train = 30 * 24
    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMAXForecaster()
        model.fit(y_train, x_train)

    coefs = model.exog_coefficients
    assert isinstance(coefs, pd.Series)
    # All returned names must be from the curated exog set.
    assert set(coefs.index).issubset(set(ARIMAX_EXOG_COLUMNS))
    # Coefficients must be finite floats.
    assert coefs.notna().all()
    assert (coefs.abs() < 1e6).all()
