from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from energy_price_forecast.evaluation.dm_test import (
    daily_mean_loss,
    dm_test,
    newey_west_long_run_variance,
)
from scripts.run_dm_test import _INPUTS, _run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# newey_west_long_run_variance
# ---------------------------------------------------------------------------


def test_newey_west_matches_hand_computation() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0])

    # gamma_0 = 1.25, gamma_1 = 0.3125 (biased, divided by n=4, not n-k=3),
    # Bartlett weight w_1 = 1 - 1/2 = 0.5 -> variance = 1.25 + 2*0.5*0.3125 = 1.5625.
    result = newey_west_long_run_variance(d, lag=1)

    assert result == pytest.approx(1.5625)


def test_newey_west_lag_zero_is_biased_sample_variance() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0])

    result = newey_west_long_run_variance(d, lag=0)

    assert result == pytest.approx(1.25)


def test_newey_west_raises_on_constant_series() -> None:
    d = np.array([5.0, 5.0, 5.0, 5.0])

    with pytest.raises(ValueError, match="zero long-run variance"):
        newey_west_long_run_variance(d, lag=1)


# ---------------------------------------------------------------------------
# dm_test
# ---------------------------------------------------------------------------


def test_dm_test_end_to_end_hand_computed() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    loss_a = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    loss_b = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)

    result = dm_test(loss_a, loss_b, hac_lag=1, horizon=1)

    assert result.n_obs == 4
    assert result.mean_loss_diff == pytest.approx(2.5)
    # long-run variance = 1.5625 (see above) -> raw DM stat = 2.5 / sqrt(1.5625 / 4) = 4.0;
    # HLN factor for h=1, n=4: sqrt((4 + 1 - 2) / 4) = sqrt(3/4).
    expected_dm_stat = 4.0 * np.sqrt(3 / 4)
    assert result.dm_stat == pytest.approx(expected_dm_stat)
    expected_p = 2 * (1 - stats.t.cdf(abs(expected_dm_stat), df=3))
    assert result.p_value == pytest.approx(expected_p)


def test_hln_factor_matches_formula_for_h_greater_than_one() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    loss_a = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], index=idx)
    loss_b = pd.Series([0.0] * 6, index=idx)
    n, h = 6, 3

    result = dm_test(loss_a, loss_b, hac_lag=0, horizon=h)

    d = loss_a.to_numpy() - loss_b.to_numpy()
    gamma0 = float(np.sum((d - d.mean()) ** 2) / n)
    dm_stat_raw = d.mean() / np.sqrt(gamma0 / n)
    expected_hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

    assert result.dm_stat == pytest.approx(dm_stat_raw * expected_hln)


def test_hln_factor_h1_special_case() -> None:
    idx = _hourly_utc("2021-01-01", 5)
    loss_a = pd.Series([1.0, 3.0, 2.0, 4.0, 2.0], index=idx)
    loss_b = pd.Series([0.0] * 5, index=idx)
    n = 5

    result = dm_test(loss_a, loss_b, hac_lag=0, horizon=1)

    d = loss_a.to_numpy() - loss_b.to_numpy()
    gamma0 = float(np.sum((d - d.mean()) ** 2) / n)
    dm_stat_raw = d.mean() / np.sqrt(gamma0 / n)
    expected_hln = np.sqrt((n - 1) / n)  # h=1 special case

    assert result.dm_stat == pytest.approx(dm_stat_raw * expected_hln)


def test_sign_convention_series_a_better_gives_negative_stat() -> None:
    idx = _hourly_utc("2021-01-01", 30)
    rng = np.random.default_rng(0)
    loss_a = pd.Series(rng.uniform(0, 1, size=30), index=idx)
    loss_b = pd.Series(rng.uniform(2, 3, size=30), index=idx)  # systematically larger

    result = dm_test(loss_a, loss_b, hac_lag=2, horizon=1)

    assert result.mean_loss_diff < 0
    assert result.dm_stat < 0


def test_dm_test_raises_on_index_mismatch() -> None:
    loss_a = pd.Series([1.0, 2.0, 3.0, 4.0], index=_hourly_utc("2021-01-01", 4))
    loss_b = pd.Series([1.0, 2.0, 3.0, 4.0], index=_hourly_utc("2021-01-02", 4))

    with pytest.raises(ValueError, match="identical index"):
        dm_test(loss_a, loss_b, hac_lag=1, horizon=1)


def test_dm_test_raises_on_nan_with_count() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    loss_a = pd.Series([1.0, float("nan"), 3.0, 4.0], index=idx)
    loss_b = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)

    with pytest.raises(ValueError, match="1 NaN"):
        dm_test(loss_a, loss_b, hac_lag=1, horizon=1)


# ---------------------------------------------------------------------------
# daily_mean_loss
# ---------------------------------------------------------------------------


def test_daily_mean_loss_known_means_and_incomplete_day_count() -> None:
    idx = _hourly_utc("2021-01-01", 24 + 12)  # one full day + one incomplete (12h) day
    loss = pd.Series([1.0] * 24 + [3.0] * 12, index=idx)

    daily, n_incomplete = daily_mean_loss(loss)

    assert len(daily) == 2
    assert daily.iloc[0] == pytest.approx(1.0)
    assert daily.iloc[1] == pytest.approx(3.0)
    assert n_incomplete == 1


# ---------------------------------------------------------------------------
# scripts.run_dm_test (script I/O smoke)
# ---------------------------------------------------------------------------


def _fixture_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        price_path=tmp_path / "hourly.parquet",
        pred_lgbm=tmp_path / "preds_lgbm_q50.parquet",
        pred_lasso=tmp_path / "backtest_lasso.parquet",
        pred_arimax=tmp_path / "preds_arimax_v2_q50.parquet",
        out=tmp_path / "dm_test.csv",
    )


def test_run_dm_test_script_writes_four_rows_with_fixed_columns(tmp_path: Path) -> None:
    idx = _hourly_utc("2021-01-01", 48)  # two complete UTC days
    rng = np.random.default_rng(1)
    price = pd.Series(50.0 + rng.normal(0, 1, size=48), index=idx)
    lgbm_pred = price + rng.normal(0, 0.5, size=48)
    lasso_pred = price + rng.normal(0, 2.0, size=48)
    arimax_pred = price + rng.normal(0, 2.0, size=48)

    args = _fixture_args(tmp_path)
    pd.DataFrame({"day_ahead_price": price}, index=idx).to_parquet(args.price_path)
    pd.DataFrame({"y_pred": lgbm_pred}, index=idx).to_parquet(args.pred_lgbm)
    pd.DataFrame({"y_pred": lasso_pred}, index=idx).to_parquet(args.pred_lasso)
    pd.DataFrame({"y_pred": arimax_pred}, index=idx).to_parquet(args.pred_arimax)

    _run(args)

    result = pd.read_csv(args.out)
    assert len(result) == 4
    assert list(result.columns) == [
        "comparison",
        "variant",
        "n_obs",
        "hac_lag",
        "horizon",
        "mean_loss_diff_eur_mwh",
        "dm_stat",
        "p_value",
    ]
    assert set(result["comparison"]) == {"lightgbm_vs_lasso", "lightgbm_vs_arimax"}
    assert set(result["variant"]) == {"daily", "hourly"}


def test_run_dm_test_script_missing_files_collects_all(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        _run(args)

    message = str(exc_info.value)
    for filename, producer in _INPUTS.values():
        assert filename in message
        assert producer in message
