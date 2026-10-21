import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.config import ConformalConfig
from energy_price_forecast.evaluation.fhs_quantiles import (
    POOL_GAP_TOLERANCE_DAYS,
    compute_fhs_quantiles_for_row,
)

_LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)


def _quarterhourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="15min", tz="UTC")


def _config(window_days: int) -> ConformalConfig:
    return ConformalConfig(
        window_days=window_days,
        embargo_days=1,
        n_neighbors=50,
        sigma_floor_fraction=0.10,
        recompute_cadence_days=1,
        min_calibration_hours=24 * 14,
    )


def test_quantiles_are_monotonic_and_finite_once_calibrated() -> None:
    # 150 days: comfortably past both the sigma calibration gate (14 days)
    # and a 90-day full pool.
    index = _quarterhourly_utc("2020-12-31 23:00", 96 * 150)
    rng = np.random.default_rng(0)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)

    quantiles = compute_fhs_quantiles_for_row(
        forecast, price, window_days=90, config=_config(90), levels=_LEVELS
    )

    assert set(quantiles) == set(_LEVELS)
    grid = pd.DataFrame({a: quantiles[a] for a in _LEVELS})
    # the last 30 days are comfortably past the 90-day pool requirement
    tail = grid.iloc[-96 * 30 :]
    assert tail.notna().all().all()
    for i in range(len(_LEVELS) - 1):
        assert (tail[_LEVELS[i]] <= tail[_LEVELS[i + 1]]).all()


def test_early_history_is_nan_at_every_level_consistently() -> None:
    """Before a full pool has accumulated, every level must be NaN
    together at a given timestamp -- never a partial row (which would be
    unsafe for rearrange_quantiles)."""
    index = _quarterhourly_utc("2020-12-31 23:00", 96 * 30)
    rng = np.random.default_rng(1)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)

    quantiles = compute_fhs_quantiles_for_row(
        forecast, price, window_days=90, config=_config(90), levels=_LEVELS
    )

    grid = pd.DataFrame({a: quantiles[a] for a in _LEVELS})
    is_na = grid.isna()
    # every row is either all-NaN or all-real, never a mix
    assert ((is_na.sum(axis=1) == 0) | (is_na.sum(axis=1) == len(_LEVELS))).all()
    # 30 days never reaches a 90-day full pool -> entirely NaN
    assert is_na.all().all()


def test_pool_gap_tolerance_survives_a_gap_up_to_the_configured_size() -> None:
    """A real gap of exactly POOL_GAP_TOLERANCE_DAYS inside the pool
    window must not blank out the quantiles -- the 2026-10-21 measurement
    (docs/sprint7_step7_1_log.md section 9) this constant is based on."""
    window_days = 90
    index = _quarterhourly_utc("2020-12-31 23:00", 96 * (window_days + 40))
    rng = np.random.default_rng(10)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)

    gap_start = index[96 * 60]
    gap_end = gap_start + pd.Timedelta(days=POOL_GAP_TOLERANCE_DAYS)
    price_gapped = price.copy()
    price_gapped.loc[(price_gapped.index >= gap_start) & (price_gapped.index < gap_end)] = float(
        "nan"
    )

    quantiles = compute_fhs_quantiles_for_row(
        forecast, price_gapped, window_days=window_days, config=_config(window_days)
    )
    # the last 5 days are comfortably past both the gap and the (reduced)
    # pool requirement.
    tail = quantiles[0.5].iloc[-96 * 5 :]
    assert tail.notna().all()


def test_pool_gap_larger_than_the_tolerance_still_blanks_the_pool() -> None:
    """The tolerance has a real boundary -- it is not unconditionally
    permissive."""
    window_days = 90
    index = _quarterhourly_utc("2020-12-31 23:00", 96 * (window_days + 40))
    rng = np.random.default_rng(11)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)

    gap_start = index[96 * 60]
    gap_end = gap_start + pd.Timedelta(days=POOL_GAP_TOLERANCE_DAYS + 5)
    price_gapped = price.copy()
    price_gapped.loc[(price_gapped.index >= gap_start) & (price_gapped.index < gap_end)] = float(
        "nan"
    )

    quantiles = compute_fhs_quantiles_for_row(
        forecast, price_gapped, window_days=window_days, config=_config(window_days)
    )
    # the last 5 days: fully in the rolling regime (window_days + 40
    # total history) and the gap (days 60-79) still sits inside their
    # own ~90-day trailing window.
    tail = quantiles[0.5].iloc[-96 * 5 :]
    assert tail.isna().all()


def test_leakage_future_realized_price_never_changes_an_earlier_days_quantiles() -> None:
    index = _quarterhourly_utc("2020-12-31 23:00", 96 * 150)
    rng = np.random.default_rng(2)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)
    config = _config(90)

    quantiles_orig = compute_fhs_quantiles_for_row(
        forecast, price, window_days=90, config=config, levels=_LEVELS
    )

    test_day_start = index[96 * 120]  # comfortably inside the calibrated tail
    test_day_end = index[96 * 121]

    price_mutated = price.copy()
    price_mutated.loc[price_mutated.index >= test_day_end] = 999_999.0
    quantiles_mutated = compute_fhs_quantiles_for_row(
        forecast, price_mutated, window_days=90, config=config, levels=_LEVELS
    )

    day_slice = (index >= test_day_start) & (index < test_day_end)
    for level in _LEVELS:
        pd.testing.assert_series_equal(
            quantiles_orig[level].loc[day_slice], quantiles_mutated[level].loc[day_slice]
        )


def test_runs_through_a_dst_fall_back_day() -> None:
    index = _quarterhourly_utc("2026-07-01", 96 * 150)  # spans 2026-10-25
    rng = np.random.default_rng(3)
    price = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)

    quantiles = compute_fhs_quantiles_for_row(
        forecast, price, window_days=90, config=_config(90), levels=_LEVELS
    )

    for level in _LEVELS:
        assert len(quantiles[level]) == len(index)
    local_days = index.tz_convert("Europe/Berlin").normalize()
    fall_back_day = pd.Timestamp("2026-10-25", tz="Europe/Berlin")
    assert int((local_days == fall_back_day).sum()) == 100


def test_requires_a_median_level() -> None:
    index = _quarterhourly_utc("2021-01-01", 96 * 10)
    price = pd.Series(50.0, index=index)
    forecast = pd.Series(50.0, index=index)

    with pytest.raises(ValueError, match="0.5"):
        compute_fhs_quantiles_for_row(
            forecast, price, window_days=5, config=_config(5), levels=(0.25, 0.75)
        )
