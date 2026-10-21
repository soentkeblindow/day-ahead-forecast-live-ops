"""Unit tests for the live, single-target-day FHS quantile step (Sprint
7.1 Teil B -> Probephase). No network access; the OOS-Store is a real
on-disk CSV under tmp_path, built via the same ops.oos_store write path
the real daily top-up job uses.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.arena.quantile_step import run_quantile_step
from energy_price_forecast.market_time import LOCAL_TZ
from energy_price_forecast.ops.oos_store import (
    OOS_MODEL_VERSION,
    OosStoreRow,
    rows_to_frame,
    write_oos_store,
)

LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)


def _local_day_index(day: dt.date) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1), tz=LOCAL_TZ)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


def _write_history(
    path: Path,
    ladder_row: str,
    first_day: dt.date,
    n_days: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Writes n_days of real-shaped OOS-store history and returns the
    matching realised-price Series over the same span (so sigma/pool
    have real, non-degenerate data to calibrate against)."""
    rows: list[OosStoreRow] = []
    price_parts: list[pd.Series] = []
    for d in range(n_days):
        day = first_day + dt.timedelta(days=d)
        idx = _local_day_index(day)
        forecast_values = 50 + rng.normal(0, 5, len(idx))
        price_values = forecast_values + rng.normal(0, 2, len(idx))
        for ts, value in zip(idx, forecast_values, strict=True):
            rows.append(
                OosStoreRow(
                    target_day=day,
                    timestamp_utc=ts,
                    ladder_row=ladder_row,
                    forecast=float(value),
                    model_version=OOS_MODEL_VERSION,
                    weather_run_init_utc=None,
                    source="backfill",
                    created_at_utc=pd.Timestamp("2026-01-01", tz="UTC"),
                )
            )
        price_parts.append(pd.Series(price_values, index=idx))
    write_oos_store(rows_to_frame(rows), path)
    return pd.concat(price_parts).sort_index()


def test_returns_monotonic_quantiles_for_today(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    first_day = dt.date(2026, 1, 1)
    oos_path = tmp_path / "oos_store.csv"
    prices_qh = _write_history(oos_path, "core_gas", first_day, 200, rng)

    target_day = first_day + dt.timedelta(days=200)
    today_values = [50.0] * 96

    quantiles = run_quantile_step(
        "core_gas", today_values, target_day, prices_qh, oos_store_path=oos_path
    )

    assert quantiles is not None
    assert set(quantiles) == set(LEVELS)
    today_index = _local_day_index(target_day)
    grid = pd.DataFrame({a: quantiles[a] for a in LEVELS}, index=today_index)
    assert grid.notna().all().all()
    for i in range(len(LEVELS) - 1):
        assert (grid[LEVELS[i]] <= grid[LEVELS[i + 1]]).all()


def test_returns_none_when_history_is_too_short(tmp_path: Path) -> None:
    """Fewer than ~166 days of history (180-day window minus the 14-day
    tolerance) -- the Probephase's skip-this-day-only policy, not an
    exception."""
    rng = np.random.default_rng(1)
    first_day = dt.date(2026, 1, 1)
    oos_path = tmp_path / "oos_store.csv"
    prices_qh = _write_history(oos_path, "core_gas", first_day, 30, rng)

    target_day = first_day + dt.timedelta(days=30)
    today_values = [50.0] * 96

    result = run_quantile_step(
        "core_gas", today_values, target_day, prices_qh, oos_store_path=oos_path
    )

    assert result is None


def test_rejects_a_wrong_value_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    first_day = dt.date(2026, 1, 1)
    oos_path = tmp_path / "oos_store.csv"
    prices_qh = _write_history(oos_path, "core_gas", first_day, 200, rng)

    target_day = first_day + dt.timedelta(days=200)
    with pytest.raises(ValueError, match="expected 96"):
        run_quantile_step("core_gas", [50.0] * 95, target_day, prices_qh, oos_store_path=oos_path)


def test_runs_on_a_dst_fall_back_target_day(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    first_day = dt.date(2026, 4, 1)  # 200 days later lands well past 2026-10-25
    oos_path = tmp_path / "oos_store.csv"
    prices_qh = _write_history(oos_path, "core_gas", first_day, 200, rng)

    target_day = dt.date(2026, 10, 25)  # DE/LU fall back, 100 quarter-hours
    today_values = [50.0] * 100

    quantiles = run_quantile_step(
        "core_gas", today_values, target_day, prices_qh, oos_store_path=oos_path
    )

    assert quantiles is not None
    for level in LEVELS:
        assert len(quantiles[level]) == 100


def test_uses_todays_point_values_not_a_recomputed_forecast(tmp_path: Path) -> None:
    """The median level (0.5) must track the exact point_values passed in
    -- spec section 1: the quantiles are built AROUND the already-
    submitted Challenge-2 median, never a separately-held series."""
    rng = np.random.default_rng(4)
    first_day = dt.date(2026, 1, 1)
    oos_path = tmp_path / "oos_store.csv"
    prices_qh = _write_history(oos_path, "core_gas", first_day, 200, rng)

    target_day = first_day + dt.timedelta(days=200)
    distinctive_values = [100.0 + i for i in range(96)]

    quantiles = run_quantile_step(
        "core_gas", distinctive_values, target_day, prices_qh, oos_store_path=oos_path
    )

    assert quantiles is not None
    today_index = _local_day_index(target_day)
    # the median quantile's own u_a is typically close to zero, but the
    # RELATIVE shape across the day must track point_values closely --
    # checked via the per-point difference staying small and structured,
    # not by exact equality (u_0.5 is not guaranteed to be exactly 0).
    median = quantiles[0.5].reindex(today_index).to_numpy()
    assert np.corrcoef(median, distinctive_values)[0, 1] > 0.99
