"""Unit tests for arena/preflight.py -- Check A, Check B, the training-
extent check, and the plausibility bands (spec 6.7.2, sections 2.1-2.6,
5.2, 5.5, 7).
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_price_forecast.arena.preflight import (
    check_capacity_factor_bounds,
    check_commodity_staleness,
    check_daily_sum_plausibility,
    check_payload_plausibility,
    check_reconstruction_inputs,
    check_target_row,
    check_training_extent,
    known_defect_tolerance_hours,
)
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES

_RADIATION = {"shortwave_radiation", "direct_normal_irradiance"}


def _valid_weather_run() -> pd.DataFrame:
    index = pd.date_range("2026-09-12T00:00", periods=24, freq="h", tz="UTC")
    data = {}
    for point in GRID_POINTS:
        for variable in HOURLY_VARIABLES:
            data[f"{point.point_id}__{variable}"] = [10.0] * 24
    return pd.DataFrame(data, index=index, dtype="float32")


# ---------------------------------------------------------------------------
# Check A -- check_reconstruction_inputs
# ---------------------------------------------------------------------------


def test_check_a_passes_with_valid_weather_anchor_and_holiday_coverage() -> None:
    result = check_reconstruction_inputs(
        weather_run=_valid_weather_run(),
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=True,
    )
    assert result.ok is True
    assert result.reasons == ()


def test_check_a_fails_when_weather_run_is_none() -> None:
    result = check_reconstruction_inputs(
        weather_run=None,
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=True,
    )
    assert result.ok is False
    assert any("weather run" in r for r in result.reasons)


def test_check_a_fails_when_weather_run_is_invalid() -> None:
    bad_run = _valid_weather_run()
    non_radiation_col = f"{GRID_POINTS[0].point_id}__wind_speed_10m"
    bad_run[non_radiation_col] = float("nan")
    result = check_reconstruction_inputs(
        weather_run=bad_run,
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=True,
    )
    assert result.ok is False
    assert any("failed validation" in r for r in result.reasons)


def test_check_a_fails_when_anchor_table_expired() -> None:
    result = check_reconstruction_inputs(
        weather_run=_valid_weather_run(),
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-09-01", tz="UTC"),  # already past
        holiday_calendar_covers_target=True,
    )
    assert result.ok is False
    assert any("anchor table" in r for r in result.reasons)


def test_check_a_fails_when_holiday_calendar_does_not_cover_target() -> None:
    result = check_reconstruction_inputs(
        weather_run=_valid_weather_run(),
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=False,
    )
    assert result.ok is False
    assert any("holiday calendar" in r for r in result.reasons)


def test_check_a_covers_holiday_calendar_both_directions() -> None:
    within = check_reconstruction_inputs(
        weather_run=_valid_weather_run(),
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=True,
    )
    beyond = check_reconstruction_inputs(
        weather_run=_valid_weather_run(),
        target_day=pd.Timestamp("2026-09-13").date(),
        anchor_valid_until=pd.Timestamp("2026-10-28", tz="UTC"),
        holiday_calendar_covers_target=False,
    )
    assert within.ok is True
    assert beyond.ok is False


# ---------------------------------------------------------------------------
# Check B -- check_target_row
# ---------------------------------------------------------------------------


def _feature_row(**columns: float) -> pd.DataFrame:
    index = pd.date_range("2026-09-13", periods=1, freq="h", tz="UTC")
    return pd.DataFrame({k: [v] for k, v in columns.items()}, index=index)


def test_check_b_passes_with_a_complete_row() -> None:
    features = _feature_row(price_lag_24h=50.0, renewable_share_forecast_nwp=0.3)
    result = check_target_row(
        features,
        pd.Timestamp("2026-09-13").date(),
        required=frozenset({"price_lag_24h", "renewable_share_forecast_nwp"}),
    )
    assert result.ok is True
    assert result.missing_features == ()


def test_check_b_fails_on_a_nan_required_column() -> None:
    features = _feature_row(price_lag_24h=float("nan"), renewable_share_forecast_nwp=0.3)
    result = check_target_row(
        features,
        pd.Timestamp("2026-09-13").date(),
        required=frozenset({"price_lag_24h", "renewable_share_forecast_nwp"}),
    )
    assert result.ok is False
    assert result.missing_features == ("price_lag_24h",)


def test_check_b_fails_on_a_missing_column_entirely() -> None:
    features = _feature_row(price_lag_24h=50.0)
    result = check_target_row(
        features,
        pd.Timestamp("2026-09-13").date(),
        required=frozenset({"price_lag_24h", "renewable_share_forecast_nwp"}),
    )
    assert result.ok is False
    assert result.missing_features == ("renewable_share_forecast_nwp",)


def test_check_b_passes_when_a_non_required_column_is_nan() -> None:
    features = _feature_row(price_lag_24h=50.0, some_unused_column=float("nan"))
    result = check_target_row(
        features, pd.Timestamp("2026-09-13").date(), required=frozenset({"price_lag_24h"})
    )
    assert result.ok is True


# ---------------------------------------------------------------------------
# check_training_extent
# ---------------------------------------------------------------------------


def _hourly_frame(start: str, n_hours: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({"x": range(n_hours)}, index=index)


def test_training_extent_passes_for_a_full_90_day_window() -> None:
    features = _hourly_frame("2026-06-15", 90 * 24)
    result = check_training_extent(
        features, expected_days=90, must_reach=pd.Timestamp("2026-09-12").date()
    )
    assert result.ok is True


def test_training_extent_fails_when_window_ends_too_early() -> None:
    """The real 2026-09-11 shape: a source freezes, the window silently
    shortens at the recent end, no NaN anywhere in what's left."""
    features = _hourly_frame("2026-06-10", 85 * 24)  # ends 5 days early
    result = check_training_extent(
        features, expected_days=90, must_reach=pd.Timestamp("2026-09-12").date()
    )
    assert result.ok is False
    assert any("must reach" in r for r in result.reasons)


def test_training_extent_fails_when_row_count_too_low() -> None:
    index = pd.date_range("2026-06-15", periods=90, freq="D", tz="UTC")  # only 90 rows, not hours
    features = pd.DataFrame({"x": range(90)}, index=index)
    result = check_training_extent(features, expected_days=90, must_reach=index.max().date())
    assert result.ok is False
    assert any("row(s)" in r for r in result.reasons)


def test_training_extent_tolerates_a_known_defect_gap() -> None:
    """The real 2026-09-12 finding: a 90-day window missing exactly the
    hours of a documented, permanent gap must still pass, given the
    matching tolerance."""
    features = _hourly_frame("2026-06-17", 88 * 24)  # 2 days short of 90
    result = check_training_extent(
        features,
        expected_days=90,
        must_reach=pd.Timestamp("2026-09-12").date(),
        tolerated_missing_hours=48,
    )
    assert result.ok is True


def test_training_extent_still_fails_beyond_the_granted_tolerance() -> None:
    """Tolerance only covers what it explains -- an extra, unexplained
    shortfall on top must still block."""
    features = _hourly_frame("2026-06-17", 88 * 24)  # 2 days short of 90
    result = check_training_extent(
        features,
        expected_days=90,
        must_reach=pd.Timestamp("2026-09-12").date(),
        tolerated_missing_hours=24,  # only explains 1 of the 2 missing days
    )
    assert result.ok is False
    assert any("row(s)" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# known_defect_tolerance_hours
# ---------------------------------------------------------------------------


def test_known_defect_tolerance_hours_explains_a_direct_defect_day() -> None:
    # KNOWN_WEATHER_DEFECTS keys 2026-06-23T00Z (corrupt) -> excludes delivery day 2026-06-24.
    hours = known_defect_tolerance_hours({pd.Timestamp("2026-06-24").date()})
    assert hours == 24


def test_known_defect_tolerance_hours_explains_the_real_knock_on_chain() -> None:
    """The real 2026-09-12 chain: 2026-06-24 is a documented defect day,
    2026-06-25 is its knock-on (no D-1 persistence-lag value) -- both
    excluded, both explained."""
    hours = known_defect_tolerance_hours(
        {pd.Timestamp("2026-06-24").date(), pd.Timestamp("2026-06-25").date()}
    )
    assert hours == 48


def test_known_defect_tolerance_hours_does_not_explain_an_unrelated_gap() -> None:
    """A local-cache gap not in KNOWN_WEATHER_DEFECTS (the real 2025-06-13
    case found the same session) contributes zero hours -- it must still
    block check_training_extent, not be silently tolerated."""
    hours = known_defect_tolerance_hours({pd.Timestamp("2025-06-13").date()})
    assert hours == 0


def test_known_defect_tolerance_hours_does_not_explain_a_knock_on_without_its_root() -> None:
    """A knock-on day alone, without the defect day it chains from also
    present in the excluded set, is not explained -- the chain must be
    unbroken back to a documented entry."""
    hours = known_defect_tolerance_hours({pd.Timestamp("2026-06-25").date()})
    assert hours == 0


# ---------------------------------------------------------------------------
# Renewables plausibility (spec section 2.6 layer 3)
# ---------------------------------------------------------------------------


def test_capacity_factor_within_bounds_passes() -> None:
    cf = pd.Series(
        [0.0, 0.5, 1.0], index=pd.date_range("2026-09-13", periods=3, freq="h", tz="UTC")
    )
    assert check_capacity_factor_bounds(cf).ok is True


def test_capacity_factor_above_one_blocks() -> None:
    cf = pd.Series([0.5, 1.2], index=pd.date_range("2026-09-13", periods=2, freq="h", tz="UTC"))
    result = check_capacity_factor_bounds(cf)
    assert result.ok is False
    assert "capacity factor" in result.reasons[0]


def test_capacity_factor_below_zero_blocks() -> None:
    cf = pd.Series([-0.01, 0.5], index=pd.date_range("2026-09-13", periods=2, freq="h", tz="UTC"))
    assert check_capacity_factor_bounds(cf).ok is False


def test_daily_sum_within_band_passes() -> None:
    history = pd.Series([100.0, 150.0, 200.0])
    result = check_daily_sum_plausibility(160.0, historical_month_sums_mw=history)
    assert result.ok is True


def test_daily_sum_far_outside_band_blocks() -> None:
    history = pd.Series([100.0, 150.0, 200.0])
    result = check_daily_sum_plausibility(10_000.0, historical_month_sums_mw=history)
    assert result.ok is False


def test_daily_sum_unusual_but_within_margin_passes() -> None:
    """Both sides of the boundary tested separately (spec section 7)."""
    history = pd.Series([100.0, 200.0])  # margin=0.5 -> band [50, 300]
    just_inside = check_daily_sum_plausibility(299.0, historical_month_sums_mw=history)
    just_outside = check_daily_sum_plausibility(301.0, historical_month_sums_mw=history)
    assert just_inside.ok is True
    assert just_outside.ok is False


# ---------------------------------------------------------------------------
# Payload plausibility (spec section 5.6)
# ---------------------------------------------------------------------------


def test_payload_plausible_values_pass() -> None:
    values = [40.0, 55.0, 60.0, 45.0]
    result = check_payload_plausibility(values, historical_min=-50.0, historical_max=800.0)
    assert result.ok is True


def test_payload_all_identical_values_blocks() -> None:
    values = [50.0] * 96
    result = check_payload_plausibility(values, historical_min=-50.0, historical_max=800.0)
    assert result.ok is False
    assert any("identical" in r for r in result.reasons)


def test_payload_absurd_outlier_blocks() -> None:
    values = [40.0, 45.0, 5000.0, 42.0]
    result = check_payload_plausibility(values, historical_min=-50.0, historical_max=800.0)
    assert result.ok is False
    assert any("plausibility band" in r for r in result.reasons)


def test_payload_real_price_spike_within_margin_passes() -> None:
    # A genuine spike, still inside historical_max + margin.
    values = [40.0, 45.0, 950.0, 42.0]
    result = check_payload_plausibility(values, historical_min=-50.0, historical_max=800.0)
    assert result.ok is True


# ---------------------------------------------------------------------------
# Commodity staleness (spec section 2.3) -- never blocking, three cases
# individually per spec section 7: 3 days back (normal weekend, no warning),
# 5 days back (the real observed case, warning), 9 days back (beyond the
# ffill limit -- NaN and a Check B failure elsewhere, out of scope here).
# ---------------------------------------------------------------------------


def _commodity_df(last_value_days_ago: int, as_of: pd.Timestamp) -> pd.DataFrame:
    last_value_ts = as_of - pd.Timedelta(days=last_value_days_ago)
    index = pd.DatetimeIndex([last_value_ts - pd.Timedelta(hours=1), last_value_ts])
    return pd.DataFrame({"ttf_gas_eur_per_mwh": [50.0, 51.0]}, index=index)


def test_commodity_staleness_three_days_back_is_not_warned() -> None:
    as_of = pd.Timestamp("2026-09-12T10:00", tz="UTC")
    df = _commodity_df(3, as_of)
    assert check_commodity_staleness(df, as_of) == {}


def test_commodity_staleness_five_days_back_is_warned() -> None:
    as_of = pd.Timestamp("2026-09-12T10:00", tz="UTC")
    df = _commodity_df(5, as_of)
    warnings = check_commodity_staleness(df, as_of)
    assert "ttf_gas_eur_per_mwh" in warnings
    assert warnings["ttf_gas_eur_per_mwh"] == pytest.approx(5.0)


def test_commodity_staleness_nine_days_back_is_also_warned() -> None:
    # Beyond the 7-day ffill limit -- this function doesn't know or care
    # about that limit, it just reports the age; Check B (via the NaN the
    # ffill limit itself produces) is what actually blocks at this age.
    as_of = pd.Timestamp("2026-09-12T10:00", tz="UTC")
    df = _commodity_df(9, as_of)
    warnings = check_commodity_staleness(df, as_of)
    assert "ttf_gas_eur_per_mwh" in warnings
    assert warnings["ttf_gas_eur_per_mwh"] == pytest.approx(9.0)


def test_commodity_staleness_missing_column_is_silently_skipped() -> None:
    as_of = pd.Timestamp("2026-09-12T10:00", tz="UTC")
    df = pd.DataFrame({"day_ahead_price": [50.0]}, index=[as_of])
    assert check_commodity_staleness(df, as_of) == {}


def test_commodity_staleness_all_nan_column_is_silently_skipped() -> None:
    as_of = pd.Timestamp("2026-09-12T10:00", tz="UTC")
    df = pd.DataFrame({"ttf_gas_eur_per_mwh": [float("nan")]}, index=[as_of])
    assert check_commodity_staleness(df, as_of) == {}
