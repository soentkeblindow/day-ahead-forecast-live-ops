"""Unit tests for arena/preflight.py -- Check A, Check B, the training-
extent check, and the plausibility bands (spec 6.7.2, sections 2.1-2.6,
5.2, 5.5, 7).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import energy_price_forecast.arena.preflight as preflight_module
from energy_price_forecast.arena.preflight import (
    CAPACITY_ANCHOR_WARN_DAYS,
    MAX_RNW_LABEL_EDGE_AGE_DAYS,
    MAX_TRAINING_EDGE_AGE_DAYS,
    MIN_TRAINING_DAYS,
    check_capacity_anchor,
    check_capacity_factor_bounds,
    check_commodity_staleness,
    check_daily_sum_plausibility,
    check_holiday_calendar,
    check_payload_plausibility,
    check_renewables_label_edge_age,
    check_target_row,
    check_training_window,
    check_weather_run,
)
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES
from energy_price_forecast.ops.windows import local_day_bounds

_RADIATION = {"shortwave_radiation", "direct_normal_irradiance"}


def _valid_weather_run() -> pd.DataFrame:
    index = pd.date_range("2026-09-12T00:00", periods=24, freq="h", tz="UTC")
    data = {}
    for point in GRID_POINTS:
        for variable in HOURLY_VARIABLES:
            data[f"{point.point_id}__{variable}"] = [10.0] * 24
    return pd.DataFrame(data, index=index, dtype="float32")


# ---------------------------------------------------------------------------
# Check A, split per spec 6.9 section 2.3: check_holiday_calendar (global),
# check_weather_run/check_capacity_anchor (per-row, NWP-only).
# ---------------------------------------------------------------------------

_TARGET_DAY = pd.Timestamp("2026-09-13").date()


def test_check_holiday_calendar_passes_for_a_covered_year() -> None:
    result = check_holiday_calendar(target_day=_TARGET_DAY, holiday_calendar_covers_target=True)
    assert result.ok is True
    assert result.reasons == ()


def test_check_holiday_calendar_fails_when_not_covered() -> None:
    result = check_holiday_calendar(target_day=_TARGET_DAY, holiday_calendar_covers_target=False)
    assert result.ok is False
    assert any("holiday calendar" in r for r in result.reasons)


def test_check_weather_run_passes_with_a_valid_run() -> None:
    assert check_weather_run(_valid_weather_run()).ok is True


def test_check_weather_run_fails_when_none() -> None:
    result = check_weather_run(None)
    assert result.ok is False
    assert any("weather run" in r for r in result.reasons)


def test_check_weather_run_fails_when_invalid() -> None:
    bad_run = _valid_weather_run()
    non_radiation_col = f"{GRID_POINTS[0].point_id}__wind_speed_10m"
    bad_run[non_radiation_col] = float("nan")
    result = check_weather_run(bad_run)
    assert result.ok is False
    assert any("failed validation" in r for r in result.reasons)


def test_check_capacity_anchor_passes_and_no_warning_when_far_from_expiry() -> None:
    result, report = check_capacity_anchor(_TARGET_DAY, pd.Timestamp("2026-10-28", tz="UTC"))
    assert result.ok is True
    assert report.warning is False


def test_check_capacity_anchor_warns_at_exactly_the_warn_window() -> None:
    """spec 6.9 section 2.3: 'Ab 14 Tagen vor Ablauf: Warnung.'"""
    run_init = run_init_for_target_day(_TARGET_DAY)
    result, report = check_capacity_anchor(
        _TARGET_DAY, run_init + pd.Timedelta(days=CAPACITY_ANCHOR_WARN_DAYS)
    )
    assert result.ok is True
    assert report.warning is True


def test_check_capacity_anchor_no_warning_one_day_beyond_the_warn_window() -> None:
    run_init = run_init_for_target_day(_TARGET_DAY)
    result, report = check_capacity_anchor(
        _TARGET_DAY, run_init + pd.Timedelta(days=CAPACITY_ANCHOR_WARN_DAYS + 1)
    )
    assert result.ok is True
    assert report.warning is False


def test_check_capacity_anchor_fails_when_expired() -> None:
    result, report = check_capacity_anchor(_TARGET_DAY, pd.Timestamp("2026-09-01", tz="UTC"))
    assert result.ok is False
    assert any("anchor table" in r for r in result.reasons)
    assert report.warning is False  # already expired, not merely approaching


def test_known_defect_tolerance_hours_and_the_old_check_a_functions_are_removed() -> None:
    """spec 6.9 section 2.6: known_defect_tolerance_hours and its chain
    logic are removed, not patched -- and check_reconstruction_inputs/
    check_training_extent are replaced by the split functions above."""
    for name in (
        "known_defect_tolerance_hours",
        "check_training_extent",
        "check_reconstruction_inputs",
    ):
        assert not hasattr(preflight_module, name)


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
# check_training_window (spec 6.9 section 2.6/5.4) -- replaces the former
# check_training_extent/known_defect_tolerance_hours.
# ---------------------------------------------------------------------------


def _day_hours(day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(day)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")


def _full_window(
    must_reach: dt.date, window_days: int, value: float = 1.0
) -> tuple[pd.DataFrame, pd.Series]:
    """A completely gap-free window_days-day window ending at must_reach,
    one required feature column ("feat_a") plus a matching price label,
    both constant-filled -- DST-free days only (a July window, matching
    this project's existing test convention), so every day contributes
    exactly 24 hours and day counts are simple integers."""
    days = [must_reach - dt.timedelta(days=window_days - 1 - i) for i in range(window_days)]
    hours = _day_hours(days[0]).append([_day_hours(d) for d in days[1:]])
    features = pd.DataFrame({"feat_a": value}, index=hours)
    labels = pd.Series(50.0, index=hours)
    return features, labels


def _drop_days(
    features: pd.DataFrame, labels: pd.Series, days: list[dt.date]
) -> tuple[pd.DataFrame, pd.Series]:
    """Removes entire days' worth of rows -- simulates a day missing from
    the built matrix/price history entirely, not a NaN cell."""
    drop_hours = _day_hours(days[0]).append([_day_hours(d) for d in days[1:]])
    return features.drop(index=drop_hours), labels.drop(index=drop_hours)


_JULY_MUST_REACH = dt.date(2026, 7, 14)  # DST-free week, matches tests/test_run_daily_submission.py


def test_training_window_passes_at_exactly_min_training_days() -> None:
    features, labels = _full_window(_JULY_MUST_REACH, 90)
    n_to_drop = 90 - MIN_TRAINING_DAYS
    drop_days = [_JULY_MUST_REACH - dt.timedelta(days=10 + i) for i in range(n_to_drop)]
    features, labels = _drop_days(features, labels, drop_days)

    result, report = check_training_window(
        features, labels, frozenset({"feat_a"}), must_reach=_JULY_MUST_REACH, window_days=90
    )

    assert result.ok is True
    assert report.n_training_days == MIN_TRAINING_DAYS


def test_training_window_fails_one_day_below_min_training_days() -> None:
    features, labels = _full_window(_JULY_MUST_REACH, 90)
    n_to_drop = 90 - MIN_TRAINING_DAYS + 1
    drop_days = [_JULY_MUST_REACH - dt.timedelta(days=10 + i) for i in range(n_to_drop)]
    features, labels = _drop_days(features, labels, drop_days)

    result, report = check_training_window(
        features, labels, frozenset({"feat_a"}), must_reach=_JULY_MUST_REACH, window_days=90
    )

    assert result.ok is False
    assert report.n_training_days == MIN_TRAINING_DAYS - 1
    assert any("complete training day" in r for r in result.reasons)


def test_training_window_passes_at_exactly_max_edge_age() -> None:
    # A wide window (97 days) so dropping edge days never also trips the
    # separate n_training_days floor -- isolates the age dimension.
    window_days = 97
    features, labels = _full_window(_JULY_MUST_REACH, window_days)
    drop_days = [_JULY_MUST_REACH - dt.timedelta(days=i) for i in range(MAX_TRAINING_EDGE_AGE_DAYS)]
    features, labels = _drop_days(features, labels, drop_days)

    result, report = check_training_window(
        features,
        labels,
        frozenset({"feat_a"}),
        must_reach=_JULY_MUST_REACH,
        window_days=window_days,
    )

    assert result.ok is True
    assert report.age_of_last_complete_day == MAX_TRAINING_EDGE_AGE_DAYS


def test_training_window_fails_one_day_beyond_max_edge_age() -> None:
    window_days = 97
    features, labels = _full_window(_JULY_MUST_REACH, window_days)
    drop_days = [
        _JULY_MUST_REACH - dt.timedelta(days=i) for i in range(MAX_TRAINING_EDGE_AGE_DAYS + 1)
    ]
    features, labels = _drop_days(features, labels, drop_days)

    result, report = check_training_window(
        features,
        labels,
        frozenset({"feat_a"}),
        must_reach=_JULY_MUST_REACH,
        window_days=window_days,
    )

    assert result.ok is False
    assert report.age_of_last_complete_day == MAX_TRAINING_EDGE_AGE_DAYS + 1
    assert any("last complete training day" in r for r in result.reasons)


def test_training_window_a_missing_label_makes_its_day_incomplete() -> None:
    features, labels = _full_window(_JULY_MUST_REACH, 90)
    gap_day = _JULY_MUST_REACH - dt.timedelta(days=20)
    labels = labels.copy()
    labels.loc[_day_hours(gap_day)] = float("nan")

    result, report = check_training_window(
        features, labels, frozenset({"feat_a"}), must_reach=_JULY_MUST_REACH, window_days=90
    )

    assert gap_day in report.missing_days
    assert report.n_training_days == 89


def test_training_window_ignores_nan_in_a_column_not_required() -> None:
    features, labels = _full_window(_JULY_MUST_REACH, 90)
    features = features.copy()
    features["unused_col"] = 1.0
    gap_day = _JULY_MUST_REACH - dt.timedelta(days=20)
    features.loc[_day_hours(gap_day), "unused_col"] = float("nan")

    result, report = check_training_window(
        features, labels, frozenset({"feat_a"}), must_reach=_JULY_MUST_REACH, window_days=90
    )

    assert result.ok is True
    assert report.n_training_days == 90
    assert gap_day not in report.missing_days


def test_training_window_reports_a_known_weather_defect_day_but_grants_no_tolerance() -> None:
    """spec 6.9 section 2.6: KNOWN_WEATHER_DEFECTS is reporting-only now --
    2026-06-24 (D-1 run_init 2026-06-23 is a documented corrupt-run entry)
    still counts against n_training_days like any other missing day, it is
    just additionally named in known_weather_defect_days."""
    features, labels = _full_window(_JULY_MUST_REACH, 90)
    defect_day = dt.date(2026, 6, 24)
    features, labels = _drop_days(features, labels, [defect_day])

    result, report = check_training_window(
        features, labels, frozenset({"feat_a"}), must_reach=_JULY_MUST_REACH, window_days=90
    )

    assert result.ok is True  # 89 of 90 -- well within MIN_TRAINING_DAYS regardless of cause
    assert defect_day in report.missing_days
    assert defect_day in report.known_weather_defect_days


# ---------------------------------------------------------------------------
# check_renewables_label_edge_age (spec 6.9 section 2.7)
# ---------------------------------------------------------------------------

_RNW_TARGET_COLUMNS = ("wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast")


def _renewables_window(must_reach: dt.date, n_days: int) -> pd.DataFrame:
    days = [must_reach - dt.timedelta(days=n_days - 1 - i) for i in range(n_days)]
    hours = _day_hours(days[0]).append([_day_hours(d) for d in days[1:]])
    return pd.DataFrame({col: 1.0 for col in _RNW_TARGET_COLUMNS}, index=hours)


def test_renewables_label_edge_age_zero_when_must_reach_is_complete() -> None:
    target_hourly = _renewables_window(_JULY_MUST_REACH, 30)
    result, age = check_renewables_label_edge_age(target_hourly, must_reach=_JULY_MUST_REACH)
    assert age == 0
    assert result.ok is True


def test_renewables_label_edge_age_passes_at_exactly_the_limit() -> None:
    target_hourly = _renewables_window(_JULY_MUST_REACH, 30)
    for i in range(MAX_RNW_LABEL_EDGE_AGE_DAYS):
        day = _JULY_MUST_REACH - dt.timedelta(days=i)
        target_hourly.loc[_day_hours(day), :] = float("nan")

    result, age = check_renewables_label_edge_age(target_hourly, must_reach=_JULY_MUST_REACH)

    assert age == MAX_RNW_LABEL_EDGE_AGE_DAYS
    assert result.ok is True


def test_renewables_label_edge_age_fails_one_day_beyond_the_limit() -> None:
    target_hourly = _renewables_window(_JULY_MUST_REACH, 30)
    for i in range(MAX_RNW_LABEL_EDGE_AGE_DAYS + 1):
        day = _JULY_MUST_REACH - dt.timedelta(days=i)
        target_hourly.loc[_day_hours(day), :] = float("nan")

    result, age = check_renewables_label_edge_age(target_hourly, must_reach=_JULY_MUST_REACH)

    assert age == MAX_RNW_LABEL_EDGE_AGE_DAYS + 1
    assert result.ok is False
    assert any("last complete renewables label day" in r for r in result.reasons)


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
