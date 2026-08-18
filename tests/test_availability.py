import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.walkforward import walk_forward_splits
from energy_price_forecast.features.availability import (
    Availability,
    LeakageError,
    assert_no_leakage,
    availability_of,
    build_matrix,
    calendar_feature,
    combine,
    forecast_for_target,
    knowledge_time,
    lag,
)
from energy_price_forecast.market_time import gate_closure_for_index


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _raw(start: str, periods: int, name: str = "col") -> pd.Series:
    idx = _hourly_utc(start, periods)
    return pd.Series(np.arange(periods, dtype=float), index=idx, name=name)


# Berlin day 2024-01-15 (CET = UTC+1, 24 hours): gate_closure = 2024-01-14 11:00 UTC
_TARGET_JAN15 = pd.date_range("2024-01-14 23:00", periods=24, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# knowledge_time per class — hand-computed reference values
# ---------------------------------------------------------------------------


def test_knowledge_time_deterministic_is_sentinel() -> None:
    idx = _hourly_utc("2024-01-15 10:00", 4)
    result = knowledge_time(Availability.DETERMINISTIC, idx)
    expected = pd.Timestamp("1900-01-01", tz="UTC")
    assert (result == expected).all()


def test_knowledge_time_rt_actual_is_t_plus_1h() -> None:
    idx = _hourly_utc("2024-01-15 10:00", 3)
    result = knowledge_time(Availability.RT_ACTUAL, idx)
    assert (result == idx + pd.Timedelta(hours=1)).all()


def test_knowledge_time_da_fixed_is_utc_midnight() -> None:
    # UTC-anchored: normalize floors to UTC midnight of the source's UTC date.
    # 2024-01-15 10:00 UTC → 2024-01-15 00:00 UTC (not local midnight 23:00 UTC prev day).
    idx = pd.DatetimeIndex(["2024-01-15 10:00"], tz="UTC")
    result = knowledge_time(Availability.DA_FIXED, idx)
    assert result[0] == pd.Timestamp("2024-01-15 00:00", tz="UTC")


def test_knowledge_time_da_forecast_equals_gate_closure_of_value_day() -> None:
    idx = pd.DatetimeIndex(["2024-01-15 10:00"], tz="UTC")
    result = knowledge_time(Availability.DA_FORECAST, idx)
    assert (result == gate_closure_for_index(idx)).all()


def test_knowledge_time_commodity_is_next_utc_midnight() -> None:
    # Formula is UTC-anchored (not local): avoids CEST fall-DST trap where
    # local midnight (22:00 UTC) can land after gate closure (10:00 UTC).
    # 2024-01-15 10:00 UTC -> next UTC midnight = 2024-01-16 00:00 UTC.
    idx = pd.DatetimeIndex(["2024-01-15 10:00"], tz="UTC")
    result = knowledge_time(Availability.COMMODITY, idx)
    assert result[0] == pd.Timestamp("2024-01-16 00:00", tz="UTC")


# ---------------------------------------------------------------------------
# availability_of — registry and prefix rules
# ---------------------------------------------------------------------------


def test_availability_of_registered_columns() -> None:
    assert availability_of("day_ahead_price") is Availability.DA_FIXED
    assert availability_of("load_actual") is Availability.RT_ACTUAL
    assert availability_of("ttf_gas_eur_per_mwh") is Availability.COMMODITY
    assert availability_of("eua_co2_eur_per_t") is Availability.COMMODITY
    assert availability_of("load_forecast_day_ahead") is Availability.DA_FORECAST
    assert availability_of("wind_onshore_forecast") is Availability.DA_FORECAST
    assert availability_of("gen_nuclear") is Availability.RT_ACTUAL
    assert availability_of("gen_solar") is Availability.RT_ACTUAL


def test_availability_of_scheduled_prefix() -> None:
    assert availability_of("scheduled_net_de_to_at") is Availability.DA_FIXED
    assert availability_of("scheduled_net_de_to_fr") is Availability.DA_FIXED


def test_availability_of_physical_prefix() -> None:
    assert availability_of("physical_net_de_to_at") is Availability.RT_ACTUAL
    assert availability_of("physical_net_de_to_nl") is Availability.RT_ACTUAL


def test_availability_of_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown_column"):
        availability_of("unknown_column")


# ---------------------------------------------------------------------------
# lag — core leakage regression
# ---------------------------------------------------------------------------


def test_price_lag_24h_passes() -> None:
    raw = _raw("2024-01-13 23:00", 24, "day_ahead_price")
    f = lag("price_lag_24h", raw, "day_ahead_price", hours=24, target_index=_TARGET_JAN15)
    assert_no_leakage([f])


def test_load_actual_lag_24h_fails_afternoon() -> None:
    # RT_ACTUAL: value at t-24h is known at t-23h; gate = 2024-01-14 11:00 UTC.
    # Targets from 2024-01-15 11:00 UTC onward have knowledge_time > gate_closure.
    raw = _raw("2024-01-13 23:00", 24, "load_actual")
    f = lag("load_actual_lag_24h", raw, "load_actual", hours=24, target_index=_TARGET_JAN15)
    with pytest.raises(LeakageError, match="load_actual_lag_24h"):
        assert_no_leakage([f])


def test_load_actual_lag_48h_passes() -> None:
    raw = _raw("2024-01-12 23:00", 48, "load_actual")
    f = lag("load_actual_lag_48h", raw, "load_actual", hours=48, target_index=_TARGET_JAN15)
    assert_no_leakage([f])


def test_lag_hours_zero_raises() -> None:
    raw = _raw("2024-01-14 00:00", 24, "load_actual")
    with pytest.raises(ValueError, match="lag hours"):
        lag("bad", raw, "load_actual", hours=0, target_index=_TARGET_JAN15)


# ---------------------------------------------------------------------------
# forecast_for_target
# ---------------------------------------------------------------------------


def test_forecast_for_target_da_forecast_passes_by_equality() -> None:
    raw = _raw("2024-01-14 23:00", 24, "load_forecast_day_ahead")
    f = forecast_for_target("load_fc", raw, "load_forecast_day_ahead", target_index=_TARGET_JAN15)
    assert_no_leakage([f])


def test_forecast_for_target_non_forecast_column_raises() -> None:
    raw = _raw("2024-01-14 23:00", 24, "day_ahead_price")
    with pytest.raises(ValueError, match="DA_FORECAST"):
        forecast_for_target("price_fc", raw, "day_ahead_price", target_index=_TARGET_JAN15)


# ---------------------------------------------------------------------------
# combine — knowledge_time is elementwise MAX
# ---------------------------------------------------------------------------


def test_combine_knowledge_time_is_max() -> None:
    raw_fc = _raw("2024-01-14 23:00", 24, "load_forecast_day_ahead")
    raw_price = _raw("2024-01-13 23:00", 24, "day_ahead_price")
    f_fc = forecast_for_target(
        "load_fc", raw_fc, "load_forecast_day_ahead", target_index=_TARGET_JAN15
    )
    f_price = lag(
        "price_lag_24h", raw_price, "day_ahead_price", hours=24, target_index=_TARGET_JAN15
    )
    f_combined = combine("spread", [f_fc, f_price], lambda a, b: a - b)
    expected_kt = pd.concat([f_fc.knowledge_time, f_price.knowledge_time], axis=1).max(axis=1)
    pd.testing.assert_series_equal(f_combined.knowledge_time, expected_kt)


def test_combine_empty_parts_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine("bad", [], lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------


def test_build_matrix_assembles_correct_columns() -> None:
    raw_price = _raw("2024-01-13 23:00", 24, "day_ahead_price")
    f1 = lag("price_lag_24h", raw_price, "day_ahead_price", hours=24, target_index=_TARGET_JAN15)
    raw_hour = pd.Series(_TARGET_JAN15.hour, index=_TARGET_JAN15, dtype=float)
    f2 = calendar_feature("hour_of_day", raw_hour)
    x = build_matrix([f1, f2])
    assert list(x.columns) == ["price_lag_24h", "hour_of_day"]
    assert len(x) == 24


def test_build_matrix_raises_leakage_error_naming_feature() -> None:
    raw = _raw("2024-01-13 23:00", 24, "load_actual")
    f = lag("load_actual_lag_24h", raw, "load_actual", hours=24, target_index=_TARGET_JAN15)
    with pytest.raises(LeakageError, match="load_actual_lag_24h"):
        build_matrix([f])


# ---------------------------------------------------------------------------
# Structural invariant note
# ---------------------------------------------------------------------------
# Feature has no default for knowledge_time; it is impossible to construct one
# without supplying it. Use the constructors (calendar_feature, lag,
# forecast_for_target, combine) — they are the public API and the sole path
# into build_matrix.


# ---------------------------------------------------------------------------
# Gate-closure consistency with walk_forward_splits (including DST)
# ---------------------------------------------------------------------------


def test_gate_closure_consistent_winter() -> None:
    idx = pd.date_range("2024-01-01 23:00", "2024-01-15 22:00", freq="h", tz="UTC")
    folds = list(walk_forward_splits(idx, test_start="2024-01-05", test_end="2024-01-14"))
    assert len(folds) > 0
    for fold in folds:
        result = gate_closure_for_index(fold.test_index)
        # Memory note: use .all(), not direct equality (DatetimeIndex vs scalar Timestamp)
        assert (result == fold.gate_closure).all(), f"Mismatch on {fold.delivery_day}"


def test_gate_closure_consistent_summer() -> None:
    idx = pd.date_range("2024-06-01 22:00", "2024-06-15 21:00", freq="h", tz="UTC")
    folds = list(walk_forward_splits(idx, test_start="2024-06-05", test_end="2024-06-14"))
    assert len(folds) > 0
    for fold in folds:
        result = gate_closure_for_index(fold.test_index)
        assert (result == fold.gate_closure).all(), f"Mismatch on {fold.delivery_day}"


def test_gate_closure_consistent_dst_spring_forward() -> None:
    # 2024-03-31: clocks spring forward, 23-hour day
    idx = pd.date_range("2024-03-28 23:00", "2024-04-01 21:00", freq="h", tz="UTC")
    folds = list(walk_forward_splits(idx, test_start="2024-03-31", test_end="2024-03-31"))
    assert len(folds) == 1
    fold = folds[0]
    assert len(fold.test_index) == 23
    result = gate_closure_for_index(fold.test_index)
    assert (result == fold.gate_closure).all()


def test_gate_closure_consistent_dst_fall_back() -> None:
    # 2024-10-27: clocks fall back, 25-hour day
    idx = pd.date_range("2024-10-24 22:00", "2024-10-28 22:00", freq="h", tz="UTC")
    folds = list(walk_forward_splits(idx, test_start="2024-10-27", test_end="2024-10-27"))
    assert len(folds) == 1
    fold = folds[0]
    assert len(fold.test_index) == 25
    result = gate_closure_for_index(fold.test_index)
    assert (result == fold.gate_closure).all()
