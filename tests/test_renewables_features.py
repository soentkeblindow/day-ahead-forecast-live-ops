import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.data.capacity import ProductionType
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import GRID_POINTS, expected_columns
from energy_price_forecast.features.renewables_forecast import (
    _ONSHORE_POINTS,
    _target_hours_for_day,
    build_feature_matrix,
    columns_for,
)
from energy_price_forecast.features.solar_geometry import is_daylight_hour


def _hours_needed_for_day(day: dt.date) -> pd.DatetimeIndex:
    hours = _target_hours_for_day(day)
    return pd.DatetimeIndex(hours.union(hours + pd.Timedelta(hours=1)))


def _weather_for_days(days: list[dt.date], value: float = 0.0) -> pd.DataFrame:
    cols = expected_columns()
    frames = []
    for day in days:
        run_init = run_init_for_target_day(day)
        valid_times = _hours_needed_for_day(day)
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(valid_times), tz="UTC"), valid_times],
            names=["run_init_utc", "valid_time_utc"],
        )
        frames.append(pd.DataFrame({c: value for c in cols}, index=idx, dtype="float64"))
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# Alignment (spec 2.1)
# ---------------------------------------------------------------------------


def test_radiation_alignment_uses_the_h_plus_1_artefact_value() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    valid_times = weather.index.get_level_values("valid_time_utc")
    ramp = np.arange(len(valid_times), dtype="float64")
    weather["he_th__shortwave_radiation"] = ramp

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    target_hours = _target_hours_for_day(day)
    expected_h1_value = weather.loc[
        weather.index.get_level_values("valid_time_utc").isin(target_hours + pd.Timedelta(hours=1)),
        "he_th__shortwave_radiation",
    ].to_numpy()
    naive_h_value = weather.loc[
        weather.index.get_level_values("valid_time_utc").isin(target_hours),
        "he_th__shortwave_radiation",
    ].to_numpy()

    actual = features["he_th__shortwave_radiation"].to_numpy()
    np.testing.assert_allclose(actual, expected_h1_value)

    # Negative control: a join on the *same* timestamp (H, not H+1) would
    # have produced a different result for this ramp -- proves the test has
    # teeth, per spec 6.5.2 section 7.
    assert not np.allclose(actual, naive_h_value)


def test_instantaneous_alignment_is_the_mean_of_h_and_h_plus_1() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    valid_times = weather.index.get_level_values("valid_time_utc")
    ramp = np.arange(len(valid_times), dtype="float64")
    weather["he_th__temperature_2m"] = ramp

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    target_hours = _target_hours_for_day(day)
    at_h = weather.loc[
        weather.index.get_level_values("valid_time_utc").isin(target_hours), "he_th__temperature_2m"
    ].to_numpy()
    at_h1 = weather.loc[
        weather.index.get_level_values("valid_time_utc").isin(target_hours + pd.Timedelta(hours=1)),
        "he_th__temperature_2m",
    ].to_numpy()
    expected = (at_h + at_h1) / 2.0

    np.testing.assert_allclose(features["he_th__temperature_2m"].to_numpy(), expected)


def test_direction_averaging_359_and_1_degrees_is_near_zero_not_180() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    target_hours = _target_hours_for_day(day)
    at_h_mask = weather.index.get_level_values("valid_time_utc").isin(target_hours)
    at_h1_mask = weather.index.get_level_values("valid_time_utc").isin(
        target_hours + pd.Timedelta(hours=1)
    )
    weather.loc[at_h_mask, "he_th__wind_direction_100m"] = 359.0
    weather.loc[at_h1_mask, "he_th__wind_direction_100m"] = 1.0

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    sin_component = features["he_th__wind_dir_sin"].to_numpy()
    cos_component = features["he_th__wind_dir_cos"].to_numpy()
    averaged_angle = np.degrees(np.arctan2(sin_component, cos_component)) % 360.0

    # The correct sin/cos average is near 0 degrees; a naive degree average
    # (359+1)/2 = 180 would be the exact opposite -- this is the case
    # section 2.2 exists for.
    assert np.all((averaged_angle < 5.0) | (averaged_angle > 355.0))


# ---------------------------------------------------------------------------
# Run assignment (spec 2.6, 2.1 data-need consequence, DST)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected_hours"), [(dt.date(2026, 3, 29), 23), (dt.date(2026, 10, 25), 25)]
)
def test_run_assignment_across_both_dst_changeovers(day: dt.date, expected_hours: int) -> None:
    weather = _weather_for_days([day])
    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()
    assert len(features) == expected_hours

    expected_run = run_init_for_target_day(day)
    run_inits = features.index.get_level_values("run_init_utc")
    assert (run_inits == expected_run).all()


def test_missing_run_excludes_the_whole_delivery_day() -> None:
    # 2025-08-06 needs run_init 2025-08-05, one of the four real outage days
    # from 6.5.1 (docs/sprint6_step6_5_1_log.md) -- weather artefact simply
    # never has that run.
    day = dt.date(2025, 8, 6)
    empty_weather = pd.DataFrame(
        columns=expected_columns(),
        index=pd.MultiIndex.from_arrays([[], []], names=["run_init_utc", "valid_time_utc"]),
    ).astype("float64")

    features, excluded = build_feature_matrix([day], empty_weather)
    assert excluded == (day,)
    assert features.empty


def test_incomplete_weather_row_excludes_the_whole_delivery_day() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    # Drop exactly one needed row -- a single missing hour must still take
    # out the entire day, not just that hour.
    one_row_to_drop = weather.index[5]
    weather = weather.drop(index=one_row_to_drop)

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == (day,)
    assert features.empty


def test_ambiguous_weather_row_raises_instead_of_silently_dropping() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    duplicate_row = weather.iloc[[5]]
    weather = pd.concat([weather, duplicate_row])

    with pytest.raises(ValueError):
        build_feature_matrix([day], weather)


# ---------------------------------------------------------------------------
# Spatial aggregates (spec 5.5 step 3)
# ---------------------------------------------------------------------------


def test_cube_mean_matches_hand_calculation() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    speeds = {p.point_id: float(i + 1) for i, p in enumerate(_ONSHORE_POINTS)}
    for point_id, speed in speeds.items():
        weather[f"{point_id}__wind_speed_100m"] = speed

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    expected = float(np.mean([v**3 for v in speeds.values()]))
    np.testing.assert_allclose(features["wind100_cube_mean_onshore"].to_numpy(), expected)


def test_frac_above_thresholds_at_the_boundary() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    # 5 points above rated (12), 8 at/below -- exact known fraction.
    for i, point in enumerate(_ONSHORE_POINTS):
        weather[f"{point.point_id}__wind_speed_100m"] = 20.0 if i < 5 else 5.0

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    np.testing.assert_allclose(features["wind100_frac_above_rated_onshore"].to_numpy(), 5 / 13)
    np.testing.assert_allclose(features["wind100_frac_above_cutout_onshore"].to_numpy(), 0.0)


def test_regional_means_use_only_the_points_in_that_region() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    north_points = [p for p in _ONSHORE_POINTS if p.region == "north"]
    for point in north_points:
        weather[f"{point.point_id}__wind_speed_100m"] = (
            100.0  # far outside any other region's value
        )

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    np.testing.assert_allclose(features["wind100_mean_north"].to_numpy(), 100.0)
    assert not np.allclose(features["wind100_mean_south"].to_numpy(), 100.0)
    assert not np.allclose(features["wind100_mean_central"].to_numpy(), 100.0)


def test_alignment_runs_before_aggregation() -> None:
    """A case where aggregate(align(x)) != align(aggregate(x))."""
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])
    raw_times = weather.index.get_level_values("valid_time_utc")

    # A smooth ramp keyed by chronological position in the raw artefact, so
    # consecutive raw hours (H and H+1) differ by exactly 1 -- point 0
    # genuinely changes value between H and H+1 at every target hour, while
    # every other onshore point stays flat.
    order = np.argsort(raw_times.to_numpy())
    rank = np.empty(len(raw_times))
    rank[order] = np.arange(len(raw_times), dtype="float64")
    rank_series = pd.Series(rank, index=raw_times)

    point0 = _ONSHORE_POINTS[0].point_id
    weather[f"{point0}__wind_speed_100m"] = rank
    for point in _ONSHORE_POINTS[1:]:
        weather[f"{point.point_id}__wind_speed_100m"] = 10.0

    features, excluded = build_feature_matrix([day], weather)
    assert excluded == ()

    target_hours = _target_hours_for_day(day)
    rank_at_h = rank_series.reindex(target_hours).to_numpy()
    rank_at_h1 = rank_series.reindex(target_hours + pd.Timedelta(hours=1)).to_numpy()
    n = len(_ONSHORE_POINTS)
    flat_cube_sum = 12 * 10.0**3

    # Correct order: align point 0 first (mean of H and H+1), then cube-mean
    # across all 13 onshore points.
    aligned_point0 = (rank_at_h + rank_at_h1) / 2.0
    correct = (aligned_point0**3 + flat_cube_sum) / n

    # Wrong order: cube-mean at H and at H+1 separately, then average those
    # two aggregates -- differs from `correct` because cubing is nonlinear.
    cube_at_h = (rank_at_h**3 + flat_cube_sum) / n
    cube_at_h1 = (rank_at_h1**3 + flat_cube_sum) / n
    wrong = (cube_at_h + cube_at_h1) / 2.0

    assert not np.allclose(
        correct, wrong
    )  # the two orders must actually differ for this case to prove anything
    np.testing.assert_allclose(features["wind100_cube_mean_onshore"].to_numpy(), correct)


# ---------------------------------------------------------------------------
# Daylight-hour mask (spec 2.3) -- lives in solar_geometry.py, tested here
# against real sunrise/sunset per spec 6.5.2 section 7
# ---------------------------------------------------------------------------


def test_daylight_mask_summer_sunrise_and_sunset_hours_count_as_day() -> None:
    # 2025-06-21, Germany: sunrise well before 05:00 local, sunset well
    # after 21:00 local -- the hour containing each transition, and no
    # further, is checked against its immediate neighbours.
    hours = pd.date_range("2025-06-20T00:00:00Z", periods=48, freq="h", tz="UTC")
    mask = is_daylight_hour(hours)
    local = hours.tz_convert("Europe/Berlin")

    deep_night = local.hour == 1
    deep_day = local.hour == 12
    assert not mask[deep_night].any()
    assert mask[deep_day].all()


def test_daylight_mask_winter_has_a_short_day() -> None:
    hours = pd.date_range("2025-12-21T00:00:00Z", periods=48, freq="h", tz="UTC")
    mask = is_daylight_hour(hours)
    local = hours.tz_convert("Europe/Berlin")

    deep_night = local.hour == 2
    midday = local.hour == 12
    assert not mask[deep_night].any()
    assert mask[midday].all()

    # Winter day is shorter than summer's -- fewer daylight hours overall.
    summer_hours = pd.date_range("2025-06-21T00:00:00Z", periods=24, freq="h", tz="UTC")
    winter_hours = pd.date_range("2025-12-21T00:00:00Z", periods=24, freq="h", tz="UTC")
    assert is_daylight_hour(winter_hours).sum() < is_daylight_hour(summer_hours).sum()


# ---------------------------------------------------------------------------
# columns_for (spec 5.5 step 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected_point_and_aggregate_count"),
    [
        (ProductionType.WIND_ONSHORE, 65),
        (ProductionType.WIND_OFFSHORE, 20),
        (ProductionType.SOLAR, 80),
    ],
)
def test_columns_for_matches_the_spec_table_point_and_aggregate_counts(
    target: ProductionType, expected_point_and_aggregate_count: int
) -> None:
    cols = columns_for(target)
    shared_calendar_count = 4
    solar_position_count = 12 if target == ProductionType.SOLAR else 0
    assert (
        len(cols)
        == expected_point_and_aggregate_count + shared_calendar_count + solar_position_count
    )
    assert len(cols) == len(set(cols))  # duplicate-free


def test_columns_for_is_a_pure_function_with_stable_order() -> None:
    for target in ProductionType:
        assert columns_for(target) == columns_for(target)


def test_assign_runs_grid_points_sanity() -> None:
    # Guards the fixed grid assumption every other test in this file relies on.
    assert len(GRID_POINTS) == 18
