"""Feature matrix for the three renewables capacity-factor models (spec
6.5.2, section 5.5) -- the part of this step where most of the possible
mistakes live.

One shared feature matrix serves all three models (wind onshore, wind
offshore, solar); `columns_for(target)` is the single place that knows
which subset of columns each model actually uses.

Run assignment (step 1): for each UTC target hour t, the local delivery day
D determines the weather run via `run_init_for_target_day` (6.5.1) -- pure
calendar arithmetic, never "the most recently available run". Fail-fast: if
the run is entirely missing, or the weather artefact is missing a row for
any hour of that local day, the *whole* delivery day is dropped (spec 2.6),
not just the affected hour, and named in the returned exclusion list.

Alignment (step 2, spec 2.1): radiation variables use the artefact value at
H+1 (already the mean over [H, H+1)); instantaneous variables use the mean
of the values at H and H+1 (a trapezoidal approximation of the interval
mean); wind direction is decomposed into sin/cos *before* any averaging, at
both the temporal-alignment stage here and the spatial-aggregation stage
below -- never averaged as raw degrees (the classic 359/1 -> 180 bug).

Spatial aggregation (step 3) always runs on the already time-aligned
point values, never the reverse order.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from energy_price_forecast.data.capacity import ProductionType
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import (
    GRID_POINTS,
    HOURLY_VARIABLES,
    VARIABLE_TIME_CONVENTION,
    GridPoint,
    TimeConvention,
)
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.solar_geometry import solar_position
from energy_price_forecast.market_time import LOCAL_TZ
from energy_price_forecast.ops.windows import local_day_bounds

logger = logging.getLogger(__name__)

# Turbine thresholds, m/s -- named constants, not literals in an expression,
# because the wind-speed unit was already the error source once in 6.5.1.
WIND_RATED_SPEED_MS = 12.0
WIND_CUTOUT_SPEED_MS = 25.0

_REGIONS = ("north", "central", "south")
_POINT_VARIABLES_EXCEPT_DIRECTION = tuple(v for v in HOURLY_VARIABLES if v != "wind_direction_100m")

_ONSHORE_POINTS: tuple[GridPoint, ...] = tuple(p for p in GRID_POINTS if p.kind == "onshore")
_OFFSHORE_POINTS: tuple[GridPoint, ...] = tuple(p for p in GRID_POINTS if p.kind == "offshore")
_SOLAR_POINTS: tuple[GridPoint, ...] = tuple(
    p for p in GRID_POINTS if p.kind in ("onshore", "solar_only")
)
assert len(_ONSHORE_POINTS) == 13
assert len(_OFFSHORE_POINTS) == 3
assert len(_SOLAR_POINTS) == 15

_WIND_ONSHORE_POINT_VARS = (
    "wind_speed_100m",
    "wind_speed_10m",
    "temperature_2m",
    "surface_pressure",
)
_WIND_OFFSHORE_POINT_VARS = _WIND_ONSHORE_POINT_VARS
_SOLAR_POINT_VARS = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "cloud_cover_low",
    "temperature_2m",
)

_SHARED_CALENDAR_COLUMNS = ("hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos")

_SOLAR_REFERENCE_POINT = next(
    p for p in GRID_POINTS if p.point_id == "he_th"
)  # spec 6.5.1 F7 point
_REGION_CENTROIDS: dict[str, tuple[float, float]] = {
    region: (
        float(np.mean([p.latitude for p in GRID_POINTS if p.region == region])),
        float(np.mean([p.longitude for p in GRID_POINTS if p.region == region])),
    )
    for region in _REGIONS
}
_SOLAR_POSITION_COLUMNS = (
    "solar_elevation_ref",
    "solar_azimuth_sin_ref",
    "solar_azimuth_cos_ref",
    *[f"solar_elevation_{r}" for r in _REGIONS],
    *[f"solar_azimuth_sin_{r}" for r in _REGIONS],
    *[f"solar_azimuth_cos_{r}" for r in _REGIONS],
)


def _target_hours_for_day(day: dt.date) -> pd.DatetimeIndex:
    """UTC hourly index of local delivery day ``day`` (23, 24, or 25 hours).

    Built from the existing DST-safe `ops.windows.local_day_bounds`, never
    from `pd.Timedelta(days=1)` on a tz-aware timestamp -- the bug class
    that has already struck four times this sprint.
    """
    start, end = local_day_bounds(day)
    local_hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DatetimeIndex(local_hours.tz_convert("UTC"))


def _align_point_variables(at_h: pd.DataFrame, at_h1: pd.DataFrame) -> pd.DataFrame:
    """Time-align every point/variable column (spec 2.1) -- step 2, before
    any spatial aggregation. Wind direction is replaced by a sin/cos pair,
    averaged component-wise, never as raw degrees.
    """
    out: dict[str, np.ndarray] = {}

    for point in GRID_POINTS:
        for variable in _POINT_VARIABLES_EXCEPT_DIRECTION:
            col = f"{point.point_id}__{variable}"
            convention = VARIABLE_TIME_CONVENTION[variable][0]
            if convention == TimeConvention.MEAN_PRECEDING_HOUR:
                out[col] = at_h1[col].to_numpy(dtype="float64")
            else:
                out[col] = (
                    at_h[col].to_numpy(dtype="float64") + at_h1[col].to_numpy(dtype="float64")
                ) / 2.0

        dir_col = f"{point.point_id}__wind_direction_100m"
        sin_h = np.sin(np.radians(at_h[dir_col].to_numpy(dtype="float64")))
        cos_h = np.cos(np.radians(at_h[dir_col].to_numpy(dtype="float64")))
        sin_h1 = np.sin(np.radians(at_h1[dir_col].to_numpy(dtype="float64")))
        cos_h1 = np.cos(np.radians(at_h1[dir_col].to_numpy(dtype="float64")))
        out[f"{point.point_id}__wind_dir_sin"] = (sin_h + sin_h1) / 2.0
        out[f"{point.point_id}__wind_dir_cos"] = (cos_h + cos_h1) / 2.0

    return pd.DataFrame(out, index=at_h.index)


def assign_runs(
    target_days: Sequence[dt.date], weather: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[dt.date, ...]]:
    """Time-aligned per-point weather columns for every hour of every day
    whose weather run exists in full (spec 5.5 steps 1-2).

    Returns (aligned, excluded_days). ``aligned`` is indexed by
    (run_init_utc, valid_time_utc); ``excluded_days`` lists, in the order
    given, every day whose run was missing or incomplete (spec 2.6) -- also
    logged individually as it happens.

    An ambiguous weather row (more than one match for the same
    (run_init_utc, valid_time_utc) pair) raises rather than silently
    dropping a row: ``DataFrame.reindex`` itself raises on a duplicate-
    labelled axis, which is exactly the fail-fast behaviour needed here.
    """
    frames: list[pd.DataFrame] = []
    excluded: list[dt.date] = []

    for day in target_days:
        run_init = run_init_for_target_day(day)
        hours = _target_hours_for_day(day)

        if run_init not in weather.index.get_level_values("run_init_utc"):
            excluded.append(day)
            logger.warning("excluding %s: no weather run for run_init=%s", day, run_init)
            continue

        # .xs() on a 2-level MultiIndex with 162 columns always returns a
        # DataFrame at runtime; the stubs type it as DataFrame | Series
        # because a single-column frame could squeeze to a Series.
        run_weather = pd.DataFrame(weather.xs(run_init, level="run_init_utc"))
        at_h = run_weather.reindex(hours)
        at_h1 = run_weather.reindex(hours + pd.Timedelta(hours=1))
        if at_h.isna().to_numpy().any() or at_h1.isna().to_numpy().any():
            excluded.append(day)
            logger.warning(
                "excluding %s: incomplete weather coverage for run_init=%s", day, run_init
            )
            continue

        aligned = _align_point_variables(at_h, at_h1)
        aligned.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(hours), tz="UTC"), hours],
            names=["run_init_utc", "valid_time_utc"],
        )
        frames.append(aligned)

    if not frames:
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["run_init_utc", "valid_time_utc"])
        )
        return empty, tuple(excluded)
    return pd.concat(frames), tuple(excluded)


def _spatial_aggregates(aligned: pd.DataFrame) -> pd.DataFrame:
    """Region/fleet aggregates from already time-aligned point columns
    (spec 5.5 step 3, table). `_onshore`/`_offshore` suffixes on the two
    aggregate names both models would otherwise share (`wind100_cube_mean`,
    `wind100_frac_above_cutout`) exist only because this is one shared
    matrix, not three -- `columns_for` picks the right one per target.
    """
    out: dict[str, np.ndarray] = {}

    onshore_speed = aligned[[f"{p.point_id}__wind_speed_100m" for p in _ONSHORE_POINTS]].to_numpy()
    out["wind100_mean_all"] = onshore_speed.mean(axis=1)
    for region in _REGIONS:
        region_onshore = [p for p in _ONSHORE_POINTS if p.region == region]
        speed_cols = [f"{p.point_id}__wind_speed_100m" for p in region_onshore]
        out[f"wind100_mean_{region}"] = aligned[speed_cols].to_numpy().mean(axis=1)
        sin_cols = [f"{p.point_id}__wind_dir_sin" for p in region_onshore]
        cos_cols = [f"{p.point_id}__wind_dir_cos" for p in region_onshore]
        out[f"wind_dir_sin_{region}"] = aligned[sin_cols].to_numpy().mean(axis=1)
        out[f"wind_dir_cos_{region}"] = aligned[cos_cols].to_numpy().mean(axis=1)
    out["wind100_cube_mean_onshore"] = (onshore_speed**3).mean(axis=1)
    out["wind100_frac_above_rated_onshore"] = (onshore_speed > WIND_RATED_SPEED_MS).mean(axis=1)
    out["wind100_frac_above_cutout_onshore"] = (onshore_speed > WIND_CUTOUT_SPEED_MS).mean(axis=1)

    offshore_speed = aligned[
        [f"{p.point_id}__wind_speed_100m" for p in _OFFSHORE_POINTS]
    ].to_numpy()
    out["wind100_cube_mean_offshore"] = (offshore_speed**3).mean(axis=1)
    out["wind100_frac_above_cutout_offshore"] = (offshore_speed > WIND_CUTOUT_SPEED_MS).mean(axis=1)

    solar_ghi_all = [f"{p.point_id}__shortwave_radiation" for p in _SOLAR_POINTS]
    out["ghi_mean_all"] = aligned[solar_ghi_all].to_numpy().mean(axis=1)
    for region in _REGIONS:
        region_solar = [p for p in _SOLAR_POINTS if p.region == region]
        cols = [f"{p.point_id}__shortwave_radiation" for p in region_solar]
        out[f"ghi_mean_{region}"] = aligned[cols].to_numpy().mean(axis=1)
    cloud_cols = [f"{p.point_id}__cloud_cover" for p in _SOLAR_POINTS]
    out["cloud_cover_mean_all"] = aligned[cloud_cols].to_numpy().mean(axis=1)

    return pd.DataFrame(out, index=aligned.index)


def _calendar_features(valid_time: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-of-local-day (reused from features/calendar.py, unchanged) plus
    day-of-year, both as sin/cos (spec 5.5 step 4). No weekday, no holiday
    flags -- deliberately excluded (spec: wind and sun have no weekly
    structure, that feature would be pure overfitting risk).
    """
    calendar_features = build_calendar_features(valid_time)
    wanted = {"hour_sin", "hour_cos"}
    out = {f.name: f.values.to_numpy() for f in calendar_features if f.name in wanted}

    local = valid_time.tz_convert(LOCAL_TZ)
    day_of_year = local.dayofyear.to_numpy().astype("float64")
    days_in_year = np.where(local.is_leap_year, 366.0, 365.0)
    angle = 2.0 * np.pi * day_of_year / days_in_year
    out["day_of_year_sin"] = np.sin(angle)
    out["day_of_year_cos"] = np.cos(angle)

    return pd.DataFrame(out, index=valid_time)


def _solar_position_block(
    valid_time: pd.DatetimeIndex, latitude: float, longitude: float, suffix: str
) -> dict[str, np.ndarray]:
    position = solar_position(valid_time, latitude, longitude)
    azimuth_rad = np.radians(position["azimuth"].to_numpy())
    return {
        f"solar_elevation_{suffix}": position["elevation"].to_numpy(),
        f"solar_azimuth_sin_{suffix}": np.sin(azimuth_rad),
        f"solar_azimuth_cos_{suffix}": np.cos(azimuth_rad),
    }


def _solar_position_features(valid_time: pd.DatetimeIndex) -> pd.DataFrame:
    """Solar-only calendar-adjacent features (spec 5.5 step 4): elevation as
    a plain value (not circular), azimuth as sin/cos (circular) -- at one
    fixed reference point plus each region's point centroid, so the model
    sees the same east-west sun-angle offset across Germany's longitude
    span that 6.5.1's own wiring-probe expectation #3 measured.
    """
    out: dict[str, np.ndarray] = {}
    out.update(
        _solar_position_block(
            valid_time, _SOLAR_REFERENCE_POINT.latitude, _SOLAR_REFERENCE_POINT.longitude, "ref"
        )
    )
    for region, (lat, lon) in _REGION_CENTROIDS.items():
        out.update(_solar_position_block(valid_time, lat, lon, region))
    return pd.DataFrame(out, index=valid_time)


def build_feature_matrix(
    target_days: Sequence[dt.date], weather: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[dt.date, ...]]:
    """The shared feature matrix for all three renewables models (spec 5.5).

    Returns (features, excluded_days): features is indexed by
    (run_init_utc, valid_time_utc), float32, one row per target hour of
    every included delivery day; excluded_days are the days dropped
    entirely because their weather run was missing or incomplete.
    """
    aligned, excluded = assign_runs(target_days, weather)
    if aligned.empty:
        return aligned.astype("float32"), excluded

    valid_time = pd.DatetimeIndex(aligned.index.get_level_values("valid_time_utc"))

    aggregates = _spatial_aggregates(aligned)
    calendar = _calendar_features(valid_time)
    solar_geo = _solar_position_features(valid_time)

    aggregates.index = aligned.index
    calendar.index = aligned.index
    solar_geo.index = aligned.index

    features = pd.concat([aligned, aggregates, calendar, solar_geo], axis=1)
    return features.astype("float32"), excluded


def columns_for(target: ProductionType) -> list[str]:
    """Duplicate-free, canonically ordered column list for one model (spec
    5.5 step 5) -- the single place the column-to-model mapping lives.
    """
    if target == ProductionType.WIND_ONSHORE:
        point_cols = [
            f"{p.point_id}__{v}" for p in _ONSHORE_POINTS for v in _WIND_ONSHORE_POINT_VARS
        ]
        aggregate_cols = [
            *[f"wind_dir_sin_{r}" for r in _REGIONS],
            *[f"wind_dir_cos_{r}" for r in _REGIONS],
            "wind100_mean_all",
            *[f"wind100_mean_{r}" for r in _REGIONS],
            "wind100_cube_mean_onshore",
            "wind100_frac_above_rated_onshore",
            "wind100_frac_above_cutout_onshore",
        ]
        return [*point_cols, *aggregate_cols, *_SHARED_CALENDAR_COLUMNS]

    if target == ProductionType.WIND_OFFSHORE:
        point_cols = [
            f"{p.point_id}__{v}" for p in _OFFSHORE_POINTS for v in _WIND_OFFSHORE_POINT_VARS
        ]
        direction_cols = [
            f"{p.point_id}__wind_dir_{comp}" for p in _OFFSHORE_POINTS for comp in ("sin", "cos")
        ]
        aggregate_cols = ["wind100_cube_mean_offshore", "wind100_frac_above_cutout_offshore"]
        return [*point_cols, *direction_cols, *aggregate_cols, *_SHARED_CALENDAR_COLUMNS]

    if target == ProductionType.SOLAR:
        point_cols = [f"{p.point_id}__{v}" for p in _SOLAR_POINTS for v in _SOLAR_POINT_VARS]
        aggregate_cols = [
            "ghi_mean_all",
            *[f"ghi_mean_{r}" for r in _REGIONS],
            "cloud_cover_mean_all",
        ]
        return [*point_cols, *aggregate_cols, *_SHARED_CALENDAR_COLUMNS, *_SOLAR_POSITION_COLUMNS]

    raise ValueError(f"unknown target: {target!r}")  # pragma: no cover
