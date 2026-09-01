"""NOAA solar position approximation: elevation and azimuth from a UTC
timestamp and a fixed latitude/longitude (spec 6.5.2, sections 2.4 and 5.4).

No new dependency: `pvlib` is the obvious package for this, but this project
pins six exact dependency versions after a free resolve broke the test suite
once (sprint 6.1, mlflow 3.11 -> 3.15), and this code runs in GitHub Actions,
where every extra dependency is another failure mode. Only two formulas are
actually needed out of a large package.

The trade for not depending on pvlib at runtime is an external accuracy
check: `tests/fixtures/solar_position_reference.csv` is a frozen set of
pvlib-computed values (several dozen timestamps across the year, across the
latitude span of the weather grid's 18 points, covering both solstices, both
equinoxes, and both DST changeover days), generated once by a throwaway
script under `scratch/` (not committed, `uv run --with pvlib ...`, same
pattern as the Open-Meteo spike in 6.5.0) and never re-run automatically.
`tests/test_solar_geometry.py` compares against it with a 0.5-degree
tolerance.

This implements the *geometric* (non-refracted) solar position -- the same
quantity pvlib's `get_solarposition()` returns in its `elevation`/`zenith`
columns, not the atmosphere-corrected `apparent_elevation`/`apparent_zenith`.
The reference fixture is built from the same non-refracted columns, so the
comparison stays apples-to-apples; refraction is not reconstructed here (it
only matters within roughly a degree of the horizon, and transcribing its
empirical correction formula from memory would be a needless source of
error for a comparison this module doesn't need to win by more than 0.5
degrees anyway).

No timezone logic anywhere in this module -- it accepts only tz-aware UTC
timestamps and raises on anything else. That is deliberate: this is the one
module in 6.5.2 that provably cannot cause a fifth DST bug this sprint,
because it contains no timezone-conversion code at all (spec section 11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_price_forecast.data.weather_grid import GRID_POINTS

# The 15 grid points that feed the solar model (spec 6.5.2 section 2.3): the
# 13 onshore land points (also used by the wind-onshore model) plus the 2
# solar-only land points. Excludes the 3 offshore points.
_SOLAR_POINTS = tuple(p for p in GRID_POINTS if p.kind in ("onshore", "solar_only"))
assert len(_SOLAR_POINTS) == 15  # spec 6.5.1 E1 -- would only change with a grid-version bump


def _require_utc(timestamps: pd.DatetimeIndex) -> None:
    if timestamps.tz is None or str(timestamps.tz) != "UTC":
        raise ValueError("timestamps must be tz-aware UTC")


def solar_position(timestamps: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Solar elevation and azimuth in degrees, NOAA approximation.

    Accepts tz-aware UTC timestamps only and raises on anything else: this
    module contains no timezone logic at all, which is why it cannot cause
    the fifth DST bug of this sprint.

    Accuracy is within 0.5 degrees of pvlib across the latitude range of the
    grid point set, verified against a frozen reference fixture (spec 2.4).
    """
    _require_utc(timestamps)

    jd = timestamps.to_julian_date().to_numpy()
    t = (jd - 2451545.0) / 36525.0  # Julian century (J2000 epoch)

    l0 = np.mod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0)  # geometric mean longitude
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)  # geometric mean anomaly
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)  # eccentricity of Earth's orbit

    m_rad = np.radians(m)
    center = (
        np.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + np.sin(3 * m_rad) * 0.000289
    )  # equation of center

    true_long = l0 + center
    app_long = (
        true_long - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * t))
    )  # apparent longitude

    mean_obliq = (
        23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    )
    obliq_corr = mean_obliq + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * t))

    declination = np.degrees(
        np.arcsin(np.sin(np.radians(obliq_corr)) * np.sin(np.radians(app_long)))
    )

    y = np.tan(np.radians(obliq_corr / 2.0)) ** 2
    equation_of_time = 4.0 * np.degrees(
        y * np.sin(2 * np.radians(l0))
        - 2 * e * np.sin(m_rad)
        + 4 * e * y * np.sin(m_rad) * np.cos(2 * np.radians(l0))
        - 0.5 * y * y * np.sin(4 * np.radians(l0))
        - 1.25 * e * e * np.sin(2 * m_rad)
    )  # minutes

    minutes_of_day = timestamps.hour * 60.0 + timestamps.minute + timestamps.second / 60.0
    # UTC has a fixed zero offset, so true solar time only needs the
    # longitude term (4 minutes per degree east) and the equation of time --
    # no local-timezone offset to add or subtract anywhere.
    true_solar_time = np.mod(minutes_of_day + equation_of_time + 4.0 * longitude, 1440.0)

    hour_angle = np.where(
        true_solar_time / 4.0 < 0.0, true_solar_time / 4.0 + 180.0, true_solar_time / 4.0 - 180.0
    )

    lat_rad = np.radians(latitude)
    declin_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    cos_zenith = np.clip(
        np.sin(lat_rad) * np.sin(declin_rad)
        + np.cos(lat_rad) * np.cos(declin_rad) * np.cos(ha_rad),
        -1.0,
        1.0,
    )
    zenith = np.degrees(np.arccos(cos_zenith))
    elevation = 90.0 - zenith

    zenith_rad = np.radians(zenith)
    azimuth_arg = np.clip(
        (np.sin(lat_rad) * np.cos(zenith_rad) - np.sin(declin_rad))
        / (np.cos(lat_rad) * np.sin(zenith_rad)),
        -1.0,
        1.0,
    )
    azimuth_base = np.degrees(np.arccos(azimuth_arg))
    azimuth = np.where(
        hour_angle > 0.0, np.mod(azimuth_base + 180.0, 360.0), np.mod(540.0 - azimuth_base, 360.0)
    )

    return pd.DataFrame({"elevation": elevation, "azimuth": azimuth}, index=timestamps)


def is_daylight_hour(hour_start_utc: pd.DatetimeIndex) -> pd.Series:
    """Daylight-hour mask for the solar model (spec 6.5.2 section 2.3).

    True for hour [H, H+1) if the solar elevation is above zero at H or at
    H+1 at any of the 15 solar-eligible grid points. Deliberately an
    interval rule, not an interval-midpoint one: the hour in which the sun
    rises or sets mid-hour genuinely has some daylight generation, and the
    interval rule never cuts that hour out.
    """
    _require_utc(hour_start_utc)

    query_points = hour_start_utc.union(hour_start_utc + pd.Timedelta(hours=1))
    any_above_horizon = pd.Series(False, index=query_points)
    for point in _SOLAR_POINTS:
        elevation = solar_position(query_points, point.latitude, point.longitude)["elevation"]
        any_above_horizon = any_above_horizon | (elevation > 0.0)

    at_h = any_above_horizon.reindex(hour_start_utc).to_numpy()
    at_h_plus_1 = any_above_horizon.reindex(hour_start_utc + pd.Timedelta(hours=1)).to_numpy()
    return pd.Series(at_h | at_h_plus_1, index=hour_start_utc, name="is_daylight_hour")
