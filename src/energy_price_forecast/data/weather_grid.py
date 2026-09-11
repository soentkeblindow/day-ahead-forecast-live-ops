"""Versioned ECMWF IFS grid point set, variable list and time-convention
provenance for the weather feature pipeline (sprint 6.5).

Pure data module: no I/O, no network, no behaviour beyond validation at
import time. The point set -- ordered exactly as it travels to the
Open-Meteo Single Runs API, because grid point order is part of the
request/response contract (spike 6.5.0, F5) -- and the variable list are
both derived in docs/sprint6_step6_5_plan.md, decision E1: 3 offshore
points, 13 onshore land points chosen by wind capacity density (used by
both the wind-onshore and solar models), plus 2 land points added only for
solar. See spec 6.5.1, §5.1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import pandas as pd

GRID_VERSION: Final[str] = "grid_v1"


@dataclass(frozen=True)
class GridPoint:
    """One fixed model grid point, versioned as part of GRID_VERSION.

    Coordinates are rounded to one decimal degree on purpose: at 9 km grid
    spacing anything finer is meaningless, because Open-Meteo snaps the
    request to the nearest grid cell anyway (spec plan, E1).
    """

    point_id: str
    latitude: float
    longitude: float
    region: Literal["north", "central", "south", "offshore"]
    kind: Literal["onshore", "offshore", "solar_only"]


GRID_POINTS: Final[tuple[GridPoint, ...]] = (
    # Offshore (3) -- plan E1
    GridPoint("offshore_north_sea_west", 54.0, 6.5, "offshore", "offshore"),
    GridPoint("offshore_north_sea_east", 54.4, 7.7, "offshore", "offshore"),
    GridPoint("offshore_baltic", 54.8, 14.1, "offshore", "offshore"),
    # Land, set by onshore wind capacity density -- used by wind onshore AND solar (13)
    GridPoint("sh_west", 54.4, 9.0, "north", "onshore"),
    GridPoint("ni_northwest", 53.3, 7.5, "north", "onshore"),
    GridPoint("ni_northeast", 53.3, 9.3, "north", "onshore"),
    GridPoint("mv_west", 53.7, 12.0, "north", "onshore"),
    GridPoint("bb_north", 53.0, 13.8, "north", "onshore"),
    GridPoint("st_center", 52.1, 11.6, "north", "onshore"),
    GridPoint("ni_south", 52.4, 9.7, "north", "onshore"),
    GridPoint("nw_center", 51.6, 7.8, "central", "onshore"),
    GridPoint("he_th", 50.8, 9.8, "central", "onshore"),
    GridPoint("sn_center", 51.2, 12.8, "central", "onshore"),
    GridPoint("rp_sl", 49.8, 7.5, "south", "onshore"),
    GridPoint("by_north", 49.6, 10.5, "south", "onshore"),
    GridPoint("bw_center", 48.7, 9.3, "south", "onshore"),
    # Land, solar only (2) -- both count as region "south" for aggregates (plan E1)
    GridPoint("by_alpine_foreland", 48.0, 11.9, "south", "solar_only"),
    GridPoint("upper_rhine", 48.4, 7.9, "south", "solar_only"),
)


HOURLY_VARIABLES: Final[tuple[str, ...]] = (
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "cloud_cover_low",
)


class TimeConvention(StrEnum):
    INSTANTANEOUS = "instantaneous"
    MEAN_PRECEDING_HOUR = "mean_preceding_hour"


class ConventionProvenance(StrEnum):
    """How we know a variable's time convention.

    ASSUMED is a placeholder that must not survive: a test fails while any
    variable still carries it (spec 6.5.1, §2.2). The first draft of that
    spec presented a documentation-based inference as a measurement, citing
    F7 as evidence for variables F7 never actually covered -- this marker
    exists so that mistake can't repeat silently.
    """

    MEASURED_F7 = "measured_f7"
    MEASURED_F11 = "measured_f11"
    ASSUMED = "assumed"


# Measured, not assumed (spec 6.5.1, §5.2, F11 -- run 2026-08-31). Two
# independent, agreeing measurements per variable: (1) whether an "_instant"
# variant exists next to the default (it does only for the two radiation
# variables; requesting it for any of the other seven returns the exact same
# generic 400 as a nonexistent variable name); (2) lead 0 of a 12 UTC run
# (bright afternoon, no night ambiguity) -- both radiation variables are None
# there (no "preceding hour" exists yet), all seven others carry ordinary
# values. shortwave_radiation keeps its original spike 6.5.0, F7 provenance;
# the other eight are now MEASURED_F11. Full measurement log:
# docs/sprint6_step6_5_1_log.md, "Schritt 8".
VARIABLE_TIME_CONVENTION: Final[Mapping[str, tuple[TimeConvention, ConventionProvenance]]] = {
    "wind_speed_100m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "wind_speed_10m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "wind_direction_100m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "temperature_2m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "surface_pressure": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "shortwave_radiation": (
        TimeConvention.MEAN_PRECEDING_HOUR,
        ConventionProvenance.MEASURED_F7,
    ),
    "direct_normal_irradiance": (
        TimeConvention.MEAN_PRECEDING_HOUR,
        ConventionProvenance.MEASURED_F11,
    ),
    "cloud_cover": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
    "cloud_cover_low": (TimeConvention.INSTANTANEOUS, ConventionProvenance.MEASURED_F11),
}

assert set(VARIABLE_TIME_CONVENTION) == set(HOURLY_VARIABLES), (
    "VARIABLE_TIME_CONVENTION must cover exactly the nine HOURLY_VARIABLES, no more, no less"
)


def _cache_fingerprint(
    points: tuple[GridPoint, ...] = GRID_POINTS,
    variables: tuple[str, ...] = HOURLY_VARIABLES,
) -> str:
    """First eight hex characters of a SHA256 over the canonical
    serialisation of ``points`` (order included) and ``variables``.

    Computed at import time (from the module-level defaults) rather than
    maintained by hand, so that a forgotten version bump cannot cause
    silently mixed schemas in the cache (spec 6.5.1, §2.5b). Accepts
    explicit arguments so tests can prove the fingerprint actually reacts to
    a changed point set or variable tuple, without mutating the frozen
    module-level constants.
    """
    canonical = json.dumps(
        {
            "points": [(p.point_id, p.latitude, p.longitude) for p in points],
            "variables": list(variables),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


CACHE_KEY: Final[str] = f"{GRID_VERSION}_{_cache_fingerprint()}"


def expected_columns() -> tuple[str, ...]:
    """The exact 162 column names a cached run must carry, in canonical order."""
    return tuple(
        f"{point.point_id}__{variable}" for point in GRID_POINTS for variable in HOURLY_VARIABLES
    )


class KnownWeatherDefectCategory(StrEnum):
    """Two, and the distinction is not cosmetic (docs/sprint6_auftrag_known_data_defects.md, §2).

    PROVIDER_UNAVAILABLE: a real refetch returns ``WeatherRunUnavailable``
    (Open-Meteo's own ``modelRunUnavailable`` reason) -- no file was ever
    written for this run. Archives are occasionally backfilled, so this
    category could in principle close on its own; that is exactly why every
    entry must be re-verified before being trusted (see KNOWN_WEATHER_DEFECTS'
    own docstring).

    PROVIDER_CORRUPT: HTTP 200, but the response is near-total NaN across
    most non-radiation variables (real, reproducible measurement -- not
    every variable is NaN, but enough are that the run is unusable; see
    validate_weather_run()). Very unlikely to self-correct.
    """

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_CORRUPT = "provider_corrupt"


@dataclass(frozen=True)
class KnownWeatherDefect:
    category: KnownWeatherDefectCategory
    checked_date: str  # ISO date -- when this entry was last confirmed by a real refetch
    note: str


# Keys are the 00 UTC RUN INITIALISATION time, NOT the delivery day --
# run_init_for_target_day(D) uses the 00 UTC run of D-1, so a defect keyed
# 2026-06-23T00:00 UTC here affects delivery day 2026-06-24, one calendar
# day later. A caller indexing by delivery day must shift by one. This
# project has hit four real off-by-one/DST bugs around exactly this kind of
# date arithmetic this sprint (docs/sprint6_auftrag_known_data_defects.md,
# §2) -- this is the next likely candidate, so it is spelled out here
# rather than left implicit.
#
# Every entry was re-verified by a real fetch_run(..., use_cache=False) call
# on the date in ``checked_date``, not carried forward from an old finding
# (docs/sprint6_auftrag_known_data_defects.md, §4: "Aufgenommen wird ein
# Defekt, wenn ein Neuabruf ihn heute reproduziert."). All six re-verified
# 2026-09-11: the four PROVIDER_UNAVAILABLE runs still raise
# WeatherRunUnavailable; the two PROVIDER_CORRUPT runs still return HTTP 200
# with 100% NaN in several non-radiation variables (e.g. 2025-08-07:
# wind_speed_100m/temperature_2m/surface_pressure/cloud_cover all 1.0,
# wind_speed_10m/cloud_cover_low 0.0 -- the failure is real but partial by
# variable, which is exactly why validate_weather_run()'s per-variable check
# (any non-radiation variable with any NaN fails the whole run) is what
# actually catches it, not a single aggregate threshold).
KNOWN_WEATHER_DEFECTS: Final[dict[pd.Timestamp, KnownWeatherDefect]] = {
    pd.Timestamp("2025-08-05", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_UNAVAILABLE,
        checked_date="2026-09-11",
        note="First noticed in 6.5.1's bulk fetch (896/900 calendar days), confirmed in 6.7.1 A9.",
    ),
    pd.Timestamp("2025-08-06", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_UNAVAILABLE,
        checked_date="2026-09-11",
        note="First noticed in 6.5.1's bulk fetch (896/900 calendar days), confirmed in 6.7.1 A9.",
    ),
    pd.Timestamp("2025-08-08", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_UNAVAILABLE,
        checked_date="2026-09-11",
        note="First noticed in 6.5.1's bulk fetch (896/900 calendar days), confirmed in 6.7.1 A9.",
    ),
    pd.Timestamp("2025-08-09", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_UNAVAILABLE,
        checked_date="2026-09-11",
        note="First noticed in 6.5.1's bulk fetch (896/900 calendar days), confirmed in 6.7.1 A9.",
    ),
    pd.Timestamp("2025-08-07", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_CORRUPT,
        checked_date="2026-09-11",
        note=(
            "Found and reproduced by a real refetch in 6.7.1 A9; the cached file was "
            "physically removed there. Affects delivery day 2025-08-08."
        ),
    ),
    pd.Timestamp("2026-06-23", tz="UTC"): KnownWeatherDefect(
        category=KnownWeatherDefectCategory.PROVIDER_CORRUPT,
        checked_date="2026-09-11",
        note=(
            "Found and reproduced by a real refetch in 6.7.1 A9; the cached file was "
            "physically removed there. Affects delivery day 2026-06-24."
        ),
    ),
}
