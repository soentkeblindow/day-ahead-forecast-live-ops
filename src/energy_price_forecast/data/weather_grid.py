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


# Provisional conventions, pending F11 (spec 6.5.1, §5.2). Only
# shortwave_radiation is an actual measurement (spike 6.5.0, F7): a timestamp
# T describes the mean over the hour preceding T. The other eight are a
# documentation-based inference, not a measurement -- flux quantities
# (radiation) are averaged over the preceding hour, state quantities
# (temperature, wind, pressure, cloud cover) are instantaneous at T -- and
# are marked ASSUMED until F11 measures each of them directly.
VARIABLE_TIME_CONVENTION: Final[Mapping[str, tuple[TimeConvention, ConventionProvenance]]] = {
    "wind_speed_100m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "wind_speed_10m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "wind_direction_100m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "temperature_2m": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "surface_pressure": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "shortwave_radiation": (
        TimeConvention.MEAN_PRECEDING_HOUR,
        ConventionProvenance.MEASURED_F7,
    ),
    "direct_normal_irradiance": (TimeConvention.MEAN_PRECEDING_HOUR, ConventionProvenance.ASSUMED),
    "cloud_cover": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
    "cloud_cover_low": (TimeConvention.INSTANTANEOUS, ConventionProvenance.ASSUMED),
}

assert set(VARIABLE_TIME_CONVENTION) == set(HOURLY_VARIABLES), (
    "VARIABLE_TIME_CONVENTION must cover exactly the nine HOURLY_VARIABLES, no more, no less"
)


def _cache_fingerprint() -> str:
    """First eight hex characters of a SHA256 over the canonical
    serialisation of GRID_POINTS (order included) and HOURLY_VARIABLES.

    Computed at import time rather than maintained by hand, so that a
    forgotten version bump cannot cause silently mixed schemas in the cache
    (spec 6.5.1, §2.5b).
    """
    canonical = json.dumps(
        {
            "points": [(p.point_id, p.latitude, p.longitude) for p in GRID_POINTS],
            "variables": list(HOURLY_VARIABLES),
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
