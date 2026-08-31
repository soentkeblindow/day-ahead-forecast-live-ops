"""Unit tests for data/weather_grid.py -- the versioned ECMWF grid point set.

Pure data module, no network, no fixtures needed.
"""

from __future__ import annotations

import math

import pytest

from energy_price_forecast.data.weather_grid import (
    CACHE_KEY,
    GRID_POINTS,
    HOURLY_VARIABLES,
    VARIABLE_TIME_CONVENTION,
    ConventionProvenance,
    GridPoint,
    _cache_fingerprint,
    expected_columns,
)

_MIN_SPACING_KM = 50.0
_EARTH_RADIUS_KM = 6371.0


def _haversine_km(a: GridPoint, b: GridPoint) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# Point set shape
# ---------------------------------------------------------------------------


def test_eighteen_points() -> None:
    assert len(GRID_POINTS) == 18


def test_point_ids_unique() -> None:
    ids = [p.point_id for p in GRID_POINTS]
    assert len(ids) == len(set(ids))


def test_regions_and_kind_fully_populated() -> None:
    allowed_regions = {"north", "central", "south", "offshore"}
    allowed_kinds = {"onshore", "offshore", "solar_only"}
    for point in GRID_POINTS:
        assert point.region in allowed_regions
        assert point.kind in allowed_kinds
    # Every kind actually occurs at least once (plan E1: 3 offshore, 13 onshore, 2 solar_only).
    kinds = [p.kind for p in GRID_POINTS]
    assert kinds.count("offshore") == 3
    assert kinds.count("onshore") == 13
    assert kinds.count("solar_only") == 2


# ---------------------------------------------------------------------------
# Minimum spacing (plan E1: 50 km is the effective-resolution floor)
# ---------------------------------------------------------------------------


def test_haversine_against_known_distance() -> None:
    # Berlin (Alexanderplatz) to Hamburg (Rathaus): ~255 km great-circle,
    # independent of anything in this module -- a correctness check on the
    # haversine implementation itself, not on the grid.
    berlin = GridPoint("berlin", 52.5219, 13.4132, "north", "onshore")
    hamburg = GridPoint("hamburg", 53.5503, 9.9937, "north", "onshore")
    assert _haversine_km(berlin, hamburg) == pytest.approx(255.0, abs=5.0)


def test_no_point_pair_closer_than_minimum_spacing() -> None:
    violations = []
    for i, a in enumerate(GRID_POINTS):
        for b in GRID_POINTS[i + 1 :]:
            distance = _haversine_km(a, b)
            if distance < _MIN_SPACING_KM:
                violations.append((a.point_id, b.point_id, distance))
    assert violations == [], f"pairs closer than {_MIN_SPACING_KM} km: {violations}"


# ---------------------------------------------------------------------------
# Time convention provenance
# ---------------------------------------------------------------------------


def test_time_convention_covers_exactly_the_nine_variables() -> None:
    assert set(VARIABLE_TIME_CONVENTION) == set(HOURLY_VARIABLES)
    assert len(HOURLY_VARIABLES) == 9


def test_no_variable_still_assumed() -> None:
    """F11 (spec 6.5.1, §5.2, docs/sprint6_step6_5_1_log.md "Schritt 8") has
    measured all eight previously-ASSUMED variables; this now stays green
    permanently rather than being the standing reminder it was before."""
    still_assumed = [
        variable
        for variable, (_, provenance) in VARIABLE_TIME_CONVENTION.items()
        if provenance is ConventionProvenance.ASSUMED
    ]
    assert still_assumed == []


# ---------------------------------------------------------------------------
# expected_columns()
# ---------------------------------------------------------------------------


def test_expected_columns_shape() -> None:
    columns = expected_columns()
    assert len(columns) == 162
    assert len(columns) == len(set(columns))


def test_expected_columns_canonical_order() -> None:
    columns = expected_columns()
    expected = tuple(
        f"{point.point_id}__{variable}" for point in GRID_POINTS for variable in HOURLY_VARIABLES
    )
    assert columns == expected


# ---------------------------------------------------------------------------
# CACHE_KEY / _cache_fingerprint sensitivity (spec §2.5b)
# ---------------------------------------------------------------------------


def test_cache_key_is_grid_version_plus_fingerprint() -> None:
    assert f"grid_v1_{_cache_fingerprint()}" == CACHE_KEY


def test_cache_fingerprint_changes_with_point_set() -> None:
    baseline = _cache_fingerprint()
    reordered = tuple(reversed(GRID_POINTS))
    assert _cache_fingerprint(points=reordered) != baseline

    with_extra_point = (
        *GRID_POINTS,
        GridPoint("extra_test_point", 51.0, 10.0, "central", "onshore"),
    )
    assert _cache_fingerprint(points=with_extra_point) != baseline


def test_cache_fingerprint_changes_with_variable_tuple() -> None:
    baseline = _cache_fingerprint()
    fewer_variables = HOURLY_VARIABLES[:-1]
    assert _cache_fingerprint(variables=fewer_variables) != baseline

    reordered_variables = tuple(reversed(HOURLY_VARIABLES))
    assert _cache_fingerprint(variables=reordered_variables) != baseline
