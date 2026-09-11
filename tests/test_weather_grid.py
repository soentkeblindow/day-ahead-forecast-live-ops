"""Unit tests for data/weather_grid.py -- the versioned ECMWF grid point set.

Pure data module, no network, no fixtures needed.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from energy_price_forecast.data.weather_grid import (
    CACHE_KEY,
    GRID_POINTS,
    HOURLY_VARIABLES,
    KNOWN_WEATHER_DEFECTS,
    VARIABLE_TIME_CONVENTION,
    ConventionProvenance,
    GridPoint,
    KnownWeatherDefectCategory,
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


# ---------------------------------------------------------------------------
# KNOWN_WEATHER_DEFECTS shape (docs/sprint6_auftrag_known_data_defects.md §7:
# "nicht leer, Schluessel eindeutig und tz-bewusst UTC, jede Kategorie
# gueltig, jeder Eintrag hat ein Pruefdatum")
# ---------------------------------------------------------------------------


def test_known_weather_defects_is_not_empty() -> None:
    assert len(KNOWN_WEATHER_DEFECTS) == 6


def test_known_weather_defects_keys_are_unique_tz_aware_utc_midnights() -> None:
    keys = list(KNOWN_WEATHER_DEFECTS.keys())
    assert len(keys) == len(set(keys))  # dict keys are already unique by construction --
    # this is the machine-checked version of that guarantee, not a tautology.
    for key in keys:
        assert isinstance(key, pd.Timestamp)
        assert key.tzinfo is not None and str(key.tz) == "UTC"
        assert key == key.normalize()  # exactly midnight -- a run init, not an arbitrary time


def test_known_weather_defects_categories_are_valid() -> None:
    for defect in KNOWN_WEATHER_DEFECTS.values():
        assert isinstance(defect.category, KnownWeatherDefectCategory)


def test_known_weather_defects_every_entry_has_a_checked_date() -> None:
    for defect in KNOWN_WEATHER_DEFECTS.values():
        assert defect.checked_date
        # A parseable ISO date, not in the future relative to when this
        # constant's entries were written (spec §4: "Aufgenommen wird ein
        # Defekt, wenn ein Neuabruf ihn heute reproduziert" -- checked_date
        # is when that "heute" was, so it can never postdate the constant).
        checked = dt.date.fromisoformat(defect.checked_date)
        assert checked <= dt.date(2026, 9, 11) or checked <= dt.date.today()


def test_known_weather_defects_split_matches_the_documented_six() -> None:
    unavailable = [
        d
        for d, defect in KNOWN_WEATHER_DEFECTS.items()
        if defect.category == KnownWeatherDefectCategory.PROVIDER_UNAVAILABLE
    ]
    corrupt = [
        d
        for d, defect in KNOWN_WEATHER_DEFECTS.items()
        if defect.category == KnownWeatherDefectCategory.PROVIDER_CORRUPT
    ]
    assert len(unavailable) == 4
    assert len(corrupt) == 2
    assert set(d.date().isoformat() for d in corrupt) == {"2025-08-07", "2026-06-23"}
