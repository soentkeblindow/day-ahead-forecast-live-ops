import copy
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.data import capacity as capacity_mod
from energy_price_forecast.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
    installed_capacity_at,
)
from scripts import build_capacity_anchors

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_ENERGY_CHARTS_FIXTURE = _FIXTURES_DIR / "energy_charts_installed_power.json"


def _synthetic_anchors(dates: list[str], values: list[float]) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(dates), tz="UTC")
    return pd.Series(values, index=index)


def _patch_source(
    monkeypatch: pytest.MonkeyPatch, source: CapacitySource, anchors: pd.Series
) -> None:
    monkeypatch.setitem(capacity_mod._ANCHOR_LOADERS, source, lambda production_type: anchors)


def _monthly_anchors(
    n: int = 8, start_value: float = 100.0, monthly_increment: float = 5.0
) -> pd.Series:
    index = pd.date_range("2020-01-31", periods=n, freq="ME", tz="UTC")
    values = [start_value + i * monthly_increment for i in range(n)]
    return pd.Series(values, index=index)


# Spacing (~4 months) well above the monthly-cadence threshold, so the
# trailing-anchor-discard rule (spec 2.11, monthly resolution only) never
# fires here -- interpolation/extrapolation tests stay isolated from it.
_WIDE_ANCHORS = _synthetic_anchors(
    ["2020-01-01", "2020-05-01", "2020-09-01", "2021-01-01", "2021-05-01"],
    [100.0, 140.0, 180.0, 220.0, 260.0],
)


def test_interpolation_between_two_anchors_hits_exact_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    lower, upper = _WIDE_ANCHORS.index[0], _WIDE_ANCHORS.index[1]
    query = lower + (upper - lower) / 4  # 25% of the way from 100.0 to 140.0
    result = installed_capacity_at(ProductionType.SOLAR, pd.DatetimeIndex([query]))
    assert result.iloc[0] == pytest.approx(110.0)


def test_extrapolation_extends_the_last_observed_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    query = last + pd.Timedelta(days=interval_days // 2)
    result = installed_capacity_at(
        ProductionType.SOLAR, pd.DatetimeIndex([query]), method=CapacityExtrapolation.LAST_INCREMENT
    )
    last_increment = _WIDE_ANCHORS.iloc[-1] - _WIDE_ANCHORS.iloc[-2]
    expected = _WIDE_ANCHORS.iloc[-1] + last_increment * (interval_days // 2) / interval_days
    assert result.iloc[0] == pytest.approx(expected)


def test_extrapolation_negative_control_differs_from_holding_last_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If extrapolation degenerated to 'hold last value', this must fail."""
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    query = pd.DatetimeIndex([last + pd.Timedelta(days=interval_days // 2)])
    result = installed_capacity_at(ProductionType.SOLAR, query)
    assert result.iloc[0] != pytest.approx(_WIDE_ANCHORS.iloc[-1])


def test_raises_beyond_max_extrapolation_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    too_far = pd.DatetimeIndex(
        [last + pd.Timedelta(days=capacity_mod.MAX_EXTRAPOLATION_INTERVALS * interval_days + 10)]
    )
    with pytest.raises(ValueError, match="beyond the capacity anchor table's validity boundary"):
        installed_capacity_at(ProductionType.SOLAR, too_far)


def test_value_returned_exactly_at_max_extrapolation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    at_boundary = pd.DatetimeIndex(
        [last + pd.Timedelta(days=capacity_mod.MAX_EXTRAPOLATION_INTERVALS * interval_days)]
    )
    result = installed_capacity_at(ProductionType.SOLAR, at_boundary)
    assert np.isfinite(result.iloc[0])


def test_raises_before_first_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    too_early = pd.DatetimeIndex([_WIDE_ANCHORS.index[0] - pd.Timedelta(days=1)])
    with pytest.raises(ValueError, match="precedes the first known anchor"):
        installed_capacity_at(ProductionType.SOLAR, too_early)


def test_leap_year_and_year_boundary_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    anchors = _synthetic_anchors(["2019-11-01", "2020-03-01"], [100.0, 200.0])
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, anchors)
    query = pd.Timestamp("2020-01-01", tz="UTC")  # spans New Year and 2020-02-29
    result = installed_capacity_at(ProductionType.SOLAR, pd.DatetimeIndex([query]))
    total_days = (anchors.index[1] - anchors.index[0]).days
    elapsed_days = (query - anchors.index[0]).days
    expected = 100.0 + (200.0 - 100.0) * elapsed_days / total_days
    assert result.iloc[0] == pytest.approx(expected)


def test_both_sources_share_the_same_code_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    _patch_source(monkeypatch, CapacitySource.ENTSOE_14_1_A, _WIDE_ANCHORS)
    query = pd.DatetimeIndex([_WIDE_ANCHORS.index[0] + pd.Timedelta(days=45)])
    from_registry = installed_capacity_at(
        ProductionType.SOLAR, query, source=CapacitySource.PUBLIC_REGISTRY
    )
    from_entsoe = installed_capacity_at(
        ProductionType.SOLAR, query, source=CapacitySource.ENTSOE_14_1_A
    )
    pd.testing.assert_series_equal(from_registry, from_entsoe)


def test_entsoe_source_rejected_alternative_without_patching() -> None:
    query = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")])
    with pytest.raises(NotImplementedError, match="rejected alternative"):
        installed_capacity_at(ProductionType.SOLAR, query, source=CapacitySource.ENTSOE_14_1_A)


def test_trailing_two_anchors_discarded_and_dont_affect_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = _monthly_anchors(n=8)
    corrupted = anchors.copy()
    corrupted.iloc[-2:] = [-999.0, -999.0]  # would change the result if these anchors were used

    query = pd.DatetimeIndex(
        [anchors.index[2] + pd.Timedelta(days=5)]
    )  # well inside the retained range

    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, anchors)
    result_clean = installed_capacity_at(ProductionType.SOLAR, query)

    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, corrupted)
    result_corrupted = installed_capacity_at(ProductionType.SOLAR, query)

    pd.testing.assert_series_equal(result_clean, result_corrupted)


def test_trailing_discard_moves_the_extrapolation_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    anchors = _monthly_anchors(n=8)
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, anchors)
    interval_days = (anchors.index[-1] - anchors.index[-2]).days
    # The retained last anchor is TRAILING_ANCHORS_DISCARDED intervals before
    # the true last anchor; the validity boundary sits MAX_EXTRAPOLATION_INTERVALS
    # past the retained one -- so relative to the *true* last anchor, it is
    # (MAX_EXTRAPOLATION_INTERVALS - TRAILING_ANCHORS_DISCARDED) intervals out.
    intervals_past_true_last = (
        capacity_mod.MAX_EXTRAPOLATION_INTERVALS - capacity_mod.TRAILING_ANCHORS_DISCARDED
    )
    boundary = anchors.index[-1] + pd.Timedelta(days=intervals_past_true_last * interval_days)
    just_beyond = pd.DatetimeIndex([boundary + pd.Timedelta(days=1)])
    with pytest.raises(ValueError, match="beyond the capacity anchor table's validity boundary"):
        installed_capacity_at(ProductionType.SOLAR, just_beyond)


def test_unknown_production_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown production_type"):
        installed_capacity_at("nuclear", pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")]))  # type: ignore[arg-type]


def test_naive_timestamp_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    naive = pd.DatetimeIndex([pd.Timestamp("2020-02-01")])
    with pytest.raises(ValueError, match="tz-aware"):
        installed_capacity_at(ProductionType.SOLAR, naive)


def test_duplicate_anchors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    dup = _synthetic_anchors(["2020-01-01", "2020-01-01", "2020-05-01"], [100.0, 100.0, 140.0])
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, dup)
    with pytest.raises(ValueError, match="duplicate"):
        installed_capacity_at(ProductionType.SOLAR, pd.DatetimeIndex([dup.index[0]]))


def test_non_monotonic_anchors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = pd.Series(
        [100.0, 140.0, 120.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-05-01", "2020-03-01"], tz="UTC"),
    )
    # Bypass _validate_anchors (only run inside the real loader) to simulate a
    # source that returns anchors out of order.
    monkeypatch.setitem(
        capacity_mod._ANCHOR_LOADERS, CapacitySource.PUBLIC_REGISTRY, lambda pt: bad
    )
    with pytest.raises(ValueError):
        installed_capacity_at(ProductionType.SOLAR, pd.DatetimeIndex([bad.index[0]]))


def test_mean_last_three_increments_matches_hand_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = _synthetic_anchors(
        ["2020-01-01", "2020-05-01", "2020-09-01", "2021-01-01", "2021-05-01"],
        [100.0, 140.0, 190.0, 220.0, 260.0],  # increments: 40, 50, 30, 40 -> mean of last 3 rates
    )
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, anchors)
    last = anchors.index[-1]
    interval_days = (anchors.index[-1] - anchors.index[-2]).days
    query = pd.DatetimeIndex([last + pd.Timedelta(days=interval_days)])
    result = installed_capacity_at(
        ProductionType.SOLAR, query, method=CapacityExtrapolation.MEAN_LAST_THREE_INCREMENTS
    )
    days = capacity_mod._days_since_epoch(pd.DatetimeIndex(anchors.index))
    values = anchors.to_numpy(dtype="float64")
    rates = np.diff(values[-4:]) / np.diff(days[-4:])
    expected = anchors.iloc[-1] + float(np.mean(rates)) * interval_days
    assert result.iloc[0] == pytest.approx(expected)


def test_linear_regression_last_three_matches_hand_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    query = pd.DatetimeIndex([last + pd.Timedelta(days=interval_days)])
    result = installed_capacity_at(
        ProductionType.SOLAR, query, method=CapacityExtrapolation.LINEAR_REGRESSION_LAST_THREE
    )
    days = capacity_mod._days_since_epoch(pd.DatetimeIndex(_WIDE_ANCHORS.index))
    values = _WIDE_ANCHORS.to_numpy(dtype="float64")
    slope, _intercept = np.polyfit(days[-3:], values[-3:], deg=1)
    expected = _WIDE_ANCHORS.iloc[-1] + slope * interval_days
    assert result.iloc[0] == pytest.approx(expected)


def test_anchor_table_valid_until_matches_hand_calculation_monthly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = _monthly_anchors(n=8)
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, anchors)
    retained = anchors.iloc[: -capacity_mod.TRAILING_ANCHORS_DISCARDED]
    interval_days = (retained.index[-1] - retained.index[-2]).days
    expected = retained.index[-1] + pd.Timedelta(
        days=interval_days * capacity_mod.MAX_EXTRAPOLATION_INTERVALS
    )
    result = capacity_mod.anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    assert result == expected


def test_anchor_table_valid_until_matches_hand_calculation_yearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _WIDE_ANCHORS spacing (~4 months) is above the monthly-cadence
    # threshold, so the discard rule does not apply (spec 2.11: "bei
    # jaehrlicher Aufloesung entfaellt die Regel") and the boundary is
    # measured straight from the true last anchor.
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    expected = _WIDE_ANCHORS.index[-1] + pd.Timedelta(
        days=interval_days * capacity_mod.MAX_EXTRAPOLATION_INTERVALS
    )
    result = capacity_mod.anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    assert result == expected


def test_staleness_warning_fires_within_21_days_of_expiry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    valid_until = capacity_mod.anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    query = pd.DatetimeIndex([valid_until - pd.Timedelta(days=20)])
    with caplog.at_level("WARNING", logger=capacity_mod.__name__):
        installed_capacity_at(ProductionType.SOLAR, query)
    assert any("only valid until" in record.message for record in caplog.records)


def test_staleness_warning_does_not_fire_outside_21_days(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    valid_until = capacity_mod.anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    query = pd.DatetimeIndex([valid_until - pd.Timedelta(days=22)])
    with caplog.at_level("WARNING", logger=capacity_mod.__name__):
        installed_capacity_at(ProductionType.SOLAR, query)
    assert not any("only valid until" in record.message for record in caplog.records)


def test_raise_boundary_matches_anchor_table_valid_until(monkeypatch: pytest.MonkeyPatch) -> None:
    """No second, diverging boundary anywhere in installed_capacity_at."""
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    valid_until = capacity_mod.anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    ok = installed_capacity_at(ProductionType.SOLAR, pd.DatetimeIndex([valid_until]))
    assert np.isfinite(ok.iloc[0])
    with pytest.raises(ValueError, match="beyond the capacity anchor table's validity boundary"):
        installed_capacity_at(
            ProductionType.SOLAR, pd.DatetimeIndex([valid_until + pd.Timedelta(days=1)])
        )


def test_public_registry_loads_real_committed_table() -> None:
    """Smoke test against the actual committed CSV, not a synthetic fixture."""
    query = pd.DatetimeIndex([pd.Timestamp("2024-06-15", tz="UTC")])
    for production_type in ProductionType:
        result = installed_capacity_at(production_type, query)
        assert result.iloc[0] > 0


# ---------------------------------------------------------------------------
# scripts/build_capacity_anchors.py (spec 6.5.2a section 5.4, optional per
# section 2.1) -- built from a frozen real API response, no network access.
# ---------------------------------------------------------------------------

_AS_OF = pd.Timestamp("2026-09-02", tz="UTC")  # matches the fixture's own retrieval date


def _load_fixture_response() -> dict[str, object]:
    with _ENERGY_CHARTS_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def test_build_capacity_anchors_matches_expected_structure_from_frozen_fixture() -> None:
    response = _load_fixture_response()
    anchors = build_capacity_anchors.parse_anchors(response, as_of=_AS_OF)

    expected_months = pd.date_range("2015-01-31", "2026-08-31", freq="ME", tz="UTC")
    assert set(anchors.columns) == {"anchor_date", "production_type", "capacity_mw"}
    assert set(anchors["production_type"]) == {p.value for p in ProductionType}
    assert len(anchors) == len(expected_months) * len(ProductionType)
    for production_type in ProductionType:
        dates = pd.DatetimeIndex(
            anchors.loc[anchors["production_type"] == production_type.value, "anchor_date"]
        )
        assert (dates == expected_months).all()

    # GW -> MW conversion against the raw fixture value, and 'Solar DC' (not
    # 'Solar AC') is the series actually used.
    by_name = {p["name"]: p["data"] for p in response["production_types"]}  # type: ignore[union-attr]
    idx = response["time"].index("01.2015")  # type: ignore[union-attr]
    expected_solar_mw = by_name["Solar DC"][idx] * 1000.0
    solar_2015 = anchors.loc[
        (anchors["production_type"] == "solar")
        & (anchors["anchor_date"] == pd.Timestamp("2015-01-31", tz="UTC")),
        "capacity_mw",
    ].iloc[0]
    assert solar_2015 == pytest.approx(expected_solar_mw)
    assert by_name["Solar DC"][idx] != by_name["Solar AC"][idx]  # sanity: the two really differ

    build_capacity_anchors.validate_monotonic_cadence(anchors)  # must not raise


def test_build_capacity_anchors_raises_on_missing_target_series() -> None:
    response = copy.deepcopy(_load_fixture_response())
    production_types = cast("list[dict[str, object]]", response["production_types"])
    response["production_types"] = [p for p in production_types if p["name"] != "Wind offshore"]
    with pytest.raises(ValueError, match="missing from the Energy-Charts response"):
        build_capacity_anchors.parse_anchors(response, as_of=_AS_OF)


def test_build_capacity_anchors_raises_on_revision_beyond_three_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _load_fixture_response()
    anchors = build_capacity_anchors.parse_anchors(response, as_of=_AS_OF)

    # An "old" committed file whose 2015-01-31 solar value deliberately
    # disagrees with the freshly parsed one -- well outside the 3-month
    # revision-check floor relative to _AS_OF.
    old_path = tmp_path / "capacity_anchors_public_registry.csv"
    old_path.write_text(
        "# fixture: pre-existing committed table for the revision-check test\n"
        "anchor_date,production_type,capacity_mw\n"
        "2015-01-31,solar,999999.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_capacity_anchors, "PUBLIC_REGISTRY_ANCHORS_PATH", old_path)

    with pytest.raises(ValueError, match="revision event"):
        build_capacity_anchors.validate_against_committed(anchors, as_of=_AS_OF)
