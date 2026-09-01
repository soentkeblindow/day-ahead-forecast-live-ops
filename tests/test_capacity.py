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


def test_raises_more_than_one_interval_beyond_last_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, CapacitySource.PUBLIC_REGISTRY, _WIDE_ANCHORS)
    last = _WIDE_ANCHORS.index[-1]
    interval_days = (_WIDE_ANCHORS.index[-1] - _WIDE_ANCHORS.index[-2]).days
    too_far = pd.DatetimeIndex([last + pd.Timedelta(days=interval_days + 10)])
    with pytest.raises(ValueError, match="more than one anchor interval"):
        installed_capacity_at(ProductionType.SOLAR, too_far)


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


def test_entsoe_source_not_yet_implemented_without_patching() -> None:
    query = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")])
    with pytest.raises(NotImplementedError, match="deferred"):
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
    # One interval beyond the true (undiscarded) last anchor -- but with the
    # last two anchors discarded, this is now two intervals beyond the
    # retained boundary and must raise.
    query = pd.DatetimeIndex([anchors.index[-1] + pd.Timedelta(days=25)])
    with pytest.raises(ValueError, match="more than one anchor interval"):
        installed_capacity_at(ProductionType.SOLAR, query)


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


def test_public_registry_loads_real_committed_table() -> None:
    """Smoke test against the actual committed CSV, not a synthetic fixture."""
    query = pd.DatetimeIndex([pd.Timestamp("2024-06-15", tz="UTC")])
    for production_type in ProductionType:
        result = installed_capacity_at(production_type, query)
        assert result.iloc[0] > 0
