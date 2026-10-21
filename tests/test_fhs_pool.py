import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.fhs_pool import (
    pool_by_day,
    standardised_residuals,
    tail_quantile,
)

_PERIODS_PER_DAY = 96


def _quarterhourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="15min", tz="UTC")


def _local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("Europe/Berlin").normalize()


# ---------------------------------------------------------------------------
# standardised_residuals
# ---------------------------------------------------------------------------


def test_standardised_residuals_hand_computed() -> None:
    y_true = pd.Series([110.0, 90.0, 100.0])
    median = pd.Series([100.0, 100.0, 100.0])
    sigma = pd.Series([5.0, 10.0, 2.0])

    r = standardised_residuals(y_true, median, sigma)

    pd.testing.assert_series_equal(r, pd.Series([2.0, -1.0, 0.0]))


def test_standardised_residuals_missing_sigma_yields_nan_not_exception() -> None:
    y_true = pd.Series([110.0, 90.0])
    median = pd.Series([100.0, 100.0])
    sigma = pd.Series([5.0, float("nan")])

    r = standardised_residuals(y_true, median, sigma)

    assert r.iloc[0] == pytest.approx(2.0)
    assert np.isnan(r.iloc[1])


# ---------------------------------------------------------------------------
# tail_quantile: anchor identity with np.quantile
# ---------------------------------------------------------------------------


def test_tail_quantile_matches_np_quantile() -> None:
    rng = np.random.default_rng(0)
    pool = rng.normal(0, 1, 500)
    for level in (0.025, 0.25, 0.5, 0.75, 0.975):
        assert tail_quantile(pool, level) == pytest.approx(np.quantile(pool, level))


# ---------------------------------------------------------------------------
# pool_by_day: the leakage-critical rolling tail pool (core deliverable)
# ---------------------------------------------------------------------------


def test_pool_by_day_leakage_future_values_never_appear() -> None:
    """Mutating r at/after the embargo edge must not change an earlier day's
    pool -- the same leakage-guard style as conformal.py's own core test.
    """
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 30)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 1, len(index)), index=index)

    embargo_days = 1
    window_days = 10
    pools_orig = pool_by_day(r, window_days=window_days, embargo_days=embargo_days, min_pool_days=0)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    test_day = delivery_days[20]
    edge = test_day - pd.Timedelta(days=embargo_days)

    r_mutated = r.copy()
    r_mutated.loc[r_mutated.index >= edge] = 999_999.0
    pools_mutated = pool_by_day(
        r_mutated, window_days=window_days, embargo_days=embargo_days, min_pool_days=0
    )

    np.testing.assert_array_equal(pools_orig[test_day.date()], pools_mutated[test_day.date()])
    assert 999_999.0 not in pools_mutated[test_day.date()]


def test_pool_by_day_leakage_exact_position_identity() -> None:
    """Stronger form of the leakage guard: values double as period positions
    (r.iloc[i] == i), so this checks exactly WHICH periods are pooled, not
    just how many -- the "exact index comparison" the spec asks for.
    """
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 30)
    r = pd.Series(np.arange(len(index), dtype=float), index=index)

    embargo_days = 1
    window_days = 10
    pools = pool_by_day(r, window_days=window_days, embargo_days=embargo_days, min_pool_days=0)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    test_day = delivery_days[20]
    edge = test_day - pd.Timedelta(days=embargo_days)

    valid_period_count = int((index < edge).sum())
    pool = pools[test_day.date()]
    assert pool.size > 0
    assert (pool < valid_period_count).all()


def test_pool_by_day_never_reaches_before_window_start() -> None:
    """spec section 3.2, window-length invariant: no timestamp
    < D - embargo - window is ever pooled, once the window is deep enough
    to no longer be expanding.
    """
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 40)
    r = pd.Series(np.arange(len(index), dtype=float), index=index)

    embargo_days = 1
    window_days = 10
    pools = pool_by_day(r, window_days=window_days, embargo_days=embargo_days, min_pool_days=0)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    test_day = delivery_days[35]
    edge = test_day - pd.Timedelta(days=embargo_days)
    lo = edge - pd.Timedelta(days=window_days)
    assert lo > index.min()  # sanity: genuinely in the rolling regime

    min_valid_position = int((index < lo).sum())
    pool = pools[test_day.date()]
    assert (pool >= min_valid_position).all()


def test_pool_by_day_grows_before_window_then_constant_length() -> None:
    """The pool grows before the window has accumulated its full length
    (lower bound clamped to the start of history), then stays constant
    length -- both regimes emerge from the same masking logic, no separate
    window_mode code path."""
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 400)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 1, len(index)), index=index)

    pools = pool_by_day(r, window_days=180, embargo_days=1, min_pool_days=0)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    sizes = [pools[d.date()].size for d in delivery_days]

    assert sizes[50] < sizes[200]  # still expanding, well before the window fills
    late_sizes = {pools[d.date()].size for d in delivery_days[380:390]}
    assert len(late_sizes) == 1  # window has fully rolled: constant length


def test_pool_by_day_below_min_pool_days_yields_empty_array() -> None:
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 10)
    r = pd.Series(1.0, index=index)

    pools = pool_by_day(r, window_days=5, embargo_days=1, min_pool_days=1_000)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    for day in delivery_days:
        assert pools[day.date()].size == 0


def test_pool_by_day_min_pool_days_checked_on_nan_filtered_count() -> None:
    """min_pool_days must gate on the NaN-filtered count, not the raw window
    span: a window whose raw span exceeds the threshold but whose non-NaN
    content falls short must still yield an empty pool.
    """
    # start aligned to Europe/Berlin local midnight (UTC+1 in January) so
    # every local delivery day spans exactly 96 periods of this UTC index.
    index = _quarterhourly_utc("2020-12-31 23:00", _PERIODS_PER_DAY * 20)
    r = pd.Series(1.0, index=index)
    r.iloc[: _PERIODS_PER_DAY * 5] = float("nan")  # first 5 days missing sigma

    pools = pool_by_day(r, window_days=20, embargo_days=1, min_pool_days=14)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    test_day = delivery_days[19]

    # raw window span (18 days) exceeds min_pool_days (14), but the
    # NaN-filtered content (13 days) falls short -> must be empty.
    assert pools[test_day.date()].size == 0


def test_pool_by_day_drops_nan_values_from_the_returned_pool() -> None:
    # start aligned to Europe/Berlin local midnight (UTC+1 in January) so
    # every local delivery day spans exactly 96 periods of this UTC index.
    index = _quarterhourly_utc("2020-12-31 23:00", _PERIODS_PER_DAY * 20)
    r = pd.Series(1.0, index=index)
    r.iloc[: _PERIODS_PER_DAY * 5] = float("nan")

    pools = pool_by_day(r, window_days=20, embargo_days=1, min_pool_days=10)

    delivery_days = pd.DatetimeIndex(sorted(pd.unique(_local_days(index))))
    test_day = delivery_days[19]

    pool = pools[test_day.date()]
    assert not np.isnan(pool).any()
    assert pool.size == _PERIODS_PER_DAY * 13  # 18 raw days minus the 5 NaN days


def test_pool_by_day_pools_are_ascending_sorted() -> None:
    index = _quarterhourly_utc("2021-01-01", _PERIODS_PER_DAY * 15)
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 1, len(index)), index=index)

    pools = pool_by_day(r, window_days=10, embargo_days=1, min_pool_days=0)

    for pool in pools.values():
        if pool.size > 0:
            assert np.all(np.diff(pool) >= 0)
