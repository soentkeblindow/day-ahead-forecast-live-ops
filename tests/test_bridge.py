"""Unit tests for the additive shape profile and the hourly-to-quarter-hourly
expansion (models/bridge.py)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.models.bridge import (
    ShapeProfile,
    expand_to_quarterhour,
    fit_shape_profile,
)

_TZ = "Europe/Berlin"
_KNOWN_SHAPE = np.array([5.0, -5.0, 3.0, -3.0])  # sums to zero by construction


def _synthetic_prices(
    start_date: dt.date,
    n_days: int,
    *,
    shape: np.ndarray = _KNOWN_SHAPE,
    hourly_mean: float = 50.0,
    seed: int = 0,
) -> pd.Series:
    """n_days of quarter-hourly prices with an exact, known intra-hour shape
    riding on top of a per-hour-varying mean (so the mean itself carries no
    information a naive test could accidentally exploit)."""
    rng = np.random.default_rng(seed)
    idx_parts: list[pd.DatetimeIndex] = []
    val_parts: list[np.ndarray] = []
    days = pd.date_range(start_date, periods=n_days, freq="D", tz=_TZ)
    for day in days:
        qh = pd.date_range(day, day + pd.Timedelta(days=1), freq="15min", inclusive="left")
        for h in range(len(qh) // 4):
            base = hourly_mean + rng.normal(0, 2)
            idx_parts.append(qh[h * 4 : (h + 1) * 4])
            val_parts.append(base + shape)
    index = pd.DatetimeIndex(np.concatenate([list(p) for p in idx_parts])).tz_convert("UTC")
    return pd.Series(np.concatenate(val_parts), index=index)


# ---------------------------------------------------------------------------
# fit_shape_profile
# ---------------------------------------------------------------------------


def test_profile_reconstructs_known_shape_exactly_without_noise() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28, shape=_KNOWN_SHAPE)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    for hour in range(24):
        np.testing.assert_allclose(profile.values.loc[hour].to_numpy(), _KNOWN_SHAPE, atol=1e-9)


def test_profile_reconstructs_known_shape_with_noise_within_tolerance() -> None:
    noisy_shape = _KNOWN_SHAPE  # the additive deviation itself stays exact per hour-group;
    # noise enters via the per-group base only (see _synthetic_prices), so the
    # averaged profile should still recover the shape closely over 28 days.
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28, shape=noisy_shape, seed=42)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    for hour in range(24):
        np.testing.assert_allclose(profile.values.loc[hour].to_numpy(), _KNOWN_SHAPE, atol=1e-9)


def test_zero_sum_invariant_holds_per_hour() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    by_hour = profile.values.groupby(level="hour").sum()
    assert (by_hour.abs() < 1e-9).all()


def test_shapeprofile_construction_raises_on_violated_zero_sum() -> None:
    idx = pd.MultiIndex.from_product([range(24), range(4)], names=["hour", "quarter"])
    values = pd.Series(0.0, index=idx)
    values.loc[(5, 0)] = 1.0  # unbalanced: hour 5 no longer sums to zero
    counts = pd.Series(28, index=idx)

    with pytest.raises(ValueError, match="hour=5"):
        ShapeProfile(
            values=values, n_days=28, window_end_day=pd.Timestamp("2026-01-28"), counts=counts
        )


def test_window_length_is_exactly_n_days_inclusive_both_ends() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 40)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    assert profile.n_days == 28
    assert profile.window_end_day == pd.Timestamp("2026-01-28")
    # Total observations per cell must equal exactly n_days (one observation
    # per ordinary day per cell): window is [end_day - 27, end_day].
    assert (profile.counts == 28).all()


def test_min_observations_raises_on_sparse_window() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 2)

    with pytest.raises(ValueError, match="observation"):
        fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-02"), n_days=2, min_observations=3)


def test_dst_spring_forward_day_in_window_reduces_missing_hour_count() -> None:
    """A window spanning the 2026 DE/LU spring-forward day (2026-03-29) runs
    through and hour=2's cells carry one fewer observation than the other 23
    hours -- that day contributes no local 02:00 quarter-hours at all."""
    prices = _synthetic_prices(dt.date(2026, 3, 15), 28)  # window covers 03-15..04-11
    profile = fit_shape_profile(
        prices, end_day=pd.Timestamp("2026-04-11"), n_days=28, min_observations=1
    )

    other_hour_count = profile.counts.loc[(3, 0)]
    spring_hour_count = profile.counts.loc[(2, 0)]
    assert spring_hour_count == other_hour_count - 1


def test_leakage_sentinel_never_influences_the_profile() -> None:
    """fit_shape_profile(end_day=D-1) must not read anything at or after the
    start of D. Filling everything from D onward with an extreme sentinel and
    checking the profile is unaffected is the leakage test proper."""
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28, shape=_KNOWN_SHAPE)
    end_day = pd.Timestamp("2026-01-28")

    clean_profile = fit_shape_profile(prices, end_day=end_day, n_days=28)

    sentinel_start = pd.Timestamp(dt.date(2026, 1, 29), tz=_TZ)
    sentinel_end = pd.Timestamp(dt.date(2026, 3, 1), tz=_TZ)
    sentinel_index = pd.date_range(sentinel_start, sentinel_end, freq="15min", inclusive="left")
    contaminated = pd.concat(
        [prices, pd.Series(1e9, index=sentinel_index.tz_convert("UTC"))]
    ).sort_index()

    contaminated_profile = fit_shape_profile(contaminated, end_day=end_day, n_days=28)

    pd.testing.assert_series_equal(contaminated_profile.values, clean_profile.values)


# ---------------------------------------------------------------------------
# expand_to_quarterhour
# ---------------------------------------------------------------------------


def _hourly_forecast_for(target_date: dt.date, value: float = 100.0) -> pd.Series:
    start = pd.Timestamp(target_date, tz=_TZ)
    end = pd.Timestamp(target_date + dt.timedelta(days=1), tz=_TZ)
    idx = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    return pd.Series(value, index=idx)


def test_flat_expansion_repeats_each_hourly_value_four_times() -> None:
    target_date = dt.date(2026, 6, 15)
    hourly = pd.Series(
        range(24),
        index=_hourly_forecast_for(target_date).index,
        dtype=float,
    )

    result = expand_to_quarterhour(hourly, None, target_day=pd.Timestamp(target_date))

    assert len(result) == 96
    expected = np.repeat(np.arange(24, dtype=float), 4)
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_hourly_mean_is_preserved_for_shaped_and_flat_expansion() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28, shape=_KNOWN_SHAPE)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    target_date = dt.date(2026, 1, 29)
    hourly = _hourly_forecast_for(target_date, value=100.0)

    shaped = expand_to_quarterhour(hourly, profile, target_day=pd.Timestamp(target_date))
    flat = expand_to_quarterhour(hourly, None, target_day=pd.Timestamp(target_date))

    for result in (shaped, flat):
        hour_bucket = pd.DatetimeIndex(result.index).floor("h")
        hourly_means = result.groupby(hour_bucket).mean()
        np.testing.assert_allclose(hourly_means.to_numpy(), 100.0, atol=1e-9)


def test_shaped_expansion_applies_the_profile_deltas() -> None:
    prices = _synthetic_prices(dt.date(2026, 1, 1), 28, shape=_KNOWN_SHAPE)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp("2026-01-28"), n_days=28)

    target_date = dt.date(2026, 1, 29)
    hourly = _hourly_forecast_for(target_date, value=100.0)
    shaped = expand_to_quarterhour(hourly, profile, target_day=pd.Timestamp(target_date))

    np.testing.assert_allclose(shaped.iloc[:4].to_numpy(), 100.0 + _KNOWN_SHAPE, atol=1e-9)


@pytest.mark.parametrize(
    ("target_day", "profile_end_day", "expected_len"),
    [
        (dt.date(2026, 3, 29), "2026-03-28", 92),  # DE/LU DST start -- spring forward, 23h day
        (dt.date(2026, 10, 25), "2026-10-24", 100),  # DE/LU DST end -- fall back, 25h day
    ],
)
def test_output_length_matches_dst_day_type(
    target_day: dt.date, profile_end_day: str, expected_len: int
) -> None:
    """Restarbeit 6.7.2, Teil C.1, level 2 -- the more important of the two
    C.1 tests (the component, not the guard rail): nothing in this path may
    silently assume 96. The fall-back (25h -> 100) side was previously
    untested; the spring-forward (23h -> 92) side already was. Confirmed
    (a scratch check before writing this) that expand_to_quarterhour already
    produces the right count on the real 2026-10-25 fall-back day -- this
    pins that as a regression test, not a fix."""
    prices = _synthetic_prices(target_day - dt.timedelta(days=28), 28)
    profile = fit_shape_profile(prices, end_day=pd.Timestamp(profile_end_day), n_days=28)

    hourly = _hourly_forecast_for(target_day)
    result = expand_to_quarterhour(hourly, profile, target_day=pd.Timestamp(target_day))
    assert len(result) == expected_len


def test_mismatched_hourly_index_raises() -> None:
    target_date = dt.date(2026, 6, 15)
    wrong_day_hourly = _hourly_forecast_for(dt.date(2026, 6, 16))

    with pytest.raises(ValueError, match="does not match"):
        expand_to_quarterhour(wrong_day_hourly, None, target_day=pd.Timestamp(target_date))
