import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.config import ConformalConfig
from energy_price_forecast.evaluation.conformal import (
    calibration_window,
    hourly_sigma,
    local_scale,
    quantile_shift,
    scaled_conformal_calibrate,
)


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# calibration_window: the leakage guard (core deliverable)
# ---------------------------------------------------------------------------


def test_calibration_window_never_reaches_embargo_edge() -> None:
    index = _hourly_utc("2021-01-01", 24 * 30)
    test_day = pd.Timestamp("2021-01-25", tz="UTC")
    embargo_days = 1

    win = calibration_window(test_day, index, window_days=10, embargo_days=embargo_days)

    edge = test_day - pd.Timedelta(days=embargo_days)
    assert len(win) > 0
    assert win.max() < edge


def test_calibration_window_bounds_are_half_open_window() -> None:
    index = _hourly_utc("2021-01-01", 24 * 30)
    test_day = pd.Timestamp("2021-01-25", tz="UTC")

    win = calibration_window(test_day, index, window_days=10, embargo_days=1)

    edge = test_day - pd.Timedelta(days=1)
    lo = edge - pd.Timedelta(days=10)
    assert win.min() >= lo
    assert win.max() < edge
    # exactly the hourly hours in [lo, edge)
    assert len(win) == 10 * 24


def test_calibration_window_shrinks_at_start_of_history() -> None:
    index = _hourly_utc("2021-01-01", 24 * 30)
    test_day = pd.Timestamp("2021-01-03", tz="UTC")  # only ~1 day of history before it

    win = calibration_window(test_day, index, window_days=90, embargo_days=1)

    edge = test_day - pd.Timedelta(days=1)
    assert win.max() < edge
    assert win.min() == index.min()
    assert len(win) < 90 * 24


def test_calibration_window_empty_when_test_day_before_history() -> None:
    index = _hourly_utc("2021-01-01", 24 * 30)
    test_day = pd.Timestamp("2020-01-01", tz="UTC")

    win = calibration_window(test_day, index, window_days=90, embargo_days=1)

    assert len(win) == 0


def test_calibration_window_edge_never_selected_across_embargo_sizes() -> None:
    """The window edge itself, and everything after it, must never be selected."""
    index = _hourly_utc("2021-01-01", 24 * 60)
    test_day = pd.Timestamp("2021-02-01", tz="UTC")
    for embargo_days in (1, 2, 7):
        win = calibration_window(test_day, index, window_days=20, embargo_days=embargo_days)
        edge = test_day - pd.Timedelta(days=embargo_days)
        assert (win < edge).all()


# ---------------------------------------------------------------------------
# local_scale: kNN-in-forecast-level sigma
# ---------------------------------------------------------------------------

# calm zone (medians 0-4): constant realizations -> std 0, floored.
# volatile zone (medians 5-9): spread-out realizations -> nonzero std.
_CALIB_MEDIAN = pd.Series([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
_CALIB_Y = pd.Series([10.0, 10, 10, 10, 10, 50, 52, 48, 55, 45])


def test_local_scale_matches_known_neighbourhood() -> None:
    # query p=8: the 3 calib points with medians {7, 8, 9} are strictly closer
    # than any other (distances 1, 0, 1 vs >= 2 for the rest).
    query = pd.Series([8.0])
    out = local_scale(query, _CALIB_MEDIAN, _CALIB_Y, k=3, floor=0.5)
    expected = float(np.std([48.0, 55.0, 45.0], ddof=1))
    assert out.iloc[0] == pytest.approx(expected)


def test_local_scale_floors_near_constant_neighbourhood() -> None:
    # query p=1: the 3 nearest calib points {0, 1, 2} are all in the calm
    # zone (y=10 each) -> raw std is 0, floor must kick in.
    query = pd.Series([1.0])
    out = local_scale(query, _CALIB_MEDIAN, _CALIB_Y, k=3, floor=0.5)
    assert out.iloc[0] == pytest.approx(0.5)


def test_local_scale_uses_fewer_than_k_when_calibration_set_is_smaller() -> None:
    small_median = pd.Series([1.0, 2.0])
    small_y = pd.Series([10.0, 20.0])
    query = pd.Series([1.5])

    out = local_scale(query, small_median, small_y, k=200, floor=0.0)

    expected = float(np.std([10.0, 20.0], ddof=1))
    assert out.iloc[0] == pytest.approx(expected)


def test_local_scale_returns_floor_for_empty_calibration_set() -> None:
    query = pd.Series([1.0, 2.0])
    out = local_scale(query, pd.Series(dtype=float), pd.Series(dtype=float), k=10, floor=0.5)
    assert (out == 0.5).all()


def test_local_scale_output_aligned_to_query_index() -> None:
    query = pd.Series([1.0, 8.0], index=["hour_a", "hour_b"])
    out = local_scale(query, _CALIB_MEDIAN, _CALIB_Y, k=3, floor=0.5)
    assert list(out.index) == ["hour_a", "hour_b"]


# ---------------------------------------------------------------------------
# quantile_shift: the (n+1)-corrected order statistic, with rank clamping
# ---------------------------------------------------------------------------

# scores == y directly: pred=0, sigma=1, 9 hand-pickable values.
_SHIFT_Y = pd.Series([-2.0, -1, 0, 1, 2, 3, 4, 5, 6])
_SHIFT_PRED = pd.Series([0.0] * 9)
_SHIFT_SIGMA = pd.Series([1.0] * 9)


def test_quantile_shift_matches_hand_computed_order_statistic() -> None:
    # n=9, alpha=0.5 -> rank = ceil(10*0.5) = 5 -> 5th smallest of
    # [-2,-1,0,1,2,3,4,5,6] (1-indexed) = 2.
    q = quantile_shift(_SHIFT_Y, _SHIFT_PRED, _SHIFT_SIGMA, 0.5)
    assert q == pytest.approx(2.0)


def test_quantile_shift_clamps_rank_at_high_alpha() -> None:
    # rank = ceil(10*0.95) = 10 > n=9 -> clamped to n -> largest score (6).
    q = quantile_shift(_SHIFT_Y, _SHIFT_PRED, _SHIFT_SIGMA, 0.95)
    assert q == pytest.approx(6.0)


def test_quantile_shift_clamps_rank_at_low_alpha() -> None:
    # rank = ceil(10*0.05) = 1 -> smallest score (-2).
    q = quantile_shift(_SHIFT_Y, _SHIFT_PRED, _SHIFT_SIGMA, 0.05)
    assert q == pytest.approx(-2.0)


def test_quantile_shift_normalizes_by_sigma_and_subtracts_pred() -> None:
    # non-trivial pred/sigma: scores = (y - pred) / sigma.
    y = pd.Series([10.0, 20.0, 30.0])
    pred = pd.Series([0.0, 10.0, 10.0])
    sigma = pd.Series([1.0, 2.0, 4.0])
    # scores = [10, 5, 5], n=3, alpha=0.5 -> rank=ceil(4*0.5)=2 -> 2nd smallest = 5.
    q = quantile_shift(y, pred, sigma, 0.5)
    assert q == pytest.approx(5.0)


def test_quantile_shift_empty_calibration_returns_nan() -> None:
    empty = pd.Series(dtype=float)
    q = quantile_shift(empty, empty, empty, 0.5)
    assert np.isnan(q)


# ---------------------------------------------------------------------------
# scaled_conformal_calibrate: the orchestrator
# ---------------------------------------------------------------------------

# All fixtures use January (no DST) so every local day is exactly 24 hours.


def _local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("Europe/Berlin").normalize()


def test_leakage_future_realizations_never_change_a_days_calibration() -> None:
    """The core deliverable: mutating y at/after the embargo edge must not
    change the calibrated output of the test day whose calibration it guards.
    """
    index = _hourly_utc("2021-01-01", 24 * 30)
    rng = np.random.default_rng(0)
    y = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    preds = {
        0.05: pd.Series(40.0, index=index),
        0.50: pd.Series(50.0, index=index),
        0.95: pd.Series(60.0, index=index),
    }
    config = ConformalConfig(
        window_days=5,
        embargo_days=1,
        n_neighbors=50,
        sigma_floor_fraction=0.10,
        recompute_cadence_days=1,
        min_calibration_hours=24,
    )

    calibrated_orig, _ = scaled_conformal_calibrate(y, preds, config=config)

    local_days = _local_days(index)
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))
    test_day = delivery_days[10]
    edge = test_day - pd.Timedelta(days=config.embargo_days)

    y_mutated = y.copy()
    y_mutated.loc[y_mutated.index >= edge] = 1_000_000.0
    calibrated_mutated, _ = scaled_conformal_calibrate(y_mutated, preds, config=config)

    d_idx = index[local_days == test_day]
    for a in preds:
        pd.testing.assert_series_equal(
            calibrated_orig[a].loc[d_idx], calibrated_mutated[a].loc[d_idx]
        )


def test_scaling_direction_larger_sigma_yields_larger_correction() -> None:
    """Within the same day/recompute group, a hour whose own median forecast
    sits in a volatile neighbourhood gets a larger |correction| than one in a
    calm neighbourhood."""
    calib_index = _hourly_utc("2021-01-01", 24 * 10)
    n = len(calib_index)
    # first half: calm neighbourhood (median ~10, tight realizations)
    # second half: volatile neighbourhood (median ~100, wide realizations)
    rng = np.random.default_rng(1)
    calib_median = pd.Series([10.0] * (n // 2) + [100.0] * (n - n // 2), index=calib_index)
    calib_y = pd.Series(
        np.concatenate(
            [
                10 + rng.normal(0, 1, n // 2),
                100 + rng.normal(0, 20, n - n // 2),
            ]
        ),
        index=calib_index,
    )

    test_day_index = _hourly_utc("2021-01-11", 24)  # contiguous, right after calib_index
    test_median = pd.Series([10.0] * 12 + [100.0] * 12, index=test_day_index)

    full_index = calib_index.append(test_day_index)
    y_true = pd.concat([calib_y, pd.Series(np.nan, index=test_day_index)])
    preds = {
        0.05: pd.concat([calib_median - 5, test_median - 5]),
        0.50: pd.concat([calib_median, test_median]),
        0.95: pd.concat([calib_median + 5, test_median + 5]),
    }
    for s in preds.values():
        s.index = full_index
    y_true.index = full_index

    config = ConformalConfig(
        window_days=10,
        embargo_days=0,
        n_neighbors=20,
        sigma_floor_fraction=0.01,
        recompute_cadence_days=1,
        min_calibration_hours=24,
    )
    calibrated, _ = scaled_conformal_calibrate(y_true, preds, config=config)

    raw_95 = preds[0.95].loc[test_day_index]
    cal_95 = calibrated[0.95].loc[test_day_index]
    correction = (cal_95 - raw_95).abs()

    calm_hours = test_day_index[:12]
    volatile_hours = test_day_index[12:]
    assert correction.loc[volatile_hours].mean() > correction.loc[calm_hours].mean()


def test_coverage_improves_after_calibration() -> None:
    """Behavioural, not exact: a systematically under-covering raw quantile
    moves closer to its nominal level after calibration."""
    index = _hourly_utc("2021-01-01", 24 * 60)
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(50, 10, len(index)), index=index)
    # Deliberately too-low q_0.90 (true q_0.90 of N(50,10) is ~62.8).
    preds = {
        0.50: pd.Series(50.0, index=index),
        0.90: pd.Series(56.4, index=index),
    }
    config = ConformalConfig(
        window_days=14,
        embargo_days=1,
        n_neighbors=200,
        sigma_floor_fraction=0.10,
        recompute_cadence_days=1,
        min_calibration_hours=24 * 10,
    )

    calibrated, diagnostics = scaled_conformal_calibrate(y, preds, config=config)

    calibrated_days = diagnostics.loc[~diagnostics["uncalibrated"], "day"].unique()
    local_days = _local_days(index)
    mask = local_days.isin(calibrated_days)
    measured_idx = index[mask]
    assert len(measured_idx) > 24 * 10  # sanity: a meaningful measurement window

    raw_cov = float((y.loc[measured_idx] <= preds[0.90].loc[measured_idx]).mean())
    cal_cov = float((y.loc[measured_idx] <= calibrated[0.90].loc[measured_idx]).mean())

    assert abs(cal_cov - 0.90) < abs(raw_cov - 0.90)


def test_crossing_is_reported_not_enforced() -> None:
    index = _hourly_utc("2021-01-01", 24 * 10)
    rng = np.random.default_rng(3)
    y = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    preds = {
        0.05: pd.Series(45.0, index=index),
        0.50: pd.Series(50.0, index=index),
        0.95: pd.Series(55.0, index=index),
    }
    # force an extreme, un-fixable crossing on one specific hour of the last day
    crossed_hour = index[-1]
    preds[0.05].loc[[crossed_hour]] = 1000.0
    preds[0.95].loc[[crossed_hour]] = 1.0

    config = ConformalConfig(
        window_days=5,
        embargo_days=1,
        n_neighbors=50,
        sigma_floor_fraction=0.10,
        recompute_cadence_days=1,
        min_calibration_hours=24,
    )
    calibrated, diagnostics = scaled_conformal_calibrate(y, preds, config=config)

    assert calibrated[0.05].loc[crossed_hour] > calibrated[0.95].loc[crossed_hour]
    last_day = _local_days(index)[-1]
    day_rows = diagnostics.loc[diagnostics["day"] == last_day]
    assert (day_rows["crossing_rate"] > 0).all()


def test_thin_calibration_window_passes_through_raw_and_flags() -> None:
    index = _hourly_utc("2021-01-01", 24 * 5)
    y = pd.Series(50.0, index=index)
    preds = {0.50: pd.Series(50.0, index=index), 0.90: pd.Series(60.0, index=index)}
    config = ConformalConfig(min_calibration_hours=10_000)  # unreachable -> always thin

    calibrated, diagnostics = scaled_conformal_calibrate(y, preds, config=config)

    assert diagnostics["uncalibrated"].all()
    for a in preds:
        pd.testing.assert_series_equal(calibrated[a], preds[a].astype(float))


def test_reproducibility_same_inputs_bit_identical_output() -> None:
    index = _hourly_utc("2021-01-01", 24 * 15)
    rng = np.random.default_rng(4)
    y = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    preds = {
        0.05: pd.Series(40.0, index=index),
        0.50: pd.Series(50.0, index=index),
        0.95: pd.Series(60.0, index=index),
    }
    config = ConformalConfig(window_days=5, embargo_days=1, min_calibration_hours=24)

    calibrated_a, diagnostics_a = scaled_conformal_calibrate(y, preds, config=config)
    calibrated_b, diagnostics_b = scaled_conformal_calibrate(y, preds, config=config)

    for a in preds:
        pd.testing.assert_series_equal(calibrated_a[a], calibrated_b[a])
    pd.testing.assert_frame_equal(diagnostics_a, diagnostics_b)


def test_diagnostics_column_contract() -> None:
    index = _hourly_utc("2021-01-01", 24 * 10)
    y = pd.Series(50.0, index=index)
    preds = {
        0.05: pd.Series(40.0, index=index),
        0.50: pd.Series(50.0, index=index),
        0.95: pd.Series(60.0, index=index),
    }
    config = ConformalConfig(window_days=5, embargo_days=1, min_calibration_hours=24)

    _, diagnostics = scaled_conformal_calibrate(y, preds, config=config)

    assert set(diagnostics.columns) == {
        "day",
        "level",
        "Q",
        "n",
        "sigma",
        "uncalibrated",
        "crossing_rate",
    }
    assert (diagnostics["n"] >= 0).all()
    assert diagnostics["n"].apply(lambda v: float(v).is_integer()).all()


def test_missing_median_level_raises() -> None:
    index = _hourly_utc("2021-01-01", 24 * 5)
    y = pd.Series(50.0, index=index)
    preds = {0.05: pd.Series(40.0, index=index), 0.95: pd.Series(60.0, index=index)}
    config = ConformalConfig()

    with pytest.raises(ValueError, match="0.5"):
        scaled_conformal_calibrate(y, preds, config=config)


# ---------------------------------------------------------------------------
# hourly_sigma: extracted so Sprint 4.4a imports it verbatim (spec 3.2)
# ---------------------------------------------------------------------------


def test_hourly_sigma_matches_scaled_conformal_calibrate_diagnostics() -> None:
    """Equivalence proof: hourly_sigma is not a parallel re-derivation of sigma."""
    index = _hourly_utc("2021-01-01", 24 * 30)
    rng = np.random.default_rng(5)
    y = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    preds = {
        0.05: pd.Series(40.0, index=index),
        0.50: pd.Series(50.0, index=index),
        0.95: pd.Series(60.0, index=index),
    }
    config = ConformalConfig(
        window_days=5,
        embargo_days=1,
        n_neighbors=50,
        sigma_floor_fraction=0.10,
        recompute_cadence_days=3,
        min_calibration_hours=24,
    )

    _, diagnostics = scaled_conformal_calibrate(y, preds, config=config)
    sigma_series = hourly_sigma(y, preds[0.5], config=config)

    local_days = _local_days(index)
    mean_by_day = sigma_series.groupby(local_days).mean().sort_index()
    diag_sigma_by_day = diagnostics.drop_duplicates("day").set_index("day")["sigma"].sort_index()

    pd.testing.assert_series_equal(mean_by_day, diag_sigma_by_day, check_names=False)


def test_hourly_sigma_constant_within_cadence_group_and_nan_when_uncalibrated() -> None:
    index = _hourly_utc("2021-01-01", 24 * 15)
    rng = np.random.default_rng(6)
    y = pd.Series(50 + rng.normal(0, 5, len(index)), index=index)
    q50 = pd.Series(50.0, index=index)  # constant query level: isolates cadence effects
    config = ConformalConfig(
        window_days=10,
        embargo_days=1,
        n_neighbors=50,
        sigma_floor_fraction=0.0,
        recompute_cadence_days=3,
        min_calibration_hours=24,
    )

    sigma = hourly_sigma(y, q50, config=config)
    local_days = _local_days(index)
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))

    # day_num 0: zero prior history -> always uncalibrated regardless of
    # threshold; day_num 1-2 reuse that same (uncalibrated) recompute.
    for day in delivery_days[:3]:
        assert sigma.loc[local_days == day].isna().all()

    # groups sharing one recompute (day_num % 3 == 0): constant within, and
    # differs from the next group once the calibration window has shifted.
    group_a_days = delivery_days[3:6]
    group_b_days = delivery_days[6:9]
    values_a = {float(sigma.loc[local_days == d].iloc[0]) for d in group_a_days}
    values_b = {float(sigma.loc[local_days == d].iloc[0]) for d in group_b_days}
    assert len(values_a) == 1
    assert len(values_b) == 1
    assert values_a != values_b
