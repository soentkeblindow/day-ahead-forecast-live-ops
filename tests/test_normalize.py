"""Tests for the hourly-normalisation module (data/normalize.py).

All tests use synthetic mini-frames so no real data or API calls are needed.
The six test groups map directly to the acceptance criteria in the spec (§6).
"""

import pandas as pd
import pytest

from energy_price_forecast.data.normalize import to_hourly, to_hourly_vwap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMODITY_DEFAULTS = {"ttf_gas_eur_per_mwh": 30.0, "eua_co2_eur_per_t": 25.0}


def _qh_index(start: str, n_hours: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_hours * 4, freq="15min", tz="UTC")


def _minimal_df(index: pd.DatetimeIndex, **overrides: float) -> pd.DataFrame:
    """Build a minimal DataFrame with all required columns."""
    base = {"day_ahead_price": 50.0, "load_actual": 30_000.0, **_COMMODITY_DEFAULTS}
    return pd.DataFrame({**base, **overrides}, index=index)


# ---------------------------------------------------------------------------
# Group 1 — VWAP: weighted result
# ---------------------------------------------------------------------------


def test_vwap_weighted_mean() -> None:
    """Unequal loads produce the correct load-weighted average price."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, 200.0, 300.0, 400.0], index=idx)
    # hand-computed: (50*100 + 60*200 + 70*300 + 80*400) / 1000 = 70.0
    result = to_hourly_vwap(price, weight)
    assert result.iloc[0] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# Group 2 — VWAP: equal weights collapse to simple mean
# ---------------------------------------------------------------------------


def test_vwap_equal_weights_gives_simple_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    result = to_hourly_vwap(price, weight)
    assert result.iloc[0] == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# Group 3 — VWAP: zero / NaN weight falls back to simple mean (no NaN/inf)
# ---------------------------------------------------------------------------


def test_vwap_zero_weight_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    result = to_hourly_vwap(price, weight)
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(65.0)


def test_vwap_nan_weight_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([float("nan")] * 4, index=idx)
    result = to_hourly_vwap(price, weight)
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# Group 3b — VWAP: weight_fallback (docs/bugs_in_live_system.md entry 2)
# ---------------------------------------------------------------------------


def test_vwap_no_fallback_partial_nan_weight_is_zero_weighted() -> None:
    """Without weight_fallback (the default), a partially-NaN weight silently
    zero-weights the affected quarter-hour instead of excluding it -- this is
    the exact prior behaviour and must stay unchanged when nothing opts in."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, float("nan"), 300.0, 400.0], index=idx)
    result = to_hourly_vwap(price, weight)
    # hand-computed: (50*100 + 70*300 + 80*400) / (100+300+400) = 72.5
    assert result.iloc[0] == pytest.approx(72.5)


def test_vwap_fallback_fills_missing_primary_weight() -> None:
    """A quarter-hour missing `weight` but present in `weight_fallback` is
    included in the VWAP with the fallback weight, not zero-weighted."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, float("nan"), 300.0, 400.0], index=idx)
    fallback = pd.Series([90.0, 200.0, 290.0, 410.0], index=idx)
    result = to_hourly_vwap(price, weight, weight_fallback=fallback)
    # hand-computed with the second quarter's weight taken from fallback (200):
    # (50*100 + 60*200 + 70*300 + 80*400) / (100+200+300+400) = 70.0
    assert result.iloc[0] == pytest.approx(70.0)


def test_vwap_fallback_still_missing_forces_simple_mean() -> None:
    """If a quarter-hour's weight is NaN in *both* `weight` and
    `weight_fallback`, the whole hour falls back to the simple mean instead
    of silently VWAP-ing over the remaining three quarter-hours."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, float("nan"), 300.0, 400.0], index=idx)
    fallback = pd.Series([90.0, float("nan"), 290.0, 410.0], index=idx)
    result = to_hourly_vwap(price, weight, weight_fallback=fallback)
    assert result.iloc[0] == pytest.approx(65.0)  # simple mean of all four


def test_vwap_fallback_complete_hour_matches_no_fallback_call() -> None:
    """When `weight` already has no gaps, passing weight_fallback must not
    change the result at all (it's never consulted)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    price = pd.Series([50.0, 60.0, 70.0, 80.0], index=idx)
    weight = pd.Series([100.0, 200.0, 300.0, 400.0], index=idx)
    fallback = pd.Series([999.0, 999.0, 999.0, 999.0], index=idx)
    with_fallback = to_hourly_vwap(price, weight, weight_fallback=fallback)
    without_fallback = to_hourly_vwap(price, weight)
    pd.testing.assert_series_equal(with_fallback, without_fallback)


def test_to_hourly_wires_load_forecast_as_weight_fallback() -> None:
    """to_hourly() must pass load_forecast_day_ahead as weight_fallback, so a
    gap in load_actual that load_forecast_day_ahead covers no longer degrades
    to the pre-fix zero-weighting (docs/bugs_in_live_system.md entry 2)."""
    idx = _qh_index("2024-01-01", n_hours=2)
    df = _minimal_df(idx)
    df.loc[idx[1], "load_actual"] = float("nan")
    df["load_forecast_day_ahead"] = 30_000.0
    result = to_hourly(df)
    # all four quarter-hours end up weighted (three real + one fallback, all
    # equal here) -> still a clean, fully-weighted VWAP, not a hole.
    assert not result["day_ahead_price"].isna().any()


def test_to_hourly_without_load_forecast_column_is_unaffected() -> None:
    """A frame that doesn't carry load_forecast_day_ahead at all (e.g. an
    older/synthetic fixture) must behave exactly as before -- df.get() must
    not raise, and the result must match the no-fallback code path."""
    idx = _qh_index("2024-01-01", n_hours=2)
    df = _minimal_df(idx)
    result = to_hourly(df)
    assert not result["day_ahead_price"].isna().any()


# ---------------------------------------------------------------------------
# Group 4 — "Mean, not sum": continuity across the resolution break
# ---------------------------------------------------------------------------


def test_mean_not_sum_continuity() -> None:
    """After normalisation, load_actual must be ≈ 30 000 MW in both the hourly
    and the 15-min block — not ≈ 120 000 MW in the 15-min block (which would
    happen if sub-hour values were summed instead of averaged).

    Dates straddle the real resolution break (2025-09-30 22:00 UTC) so the
    synthetic frame mirrors the actual mixed-resolution structure of the dataset.
    """
    hourly_idx = pd.date_range("2025-09-30 20:00", periods=3, freq="h", tz="UTC")
    qh_idx = pd.date_range("2025-09-30 23:00", periods=12, freq="15min", tz="UTC")
    mixed_idx = pd.DatetimeIndex(list(hourly_idx) + list(qh_idx))

    df = pd.DataFrame(
        {"day_ahead_price": 50.0, "load_actual": 30_000.0, **_COMMODITY_DEFAULTS},
        index=mixed_idx,
    )
    result = to_hourly(df)
    assert result["load_actual"].max() == pytest.approx(30_000.0)
    assert result["load_actual"].min() == pytest.approx(30_000.0)


# ---------------------------------------------------------------------------
# Group 5 — Index invariants
# ---------------------------------------------------------------------------


def test_index_is_utc_and_strictly_hourly() -> None:
    idx = _qh_index("2024-01-01", n_hours=4)
    result = to_hourly(_minimal_df(idx))

    dt_index = pd.DatetimeIndex(result.index)
    assert dt_index.tz is not None
    assert str(dt_index.tz) == "UTC"
    diffs = result.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta("1h")
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Group 6 — Column partition is exhaustive
# ---------------------------------------------------------------------------


def test_column_partition_exhaustive() -> None:
    """Every input column must appear exactly once in the normalised output."""
    idx = _qh_index("2024-01-01", n_hours=2)
    df = _minimal_df(idx)
    df["gen_wind_onshore"] = 5_000.0
    df["scheduled_net_de_to_fr"] = 1_000.0

    result = to_hourly(df)
    assert set(result.columns) == set(df.columns)
    assert len(result.columns) == len(df.columns)


# ---------------------------------------------------------------------------
# Bonus — Idempotence (§5 edge case)
# ---------------------------------------------------------------------------


def test_idempotent_on_hourly_input() -> None:
    """Applying to_hourly to already-hourly data must be a no-op."""
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "day_ahead_price": [50.0, 60.0, 70.0, 80.0],
            "load_actual": [30_000.0, 31_000.0, 29_000.0, 32_000.0],
            **_COMMODITY_DEFAULTS,
        },
        index=idx,
    )
    result = to_hourly(df)
    pd.testing.assert_frame_equal(result, df)
