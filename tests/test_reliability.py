import math

import pandas as pd
import pytest

from energy_price_forecast.evaluation.metrics import (
    interval_coverage,
    interval_width,
    pinball,
    quantile_coverage,
)
from energy_price_forecast.evaluation.reliability import (
    FORECAST_LEVEL_BUCKETS,
    NESTED_BANDS,
    band_metrics,
    reliability_curve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _full_grid(idx: pd.DatetimeIndex, values: dict[float, list[float]]) -> dict[float, pd.Series]:
    return {a: pd.Series(vals, index=idx) for a, vals in values.items()}


# ---------------------------------------------------------------------------
# Anchor identity
# ---------------------------------------------------------------------------


def test_reliability_curve_overall_matches_atomic_metrics() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    y = pd.Series([50.0, 52.0, 48.0, 55.0, 45.0, 50.0], index=idx)
    preds = _full_grid(
        idx,
        {
            0.05: [40.0] * 6,
            0.10: [42.0] * 6,
            0.25: [46.0] * 6,
            0.50: [50.0] * 6,
            0.75: [53.0] * 6,
            0.90: [56.0] * 6,
            0.95: [58.0] * 6,
        },
    )

    out = reliability_curve(y, preds)

    assert set(out["bucket"]) == {"overall"}
    for a in LEVELS:
        row = out.loc[out["level"] == a].iloc[0]
        assert row["coverage"] == pytest.approx(quantile_coverage(y, preds[a]))
        assert row["pinball"] == pytest.approx(pinball(y, preds[a], a))
        assert row["n"] == 6


def test_band_metrics_overall_matches_atomic_metrics() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    y = pd.Series([50.0, 52.0, 48.0, 55.0, 45.0, 50.0], index=idx)
    preds = _full_grid(
        idx,
        {
            0.05: [40.0] * 6,
            0.10: [42.0] * 6,
            0.25: [46.0] * 6,
            0.50: [50.0] * 6,
            0.75: [53.0] * 6,
            0.90: [56.0] * 6,
            0.95: [58.0] * 6,
        },
    )

    out = band_metrics(y, preds)

    for name, low, high in NESTED_BANDS:
        row = out.loc[out["band"] == name].iloc[0]
        assert row["coverage"] == pytest.approx(interval_coverage(y, preds[low], preds[high]))
        assert row["width"] == pytest.approx(interval_width(preds[low], preds[high]))
        assert row["nominal_coverage"] == pytest.approx(high - low)
        assert row["n"] == 6


# ---------------------------------------------------------------------------
# Bucket partition invariant
# ---------------------------------------------------------------------------


def test_reliability_curve_bucket_partition_invariant() -> None:
    idx = _hourly_utc("2021-01-01", 7)
    y = pd.Series([10.0] * 7, index=idx)
    median = pd.Series([-5.0, 10.0, 60.0, 120.0, 200.0, 300.0, 500.0], index=idx)
    preds = _full_grid(
        idx,
        {a: [10.0] * 7 for a in LEVELS if a != 0.50},
    )
    preds[0.50] = median

    out = reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS)

    # every fixed bucket edge is (-inf, inf) covering, i.e. a true partition
    bucket_names = [name for name, _, _ in FORECAST_LEVEL_BUCKETS]
    assert set(out["bucket"]) == {"overall", *bucket_names}
    for a in LEVELS:
        overall_n = out.loc[(out["bucket"] == "overall") & (out["level"] == a), "n"].iloc[0]
        bucket_n_sum = out.loc[(out["bucket"].isin(bucket_names)) & (out["level"] == a), "n"].sum()
        assert bucket_n_sum == overall_n
    # every bucket exists as a row, even ones with n == 0
    assert set(out.loc[out["bucket"] != "overall", "bucket"]).issuperset(bucket_names)


def test_band_metrics_bucket_partition_invariant() -> None:
    idx = _hourly_utc("2021-01-01", 7)
    y = pd.Series([10.0] * 7, index=idx)
    median = pd.Series([-5.0, 10.0, 60.0, 120.0, 200.0, 300.0, 500.0], index=idx)
    preds = _full_grid(
        idx,
        {a: [10.0] * 7 for a in LEVELS if a != 0.50},
    )
    preds[0.50] = median

    out = band_metrics(y, preds, buckets=FORECAST_LEVEL_BUCKETS)

    bucket_names = [name for name, _, _ in FORECAST_LEVEL_BUCKETS]
    for name, _, _ in NESTED_BANDS:
        overall_n = out.loc[(out["bucket"] == "overall") & (out["band"] == name), "n"].iloc[0]
        bucket_n_sum = out.loc[
            (out["bucket"].isin(bucket_names)) & (out["band"] == name), "n"
        ].sum()
        assert bucket_n_sum == overall_n


# ---------------------------------------------------------------------------
# Value correctness on a subset
# ---------------------------------------------------------------------------


def test_reliability_curve_value_correctness_on_bucket() -> None:
    idx = _hourly_utc("2021-01-01", 5)
    # all 5 hours fall in the 250_400 bucket via q_0.50
    median = pd.Series([300.0] * 5, index=idx)
    q95 = pd.Series([310.0, 310.0, 310.0, 310.0, 305.0], index=idx)
    y = pd.Series([300.0, 300.0, 300.0, 300.0, 320.0], index=idx)  # last hour exceeds q95
    preds = _full_grid(idx, {a: [300.0] * 5 for a in LEVELS if a not in (0.50, 0.95)})
    preds[0.50] = median
    preds[0.95] = q95

    out = reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS)
    row = out.loc[(out["bucket"] == "250_400") & (out["level"] == 0.95)].iloc[0]

    assert row["n"] == 5
    assert row["coverage"] == pytest.approx(0.8)  # 4 of 5 hours have y <= q95


# ---------------------------------------------------------------------------
# Leakage-free / diagnosis-only bucketing
# ---------------------------------------------------------------------------


def test_bucketing_uses_forecast_not_realised_price() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    median = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)  # all in 0_50 bucket
    y_normal = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)
    y_shifted = pd.Series([500.0, 500.0, 500.0, 500.0], index=idx)  # would be 400_plus if used
    preds = _full_grid(idx, {a: [10.0] * 4 for a in LEVELS if a != 0.50})
    preds[0.50] = median

    out_normal = reliability_curve(y_normal, preds, buckets=FORECAST_LEVEL_BUCKETS)
    out_shifted = reliability_curve(y_shifted, preds, buckets=FORECAST_LEVEL_BUCKETS)

    ns_normal = out_normal.loc[out_normal["level"] == 0.50].set_index("bucket")["n"]
    ns_shifted = out_shifted.loc[out_shifted["level"] == 0.50].set_index("bucket")["n"]
    assert ns_normal.equals(ns_shifted)
    assert ns_normal["0_50"] == 4
    assert ns_normal["400_plus"] == 0


# ---------------------------------------------------------------------------
# Band nominal value and missing level
# ---------------------------------------------------------------------------


def test_band_nominal_coverage_values() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {a: [5.0] * 3 for a in LEVELS})

    out = band_metrics(y, preds)
    nominal = out.set_index("band")["nominal_coverage"]
    assert nominal["90"] == pytest.approx(0.90)
    assert nominal["80"] == pytest.approx(0.80)
    assert nominal["50"] == pytest.approx(0.50)


def test_band_metrics_skips_missing_levels() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {0.05: [3.0] * 3, 0.50: [5.0] * 3, 0.95: [8.0] * 3})

    out = band_metrics(y, preds)

    assert set(out["band"]) == {"90"}


# ---------------------------------------------------------------------------
# Empty subset
# ---------------------------------------------------------------------------


def test_reliability_curve_empty_bucket_no_crash() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    median = pd.Series([10.0] * 3, index=idx)  # never in negative or 400_plus
    preds = _full_grid(idx, {a: [5.0] * 3 for a in LEVELS if a != 0.50})
    preds[0.50] = median

    out = reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS)

    empty_rows = out.loc[out["bucket"] == "negative"]
    assert (empty_rows["n"] == 0).all()
    assert empty_rows["coverage"].isna().all()
    assert empty_rows["pinball"].isna().all()


def test_band_metrics_empty_bucket_no_crash() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    median = pd.Series([10.0] * 3, index=idx)
    preds = _full_grid(idx, {a: [5.0] * 3 for a in LEVELS if a != 0.50})
    preds[0.50] = median

    out = band_metrics(y, preds, buckets=FORECAST_LEVEL_BUCKETS)

    empty_rows = out.loc[out["bucket"] == "negative"]
    assert (empty_rows["n"] == 0).all()
    assert empty_rows["coverage"].isna().all()
    assert empty_rows["width"].isna().all()


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


def test_reliability_curve_column_contract() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {a: [5.0] * 3 for a in LEVELS})

    out = reliability_curve(y, preds)

    assert set(out.columns) == {"bucket", "level", "coverage", "pinball", "n"}
    assert (out["n"] >= 0).all()
    assert out["n"].apply(lambda v: float(v).is_integer()).all()


def test_band_metrics_column_contract() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {a: [5.0] * 3 for a in LEVELS})

    out = band_metrics(y, preds)

    assert set(out.columns) == {"bucket", "band", "nominal_coverage", "coverage", "width", "n"}
    assert (out["n"] >= 0).all()
    assert out["n"].apply(lambda v: float(v).is_integer()).all()


# ---------------------------------------------------------------------------
# Fail-fast contracts
# ---------------------------------------------------------------------------


def test_reliability_curve_bucketing_without_median_raises() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {0.05: [3.0] * 3, 0.95: [8.0] * 3})  # no 0.5

    with pytest.raises(ValueError, match="0.5"):
        reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS)


def test_band_metrics_bucketing_without_median_raises() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {0.05: [3.0] * 3, 0.95: [8.0] * 3})  # no 0.5

    with pytest.raises(ValueError, match="0.5"):
        band_metrics(y, preds, buckets=FORECAST_LEVEL_BUCKETS)


def test_bucketing_with_explicit_forecast_level_bypasses_median_requirement() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    preds = _full_grid(idx, {0.05: [3.0] * 3, 0.95: [8.0] * 3})
    forecast_level = pd.Series([10.0, 60.0, 300.0], index=idx)

    out = reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS, forecast_level=forecast_level)
    assert not math.isnan(out.loc[(out["bucket"] == "0_50") & (out["level"] == 0.05), "n"].iloc[0])
