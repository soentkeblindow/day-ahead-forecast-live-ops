import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.data.loaders import load_processed_features
from energy_price_forecast.evaluation.config import RegimeConfig
from energy_price_forecast.evaluation.regimes import (
    MACRO_REGIME_COLUMN,
    REGIME_FLAG_COLUMNS,
    tag_regimes,
)
from energy_price_forecast.market_time import LOCAL_TZ

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _col(value: float | list[float], n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if len(arr) != n:
        raise ValueError(f"expected length {n}, got {len(arr)}")
    return arr


def _frame(
    index: pd.DatetimeIndex,
    *,
    load: float | list[float] = 50_000.0,
    wind_onshore: float | list[float] = 6_000.0,
    wind_offshore: float | list[float] = 3_000.0,
    solar: float | list[float] = 2_000.0,
    price: float | list[float] = 50.0,
) -> pd.DataFrame:
    """An unremarkable interim frame: no flag should fire on these defaults.

    residual share = (50_000 - 11_000) / 50_000 = 0.78 (< 0.90), wind = 9_000 MW
    (< 20_000), price = 50 (not negative, not a group outlier).
    """
    n = len(index)
    return pd.DataFrame(
        {
            "load_actual": _col(load, n),
            "gen_wind_onshore": _col(wind_onshore, n),
            "gen_wind_offshore": _col(wind_offshore, n),
            "gen_solar": _col(solar, n),
            "day_ahead_price": _col(price, n),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Threshold strictness
# ---------------------------------------------------------------------------


def test_renewable_scarcity_threshold_strict() -> None:
    idx = _hourly_utc("2020-01-01", 3)
    df = _frame(idx, load=100.0, wind_onshore=[9.5, 10.0, 10.5], wind_offshore=0.0, solar=0.0)
    out = tag_regimes(df)
    assert out["renewable_scarcity"].tolist() == [True, False, False]


def test_high_wind_threshold_strict() -> None:
    idx = _hourly_utc("2020-01-01", 3)
    df = _frame(
        idx,
        load=100_000.0,
        wind_onshore=[20_000.5, 20_000.0, 19_999.5],
        wind_offshore=0.0,
        solar=0.0,
    )
    out = tag_regimes(df)
    assert out["high_wind"].tolist() == [True, False, False]


def test_negative_price_boundary() -> None:
    idx = _hourly_utc("2020-01-01", 3)
    df = _frame(idx, price=[-0.01, 0.0, 5.0])
    out = tag_regimes(df)
    assert out["negative_price"].tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# Price spike: regime-relative percentile
# ---------------------------------------------------------------------------


def test_price_spike_is_regime_relative() -> None:
    # One contiguous hourly frame (a real interim frame has no calendar gaps).
    # Start exactly at Europe/Berlin local midnight so two 24h blocks line up
    # with clean local calendar days -> a "calm" day and a "crisis" day.
    cfg = RegimeConfig(
        crisis_start=pd.Timestamp("2021-01-03", tz=LOCAL_TZ),
        post_crisis_start=pd.Timestamp("2099-01-01", tz=LOCAL_TZ),
    )
    idx = _hourly_utc("2021-01-01 23:00", 48)  # UTC 23:00 == Berlin midnight (winter)
    calm_idx, crisis_idx = idx[:24], idx[24:]

    calm_prices = [float(p) for p in range(1, 25)]  # 1..24, 95th pct ~ 22.85 -> top two spike
    crisis_prices = [p + 1_000 for p in calm_prices]  # same shape, shifted level

    df = _frame(idx, price=calm_prices + crisis_prices)
    out = tag_regimes(df, config=cfg)

    assert out.loc[calm_idx, MACRO_REGIME_COLUMN].eq("calm").all()
    assert out.loc[crisis_idx, MACRO_REGIME_COLUMN].eq("crisis").all()

    calm_flags = out.loc[calm_idx, "price_spike"]
    crisis_flags = out.loc[crisis_idx, "price_spike"]

    # Same relative shape in both groups -> same count flagged, regardless of
    # the crisis group's absolute price level being 1000 higher throughout.
    assert calm_flags.sum() == 2
    assert calm_flags.iloc[-2:].all()
    assert not calm_flags.iloc[0]
    assert crisis_flags.sum() == 2
    assert crisis_flags.iloc[-2:].all()
    assert not crisis_flags.iloc[0]


# ---------------------------------------------------------------------------
# Overlap (Entscheidung 0) and the normal catch-all / partition invariant
# ---------------------------------------------------------------------------


def test_flags_can_overlap() -> None:
    idx = _hourly_utc("2020-01-01", 21)
    prices = [50.0] * 20 + [500.0]
    df = _frame(
        idx,
        wind_onshore=[6_000.0] * 20 + [0.0],
        wind_offshore=[3_000.0] * 20 + [0.0],
        solar=[2_000.0] * 20 + [0.0],
        price=prices,
    )

    out = tag_regimes(df)
    last = out.iloc[-1]

    assert bool(last["renewable_scarcity"])
    assert bool(last["price_spike"])
    assert not bool(last["normal"])


def test_normal_catchall_and_partition_invariant() -> None:
    idx = _hourly_utc("2020-01-01", 5)
    df = _frame(idx)
    out = tag_regimes(df)

    assert out["normal"].all()

    special_columns = list(REGIME_FLAG_COLUMNS[:-1])  # exclude "normal" itself
    specials = out[special_columns].any(axis=1)
    assert (out["normal"] == ~specials).all()


# ---------------------------------------------------------------------------
# Macro regime axis
# ---------------------------------------------------------------------------


def test_macro_regime_labels_before_during_after() -> None:
    calm_idx = _hourly_utc("2021-08-30 22:00", 3)
    crisis_idx = _hourly_utc("2022-06-15 08:00", 3)
    post_idx = _hourly_utc("2023-04-01 08:00", 3)

    assert tag_regimes(_frame(calm_idx))[MACRO_REGIME_COLUMN].eq("calm").all()
    assert tag_regimes(_frame(crisis_idx))[MACRO_REGIME_COLUMN].eq("crisis").all()
    assert tag_regimes(_frame(post_idx))[MACRO_REGIME_COLUMN].eq("post_crisis").all()


def test_macro_regime_partition_is_exhaustive_and_disjoint() -> None:
    # Synthetic, close-together boundaries so the whole test stays a small,
    # by-hand-checkable frame instead of spanning the real ~19-month gap.
    cfg = RegimeConfig(
        crisis_start=pd.Timestamp("2021-01-02", tz=LOCAL_TZ),
        post_crisis_start=pd.Timestamp("2021-01-03", tz=LOCAL_TZ),
    )
    idx = _hourly_utc("2021-01-01 00:00", 96)  # 4 days, spans both boundaries
    out = tag_regimes(_frame(idx), config=cfg)
    labels = out[MACRO_REGIME_COLUMN]

    assert labels.isna().sum() == 0
    assert set(labels.unique().tolist()) == {"calm", "crisis", "post_crisis"}
    assert labels.value_counts().sum() == len(idx)


# ---------------------------------------------------------------------------
# NaN / zero-load robustness
# ---------------------------------------------------------------------------


def test_nan_generation_and_zero_load_no_crash() -> None:
    idx = _hourly_utc("2020-01-01", 2)
    df = _frame(idx, load=[50_000.0, 0.0])
    df.loc[idx[0], "gen_wind_onshore"] = np.nan

    out = tag_regimes(df)

    assert not out["renewable_scarcity"].any()
    assert not out["high_wind"].any()
    assert out["normal"].all()


# ---------------------------------------------------------------------------
# Index identity + dtypes
# ---------------------------------------------------------------------------


def test_index_identity_and_dtypes() -> None:
    idx = _hourly_utc("2020-01-01", 5)
    df = _frame(idx)
    out = tag_regimes(df)

    assert out.index.equals(df.index)
    for col in REGIME_FLAG_COLUMNS:
        assert out[col].dtype == bool
    assert isinstance(out[MACRO_REGIME_COLUMN].dtype, pd.CategoricalDtype)


def test_empty_frame_returns_typed_empty_table() -> None:
    idx = _hourly_utc("2020-01-01", 0)
    df = _frame(idx)
    out = tag_regimes(df)

    assert len(out) == 0
    assert list(out.columns) == [*REGIME_FLAG_COLUMNS, MACRO_REGIME_COLUMN]


# ---------------------------------------------------------------------------
# Error contracts
# ---------------------------------------------------------------------------


def test_missing_required_column_raises() -> None:
    idx = _hourly_utc("2020-01-01", 3)
    df = _frame(idx).drop(columns=["load_actual"])
    with pytest.raises(ValueError, match="load_actual"):
        tag_regimes(df)


def test_non_hourly_index_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="2h", tz="UTC")
    df = _frame(idx)
    with pytest.raises(ValueError, match="hourly"):
        tag_regimes(df)


def test_non_utc_index_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="h")  # tz-naive
    df = _frame(idx)
    with pytest.raises(ValueError, match="UTC"):
        tag_regimes(df)


# ---------------------------------------------------------------------------
# Diagnostic-only contract (the leakage guard)
# ---------------------------------------------------------------------------


def test_flag_columns_disjoint_from_feature_matrix(tmp_path: pytest.TempPathFactory) -> None:
    # data/processed/ is gitignored, so CI has no real feature matrix. This
    # fixture stands in for it -- the contract under test ("regime flag names
    # never appear as feature columns") is independent of the fixture's
    # concrete column names.
    idx = _hourly_utc("2020-01-01", 4)
    features = pd.DataFrame({"lag_price_24h": 1.0, "is_weekend": 0}, index=idx)
    path = tmp_path / "features.parquet"  # type: ignore[operator]
    features.to_parquet(path)

    loaded = load_processed_features(path)
    all_regime_columns = set(REGIME_FLAG_COLUMNS) | {MACRO_REGIME_COLUMN}
    assert all_regime_columns.isdisjoint(loaded.columns)


# ---------------------------------------------------------------------------
# Config effect
# ---------------------------------------------------------------------------


def test_config_thresholds_are_not_hardcoded() -> None:
    idx = _hourly_utc("2020-01-01", 1)
    df = _frame(idx, wind_onshore=12_000.0, wind_offshore=0.0)

    default_out = tag_regimes(df)
    custom_out = tag_regimes(df, config=RegimeConfig(high_wind_generation_mw=10_000.0))

    assert not bool(default_out["high_wind"].iloc[0])
    assert bool(custom_out["high_wind"].iloc[0])
