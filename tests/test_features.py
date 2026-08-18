import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.features.availability import (
    LeakageError,
    assert_no_leakage,
    rolling_mean,
)
from energy_price_forecast.features.build import build_feature_matrix, trim_warmup
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig  # noqa: F401
from energy_price_forecast.features.fundamentals import (
    build_commodity_features,
    build_forecast_fundamentals,
)
from energy_price_forecast.features.lags import (
    build_actual_lags,
    build_cross_border_lags,
    build_forecast_error_lags,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


def test_feature_config_defaults() -> None:
    cfg = FeatureConfig()
    assert cfg.commodity_lag_hours == 48
    assert cfg.crisis_start == pd.Timestamp("2021-09-01", tz="Europe/Berlin")
    assert cfg.post_crisis_start == pd.Timestamp("2023-04-01", tz="Europe/Berlin")


# ---------------------------------------------------------------------------
# Cyclic calendar features
# ---------------------------------------------------------------------------


def test_hour_sin_zero_at_local_midnight() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET (local hour 0); sin(0) = 0
    idx = _hourly_utc("2024-01-14 23:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["hour_sin"].values.iloc[0] == pytest.approx(0.0)


def test_hour_sin_one_at_local_hour_6() -> None:
    # 2024-01-15 05:00 UTC = 2024-01-15 06:00 CET; sin(2π*6/24) = sin(π/2) = 1
    idx = _hourly_utc("2024-01-15 05:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["hour_sin"].values.iloc[0] == pytest.approx(1.0, abs=1e-10)


def test_cyclic_values_in_range() -> None:
    idx = _hourly_utc("2024-01-01 00:00", 24 * 7)
    feats = {f.name: f for f in build_calendar_features(idx)}
    for name in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"):
        vals = feats[name].values
        assert vals.min() >= -1.0 - 1e-12 and vals.max() <= 1.0 + 1e-12, f"{name} out of [-1, 1]"


def test_month_cos_january_is_one() -> None:
    # January: position 0; cos(2π*0/12) = 1
    idx = _hourly_utc("2024-01-01 00:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["month_cos"].values.iloc[0] == pytest.approx(1.0)


def test_weekday_cos_monday_is_one() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET = Monday (dayofweek 0); cos(0) = 1
    idx = _hourly_utc("2024-01-14 23:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["weekday_cos"].values.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_weekend
# ---------------------------------------------------------------------------


def test_is_weekend_saturday() -> None:
    # 2024-01-06 00:00 UTC = 2024-01-06 01:00 CET = Saturday
    idx = _hourly_utc("2024-01-06 00:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_weekend"].values == 1).all()


def test_is_weekend_wednesday() -> None:
    # 2024-01-09 23:00 UTC = 2024-01-10 00:00 CET = Wednesday
    idx = _hourly_utc("2024-01-09 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_weekend"].values == 0).all()


# ---------------------------------------------------------------------------
# is_holiday (nationwide) and is_regional_holiday
# ---------------------------------------------------------------------------


def test_is_holiday_tag_der_deutschen_einheit() -> None:
    # 2024-10-02 22:00 UTC = 2024-10-03 00:00 CEST = Oct 3 (nationwide)
    idx = _hourly_utc("2024-10-02 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_holiday"].values == 1).all()


def test_is_holiday_zero_on_fronleichnam() -> None:
    # Fronleichnam is regional only; 2024-05-29 22:00 UTC = 2024-05-30 00:00 CEST
    idx = _hourly_utc("2024-05-29 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_holiday"].values == 0).all()


def test_is_regional_holiday_fronleichnam_2024() -> None:
    # Easter 2024 = 2024-03-31; +60d = 2024-05-30 (Fronleichnam)
    idx = _hourly_utc("2024-05-29 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 1).all()


def test_is_regional_holiday_allerheiligen() -> None:
    # 2024-10-31 23:00 UTC = 2024-11-01 00:00 CET = Allerheiligen
    idx = _hourly_utc("2024-10-31 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 1).all()


def test_is_regional_holiday_zero_on_nationwide_holiday() -> None:
    # Oct 3 is nationwide; regional flag must be 0
    idx = _hourly_utc("2024-10-02 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 0).all()


def test_is_regional_holiday_zero_on_normal_weekday() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET = Monday, no holiday
    idx = _hourly_utc("2024-01-14 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 0).all()


# ---------------------------------------------------------------------------
# Local-time correctness
# ---------------------------------------------------------------------------


def test_local_time_near_utc_midnight() -> None:
    # 2024-01-07 23:30 UTC = 2024-01-08 00:30 CET → Monday (is_weekend = 0)
    # Without TZ conversion Jan 7 UTC is Sunday → is_weekend would be 1 (wrong)
    idx = pd.DatetimeIndex(["2024-01-07 23:30"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_weekend"].values.iloc[0] == 0


# ---------------------------------------------------------------------------
# Regime dummies
# ---------------------------------------------------------------------------


def test_regime_before_crisis() -> None:
    # 2021-08-31 21:59 UTC = 2021-08-31 23:59 CEST → local day 2021-08-31 → calm
    idx = pd.DatetimeIndex(["2021-08-31 21:59"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 0
    assert feats["is_post_crisis"].values.iloc[0] == 0


def test_regime_at_crisis_start() -> None:
    # 2021-08-31 22:00 UTC = 2021-09-01 00:00 CEST → crisis_start boundary → is_crisis = 1
    idx = pd.DatetimeIndex(["2021-08-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 1
    assert feats["is_post_crisis"].values.iloc[0] == 0


def test_regime_at_post_crisis_start() -> None:
    # 2023-04-01 00:00 CEST = 2023-03-31 22:00 UTC → post_crisis boundary
    idx = pd.DatetimeIndex(["2023-03-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 0
    assert feats["is_post_crisis"].values.iloc[0] == 1


def test_regime_custom_config_boundaries() -> None:
    cfg = FeatureConfig(
        crisis_start=pd.Timestamp("2022-01-01", tz="Europe/Berlin"),
        post_crisis_start=pd.Timestamp("2023-01-01", tz="Europe/Berlin"),
    )
    # 2022-06-01 00:00 CEST = 2022-05-31 22:00 UTC → inside crisis window
    idx = pd.DatetimeIndex(["2022-05-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx, cfg)}
    assert feats["is_crisis"].values.iloc[0] == 1
    assert feats["is_post_crisis"].values.iloc[0] == 0


# ---------------------------------------------------------------------------
# Forecast fundamentals
# ---------------------------------------------------------------------------


def _make_fc_df(periods: int = 48) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "solar_forecast": np.full(periods, 5000.0),
        },
        index=idx,
    )
    return df, idx


def test_residual_load_forecast_value() -> None:
    df, idx = _make_fc_df()
    feats = {f.name: f for f in build_forecast_fundamentals(df, idx)}
    expected = 40000.0 - 8000.0 - 2000.0 - 5000.0
    assert feats["residual_load_forecast"].values.to_numpy() == pytest.approx(expected)


def test_renewable_share_forecast_value() -> None:
    df, idx = _make_fc_df()
    feats = {f.name: f for f in build_forecast_fundamentals(df, idx)}
    expected = (8000.0 + 2000.0 + 5000.0) / 40000.0
    assert feats["renewable_share_forecast"].values.to_numpy() == pytest.approx(expected)


def test_forecast_fundamentals_pass_leakage() -> None:
    df, idx = _make_fc_df()
    assert_no_leakage(build_forecast_fundamentals(df, idx))


# ---------------------------------------------------------------------------
# Commodity features
# ---------------------------------------------------------------------------


def _make_commodity_df(periods: int = 5 * 24) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    eua_vals = np.full(periods, 70.0)
    eua_vals[:24] = np.nan  # first 24 hours NaN (pre-EUA placeholder)
    df = pd.DataFrame(
        {
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": eua_vals,
        },
        index=idx,
    )
    return df, idx


def test_commodity_features_pass_leakage() -> None:
    df, idx = _make_commodity_df()
    assert_no_leakage(build_commodity_features(df, idx))


def test_eua_missing_flag_where_nan() -> None:
    df, idx = _make_commodity_df()
    feats = {f.name: f for f in build_commodity_features(df, idx)}
    # target[48] -> source[0] (NaN eua) -> missing = 1
    assert feats["eua_missing"].values.iloc[48] == 1
    # target[72] -> source[24] (eua = 70.0) -> missing = 0
    assert feats["eua_missing"].values.iloc[72] == 0


def test_commodity_feature_name_from_config() -> None:
    cfg = FeatureConfig(commodity_lag_hours=72)
    df, idx = _make_commodity_df(periods=7 * 24)
    feats = {f.name: f for f in build_commodity_features(df, idx, cfg)}
    assert "ttf_gas_lag_72h" in feats
    assert "eua_co2_lag_72h" in feats


def test_commodity_lag_24h_fails_leakage() -> None:
    cfg = FeatureConfig(commodity_lag_hours=24)
    df, idx = _make_commodity_df()
    with pytest.raises(LeakageError):
        assert_no_leakage(build_commodity_features(df, idx, cfg))


# ---------------------------------------------------------------------------
# build_feature_matrix — wiring (all builders together, synthetic data)
# ---------------------------------------------------------------------------


def _make_full_df(periods: int = 5 * 24) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "solar_forecast": np.full(periods, 5000.0),
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": np.full(periods, 70.0),
        },
        index=idx,
    )


def test_build_feature_matrix_passes_leakage_and_shape() -> None:
    df = _make_full_df_with_lags()
    x = build_feature_matrix(df)
    assert len(x) == len(df)


def _make_full_df_with_lags(periods: int = 10 * 24) -> pd.DataFrame:
    """Extended DataFrame with all columns needed for 2.3.3 features."""
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "day_ahead_price": np.full(periods, 50.0),
            "load_actual": np.full(periods, 40000.0),
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "gen_wind_onshore": np.full(periods, 8000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "gen_wind_offshore": np.full(periods, 2000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "gen_solar": np.full(periods, 5000.0),
            "solar_forecast": np.full(periods, 5000.0),
            "scheduled_net_de_to_AT": np.full(periods, 1000.0),
            "scheduled_net_de_to_BE": np.full(periods, 500.0),
            "physical_net_de_to_AT": np.full(periods, 1200.0),
            "physical_net_de_to_BE": np.full(periods, 600.0),
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": np.full(periods, 70.0),
        },
        index=idx,
    )
    return df


def test_build_feature_matrix_expected_columns() -> None:
    df = _make_full_df_with_lags()
    x = build_feature_matrix(df)
    expected = {
        # 2.3.2 features
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
        "is_regional_holiday",
        "is_crisis",
        "is_post_crisis",
        "load_forecast_day_ahead",
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
        "residual_load_forecast",
        "renewable_share_forecast",
        "ttf_gas_lag_48h",
        "eua_co2_lag_48h",
        "eua_missing",
        # 2.3.3 features
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "price_roll_mean_24h",
        "price_roll_mean_168h",
        "load_actual_lag_48h",
        "load_actual_lag_168h",
        "load_forecast_error_lag_48h",
        "wind_onshore_forecast_error_lag_48h",
        "wind_offshore_forecast_error_lag_48h",
        "solar_forecast_error_lag_48h",
        "scheduled_net_export_lag_24h",
        "physical_net_export_lag_48h",
        "cross_border_deviation_lag_48h",
    }
    assert set(x.columns) == expected


def test_rolling_mean_value_correctness() -> None:
    """Test rolling_mean values against manual calculation."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    raw = pd.Series(range(48), index=idx, name="test")  # 0, 1, 2, ..., 47
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")  # 48h later

    # window=3, lag=2: value(t) = mean(raw[t-2-3+1 : t-2])
    feature = rolling_mean(
        "test_roll", raw, "day_ahead_price", window_hours=3, lag_hours=2, target_index=target_idx
    )

    # target_idx[0] = 2024-01-03 00:00 -> source = [2024-01-02 20:00, 21:00, 22:00] = [44, 45, 46]
    # mean = (44 + 45 + 46) / 3 = 45.0 (Pandas rolling is inclusive)
    assert feature.values.iloc[0] == 45.0

    # target_idx[1] = 2024-01-03 01:00 -> source = [21:00, 22:00, 23:00] = [45, 46, 47]
    # mean = (45 + 46 + 47) / 3 = 46.0
    assert feature.values.iloc[1] == 46.0


def test_rolling_mean_knowledge_time() -> None:
    """Test that knowledge_time = leading edge (independent of window size)."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    raw = pd.Series(range(48), index=idx, name="test")
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    # window=3, lag=2
    feature = rolling_mean(
        "test_roll", raw, "day_ahead_price", window_hours=3, lag_hours=2, target_index=target_idx
    )

    # Leading edge for target_idx[0] = 2024-01-03 00:00 UTC - 2h = 2024-01-02 22:00 UTC.
    # DA_FIXED (UTC-anchored): normalize(2024-01-02 22:00 UTC) = 2024-01-02 00:00 UTC.
    expected_kt = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert feature.knowledge_time.iloc[0] == expected_kt

    # Same for window=6 (knowledge_time unchanged)
    feature_wide = rolling_mean(
        "test_roll_wide",
        raw,
        "day_ahead_price",
        window_hours=6,
        lag_hours=2,
        target_index=target_idx,
    )
    assert feature_wide.knowledge_time.iloc[0] == expected_kt


def test_rolling_mean_leakage_safe() -> None:
    """Test that rolling_mean with DA_FIXED (24h lag) passes leakage check."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    raw = pd.Series(range(48), index=idx, name="test")
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    feature = rolling_mean(
        "test_roll", raw, "day_ahead_price", window_hours=24, lag_hours=24, target_index=target_idx
    )
    assert_no_leakage([feature])


def test_actual_lag_24h_fails_leakage() -> None:
    """Test that 24h Actual-Lag leaks in the afternoon (RT_ACTUAL)."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"load_actual": np.full(48, 40000.0)}, index=idx)
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    # 24h lag for RT_ACTUAL should leak
    feature = build_actual_lags(df, target_idx, FeatureConfig(actual_lags_hours=(24,)))[0]
    with pytest.raises(LeakageError, match="known after gate closure"):
        assert_no_leakage([feature])


def test_actual_lag_48h_passes_leakage() -> None:
    """Test that 48h Actual-Lag is leakage-safe."""
    idx = pd.date_range("2024-01-01 00:00", periods=72, freq="h", tz="UTC")
    df = pd.DataFrame({"load_actual": np.full(72, 40000.0)}, index=idx)
    target_idx = pd.date_range("2024-01-04 00:00", periods=24, freq="h", tz="UTC")

    feature = build_actual_lags(df, target_idx, FeatureConfig(actual_lags_hours=(48,)))[0]
    assert_no_leakage([feature])


def test_forecast_error_lag_24h_fails_leakage() -> None:
    """Test that 24h Forecast-Error-Lag leaks (inherits RT_ACTUAL binding)."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_actual": np.full(48, 40000.0),
            "load_forecast_day_ahead": np.full(48, 39000.0),
            "gen_wind_onshore": np.full(48, 8000.0),
            "wind_onshore_forecast": np.full(48, 8000.0),
            "gen_wind_offshore": np.full(48, 2000.0),
            "wind_offshore_forecast": np.full(48, 2000.0),
            "gen_solar": np.full(48, 5000.0),
            "solar_forecast": np.full(48, 5000.0),
        },
        index=idx,
    )
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    # Filter to only load_forecast_error (avoids missing columns for other errors)
    feature = build_forecast_error_lags(
        df, target_idx, FeatureConfig(forecast_error_lags_hours=(24,))
    )[0]  # load_forecast_error_lag_24h
    with pytest.raises(LeakageError, match="known after gate closure"):
        assert_no_leakage([feature])


def test_forecast_error_lag_48h_passes_leakage() -> None:
    """Test that 48h Forecast-Error-Lag is leakage-safe."""
    idx = pd.date_range("2024-01-01 00:00", periods=72, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_actual": np.full(72, 40000.0),
            "load_forecast_day_ahead": np.full(72, 39000.0),
            "gen_wind_onshore": np.full(72, 8000.0),
            "wind_onshore_forecast": np.full(72, 8000.0),
            "gen_wind_offshore": np.full(72, 2000.0),
            "wind_offshore_forecast": np.full(72, 2000.0),
            "gen_solar": np.full(72, 5000.0),
            "solar_forecast": np.full(72, 5000.0),
        },
        index=idx,
    )
    target_idx = pd.date_range("2024-01-04 00:00", periods=24, freq="h", tz="UTC")

    feature = build_forecast_error_lags(
        df, target_idx, FeatureConfig(forecast_error_lags_hours=(48,))
    )[0]  # load_forecast_error_lag_48h
    assert_no_leakage([feature])


def test_cross_border_lag_24h_passes_leakage() -> None:
    """Test that 24h Scheduled-Flow-Lag is leakage-safe (DA_FIXED)."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "scheduled_net_de_to_AT": np.full(48, 1000.0),
            "scheduled_net_de_to_BE": np.full(48, 500.0),
            "physical_net_de_to_AT": np.full(48, 1200.0),  # Required for deviation
            "physical_net_de_to_BE": np.full(48, 600.0),  # Required for deviation
        },
        index=idx,
    )
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    feature = build_cross_border_lags(
        df, target_idx, FeatureConfig(scheduled_flow_lags_hours=(24,))
    )[0]
    assert_no_leakage([feature])


def test_cross_border_physical_lag_24h_fails_leakage() -> None:
    """Test that 24h Physical-Flow-Lag leaks (RT_ACTUAL)."""
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "scheduled_net_de_to_AT": np.full(48, 1000.0),  # Required for deviation
            "scheduled_net_de_to_BE": np.full(48, 500.0),  # Required for deviation
            "physical_net_de_to_AT": np.full(48, 1200.0),
            "physical_net_de_to_BE": np.full(48, 600.0),
        },
        index=idx,
    )
    target_idx = pd.date_range("2024-01-03 00:00", periods=24, freq="h", tz="UTC")

    # Filter to physical_net_export_lag_24h (avoids deviation feature)
    features = build_cross_border_lags(
        df, target_idx, FeatureConfig(physical_flow_lags_hours=(24,))
    )
    feature = [f for f in features if f.name == "physical_net_export_lag_24h"][0]
    with pytest.raises(LeakageError, match="known after gate closure"):
        assert_no_leakage([feature])


def test_trim_warmup_removes_correct_rows() -> None:
    """Test that trim_warmup removes only the first max_lookback_hours rows."""
    df = _make_full_df_with_lags(periods=200)
    matrix = build_feature_matrix(df)

    # Default config: max_lookback_hours = 191 (168h lag + 24h rolling window - 1)
    trimmed = trim_warmup(matrix)
    expected_start = df.index[0] + pd.Timedelta(hours=FeatureConfig().max_lookback_hours())
    assert trimmed.index[0] == expected_start
    assert len(trimmed) == len(matrix) - FeatureConfig().max_lookback_hours()


def test_trim_warmup_preserves_eua_nan_region() -> None:
    """Test that trim_warmup does NOT drop the intentional EUA NaN region."""
    idx = pd.date_range("2021-09-01 00:00", periods=200, freq="h", tz="UTC")  # Post-crisis start
    eua_vals = np.full(200, 70.0)
    # max_lookback_hours = 191; eua_co2_lag_48h at row 191 looks back to row 143.
    # NaN must extend past row 143 to survive the trim.
    eua_vals[:150] = np.nan
    df = pd.DataFrame(
        {
            "day_ahead_price": np.full(200, 50.0),
            "load_actual": np.full(200, 40000.0),
            "load_forecast_day_ahead": np.full(200, 40000.0),
            "gen_wind_onshore": np.full(200, 8000.0),
            "wind_onshore_forecast": np.full(200, 8000.0),
            "gen_wind_offshore": np.full(200, 2000.0),
            "wind_offshore_forecast": np.full(200, 2000.0),
            "gen_solar": np.full(200, 5000.0),
            "solar_forecast": np.full(200, 5000.0),
            "scheduled_net_de_to_AT": np.full(200, 1000.0),
            "physical_net_de_to_AT": np.full(200, 1200.0),
            "ttf_gas_eur_per_mwh": np.full(200, 30.0),
            "eua_co2_eur_per_t": eua_vals,
        },
        index=idx,
    )

    matrix = build_feature_matrix(df)
    trimmed = trim_warmup(matrix)

    # EUA NaN region should survive (pre-Oct-2021)
    assert trimmed["eua_co2_lag_48h"].isna().any()
    # Other features should have no NaN after warm-up
    assert not trimmed["price_lag_24h"].isna().any()
