from __future__ import annotations

import pandas as pd

from .availability import Feature, combine, forecast_for_target, lag
from .config import FeatureConfig


def build_forecast_fundamentals(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> list[Feature]:
    """Day-ahead forecast fundamentals for the target day (all DA_FORECAST).

    The raw forecasts are used directly; residual load and renewable share are
    derived via `combine` (knowledge time = max over the forecasts = gate closure).
    Leakage-correct replacement for the actuals-based EDA quantities. Needs no
    config (no tunable knobs here).
    """
    load = forecast_for_target(
        "load_forecast_day_ahead",
        df["load_forecast_day_ahead"],
        "load_forecast_day_ahead",
        target_index=target_index,
    )
    won = forecast_for_target(
        "wind_onshore_forecast",
        df["wind_onshore_forecast"],
        "wind_onshore_forecast",
        target_index=target_index,
    )
    woff = forecast_for_target(
        "wind_offshore_forecast",
        df["wind_offshore_forecast"],
        "wind_offshore_forecast",
        target_index=target_index,
    )
    solar = forecast_for_target(
        "solar_forecast",
        df["solar_forecast"],
        "solar_forecast",
        target_index=target_index,
    )
    residual = combine(
        "residual_load_forecast",
        [load, won, woff, solar],
        lambda load_, won_, woff_, solar_: load_ - won_ - woff_ - solar_,
    )
    share = combine(
        "renewable_share_forecast",
        [load, won, woff, solar],
        lambda load_, won_, woff_, solar_: (won_ + woff_ + solar_) / load_,
    )
    return [load, won, woff, solar, residual, share]


def build_commodity_features(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    config: FeatureConfig | None = None,
) -> list[Feature]:
    """TTF gas and EUA CO2 as <commodity_lag_hours> lags (latest settled close at
    gate closure), plus an eua_missing flag (1 where the lagged EUA value is NaN,
    e.g. pre-Oct-2021). The lag horizon comes from the FeatureConfig.
    """
    cfg = config if config is not None else FeatureConfig()
    lag_h = cfg.commodity_lag_hours
    ttf = lag(
        f"ttf_gas_lag_{lag_h}h",
        df["ttf_gas_eur_per_mwh"],
        "ttf_gas_eur_per_mwh",
        hours=lag_h,
        target_index=target_index,
    )
    eua = lag(
        f"eua_co2_lag_{lag_h}h",
        df["eua_co2_eur_per_t"],
        "eua_co2_eur_per_t",
        hours=lag_h,
        target_index=target_index,
    )
    eua_missing = combine("eua_missing", [eua], lambda s: s.isna().astype("int8"))
    return [ttf, eua, eua_missing]
