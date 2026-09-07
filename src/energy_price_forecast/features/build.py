from __future__ import annotations

import datetime as dt

import pandas as pd

from ..ops.windows import local_day_bounds
from .availability import build_matrix
from .calendar import build_calendar_features
from .config import FeatureConfig
from .fundamentals import build_commodity_features, build_forecast_fundamentals
from .lags import (
    build_actual_lags,
    build_cross_border_lags,
    build_forecast_error_lags,
    build_price_lags,
)
from .nwp_fundamentals import build_nwp_forecast_fundamentals


def build_feature_matrix(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Assemble the full Sprint-2.3 feature matrix and leakage-check it.

    Order: deterministic calendar/regime, forecast fundamentals, commodities
    (2.3.2), then lags, rolling means, forecast errors, cross-border lags (2.3.3).
    build_matrix runs assert_no_leakage before concatenating. Returns the matrix
    on the full hourly index (no warm-up trimming here -- see trim_warmup).
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = pd.DatetimeIndex(df.index)
    features = [
        *build_calendar_features(target_index, cfg),
        *build_forecast_fundamentals(df, target_index),
        *build_commodity_features(df, target_index, cfg),
        *build_price_lags(df, target_index, cfg),
        *build_actual_lags(df, target_index, cfg),
        *build_forecast_error_lags(df, target_index, cfg),
        *build_cross_border_lags(df, target_index, cfg),
    ]
    return build_matrix(features)  # asserts no leakage, then concatenates


def _hourly_utc_index_for_local_day(target_day: dt.date) -> pd.DatetimeIndex:
    """UTC hourly timestamps of one local (Europe/Berlin) delivery day --
    23/24/25 rows across DST, derived purely from the calendar (spec 6.5.3
    section 3.4, E7: never derived from the system clock)."""
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _build_day_matrix(
    target_index: pd.DatetimeIndex,
    fundamentals: list,
    df: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Shared Bausteinliste for one day's feature rows (spec 6.6 section
    5.1: "derselbe target_index, dieselbe Bausteinliste" for both the live
    and the original feature-set builder below). Only the fundamentals
    (forecast columns) differ between the two callers; calendar,
    commodities and lags are identical."""
    features = [
        *build_calendar_features(target_index, cfg),
        *fundamentals,
        *build_commodity_features(df, target_index, cfg),
        *build_price_lags(df, target_index, cfg),
        *build_actual_lags(df, target_index, cfg),
        *build_forecast_error_lags(df, target_index, cfg),
        *build_cross_border_lags(df, target_index, cfg),
    ]
    return build_matrix(features)  # asserts no leakage, then concatenates


def build_feature_set_for_day(
    target_day: dt.date,
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """The complete, live-viable feature set for one local delivery day
    (spec 6.5.3, sections 3.4/4.2 -- E7 in code).

    Calendar/regime, NWP-reconstruction-based forecast fundamentals
    (features/nwp_fundamentals.py), commodities, and the unchanged
    price/actual/forecast-error/cross-border lags (fundamentals.py/lags.py,
    untouched) -- for exactly target_day's hourly rows. Contains no
    DA_FORECAST-class renewables column (spec 6.5.3 section 5.3).

    Backtest and live call this function identically, with target_day as
    the only day-identifying argument -- never the system clock. What
    differs between the two contexts is only which rows `df` and
    `renewables_predictions` happen to contain at call time, never the
    code path (spec 6.5.3 section 3.4).

    Raises IncompleteReconstructionError (features.nwp_fundamentals) if the
    renewables reconstruction does not fully cover target_day -- the whole
    day is unusable per spec 6.5.3 section 3.3, not partially fillable.
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = _hourly_utc_index_for_local_day(target_day)
    fundamentals = build_nwp_forecast_fundamentals(df, renewables_predictions, target_index)
    return _build_day_matrix(target_index, fundamentals, df, cfg)


def build_original_feature_set_for_day(
    target_day: dt.date,
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """The 'original' (upper-bound) feature set for one local delivery day
    (spec 6.6 section 5.1) -- the same day-by-day construction as
    build_feature_set_for_day above, sharing its Bausteinliste, but with
    the TSO-sourced fundamentals (build_forecast_fundamentals) instead of
    the NWP reconstruction.

    Never a live candidate: contains DA_FORECAST-class renewables columns
    that the 6.2 audit found unavailable at gate closure. Exists so 6.6 can
    measure both feature sets through the identical per-day code path --
    not by reading the precomputed features.parquet, a different code path
    (spec 6.6 section 5.1).
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = _hourly_utc_index_for_local_day(target_day)
    fundamentals = build_forecast_fundamentals(df, target_index)
    return _build_day_matrix(target_index, fundamentals, df, cfg)


def trim_warmup(matrix: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Drop the initial warm-up rows where the longest lag/rolling window is NaN.

    Only the leading `max_lookback_hours` rows are removed. This is NOT a dropna:
    the intentional EUA-CO2 NaN region (pre-Oct-2021, D4) must survive untouched
    -- model-side imputation happens in 2.4. The matrix stays model-agnostic.
    """
    cfg = config if config is not None else FeatureConfig()
    cutoff = matrix.index[0] + pd.Timedelta(hours=cfg.max_lookback_hours())
    return matrix.loc[matrix.index >= cutoff]
