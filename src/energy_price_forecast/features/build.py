from __future__ import annotations

import pandas as pd

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


def trim_warmup(matrix: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Drop the initial warm-up rows where the longest lag/rolling window is NaN.

    Only the leading `max_lookback_hours` rows are removed. This is NOT a dropna:
    the intentional EUA-CO2 NaN region (pre-Oct-2021, D4) must survive untouched
    -- model-side imputation happens in 2.4. The matrix stays model-agnostic.
    """
    cfg = config if config is not None else FeatureConfig()
    cutoff = matrix.index[0] + pd.Timedelta(hours=cfg.max_lookback_hours())
    return matrix.loc[matrix.index >= cutoff]
