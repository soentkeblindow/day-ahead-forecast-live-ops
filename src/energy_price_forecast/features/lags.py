"""Lagged, rolling, and forecast-error features (Sprint 2.3.3).

All builders use the 2.3.1 availability machinery to ensure point-in-time correctness.
"""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING

import pandas as pd

from .availability import Feature, combine, lag, rolling_mean
from .config import FeatureConfig

if TYPE_CHECKING:
    pass


_SCHEDULED_PREFIX = "scheduled_net_de_to_"
_PHYSICAL_PREFIX = "physical_net_de_to_"


# (actual column, forecast column, feature stem)
_ERROR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("load_actual", "load_forecast_day_ahead", "load_forecast_error"),
    ("gen_wind_onshore", "wind_onshore_forecast", "wind_onshore_forecast_error"),
    ("gen_wind_offshore", "wind_offshore_forecast", "wind_offshore_forecast_error"),
    ("gen_solar", "solar_forecast", "solar_forecast_error"),
)


def build_price_lags(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, config: FeatureConfig | None = None
) -> list[Feature]:
    """Lagged prices and trailing rolling-mean prices (all DA_FIXED, >= 24h).

    Lags at the ACF/PACF-justified horizons (24h, 48h, 168h); rolling means over
    24h and 168h windows with a 24h-lagged leading edge.
    """
    cfg = config if config is not None else FeatureConfig()
    feats: list[Feature] = [
        lag(
            f"price_lag_{h}h",
            df["day_ahead_price"],
            "day_ahead_price",
            hours=h,
            target_index=target_index,
        )
        for h in cfg.price_lags_hours
    ]
    feats += [
        rolling_mean(
            f"price_roll_mean_{w}h",
            df["day_ahead_price"],
            "day_ahead_price",
            window_hours=w,
            lag_hours=cfg.rolling_base_lag_hours,
            target_index=target_index,
        )
        for w in cfg.rolling_windows_hours
    ]
    return feats


def build_actual_lags(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, config: FeatureConfig | None = None
) -> list[Feature]:
    """Lagged realised load (RT_ACTUAL -> lags must be >= 48h).

    Realised load from D-2 / D-7 carries demand-pattern information that the
    day-ahead load forecast (already a feature in 2.3.2) does not fully capture.
    """
    cfg = config if config is not None else FeatureConfig()
    return [
        lag(
            f"load_actual_lag_{h}h",
            df["load_actual"],
            "load_actual",
            hours=h,
            target_index=target_index,
        )
        for h in cfg.actual_lags_hours
    ]


def _forecast_error_lag(
    name: str,
    df: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
    *,
    hours: int,
    target_index: pd.DatetimeIndex,
) -> Feature:
    """A lagged forecast error: lag(actual) - lag(forecast), both by `hours`.

    knowledge_time = max over the two inputs. The RT_ACTUAL actual (t+1h at the
    source) binds, so the error inherits the actual's >= 48h requirement and the
    leakage test enforces it automatically. Names of the two intermediate lags
    are local-only; just the combined feature is returned.
    """
    actual_l = lag(
        f"_{actual_col}_lag{hours}",
        df[actual_col],
        actual_col,
        hours=hours,
        target_index=target_index,
    )
    forecast_l = lag(
        f"_{forecast_col}_lag{hours}",
        df[forecast_col],
        forecast_col,
        hours=hours,
        target_index=target_index,
    )
    return combine(name, [actual_l, forecast_l], lambda a, f: a - f)


def build_forecast_error_lags(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, config: FeatureConfig | None = None
) -> list[Feature]:
    """Lagged forecast errors for load and the three renewable sources (>= 48h).

    Solar error keeps its night-time zeros: the EDA's `solar_forecast > 100 MW`
    filter was an ACF-analysis artefact, not a feature rule. SHAP prunes the weak
    ones in Sprint 3 (D6).
    """
    cfg = config if config is not None else FeatureConfig()
    feats: list[Feature] = []
    for h in cfg.forecast_error_lags_hours:
        for actual_col, forecast_col, stem in _ERROR_PAIRS:
            feats.append(
                _forecast_error_lag(
                    f"{stem}_lag_{h}h",
                    df,
                    actual_col,
                    forecast_col,
                    hours=h,
                    target_index=target_index,
                )
            )
    return feats


def _total_flow_lag(
    name: str,
    df: pd.DataFrame,
    prefix: str,
    *,
    hours: int,
    target_index: pd.DatetimeIndex,
) -> Feature:
    """Total net export = sum over all corridor columns matching `prefix`, lagged.

    Each corridor is lagged individually (so availability_of picks up its class
    via the prefix rule), then summed via combine. Sum-of-lags == lag-of-sum
    (linear), and combine's max-composition gives the correct knowledge time.
    """
    cols = sorted(c for c in df.columns if c.startswith(prefix))
    if not cols:
        raise KeyError(f"no columns matching prefix {prefix!r} in frame")
    parts = [
        lag(f"_{c}_lag{hours}", df[c], c, hours=hours, target_index=target_index) for c in cols
    ]
    return combine(name, parts, lambda *vals: functools.reduce(operator.add, vals))


def build_cross_border_lags(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, config: FeatureConfig | None = None
) -> list[Feature]:
    """Lagged total net exports plus the plan-vs-actual deviation.

    Scheduled flows are DA_FIXED (24h ok); physical flows are RT_ACTUAL (>= 48h).
    The deviation (physical - scheduled) inherits the physical lag (>= 48h).
    Cross-border is the weakest signal in the EDA (4.2) -> drop if SHAP confirms.
    """
    cfg = config if config is not None else FeatureConfig()
    feats: list[Feature] = []

    for h in cfg.scheduled_flow_lags_hours:
        feats.append(
            _total_flow_lag(
                f"scheduled_net_export_lag_{h}h",
                df,
                _SCHEDULED_PREFIX,
                hours=h,
                target_index=target_index,
            )
        )
    for h in cfg.physical_flow_lags_hours:
        feats.append(
            _total_flow_lag(
                f"physical_net_export_lag_{h}h",
                df,
                _PHYSICAL_PREFIX,
                hours=h,
                target_index=target_index,
            )
        )

    # Deviation: physical and scheduled at the same (physical) lag, so the
    # difference compares like-for-like value times.
    for h in cfg.physical_flow_lags_hours:
        sched = _total_flow_lag(
            f"_sched_dev_{h}h",
            df,
            _SCHEDULED_PREFIX,
            hours=h,
            target_index=target_index,
        )
        phys = _total_flow_lag(
            f"_phys_dev_{h}h",
            df,
            _PHYSICAL_PREFIX,
            hours=h,
            target_index=target_index,
        )
        feats.append(
            combine(
                f"cross_border_deviation_lag_{h}h",
                [phys, sched],
                lambda p, s: p - s,
            )
        )
    return feats
