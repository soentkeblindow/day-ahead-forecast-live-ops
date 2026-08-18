from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

from ..market_time import LOCAL_TZ
from .availability import Feature, calendar_feature
from .config import FeatureConfig


def _cyclical(
    name: str, position: np.ndarray, period: int, index: pd.DatetimeIndex
) -> list[Feature]:
    """sin/cos encoding of a cyclical integer position in [0, period)."""
    angle = 2.0 * np.pi * position / period
    return [
        calendar_feature(f"{name}_sin", pd.Series(np.sin(angle), index=index)),
        calendar_feature(f"{name}_cos", pd.Series(np.cos(angle), index=index)),
    ]


def _regional_holiday_dates(years: range) -> set[dt.date]:
    """Fronleichnam (Easter + 60 days) and Allerheiligen (fixed 1 Nov) per year.

    Computed explicitly rather than via state subdivisions, so the set contains
    exactly these two regional holidays (no Heilige Drei Könige, Reformationstag, ...).
    Both fall in energy-heavy Catholic states (BW, BY, NRW, HE, RP, SL).
    """
    out: set[dt.date] = set()
    for y in years:
        out.add(easter(y) + dt.timedelta(days=60))  # Fronleichnam
        out.add(dt.date(y, 11, 1))  # Allerheiligen
    return out


def build_calendar_features(
    target_index: pd.DatetimeIndex,
    config: FeatureConfig | None = None,
) -> list[Feature]:
    """Deterministic calendar + regime features, in Europe/Berlin local time.

    Cyclical hour/weekday/month, is_weekend, is_holiday (nationwide DE),
    is_regional_holiday (Fronleichnam + Allerheiligen), is_crisis, is_post_crisis.
    Regime boundaries come from the FeatureConfig.
    """
    cfg = config if config is not None else FeatureConfig()
    local = target_index.tz_convert(LOCAL_TZ)
    local_dates = local.date
    years = range(int(local.year.min()), int(local.year.max()) + 1)

    feats: list[Feature] = []
    feats += _cyclical("hour", local.hour.to_numpy(), 24, target_index)
    feats += _cyclical("weekday", local.dayofweek.to_numpy(), 7, target_index)
    feats += _cyclical("month", (local.month - 1).to_numpy(), 12, target_index)

    is_weekend = pd.Series((local.dayofweek >= 5).astype("int8"), index=target_index)
    feats.append(calendar_feature("is_weekend", is_weekend))

    de = holidays.country_holidays("DE", years=years)  # nationwide federal holidays only
    regional = _regional_holiday_dates(years)
    is_holiday = pd.Series([d in de for d in local_dates], index=target_index).astype("int8")
    is_regional = pd.Series([d in regional for d in local_dates], index=target_index).astype("int8")
    feats.append(calendar_feature("is_holiday", is_holiday))
    feats.append(calendar_feature("is_regional_holiday", is_regional))

    local_day = local.normalize()
    is_crisis = pd.Series(
        ((local_day >= cfg.crisis_start) & (local_day < cfg.post_crisis_start)).astype("int8"),
        index=target_index,
    )
    is_post = pd.Series((local_day >= cfg.post_crisis_start).astype("int8"), index=target_index)
    feats.append(calendar_feature("is_crisis", is_crisis))
    feats.append(calendar_feature("is_post_crisis", is_post))

    return feats
