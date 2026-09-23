import logging

import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COL = "day_ahead_price"
_COMMODITY_COLS: list[str] = ["ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"]
_WEIGHT_COL = "load_actual"
_WEIGHT_FALLBACK_COL = "load_forecast_day_ahead"


def to_hourly_vwap(
    price: pd.Series,
    weight: pd.Series,
    weight_fallback: pd.Series | None = None,
) -> pd.Series:
    """Collapse a (possibly sub-hourly) price series to hourly via a
    volume-weighted average price (VWAP), using `weight` as a volume proxy.

    On the hourly era (one observation per hour) the weight cancels and the
    price is returned unchanged. On the 15-min era the four quarter-hour prices
    are weighted by their volume proxy. Hours whose total weight is zero or
    missing fall back to the simple (unweighted) mean of the available prices.

    ``weight_fallback``, if given, fills individual missing ``weight`` values
    (e.g. load_actual not yet settled at the live edge, docs/bugs_in_live_system.md
    entry 2) before aggregation. If a quarter-hour's weight is still missing after
    that -- both ``weight`` and ``weight_fallback`` are NaN there -- the whole hour
    falls back to the simple mean rather than silently VWAP-ing over only the
    quarter-hours that do have a weight. Passing no ``weight_fallback`` (the
    default) reproduces the exact prior behaviour: an unfilled quarter-hour is
    zero-weighted, not hour-excluding.
    """
    df = pd.concat([price.rename("p"), weight.rename("w")], axis=1)
    df = df[df["p"].notna()]
    w_raw = df["w"]
    if weight_fallback is not None:
        w_raw = w_raw.fillna(weight_fallback.reindex(df.index))

    w = w_raw.clip(lower=0).fillna(0.0)
    num = (df["p"] * w).resample("h").sum()
    den = w.resample("h").sum()
    simple = df["p"].resample("h").mean()
    vwap = (num / den).where(den > 0, simple)

    if weight_fallback is not None:
        hour_incomplete = w_raw.isna().astype(int).resample("h").sum() > 0
        vwap = vwap.where(~hour_incomplete.reindex(vwap.index).fillna(False), simple)

    return vwap


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a mixed-resolution (hourly + 15-min) frame to one hourly grid.

    Applies three aggregation rules:
    - day_ahead_price  →  VWAP weighted by load_actual (falls back to
      load_forecast_day_ahead for still-missing weights, then to the simple
      mean if that is missing too -- see to_hourly_vwap)
    - MW columns (all others except commodities)  →  time-average (mean)
    - commodity columns  →  time-average (intra-hour constant; mean is a no-op)

    The input must have a UTC-aware DatetimeIndex. On an already-hourly input
    this function is a no-op.
    """
    assert str(pd.DatetimeIndex(df.index).tz) == "UTC", "expects a UTC tz-aware index"

    mw_cols = [c for c in df.columns if c not in {_PRICE_COL, *_COMMODITY_COLS}]
    logger.info("MW columns being time-averaged: %s", mw_cols)

    out = df[mw_cols].resample("h").mean()
    out[_COMMODITY_COLS] = df[_COMMODITY_COLS].resample("h").mean()
    out[_PRICE_COL] = to_hourly_vwap(
        df[_PRICE_COL], df[_WEIGHT_COL], weight_fallback=df.get(_WEIGHT_FALLBACK_COL)
    )

    diffs = out.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta("1h"), "index not strictly hourly"
    return out[df.columns]
