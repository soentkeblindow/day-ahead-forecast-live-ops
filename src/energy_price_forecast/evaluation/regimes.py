"""Realised-market regime tagging (Sprint 4.1).

DIAGNOSTIC ONLY. Every flag produced here is computed from REALISED market
data (actual load, actual wind/solar generation, actual day-ahead price).
These labels exist solely to slice already-computed forecasts and metrics
after the fact (Sprint 4.2 / 4.4). They MUST NOT be used as model features:
they encode information unknown at gate-closure (D-1) -- the price-spike
flag even uses the full per-regime price distribution -- so feeding them to
the forecaster would be look-ahead leakage. This separation is a contract,
enforced by ``tests/test_regimes.py`` (which asserts the flag column names
are disjoint from the processed feature matrix).
"""

from __future__ import annotations

import pandas as pd

from ..market_time import LOCAL_TZ
from .config import RegimeConfig

REGIME_FLAG_COLUMNS: tuple[str, ...] = (
    "renewable_scarcity",
    "high_wind",
    "negative_price",
    "price_spike",
    "normal",
)
MACRO_REGIME_COLUMN: str = "macro_regime"

_REQUIRED_COLUMNS = (
    "load_actual",
    "gen_wind_onshore",
    "gen_wind_offshore",
    "gen_solar",
    "day_ahead_price",
)


def _check_index(index: pd.DatetimeIndex) -> None:
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError(
            f"interim frame index must be UTC tz-aware, got tz={index.tz!r}. "
            "This is a Sprint 2.1 (load_interim_hourly) invariant."
        )
    if len(index) > 1:
        diffs = index.to_series().diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != pd.Timedelta("1h"):
            raise ValueError(
                "interim frame index must be strictly hourly (regular 1h grid); "
                f"found irregular spacing: {list(diffs)}. This may be the known "
                "15-min resolution break (from 2025-09-30) -- resampling to a "
                "canonical hourly grid is a Sprint 2.1 responsibility, not "
                "evaluation.regimes."
            )


def tag_regimes(
    interim: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Tag each hourly timestamp with overlapping market-regime flags.

    DIAGNOSTIC ONLY. Every flag is computed from REALISED market data
    (actual load, actual wind/solar generation, actual price). These labels
    exist solely to slice already-computed forecasts and metrics after the
    fact (Sprint 4.2 / 4.4). They MUST NOT be used as model features: they
    encode information unknown at gate-closure (D-1) -- the price-spike flag
    even uses the full per-regime price distribution -- so feeding them to the
    forecaster would be look-ahead leakage. This separation is a contract,
    enforced by tests/test_regimes.py.

    Parameters
    ----------
    interim:
        Canonical hourly UTC frame from ``load_interim_hourly()``. Must expose
        the realised columns load_actual, gen_wind_onshore, gen_wind_offshore,
        gen_solar, day_ahead_price, all in MW / EUR-MWh. Pass the FULL
        history: the price-spike percentile is computed over the whole frame,
        per macro regime.
    config:
        Thresholds and macro boundaries. Defaults to ``RegimeConfig()``.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ``interim``. Columns: the four special boolean
        flags, the boolean ``normal`` catch-all (True where no special flag is
        active), and the categorical ``macro_regime`` (calm / crisis /
        post_crisis).

    Notes
    -----
    Flags overlap by design: a single hour can be both ``renewable_scarcity``
    and ``price_spike``. ``normal`` is the exact logical complement of the OR
    of the four special flags, so (normal) XOR (any special) holds per row.
    ``macro_regime`` is an orthogonal, mutually-exclusive partition over the
    whole index (every hour gets exactly one label), using Europe/Berlin
    local-midnight boundaries -- identical to the ``is_crisis`` /
    ``is_post_crisis`` model feature in ``features/calendar.py``.
    """
    cfg = config if config is not None else RegimeConfig()

    missing = [c for c in _REQUIRED_COLUMNS if c not in interim.columns]
    if missing:
        raise ValueError(f"interim frame is missing required columns: {missing}")

    index = pd.DatetimeIndex(interim.index)
    _check_index(index)

    load = interim["load_actual"]
    wind = interim["gen_wind_onshore"] + interim["gen_wind_offshore"]
    solar = interim["gen_solar"]
    price = interim["day_ahead_price"]

    residual_load = load - (wind + solar)
    residual_share = residual_load / load

    # 1) Macro regime first -- needed for the per-regime spike threshold.
    # Boundaries are Europe/Berlin local dates (see RegimeConfig): convert
    # the UTC index to local time and normalise to local midnight, exactly
    # like features/calendar.py's is_crisis / is_post_crisis.
    local_day = index.tz_convert(LOCAL_TZ).normalize()
    macro = pd.Series("calm", index=interim.index, dtype="object")
    macro[local_day >= cfg.crisis_start] = "crisis"
    macro[local_day >= cfg.post_crisis_start] = "post_crisis"

    # 2) Special flags (overlapping). NaN comparisons yield False.
    renewable_scarcity = (residual_share > cfg.renewable_scarcity_residual_share).astype(bool)
    high_wind = (wind > cfg.high_wind_generation_mw).astype(bool)
    negative_price = (price < 0.0).astype(bool)

    # Price spike: above the per-macro-regime quantile over the full frame.
    # groupby(...).transform broadcasts each group's threshold back to rows.
    # On an empty frame, transform drops the DatetimeIndex for a RangeIndex;
    # set_axis restores it (transform preserves row order/count, so this is safe).
    spike_threshold = (
        price.groupby(macro)
        .transform(lambda s: s.quantile(cfg.price_spike_quantile))
        .set_axis(price.index)
    )
    price_spike = (price > spike_threshold).astype(bool)

    special = renewable_scarcity | high_wind | negative_price | price_spike
    normal = ~special

    return pd.DataFrame(
        {
            "renewable_scarcity": renewable_scarcity,
            "high_wind": high_wind,
            "negative_price": negative_price,
            "price_spike": price_spike,
            "normal": normal,
            MACRO_REGIME_COLUMN: macro.astype("category"),
        },
        index=interim.index,
    )
