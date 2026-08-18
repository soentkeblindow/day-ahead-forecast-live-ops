"""Trading book, P&L, VaR and Expected Shortfall (Sprint 4.4a).

Two hypothetical books (long/short, flat 10 MWh, mark-to-model at the
calibrated median) turn the residual pool from ``evaluation.residuals`` into
euro risk numbers: VaR and Expected Shortfall, ex ante from three separate
threshold sources, plus the ex-post, model-free ES as a descriptive
benchmark. ONE tail estimator (``residuals.lower_tail_mean`` /
``upper_tail_mean``), THREE thresholds -- ``variant`` selects only where the
standardised threshold ``u(t)`` comes from, never a second implementation.

Mark invariance (tested in ``tests/test_risk.py``): shifting ``mark`` by a
forecast-time-known constant ``c`` leaves every breach indicator unchanged
and shifts ``risk_measures``'s VaR by exactly ``quantity * c`` (variants with
an independent ``threshold_quantile``). ``risk_measures``'s ES does NOT shift
by ``quantity * c`` under that construction -- it is unchanged ("fhs": both
VaR and ES are fully mark-independent instead). The "both var and es shift by
quantity * c" property holds unconditionally for ``realised_es`` on
``book_pnl``'s own P&L output, not for ``risk_measures``'s ex-ante ES. The
book is a measurement device, not a strategy -- no alpha is claimed.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ
from .residuals import lower_tail_mean, tail_quantile, threshold_position, upper_tail_mean

Side = Literal["long", "short"]
Variant = Literal["raw", "calibrated", "fhs"]
QUANTITY_MWH: float = 10.0


def book_pnl(
    price: pd.Series, mark: pd.Series, *, side: Side, quantity: float = QUANTITY_MWH
) -> pd.Series:
    """Mark-to-model P&L of a flat 10 MW book, one position per delivery hour.

    long: Q*(P - m)    short: Q*(m - P)

    Because `mark` is the predictive MEDIAN, expected P&L is ~0 by construction.
    This book is a measurement device, not a strategy: no alpha is claimed.

    Mark invariance (tested, corrected 2026-07-09 -- see spec 2.2): shifting
    `mark` by a constant c leaves every breach indicator unchanged and shifts
    risk_measures's VaR by exactly Q * c (variants with an independent
    threshold_quantile). risk_measures's ES does NOT shift by Q * c under
    that construction -- it is unchanged (fhs: both VaR and ES are fully
    mark-independent instead). The "both var and es shift by Q * c" property
    holds unconditionally for realised_es on THIS function's own P&L output,
    not for risk_measures's ex-ante ES.
    """
    if side == "long":
        return quantity * (price - mark)
    return quantity * (mark - price)


# Harmonised maximum/minimum clearing price for Single Day-Ahead Coupling, set
# by ACER under CACM Regulation (EU) 2015/1222 Art. 41/54. NOT exchange limits:
# the DE-LU day-ahead price clears in SDAC and EPEX SPOT is only one of several
# NEMOs. The limits move via an automatic adjustment mechanism, so they form a
# STEP FUNCTION in time -- applying today's floor to a 2019 backtest would be an
# anachronistic (look-ahead) constraint. The three edges below are FIXED: they
# were verified by the owner against the NEMO Committee communication notes.
# Do not add, guess or extrapolate edges.
SDAC_PRICE_LIMITS: tuple[tuple[str, float, float], ...] = (
    ("1900-01-01", -500.0, 3000.0),
    ("2022-05-10", -500.0, 4000.0),
    ("2026-05-28", -600.0, 4000.0),
)


def sdac_limits_asof(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """(floor, cap) per delivery hour, looked up as-of the delivery DAY.

    The delivery day is Europe/Berlin LOCAL time, consistent with the rest of
    the project's delivery-day convention (market_time.py, LOCAL_TZ) -- not
    the UTC calendar day.
    """
    local_days = index.tz_convert(LOCAL_TZ).normalize()
    edges = pd.DatetimeIndex([pd.Timestamp(d, tz=LOCAL_TZ) for d, _, _ in SDAC_PRICE_LIMITS])
    positions = edges.get_indexer(local_days, method="pad")

    floors = np.array([f for _, f, _ in SDAC_PRICE_LIMITS])
    caps = np.array([c for _, _, c in SDAC_PRICE_LIMITS])
    floor = pd.Series(floors[positions], index=index)
    cap = pd.Series(caps[positions], index=index)
    return floor, cap


_RISK_COLUMNS: tuple[str, ...] = (
    "u",
    "threshold_price",
    "var",
    "es",
    "es_unclipped",
    "pi",
    "n_pool",
    "n_tail",
    "n_clipped",
    "clip_impact_es",
    "breach",
)


def _nan_row() -> dict[str, object]:
    """Row for an hour with an empty pool or missing sigma: NaN, n_pool = 0."""
    return {
        "u": float("nan"),
        "threshold_price": float("nan"),
        "var": float("nan"),
        "es": float("nan"),
        "es_unclipped": float("nan"),
        "pi": float("nan"),
        "n_pool": 0,
        "n_tail": 0,
        "n_clipped": 0,
        "clip_impact_es": float("nan"),
        "breach": float("nan"),
    }


def risk_measures(
    *,
    price: pd.Series,
    mark: pd.Series,
    sigma: pd.Series,
    pool: dict[dt.date, np.ndarray],
    variant: Variant,
    side: Side,
    level: float = 0.95,
    threshold_quantile: pd.Series | None = None,
    quantity: float = QUANTITY_MWH,
) -> pd.DataFrame:
    """VaR and ES for one (variant, side) pair, one row per delivery hour.

    ONE tail estimator, THREE thresholds. `variant` chooses only where the
    standardised threshold u(t) comes from:

      "calibrated"  u = (threshold_quantile(t) - mark(t)) / sigma(t), from the
               CALIBRATED, rearranged grid (4.3c). PRIMARY: its threshold is the
               only one whose coverage is evidenced (0.050 vs nominal 0.05).
      "fhs"    u = tail_quantile(pool[day], ...). Internally consistent --
               threshold and tail mean from ONE distribution -- but rests on the
               median-anchored location-scale assumption. Also the only variant
               on which the skew ratio is readable (sigma cancels).
      "raw"    u from the UNCALIBRATED grid. A COUNTERFACTUAL, not a risk
               number: a raw quantile grid cannot produce an ES at all, it stops
               at its own boundary. It answers only "what would a manager using
               the raw q05 as their 5% threshold have reported, with our tail
               behind it?"

    `threshold_quantile` is required for "raw"/"calibrated", forbidden for "fhs"
    (fail fast, never silently ignore).

    Threshold and every tail scenario are clipped to the SDAC limits of the
    delivery day. `clip` is monotone in the scenario price and the pool is
    sorted, so the tail slice is the SAME index set before and after -- no
    re-sorting, no materialisation beyond the slice.

    Raises ValueError if `level` is outside [0.90, 0.95] (spec 2.7): ES95 rests
    on ~438 tail points, ES99.9 on ~9, and that number would essentially BE
    2023-07-02. Refusing to print it is the correct behaviour.

    Columns: u, threshold_price, var, es, es_unclipped, pi, n_pool, n_tail,
             n_clipped, clip_impact_es, breach.
    Rows with an empty pool or missing sigma carry NaN and n_pool = 0.
    """
    if not (0.90 <= level <= 0.95):
        raise ValueError(f"level must be in [0.90, 0.95] (spec 2.7), got {level}")
    if variant == "fhs":
        if threshold_quantile is not None:
            raise ValueError("threshold_quantile must be None for variant='fhs'")
    elif threshold_quantile is None:
        raise ValueError(f"threshold_quantile is required for variant={variant!r}")

    index = pd.DatetimeIndex(price.index)
    local_days = index.tz_convert(LOCAL_TZ).normalize()
    alpha = (1.0 - level) if side == "long" else level

    if variant == "fhs":
        u_by_day: dict[dt.date, float] = {}
        for day_ts in pd.unique(local_days):
            day = pd.Timestamp(day_ts).date()
            day_pool = pool.get(day, np.array([], dtype=float))
            u_by_day[day] = tail_quantile(day_pool, alpha) if day_pool.size > 0 else float("nan")
        u = pd.Series([u_by_day[d.date()] for d in local_days], index=index)
    else:
        assert threshold_quantile is not None  # validated above
        u = (threshold_quantile - mark) / sigma

    floor, cap = sdac_limits_asof(index)

    rows: list[dict[str, object]] = []
    for t, day_ts in zip(index, local_days, strict=True):
        day = day_ts.date()
        day_pool = pool.get(day, np.array([], dtype=float))
        n_pool_t = int(day_pool.size)
        u_t = float(u.loc[t])
        mark_t = float(mark.loc[t])
        sigma_t = float(sigma.loc[t])

        if n_pool_t == 0 or np.isnan(u_t) or np.isnan(mark_t) or np.isnan(sigma_t):
            rows.append(_nan_row())
            continue

        floor_t = float(floor.loc[t])
        cap_t = float(cap.loc[t])

        threshold_price = float(np.clip(mark_t + sigma_t * u_t, floor_t, cap_t))
        var_t = quantity * abs(mark_t - threshold_price)

        if side == "long":
            tail_mean_r, n_tail_t = lower_tail_mean(day_pool, u_t)
            pi_t = threshold_position(day_pool, u_t, lower=True)
            tail_r = day_pool[day_pool <= u_t]
        else:
            tail_mean_r, n_tail_t = upper_tail_mean(day_pool, u_t)
            pi_t = threshold_position(day_pool, u_t, lower=False)
            tail_r = day_pool[day_pool >= u_t]

        if n_tail_t == 0:
            es_t = float("nan")
            es_unclipped_t = float("nan")
            n_clipped_t = 0
            clip_impact_es_t = float("nan")
        else:
            unclipped_price = mark_t + sigma_t * tail_mean_r
            es_unclipped_t = quantity * abs(mark_t - unclipped_price)

            scenario_prices = mark_t + sigma_t * tail_r
            clipped_prices = np.clip(scenario_prices, floor_t, cap_t)
            n_clipped_t = int(np.sum(clipped_prices != scenario_prices))
            es_t = quantity * abs(mark_t - float(clipped_prices.mean()))

            clip_impact_es_t = (
                (es_unclipped_t - es_t) / es_unclipped_t if es_unclipped_t != 0 else float("nan")
            )

        price_t = float(price.loc[t])
        breach_t = price_t < threshold_price if side == "long" else price_t > threshold_price

        rows.append(
            {
                "u": u_t,
                "threshold_price": threshold_price,
                "var": var_t,
                "es": es_t,
                "es_unclipped": es_unclipped_t,
                "pi": pi_t,
                "n_pool": n_pool_t,
                "n_tail": n_tail_t,
                "n_clipped": n_clipped_t,
                "clip_impact_es": clip_impact_es_t,
                "breach": float(breach_t),
            }
        )

    return pd.DataFrame(rows, index=index, columns=list(_RISK_COLUMNS))


def realised_es(pnl: pd.Series, *, level: float = 0.95) -> dict[str, float]:
    """Ex-post, model-free, UNCONDITIONAL ES of the realised P&L distribution.

    Descriptive only. It is NOT directly comparable to mean(ES_t): ES_t is a
    CONDITIONAL measure given sigma(t), and the mean of conditional ES is not
    the unconditional ES of the sigma-mixture (see spec 6). The comparable
    quantity is `es_ratio_conditional`.

    Returns {"var": ..., "es": ..., "n_tail": ..., "n": ...}. `var`/`es` are
    positive LOSS magnitudes (same sign convention as `risk_measures`), read
    off the bottom `1 - level` tail of the realised P&L distribution.
    """
    clean = pnl.dropna().to_numpy()
    alpha = 1.0 - level
    threshold = float(np.quantile(clean, alpha))
    tail = clean[clean <= threshold]
    n_tail = int(tail.size)
    var = -threshold
    es = float(-tail.mean()) if n_tail > 0 else float("nan")
    return {"var": var, "es": es, "n_tail": float(n_tail), "n": float(clean.size)}


def es_ratio_conditional(pnl: pd.Series, risk: pd.DataFrame) -> float:
    """mean(loss_t / es_t | breach_t): the CONDITIONAL comparison of ex-ante
    ES against realised loss, restricted to hours that actually breached
    (spec section 6).

    NOT `mean_es / realised_es`: ES(t) is conditional on sigma(t), and the
    mean of conditional ES is not the unconditional ES of a sigma-mixture --
    for a perfectly calibrated model with sigma in {1, 10} the naive ratio
    would read as a ~35% underestimate, a pure mixture artefact. sigma
    cancels INSIDE this ratio (per hour), not across a ratio of two means,
    which is why this is the comparable quantity: `es_ratio_conditional - 1`
    is, in essence, the Acerbi-Szekely Z2 statistic (4.4b hangs the null
    distribution on it).

    `risk` is the output of `risk_measures`, sharing `pnl`'s index. Only rows
    where `risk["breach"] == 1.0` and `risk["es"]` is usable (non-NaN,
    non-zero) contribute. Returns NaN if there are no such rows.
    """
    breach = risk["breach"] == 1.0
    es = risk["es"]
    valid = breach & es.notna() & (es != 0)
    if not valid.any():
        return float("nan")
    loss = -pnl.loc[valid]
    ratio = loss / es.loc[valid]
    return float(ratio.mean())
