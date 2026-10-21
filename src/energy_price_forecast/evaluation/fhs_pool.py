"""Filtered Historical Simulation: the residual filter and the tail pool
(Sprint 7.1 Teil B, spec section 3.2/4.1).

Ported from the sibling research repo
(``energy-price-forecast-risk-validation``,
``model_risk_regimes/evaluation/residuals.py``, Sprint 4.4a) per spec
section 4.1 -- only the three functions the quantile track actually needs
(``standardised_residuals``, ``pool_by_day``, ``tail_quantile``; the sibling
module's VaR/ES-only helpers ``lower_tail_mean``/``upper_tail_mean``/
``threshold_position`` are not ported, this repo has no VaR/ES use for
them). Adapted for quarter-hourly data (spec section 3.2's own instruction):
the sibling's hour-count minimum (``_MIN_POOL_HOURS = 180 * 24``) is
replaced by an explicit ``min_pool_days`` parameter converted internally via
``periods_per_day`` -- so the minimum stays "N days" regardless of how many
periods a day actually has, rather than quietly gating four times too early
on 96-period quarter-hourly data the way a flat period count would.

Filtered Historical Simulation (FHS) in one line: divide out the
conditional location and scale (``standardised_residuals``) so shocks from
a calm spring day and a winter price spike become samples from
(approximately) the same distribution, then pool the standardised shocks
across a long history (``pool_by_day``) to get enough tail mass for a
quantile estimate.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ


def standardised_residuals(y_true: pd.Series, median: pd.Series, sigma: pd.Series) -> pd.Series:
    """Filter the price by its conditional location and scale: r = (P - q50) / sigma.

    The "filtered" step of Filtered Historical Simulation. Dividing out the
    conditional scale is what makes shocks from calm and volatile periods
    samples from (approximately) the same distribution, hence poolable.
    ``median`` MUST be the same series actually submitted; ``sigma`` MUST
    come from ``conformal.quarterhour_sigma`` and is never re-derived here.

    Periods with missing sigma (an uncalibrated start-of-history window)
    yield NaN, not an exception.
    """
    return (y_true - median) / sigma


def pool_by_day(
    r: pd.Series,
    *,
    window_days: int,
    embargo_days: int,
    min_pool_days: int,
    periods_per_day: int = 96,
) -> dict[dt.date, np.ndarray]:
    """One ASCENDING-SORTED pool of past shocks per delivery day.

    Leakage contract: ``pool_by_day(D)`` contains no timestamp
    ``>= D - embargo_days``. NaN entries in ``r`` (missing sigma -- e.g. the
    first uncalibrated periods of a backtest) are dropped BEFORE sorting:
    never counted toward the minimum, never present in the returned array.
    Days whose NaN-filtered count falls below ``min_pool_days *
    periods_per_day`` map to an EMPTY array (callers return NaN). Never
    winsorise, trim, or otherwise treat outliers.

    The window is a plain boolean mask against ``r``'s real (NaN-dropped)
    index -- exactly like ``conformal.calibration_window`` -- which already
    "expands" for free whenever ``window_days`` reaches before the earliest
    available residual, and becomes a fixed-width rolling window once
    enough history has accumulated. No separate mode is needed: an
    unclamped window request cannot select data that does not exist; the
    embargo cutoff (the window's upper bound) stays strictly
    ``day - embargo_days`` regardless.
    """
    min_pool_periods = min_pool_days * periods_per_day

    r_clean = r.dropna()
    clean_index = pd.DatetimeIndex(r_clean.index)
    clean_values = r_clean.to_numpy()

    full_index = pd.DatetimeIndex(r.index)
    local_days = full_index.tz_convert(LOCAL_TZ).normalize()
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))

    pools: dict[dt.date, np.ndarray] = {}
    for day in delivery_days:
        edge = day - pd.Timedelta(days=embargo_days)
        lo = edge - pd.Timedelta(days=window_days)
        mask = (clean_index >= lo) & (clean_index < edge)
        window_values = clean_values[mask]
        if window_values.size < min_pool_periods:
            pools[day.date()] = np.array([], dtype=float)
        else:
            pools[day.date()] = np.sort(window_values)

    return pools


def tail_quantile(pool: np.ndarray, level: float) -> float:
    """The ``level``-quantile of the pool (anchor identity with ``np.quantile``)."""
    return float(np.quantile(pool, level))
