"""Filtered Historical Simulation: the residual filter and the tail pool (Sprint 4.4a).

This module holds a DISTRIBUTION OBJECT, not a risk measure -- the same
separation ``rearrangement.py`` drew out of ``conformal.py`` in 4.3c. It turns
realised prices into standardised, (approximately) poolable shocks and
collects them into a leakage-free rolling pool per delivery day.
``evaluation.risk`` consumes that pool but never touches its window/embargo
logic; that logic lives here, in one place, so it can be tested in isolation.

Filtered Historical Simulation (FHS) in one line: divide out the
conditional location and scale (``standardised_residuals``) so shocks from a
calm May and a December Dunkelflaute become samples from (approximately) the
same distribution, then pool the standardised shocks across a long history
(``pool_by_day``) to get enough tail mass for an Expected Shortfall.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ

_TAIL_WINDOW_DAYS: int = 365  # separate from ConformalConfig.window_days (E-c)
_MIN_POOL_HOURS: int = 180 * 24


def standardised_residuals(y_true: pd.Series, median: pd.Series, sigma: pd.Series) -> pd.Series:
    """Filter the price by its conditional location and scale: r = (P - q50) / sigma.

    The "filtered" step of Filtered Historical Simulation. Dividing out the
    conditional scale is what makes shocks from a calm May and a December
    Dunkelflaute samples from (approximately) the same distribution, hence
    poolable. `median` MUST be the same series used as the book's mark; `sigma`
    MUST come from conformal.py and is never re-derived here.

    Known violation of the i.i.d. premise (measured, not fixed -- see spec 2.5):
    std(r) is ~0.87 in local night hours and ~1.29 in the 19:00 evening ramp.
    Sigma conditions on forecast LEVEL, not on ramp STEEPNESS.

    Hours with missing sigma (15 uncalibrated days) yield NaN, not an exception.
    """
    return (y_true - median) / sigma


def pool_by_day(
    r: pd.Series,
    *,
    window_days: int = _TAIL_WINDOW_DAYS,
    embargo_days: int,
    min_pool_hours: int = _MIN_POOL_HOURS,
) -> dict[dt.date, np.ndarray]:
    """One ASCENDING-SORTED pool of past shocks per delivery day.

    Leakage contract: pool_by_day(D) contains no timestamp >= D - embargo_days.
    NaN entries in `r` (missing sigma -- e.g. the first ~14-15 uncalibrated
    days of a backtest, see spec 2.5) are dropped BEFORE sorting: never
    counted toward `min_pool_hours`, never present in the returned array.
    Days below `min_pool_hours` (measured on the NaN-filtered count) map to an
    EMPTY array (callers return NaN). Never winsorise, trim or otherwise treat
    outliers: the -14.08 shock of 2023-07-02 happened and a long book lost
    the money.

    The window is a plain boolean mask against `r`'s real (NaN-dropped) index
    -- exactly like ``conformal.calibration_window`` -- which already
    "expands" for free whenever `window_days` reaches before the earliest
    available residual, and becomes a fixed-width rolling window once enough
    history has accumulated. No separate mode is needed: an unclamped window
    request cannot select data that does not exist; the embargo cutoff (the
    window's upper bound) stays strictly `day - embargo_days` regardless.
    """
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
        if window_values.size < min_pool_hours:
            pools[day.date()] = np.array([], dtype=float)
        else:
            pools[day.date()] = np.sort(window_values)

    return pools


def lower_tail_mean(pool: np.ndarray, threshold: float) -> tuple[float, int]:
    """mean(pool[pool <= threshold]) and the tail count.

    Threshold-based, NOT count-based: the threshold may come from outside the
    pool (the calibrated grid), where no `ceil(alpha * N)` rule applies.
    Empty tail -> (nan, 0). Ties are INCLUDED (`<=`).
    """
    tail = pool[pool <= threshold]
    if tail.size == 0:
        return float("nan"), 0
    return float(tail.mean()), int(tail.size)


def upper_tail_mean(pool: np.ndarray, threshold: float) -> tuple[float, int]:
    """mean(pool[pool >= threshold]) and the tail count.

    Mirror of `lower_tail_mean` for the short side's upper tail. Ties are
    INCLUDED (`>=`). Empty tail -> (nan, 0).
    """
    tail = pool[pool >= threshold]
    if tail.size == 0:
        return float("nan"), 0
    return float(tail.mean()), int(tail.size)


def tail_quantile(pool: np.ndarray, level: float) -> float:
    """The `level`-quantile of the pool (anchor identity with `np.quantile`)."""
    return float(np.quantile(pool, level))


def threshold_position(pool: np.ndarray, threshold: float, *, lower: bool) -> float:
    """pi = share of the pool beyond `threshold`.

    For variant "fhs" this returns the nominal level by construction. For
    "calibrated" it MEASURES where the calibrated quantile actually sits inside the
    median-anchored location-scale pool. That measurement is the point.

    `lower=True` uses the same `<=` convention as `lower_tail_mean`; `lower=False`
    uses the same `>=` convention as `upper_tail_mean`. Empty pool -> NaN.
    """
    if pool.size == 0:
        return float("nan")
    if lower:
        return float(np.mean(pool <= threshold))
    return float(np.mean(pool >= threshold))
