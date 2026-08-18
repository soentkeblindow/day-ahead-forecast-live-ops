"""Diebold-Mariano significance test on point-forecast loss differentials (Sprint 5.6).

Self-implemented (not pulled from a library, per Sprint 5.6 Decision 5): the
Newey-West HAC long-run variance estimator and the classic DM statistic with
the Harvey/Leybourne/Newbold small-sample correction. Pure functions only --
no file I/O, no MLflow; the thin I/O layer (``scripts/run_dm_test.py``) loads
the persisted predictions and writes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    """Result of a Diebold-Mariano test on a loss-differential series.

    Sign convention: `mean_loss_diff = mean(loss_a - loss_b)`. A NEGATIVE
    value means model A (the first argument to `dm_test`) has the lower
    expected loss.
    """

    n_obs: int
    hac_lag: int
    horizon: int
    mean_loss_diff: float
    dm_stat: float
    p_value: float


def newey_west_long_run_variance(d: np.ndarray, lag: int) -> float:
    """Bartlett-kernel HAC estimate of the long-run variance of `d`.

    `gamma_0 + 2 * sum_{k=1..lag} (1 - k / (lag + 1)) * gamma_k`, where
    `gamma_k` is the BIASED (divided by `n`, not by the number of overlapping
    pairs `n - k`) sample autocovariance of `d` at lag `k`. The Bartlett
    weights guarantee a non-negative estimate; a value of exactly 0 (e.g. a
    constant series) raises `ValueError` instead of dividing by zero downstream.
    """
    n = len(d)
    centered = d - d.mean()
    variance = float(np.sum(centered**2) / n)
    for k in range(1, lag + 1):
        gamma_k = float(np.sum(centered[k:] * centered[:-k]) / n)
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * gamma_k
    if variance <= 0:
        raise ValueError("zero long-run variance -- cannot compute a DM statistic.")
    return variance


def dm_test(
    loss_a: pd.Series,
    loss_b: pd.Series,
    *,
    hac_lag: int,
    horizon: int,
) -> DMResult:
    """Diebold-Mariano test on the loss differential `d_t = loss_a - loss_b`.

    Inputs are per-period losses (here: absolute errors), already aligned on
    an IDENTICAL index -- raises `ValueError` on index mismatch or on any NaN
    (alignment is the caller's job; silently dropping rows here would hide
    bugs). Applies the Harvey/Leybourne/Newbold small-sample correction
    `sqrt((n + 1 - 2*h + h*(h - 1)/n) / n)` to the classic DM statistic and
    returns a two-sided p-value from a Student-t distribution with `n - 1`
    degrees of freedom.
    """
    if not loss_a.index.equals(loss_b.index):
        raise ValueError("loss_a and loss_b must share an identical index.")

    combined_nan = loss_a.isna() | loss_b.isna()
    if combined_nan.any():
        first_ts = loss_a.index[combined_nan][0]
        raise ValueError(
            f"{int(combined_nan.sum())} NaN value(s) in the loss series after "
            f"alignment, first at {first_ts}."
        )

    d = (loss_a - loss_b).to_numpy()
    n = len(d)
    mean_d = float(d.mean())
    long_run_var = newey_west_long_run_variance(d, hac_lag)

    h = horizon
    dm_stat_raw = mean_d / np.sqrt(long_run_var / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = float(dm_stat_raw * hln_factor)
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1)))

    return DMResult(
        n_obs=n,
        hac_lag=hac_lag,
        horizon=horizon,
        mean_loss_diff=mean_d,
        dm_stat=dm_stat,
        p_value=p_value,
    )


def daily_mean_loss(loss: pd.Series) -> tuple[pd.Series, int]:
    """Aggregate an hourly, UTC-indexed loss series to calendar-day means.

    Plain groupby-mean on the UTC date. Returns the daily series plus the
    count of incomplete days (fewer than 24 hours) so the caller can log it;
    incomplete days are KEPT (dropping them would silently change `n`).
    """
    day = pd.DatetimeIndex(loss.index).normalize()
    grouped = loss.groupby(day)
    daily = grouped.mean()
    n_incomplete = int((grouped.size() < 24).sum())
    return daily, n_incomplete
