"""Scaled (normalized) per-quantile conformal recalibration (Sprint 4.3b).

Step 4.3a measured that the raw quantile forecasts are under-covered, most
severely at the high end of the forecast-level range. This module CORRECTS
that: it turns the seven raw quantile series into a calibrated set whose
marginal coverage holds again, before step 4.4 computes risk (ES/VaR) on the
calibrated bands.

Theory (scaled / normalized split conformal, one-sided, per quantile level)
----------------------------------------------------------------------------
Goal: a calibrated quantile q_tilde_a such that marginal coverage
P(y <= q_tilde_a) ~ a (averaged over all calibrated hours).

Nonconformity score. For a calibration hour i, the signed, NORMALIZED
residual::

    s_i = (y_i - q_hat_a(x_i)) / sigma_i,  sigma_i = sigma(q_hat_0.5(x_i))

sigma is the level-conditional spread of realizations (see ``local_scale``).

Correction. Q_a is the finite-sample-corrected empirical alpha-quantile of
the scores: the ``ceil((n+1)*alpha)``-th order statistic of ``{s_i}`` (rank
clamped to ``[1, n]``); the ``(n+1)`` detail makes the coverage guarantee
finite-sample rather than only asymptotic.

Application (scaled). ``q_tilde_a(x) = q_hat_a(x) + Q_a * sigma(q_hat_0.5(x))``.
For a high-price hour sigma is large, so the correction is automatically
larger there -- the outer bands widen exactly where 4.3a found them
under-covered; in calm hours sigma stays small and the bands stay tight.

Two honest caveats (kept here, not just in the spec):
- The coverage guarantee holds for ANY sigma -- sigma improves CONDITIONAL
  coverage and sharpness, not validity. The sigma choice is a quality
  question, not a validity question.
- The guarantee relies on exchangeability of calibration and test points. In
  a drifting time series that only holds approximately; the rolling window
  + embargo keeps causal leakage clean but does not undo drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ
from .config import ConformalConfig


def calibration_window(
    test_day: pd.Timestamp,
    index: pd.DatetimeIndex,
    *,
    window_days: int,
    embargo_days: int,
) -> pd.DatetimeIndex:
    """Hours usable to calibrate ``test_day``, strictly before the embargo edge.

    Returns the subset of ``index`` in ``[edge - window_days, edge)`` where
    ``edge = test_day - embargo_days``. The half-open upper bound is the
    leakage guard: no returned timestamp reaches ``edge``, so no realization
    that is unknown at the test day's gate closure can enter the calibration
    set. Pure and side-effect-free; the leakage test targets this contract.
    """
    edge = test_day - pd.Timedelta(days=embargo_days)
    lo = edge - pd.Timedelta(days=window_days)
    return index[(index >= lo) & (index < edge)]


def local_scale(
    query_level: pd.Series,
    calib_median: pd.Series,
    calib_y: pd.Series,
    *,
    k: int,
    floor: float,
) -> pd.Series:
    """kNN-in-forecast-level estimate of the local realization std (sigma).

    For each query value ``p`` in ``query_level`` (a median forecast q_0.5),
    take the ``k`` calibration hours whose own ``calib_median`` is closest to
    ``p`` (distance ``|calib_median - p|``) and return the sample std
    (ddof=1) of their realized prices ``calib_y``, floored at ``floor``. This
    is the local, level-conditional volatility that makes the conformal
    correction breathe: large where the forecast level is volatile (the
    tail), small in calm hours.

    Uses fewer than ``k`` neighbours only if the calibration set is smaller.
    Returns a Series aligned to ``query_level``; each value is >= ``floor``.
    Self-inclusive: a query point that is itself part of ``calib_median`` may
    be its own nearest neighbour (see the module docstring's exchangeability
    note) -- sigma is a fixed level-to-scale function over the window, not a
    leave-one-out estimate.
    """
    paired = pd.concat([calib_median.rename("med"), calib_y.rename("y")], axis=1).dropna()
    med = paired["med"].to_numpy()
    y = paired["y"].to_numpy()

    if med.size == 0:
        return pd.Series(floor, index=query_level.index)

    n_neighbors = min(k, med.size)

    def _sigma(p: float) -> float:
        dist = np.abs(med - p)
        nearest = np.argpartition(dist, n_neighbors - 1)[:n_neighbors]
        std = float(np.std(y[nearest], ddof=1)) if n_neighbors > 1 else 0.0
        return max(std, floor)

    values = [_sigma(p) for p in query_level.to_numpy()]
    return pd.Series(values, index=query_level.index)


def quantile_shift(
    calib_y: pd.Series,
    calib_pred: pd.Series,
    calib_sigma: pd.Series,
    alpha: float,
) -> float:
    """Finite-sample conformal shift Q_alpha on normalized one-sided scores.

    Scores ``s_i = (calib_y - calib_pred) / calib_sigma``; returns the
    ``ceil((n+1)*alpha)``-th order statistic of the scores (rank clamped to
    ``[1, n]``). Applying ``q_tilde = q_pred + Q_alpha * sigma(x)`` yields
    marginal coverage P(y <= q_tilde) ~ alpha under exchangeability.
    """
    paired = pd.concat(
        [calib_y.rename("y"), calib_pred.rename("pred"), calib_sigma.rename("sigma")], axis=1
    ).dropna()
    n = len(paired)
    if n == 0:
        return float("nan")

    scores = (paired["y"] - paired["pred"]) / paired["sigma"]
    rank = int(np.ceil((n + 1) * alpha))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores.to_numpy())[rank - 1])


@dataclass
class _CalibrationState:
    """One recompute day's calibration-window snapshot, internal to this module."""

    cal_idx: pd.DatetimeIndex
    cal_med: pd.Series | None
    cal_y: pd.Series | None
    floor: float
    uncalibrated: bool
    n_cal: int


def _calibration_state(
    day: pd.Timestamp,
    q50: pd.Series,
    y_true: pd.Series,
    *,
    config: ConformalConfig,
    window_days: int,
) -> _CalibrationState:
    """Trailing calibration window + local-scale floor for one recompute day.

    Single source for the window-selection/floor logic shared by
    ``scaled_conformal_calibrate`` and ``hourly_sigma`` -- factored out so the
    two never drift apart (Sprint 4.4a). Below ``config.min_calibration_hours``
    usable hours, returns an uncalibrated state (``cal_med``/``cal_y`` None,
    ``floor`` 0.0) rather than raising.
    """
    cal_idx = calibration_window(
        day,
        pd.DatetimeIndex(y_true.index),
        window_days=window_days,
        embargo_days=config.embargo_days,
    )
    n_cal = len(cal_idx)
    if n_cal < config.min_calibration_hours:
        return _CalibrationState(cal_idx, None, None, 0.0, True, n_cal)
    cal_y = y_true.loc[cal_idx]
    cal_med = q50.loc[cal_idx]
    floor = config.sigma_floor_fraction * float(cal_y.std(ddof=1))
    return _CalibrationState(cal_idx, cal_med, cal_y, floor, False, n_cal)


def hourly_sigma(
    y_true: pd.Series,
    q50_raw: pd.Series,
    *,
    config: ConformalConfig,
    window_days: int | None = None,
) -> pd.Series:
    """Per-hour local-scale sigma(t) over q50_raw's full index.

    The SAME sigma ``scaled_conformal_calibrate`` applies to the seven
    quantile levels -- factored out so downstream consumers (Sprint 4.4a's
    residual pool) import it verbatim instead of re-deriving it. Recomputed
    every ``config.recompute_cadence_days`` from a trailing calibration
    window (``_calibration_state``); hours whose window is uncalibrated get
    NaN, not an exception.
    """
    w = window_days if window_days is not None else config.window_days
    test_index = pd.DatetimeIndex(q50_raw.index)
    local_days = test_index.tz_convert(LOCAL_TZ).normalize()
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))

    sigma = pd.Series(float("nan"), index=test_index)
    state: _CalibrationState | None = None
    for day_num, day in enumerate(delivery_days):
        if day_num % config.recompute_cadence_days == 0:
            state = _calibration_state(day, q50_raw, y_true, config=config, window_days=w)
        assert state is not None  # day_num == 0 always recomputes
        mask = local_days == day
        d_idx = test_index[mask]
        if state.uncalibrated or state.cal_med is None or state.cal_y is None:
            continue
        sigma.loc[mask] = local_scale(
            q50_raw.loc[d_idx], state.cal_med, state.cal_y, k=config.n_neighbors, floor=state.floor
        ).to_numpy()
    return sigma


def scaled_conformal_calibrate(
    y_true: pd.Series,
    preds: dict[float, pd.Series],
    *,
    config: ConformalConfig,
    window_days: int | None = None,
) -> tuple[dict[float, pd.Series], pd.DataFrame]:
    """Apply scaled per-quantile conformal recalibration over a walk-forward.

    For each test day (grouped by delivery day in local time), build the
    calibration window (``calibration_window``), estimate the
    level-conditional scale (``local_scale``) at each calibration hour's own
    q_0.5, compute one shift per level (``quantile_shift``), and apply
    ``q_tilde_a(x) = q_a(x) + Q_a * sigma(q_0.5(x))`` to all hours of the day.
    sigma and the Q_a are recomputed every ``config.recompute_cadence_days``
    days; on the days in between, the most recently computed values are
    reused (sigma is still re-evaluated at each day's own q_0.5, so the
    applied correction keeps breathing with the forecast level even though
    Q_a is fixed between recomputes).

    Days with fewer than ``config.min_calibration_hours`` usable calibration
    hours are passed through UNCHANGED (raw) and flagged ``uncalibrated`` in
    the diagnostics -- typically the first days of the test period, before a
    full calibration window has accumulated.

    Parameters
    ----------
    y_true:
        Realised price series. Must cover both the calibration windows (the
        earlier portion of the same walk-forward test period) and the test
        index of ``preds``.
    preds:
        Maps each level to its raw prediction series (one per backtest run).
        Must contain 0.5 -- it is both a calibrated level and the sigma key.
    window_days:
        Overrides ``config.window_days`` when given (wired to the CLI).

    Returns
    -------
    calibrated:
        Same levels as ``preds``; each a Series q_tilde_a over ``preds``'
        index. Uncalibrated days keep their raw (unmodified) values.
    diagnostics:
        One row per (day, level): the shift ``Q``, the calibration-window
        hour count ``n``, the mean applied ``sigma``, the ``uncalibrated``
        flag, and the day's ``crossing_rate`` over the calibrated levels
        (reported, never enforced -- quantiles are not re-sorted).
    """
    if 0.5 not in preds:
        raise ValueError(
            "median forecast (level 0.5) required for calibration: it is the sigma key"
        )

    levels = sorted(preds)
    q50 = preds[0.5]
    test_index = pd.DatetimeIndex(q50.index)
    w = window_days if window_days is not None else config.window_days

    local_days = test_index.tz_convert(LOCAL_TZ).normalize()
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))

    calibrated: dict[float, pd.Series] = {a: preds[a].astype(float).copy() for a in levels}
    diagnostics_rows: list[dict[str, object]] = []

    # The per-hour sigma applied below is computed once, up front, via the
    # SAME function Sprint 4.4a imports for its residual pool (spec 3.2) --
    # not re-derived here, so the two can never drift apart.
    sigma_series = hourly_sigma(y_true, q50, config=config, window_days=w)

    state: _CalibrationState | None = None
    shifts: dict[float, float] = dict.fromkeys(levels, float("nan"))

    for day_num, day in enumerate(delivery_days):
        if day_num % config.recompute_cadence_days == 0:
            state = _calibration_state(day, q50, y_true, config=config, window_days=w)
            if state.uncalibrated or state.cal_med is None or state.cal_y is None:
                shifts = dict.fromkeys(levels, float("nan"))
            else:
                # cal_sigma is a SEPARATE query of local_scale -- sigma AT the
                # calibration hours themselves (self-referential), used only
                # to normalize the quantile_shift scores. Not the same
                # quantity as sigma_series, which is queried at test hours.
                cal_sigma = local_scale(
                    state.cal_med,
                    state.cal_med,
                    state.cal_y,
                    k=config.n_neighbors,
                    floor=state.floor,
                )
                shifts = {
                    a: quantile_shift(state.cal_y, preds[a].loc[state.cal_idx], cal_sigma, a)
                    for a in levels
                }
        assert state is not None  # day_num == 0 always recomputes

        d_idx = test_index[local_days == day]
        sigma_day = sigma_series.loc[d_idx]

        if not state.uncalibrated:
            for a in levels:
                calibrated[a].loc[d_idx] = preds[a].loc[d_idx] + shifts[a] * sigma_day

        day_frame = pd.DataFrame({a: calibrated[a].loc[d_idx] for a in levels})
        if len(day_frame) > 0:
            monotonic = day_frame.apply(lambda row: row.is_monotonic_increasing, axis=1)
            crossing_rate = float((~monotonic).mean())
        else:
            crossing_rate = float("nan")
        mean_sigma = float(sigma_day.mean()) if len(sigma_day) > 0 else float("nan")

        for a in levels:
            diagnostics_rows.append(
                {
                    "day": day,
                    "level": a,
                    "Q": shifts[a],
                    "n": state.n_cal,
                    "sigma": mean_sigma,
                    "uncalibrated": state.uncalibrated,
                    "crossing_rate": crossing_rate,
                }
            )

    return calibrated, pd.DataFrame(diagnostics_rows)
