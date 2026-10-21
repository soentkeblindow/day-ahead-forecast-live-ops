"""Filtered Historical Simulation quantile construction (Sprint 7.1 Teil B,
spec section 1/3.2):

``q_a(t) = median(t) + u_a(day(t)) * sigma(t)``

where ``median(t)`` is the already-submitted point forecast, ``sigma(t)``
comes from ``conformal.quarterhour_sigma``, and ``u_a(day(t))`` is the
empirical a-quantile of the day's rolling pool of standardised residuals
(``fhs_pool.pool_by_day`` / ``fhs_pool.tail_quantile``).

``compute_fhs_quantiles_for_row`` is the one orchestration this module
exposes -- the same function a later live-wiring step would call for a
single target day (spec section 3.5), and the one
``scripts/compute_fhs_quantiles.py`` calls once per (ladder row, pool
window) over the whole OOS-Store backfill range (same "hoist the walk-
forward over the whole range, don't loop per day" lesson Sprint 7.1 Teil A
already learned -- ``conformal.quarterhour_sigma`` and this function's own
per-day pool loop both already work that way internally).
"""

from __future__ import annotations

import pandas as pd

from ..market_time import LOCAL_TZ
from .config import ConformalConfig
from .conformal import quarterhour_sigma
from .fhs_pool import pool_by_day, standardised_residuals, tail_quantile
from .rearrangement import rearrange_quantiles

DEFAULT_LEVELS: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)

# Sprint 7.1 Teil B follow-up measurement (scripts/check_fhs_pool_gap_
# sensitivity.py, docs/sprint7_step7_1_log.md section 9): a real finding
# showed a single missing calendar day inside the pool window (a known
# weather defect) blocks the pool for every subsequent day until the
# window rolls past it -- window_days (e.g. 180) days of silence from
# ONE bad day. Measured sensitivity is low and grows smoothly with k (no
# cliff); gap position barely matters (pool_by_day has no recency
# weighting). 14 is comfortably above the real case (k=1, ~0.1% WIS
# change) and comfortably below where real degradation starts (k>=20,
# concentrated in trailing-edge coverage_95) -- the same "stay
# conservative, don't go to the measured edge" convention
# arena/preflight.py's MAX_TRAINING_EDGE_AGE_DAYS=7 already follows.
POOL_GAP_TOLERANCE_DAYS: int = 14


def compute_fhs_quantiles_for_row(
    forecast: pd.Series,
    realized_price: pd.Series,
    *,
    window_days: int,
    config: ConformalConfig,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    periods_per_day: int = 96,
    pool_gap_tolerance_days: int = POOL_GAP_TOLERANCE_DAYS,
) -> dict[float, pd.Series]:
    """FHS quantiles for one ladder row over the full range covered by
    ``forecast``/``realized_price``'s common index.

    ``forecast`` MUST already be filtered to a single ``model_version``
    (spec section 3.2's rule 1, enforced by the caller -- this function
    has no model_version concept of its own, it just computes on whatever
    series it is given). ``realized_price`` is the realised quarter-hourly
    price (e.g. ``arena.live_inputs.read_quarterhourly_prices()``).

    Returns one Series per level, aligned to the common (forecast,
    realized_price) index, monotonically increasing per timestamp
    (``rearrangement.rearrange_quantiles``). A timestamp whose day's pool
    is too small (``fhs_pool.pool_by_day``'s own ``min_pool_days`` gate,
    called here with ``min_pool_days=window_days - pool_gap_tolerance_
    days`` -- a measured tolerance, not a literally-full-pool requirement,
    see ``POOL_GAP_TOLERANCE_DAYS``) or whose sigma is not yet calibrated
    (``conformal.quarterhour_sigma``) is NaN at every level, never
    partially -- both gates share one sigma series and one per-day pool,
    so a timestamp's NaN-ness never differs by level (safe for
    ``rearrange_quantiles``, which would otherwise risk misplacing a real
    value against a NaN one in the same row).
    """
    if 0.5 not in levels:
        raise ValueError("compute_fhs_quantiles_for_row requires a median (0.5) level")

    common_index = pd.DatetimeIndex(forecast.index.intersection(realized_price.index)).sort_values()
    forecast_aligned = forecast.loc[common_index]
    price_aligned = realized_price.loc[common_index]

    sigma = quarterhour_sigma(
        price_aligned,
        forecast_aligned,
        config=config,
        window_days=window_days,
        periods_per_day=periods_per_day,
    )
    residuals = standardised_residuals(price_aligned, forecast_aligned, sigma)
    pools = pool_by_day(
        residuals,
        window_days=window_days,
        embargo_days=config.embargo_days,
        min_pool_days=window_days - pool_gap_tolerance_days,
        periods_per_day=periods_per_day,
    )

    local_days = common_index.tz_convert(LOCAL_TZ).normalize()
    delivery_days = pd.DatetimeIndex(sorted(pd.unique(local_days)))
    u_by_level_and_day: dict[float, dict[pd.Timestamp, float]] = {level: {} for level in levels}
    for day in delivery_days:
        pool = pools[day.date()]
        for level in levels:
            u_by_level_and_day[level][day] = (
                tail_quantile(pool, level) if pool.size > 0 else float("nan")
            )

    raw: dict[float, pd.Series] = {}
    for level in levels:
        u_series = pd.Series(
            [u_by_level_and_day[level][day] for day in local_days],
            index=common_index,
            dtype=float,
        )
        raw[level] = forecast_aligned + u_series * sigma

    return rearrange_quantiles(raw)
