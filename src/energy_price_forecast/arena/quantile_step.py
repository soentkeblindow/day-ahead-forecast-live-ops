"""Live, single-target-day FHS quantile step (Sprint 7.1 Teil B ->
Probephase) -- the one function the daily submission job calls for
"today", mirroring how ``scripts/run_daily_submission.py::
run_renewables_step`` wraps ``evaluation.renewables_walkforward.
run_renewables_backtest`` for a single day instead of re-implementing
the walk-forward.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..evaluation.config import ConformalConfig
from ..evaluation.fhs_quantiles import DEFAULT_LEVELS, compute_fhs_quantiles_for_row
from ..market_time import LOCAL_TZ
from ..ops.oos_store import OOS_MODEL_VERSION, OOS_STORE_PATH, load_oos_store

# docs/sprint7_step7_1_log.md section 8.2 (the preregistered 90-vs-180-day
# comparison) and section 3.2's own n_neighbors -- the same values
# scripts/compute_fhs_quantiles.py and scripts/check_fhs_pool_gap_
# sensitivity.py already use for this exact mechanism.
POOL_WINDOW_DAYS = 180
CONFIG = ConformalConfig(n_neighbors=800)


def _local_day_index(target_day: dt.date) -> pd.DatetimeIndex:
    """Quarter-hourly UTC index covering local calendar day ``target_day``
    -- same construction as ``models.arena_baseline._local_day_index``,
    duplicated rather than imported since each caller in this codebase
    already keeps its own small copy of this (``oos_store_parity_check.
    py``'s ``_expected_quarterhour_index`` is a third)."""
    start = pd.Timestamp(target_day, tz=LOCAL_TZ)
    end = pd.Timestamp(target_day + dt.timedelta(days=1), tz=LOCAL_TZ)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


def run_quantile_step(
    ladder_row: str,
    point_values: list[float],
    target_day: dt.date,
    prices_qh: pd.Series,
    *,
    oos_store_path: Path = OOS_STORE_PATH,
) -> dict[float, pd.Series] | None:
    """Today's FHS quantiles for ``ladder_row`` (the point-forecast
    ladder's selected candidate), built around ``point_values`` -- the
    EXACT values already submitted for Challenge 2 (spec section 1:
    "median(t): die Viertelstundenprognose, die heute schon für Challenge
    2 eingereicht wird"), never a separately-held forecast series.

    Returns ``None`` if today's pool/sigma turns out unusable (spec
    section 3.4's trigger condition) -- the Probephase's policy (owner
    decision, 2026-10-21) is to skip Challenge 8 for this day only, never
    to borrow another row's pool; Challenge 2 is entirely unaffected
    either way, since this function is only ever called AFTER Challenge
    2's own outcome is already final.

    Merges ``logs/oos_store.csv``'s history for ``ladder_row`` with
    today's values so the pool/sigma calculation sees a continuous
    series through today -- the OOS-Store itself is only ever updated
    the FOLLOWING day by ``scripts/oos_store_job.py``'s daily top-up, so
    today's own value is never in it yet. ``prices_qh`` naturally has no
    value for ``target_day`` either (it has not happened yet); that is
    exactly the case ``compute_fhs_quantiles_for_row`` was fixed to
    support (see its own docstring).
    """
    today_index = _local_day_index(target_day)
    if len(point_values) != len(today_index):
        raise ValueError(
            f"point_values has {len(point_values)} value(s), expected {len(today_index)} "
            f"for {target_day}"
        )
    today_forecast = pd.Series(point_values, index=today_index, dtype=float)

    oos_store = load_oos_store(oos_store_path)
    history = oos_store.loc[
        (oos_store["ladder_row"] == ladder_row) & (oos_store["model_version"] == OOS_MODEL_VERSION)
    ]
    history_forecast = history.set_index("timestamp_utc")["forecast"].astype(float).sort_index()

    forecast = today_forecast.combine_first(history_forecast).sort_index()

    quantiles = compute_fhs_quantiles_for_row(
        forecast,
        prices_qh,
        window_days=POOL_WINDOW_DAYS,
        config=CONFIG,
        levels=DEFAULT_LEVELS,
    )

    today_quantiles = {level: series.loc[today_index] for level, series in quantiles.items()}
    if today_quantiles[0.5].isna().any():
        return None
    return today_quantiles
