"""Sprint 7.1 Teil B follow-up measurement: how sensitive is the FHS pool's
empirical quantile (``fhs_pool.tail_quantile``) -- and the resulting
WIS/coverage -- to k missing calendar days inside the pool window?

Triggered by a real finding (docs/sprint7_step7_1_log.md section 8.3):
``compute_fhs_quantiles_for_row``'s current ``min_pool_days=window_days``
(zero tolerance) means a single known weather-defect day blocks
``core_gas``/``core_gas_loadpatch``'s quantile pool for the full
window_days that follow it. Before loosening that rule, this measures the
actual sensitivity -- the same "measure before replacing an exact
threshold with a tolerance" approach ``arena/preflight.py::
check_training_window`` took (spec 6.9 section 2.6, "Messung B"), applied
here to the pool rather than the training window.

Row: ``base`` (continuous OOS-Store coverage, zero real gaps) -- so a
synthetic gap is the ONLY difference between the baseline and degraded
pool, with no real-gap confound. Window: 180 days (the Teil B headline
choice). Two gap positions per Messung B's own distinction (mid-window
vs. trailing-edge, i.e. closest to the pool's own embargo cutoff) --
pool_by_day has no recency weighting at all (unlike a fitted model's
training window), so the working hypothesis is LOW sensitivity to
position, unlike Messung B's own finding for the price model's training
window; this measures that, rather than assuming it.

Not part of any daily job -- a one-off analysis script, same category as
``scripts/check_seasonal_error.py``/``check_capacity_extrapolation.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from energy_price_forecast.arena.live_inputs import read_quarterhourly_prices
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.evaluation.config import ConformalConfig
from energy_price_forecast.evaluation.conformal import quarterhour_sigma
from energy_price_forecast.evaluation.fhs_pool import standardised_residuals, tail_quantile
from energy_price_forecast.evaluation.fhs_quantiles import DEFAULT_LEVELS
from energy_price_forecast.evaluation.metrics import interval_coverage, weighted_interval_score
from energy_price_forecast.evaluation.rearrangement import rearrange_quantiles
from energy_price_forecast.market_time import LOCAL_TZ
from energy_price_forecast.ops import store
from energy_price_forecast.ops.oos_store import OOS_MODEL_VERSION, load_oos_store

logger = logging.getLogger(__name__)

ROW = "base"
WINDOW_DAYS = 180
EMBARGO_DAYS = 1
PERIODS_PER_DAY = 96
CONFIG = ConformalConfig(n_neighbors=800)

# Mirrors Messung B's own grid (spec 6.8), extended slightly since this
# pool's window (180 days = 17280 periods) is far larger than the price
# model's 90-day training window.
GAP_SIZES_DAYS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 14, 20, 30)
GAP_POSITIONS: tuple[str, ...] = ("mid_window", "trailing_edge")

# Last 20 real days of the backfill, all within `base`'s continuous,
# gap-free coverage -- real realised prices, real forecasts, only the
# synthetic pool gap differs between scenarios.
N_EVALUATION_DAYS = 20

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "fhs_pool_gap_sensitivity.csv"


def _window_bounds(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    edge = day - pd.Timedelta(days=EMBARGO_DAYS)
    lo = edge - pd.Timedelta(days=WINDOW_DAYS)
    return lo, edge


def _gap_bounds(
    lo: pd.Timestamp, edge: pd.Timestamp, k: int, position: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if k == 0:
        return edge, edge  # empty range
    if position == "trailing_edge":
        return edge - pd.Timedelta(days=k), edge
    # mid_window: centered in [lo, edge)
    span_days = (edge - lo).days
    start_offset = (span_days - k) // 2
    gap_start = lo + pd.Timedelta(days=start_offset)
    return gap_start, gap_start + pd.Timedelta(days=k)


def _pool_for_window(
    residuals: pd.Series,
    lo: pd.Timestamp,
    edge: pd.Timestamp,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
) -> pd.Series:
    window = residuals.loc[(residuals.index >= lo) & (residuals.index < edge)]
    if gap_start < gap_end:
        window = window.loc[(window.index < gap_start) | (window.index >= gap_end)]
    return window.dropna()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    t0 = time.monotonic()
    store.load_store(PROJECT_ROOT)
    oos_store = load_oos_store()
    rows = oos_store.loc[
        (oos_store["ladder_row"] == ROW) & (oos_store["model_version"] == OOS_MODEL_VERSION)
    ]
    forecast = rows.set_index("timestamp_utc")["forecast"].astype(float).sort_index()
    price = read_quarterhourly_prices()["day_ahead_price"]

    common_index = pd.DatetimeIndex(forecast.index.intersection(price.index)).sort_values()
    forecast = forecast.loc[common_index]
    price = price.loc[common_index]

    sigma = quarterhour_sigma(
        price, forecast, config=CONFIG, window_days=WINDOW_DAYS, periods_per_day=PERIODS_PER_DAY
    )
    residuals = standardised_residuals(price, forecast, sigma)

    local_days = common_index.tz_convert(LOCAL_TZ).normalize()
    delivery_days = sorted(pd.unique(local_days))
    eval_days = delivery_days[-N_EVALUATION_DAYS:]
    logger.info(
        "evaluating %d day(s): %s .. %s", len(eval_days), eval_days[0].date(), eval_days[-1].date()
    )

    rows_out: list[dict[str, object]] = []
    for position in GAP_POSITIONS:
        for k in GAP_SIZES_DAYS:
            if k == 0 and position != GAP_POSITIONS[0]:
                continue  # k=0 baseline is identical regardless of "position" -- report it once
            u_diffs: list[float] = []
            wis_base: list[float] = []
            wis_degraded: list[float] = []
            cov50_base: list[float] = []
            cov50_degraded: list[float] = []
            cov95_base: list[float] = []
            cov95_degraded: list[float] = []

            for day in eval_days:
                day_mask = local_days == day
                day_idx = common_index[day_mask]
                y = price.loc[day_idx]
                f = forecast.loc[day_idx]
                s = sigma.loc[day_idx]

                lo, edge = _window_bounds(day)
                gap_start, gap_end = _gap_bounds(lo, edge, k, position)
                pool_clean = _pool_for_window(residuals, lo, edge, edge, edge).to_numpy()
                pool_degraded = _pool_for_window(residuals, lo, edge, gap_start, gap_end).to_numpy()

                preds_base: dict[float, pd.Series] = {}
                preds_degraded: dict[float, pd.Series] = {}
                level_u_diff: list[float] = []
                for level in DEFAULT_LEVELS:
                    u_clean = tail_quantile(pool_clean, level)
                    u_degraded = tail_quantile(pool_degraded, level)
                    level_u_diff.append(abs(u_degraded - u_clean))
                    preds_base[level] = f + u_clean * s
                    preds_degraded[level] = f + u_degraded * s
                u_diffs.append(sum(level_u_diff) / len(level_u_diff))

                preds_base = rearrange_quantiles(preds_base)
                preds_degraded = rearrange_quantiles(preds_degraded)

                wis_base.append(weighted_interval_score(y, preds_base))
                wis_degraded.append(weighted_interval_score(y, preds_degraded))
                cov50_base.append(interval_coverage(y, preds_base[0.25], preds_base[0.75]))
                cov50_degraded.append(
                    interval_coverage(y, preds_degraded[0.25], preds_degraded[0.75])
                )
                cov95_base.append(interval_coverage(y, preds_base[0.025], preds_base[0.975]))
                cov95_degraded.append(
                    interval_coverage(y, preds_degraded[0.025], preds_degraded[0.975])
                )

            n = len(eval_days)
            rows_out.append(
                {
                    "position": position,
                    "k_missing_days": k,
                    "mean_abs_u_diff": sum(u_diffs) / n,
                    "mean_wis_baseline": sum(wis_base) / n,
                    "mean_wis_degraded": sum(wis_degraded) / n,
                    "wis_degradation_pct": (
                        (sum(wis_degraded) - sum(wis_base)) / sum(wis_base) * 100
                        if sum(wis_base) > 0
                        else float("nan")
                    ),
                    "coverage_50_baseline": sum(cov50_base) / n,
                    "coverage_50_degraded": sum(cov50_degraded) / n,
                    "coverage_95_baseline": sum(cov95_base) / n,
                    "coverage_95_degraded": sum(cov95_degraded) / n,
                }
            )

    report = pd.DataFrame(rows_out)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    runtime = time.monotonic() - t0
    logger.info(
        "check_fhs_pool_gap_sensitivity: %d scenario(s), %.1fs\n%s",
        len(report),
        runtime,
        report.to_string(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
