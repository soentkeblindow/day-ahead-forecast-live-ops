"""Sprint 7.1 Teil B, offline step (spec section 6, step 2): compute FHS
quantiles over the OOS-Store backfill, decide the residual-pool window
length (90 vs 180 days, spec section 3.3), and write the comparison +
backtest coverage report.

Not part of any daily job or the submit workflow (spec section 6: Teil B's
submission wiring is a later, separately-gated step). Reads only
already-committed/already-synced disk state (``logs/oos_store.csv`` via
``ops.oos_store``, the realised quarter-hourly price series via
``arena.live_inputs.read_quarterhourly_prices`` -- which itself needs the
data store already loaded, see ``ops.store.load_store``) -- no live
ENTSO-E/weather/commodity fetch.

Two phases, both run by ``main()``:

1. **Window selection** (spec section 3.3): ``core_gas`` only, both
   candidate windows (90/180 days), restricted to target days where BOTH
   candidates have a full pool. Mean WIS decides; a <2% difference prefers
   180 days (more stable tails). Written to
   ``outputs/results/fhs_window_selection.csv``.
2. **Full backtest report**: all three ladder rows, the CHOSEN window,
   over the whole backfill. Coverage 50%/95% overall and separately for
   October 2026 (the already-documented gas-price window-extrapolation
   degradation). Written to ``outputs/results/fhs_quantile_backtest.csv``.
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
from energy_price_forecast.evaluation.fhs_quantiles import (
    DEFAULT_LEVELS,
    compute_fhs_quantiles_for_row,
)
from energy_price_forecast.evaluation.metrics import (
    interval_coverage,
    pinball,
    weighted_interval_score,
)
from energy_price_forecast.market_time import LOCAL_TZ
from energy_price_forecast.models.arena_baseline import baseline_quantiles
from energy_price_forecast.ops import store
from energy_price_forecast.ops.oos_store import LADDER_ROW_NAMES, OOS_MODEL_VERSION, load_oos_store

logger = logging.getLogger(__name__)

# spec section 3.3: only the two preregistered candidates (365 is out --
# quarter-hourly prices only start 2025-09-30).
WINDOW_CANDIDATES: tuple[int, ...] = (90, 180)

# spec section 3.3: a <2% difference in mean WIS prefers 180 days (more
# stable tails) over the headline-cheaper 90-day window.
WINDOW_SELECTION_TOLERANCE: float = 0.02

# spec section 3.2: "n_neighbors rund 800 (200 Stunden x 4)". embargo_days
# (1), sigma_floor_fraction (0.10), min_calibration_hours (24*14, "14
# Tage fuer sigma") and recompute_cadence_days (1) all stay the
# ConformalConfig default -- spec names them explicitly as the values to
# keep, not to retune.
CONFIG = ConformalConfig(n_neighbors=800)

WINDOW_SELECTION_PATH = PROJECT_ROOT / "outputs" / "results" / "fhs_window_selection.csv"
BACKTEST_PATH = PROJECT_ROOT / "outputs" / "results" / "fhs_quantile_backtest.csv"


def _oos_forecast_series(oos_store: pd.DataFrame, ladder_row: str) -> pd.Series:
    rows = oos_store.loc[
        (oos_store["ladder_row"] == ladder_row) & (oos_store["model_version"] == OOS_MODEL_VERSION)
    ]
    return rows.set_index("timestamp_utc")["forecast"].astype(float).sort_index()


def _fully_valid_days(quantiles: dict[float, pd.Series]) -> set[pd.Timestamp]:
    """Local calendar days where every period is non-NaN (a full pool) --
    sigma/pool NaN-ness is constant within a day (see fhs_quantiles'
    own docstring), so checking the median level alone is sufficient."""
    s = quantiles[0.5]
    local_days = pd.DatetimeIndex(s.index).tz_convert(LOCAL_TZ).normalize()
    valid_by_day = pd.Series(s.notna().to_numpy(), index=local_days).groupby(level=0).all()
    return set(valid_by_day[valid_by_day].index)


def _restrict_to_days(series: pd.Series, days: set[pd.Timestamp]) -> pd.Series:
    local_days = pd.DatetimeIndex(series.index).tz_convert(LOCAL_TZ).normalize()
    return series.loc[pd.Index(local_days).isin(days)]


def _metrics_per_day(
    quantiles: dict[float, pd.Series], price: pd.Series, days: set[pd.Timestamp]
) -> pd.DataFrame:
    """One row per day in ``days``: WIS, 50%/95% interval coverage, and
    per-level pinball loss, each computed only over that day's own
    periods -- the per-day granularity the report's "Verlauf ueber die
    Zeit" (spec section 7 item 4) and the October-2026 breakout both need.
    """
    rows: list[dict[str, object]] = []
    for day in sorted(days):
        day_mask = pd.DatetimeIndex(price.index).tz_convert(LOCAL_TZ).normalize() == day
        idx = price.index[day_mask]
        y = price.loc[idx]
        preds = {a: quantiles[a].loc[idx] for a in quantiles}
        row: dict[str, object] = {
            "day": day.date(),
            "wis": weighted_interval_score(y, preds),
            "coverage_50": interval_coverage(y, preds[0.25], preds[0.75]),
            "coverage_95": interval_coverage(y, preds[0.025], preds[0.975]),
        }
        for level in sorted(preds):
            row[f"pinball_{level:.3f}"] = pinball(y, preds[level], level)
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_wis_per_day(price: pd.Series, days: set[pd.Timestamp]) -> pd.DataFrame:
    """Arena quantile-track baseline WIS (spec section 3.1: empirical
    quantiles of the last 10 days) for the same days, for comparison.
    A day whose own 10-day lookback hits a real price gap is skipped
    (reported as a note), not fatal -- this is a comparison reference,
    not the headline result.
    """
    rows: list[dict[str, object]] = []
    skipped = 0
    for day in sorted(days):
        day_mask = pd.DatetimeIndex(price.index).tz_convert(LOCAL_TZ).normalize() == day
        idx = price.index[day_mask]
        y = price.loc[idx]
        try:
            baseline = baseline_quantiles(
                price, pd.Timestamp(day), DEFAULT_LEVELS, lookback_days=10
            )
        except ValueError:
            skipped += 1
            continue
        preds = {a: baseline[a].reindex(idx) for a in baseline}
        rows.append({"day": day.date(), "baseline_wis": weighted_interval_score(y, preds)})
    if skipped:
        logger.info(
            "baseline_quantiles skipped %d day(s) with an incomplete 10-day lookback", skipped
        )
    return pd.DataFrame(rows)


def select_window(oos_store: pd.DataFrame, price: pd.Series) -> tuple[int, pd.DataFrame]:
    """spec section 3.3: ``core_gas`` only, the two candidate windows,
    restricted to days where both have a full pool. Returns the chosen
    window and the comparison table (written to
    ``fhs_window_selection.csv``)."""
    forecast = _oos_forecast_series(oos_store, "core_gas")

    quantiles_by_window = {
        window: compute_fhs_quantiles_for_row(forecast, price, window_days=window, config=CONFIG)
        for window in WINDOW_CANDIDATES
    }
    comparable_days = set.intersection(
        *(_fully_valid_days(q) for q in quantiles_by_window.values())
    )
    logger.info(
        "window selection: %d day(s) with a full pool for every candidate", len(comparable_days)
    )

    baseline_per_day = _baseline_wis_per_day(price, comparable_days)
    baseline_wis = (
        float(baseline_per_day["baseline_wis"].mean())
        if not baseline_per_day.empty
        else float("nan")
    )

    report_rows: list[dict[str, object]] = []
    mean_wis: dict[int, float] = {}
    for window, quantiles in quantiles_by_window.items():
        per_day = _metrics_per_day(quantiles, price, comparable_days)
        mean_wis[window] = float(per_day["wis"].mean())
        report_rows.append(
            {
                "window_days": window,
                "n_days": len(comparable_days),
                "mean_wis": mean_wis[window],
                "coverage_50": float(per_day["coverage_50"].mean()),
                "coverage_95": float(per_day["coverage_95"].mean()),
                "baseline_wis": baseline_wis,
                **{
                    f"mean_pinball_{level:.3f}": float(per_day[f"pinball_{level:.3f}"].mean())
                    for level in DEFAULT_LEVELS
                },
            }
        )

    wis_90, wis_180 = mean_wis[90], mean_wis[180]
    relative_diff = (
        abs(wis_180 - wis_90) / min(wis_90, wis_180) if min(wis_90, wis_180) > 0 else 0.0
    )
    if relative_diff < WINDOW_SELECTION_TOLERANCE:
        chosen = 180
        reason = f"difference {relative_diff:.1%} < {WINDOW_SELECTION_TOLERANCE:.0%} -> prefer 180 (stabler tails)"
    else:
        chosen = min(mean_wis, key=lambda w: mean_wis[w])
        reason = f"difference {relative_diff:.1%} >= {WINDOW_SELECTION_TOLERANCE:.0%} -> lower mean WIS wins"
    logger.info("window selection: chosen=%d days (%s)", chosen, reason)

    report = pd.DataFrame(report_rows)
    report["chosen"] = report["window_days"] == chosen
    return chosen, report


def full_backtest_report(
    oos_store: pd.DataFrame, price: pd.Series, window_days: int
) -> pd.DataFrame:
    """All three ladder rows, the chosen window, over the whole backfill
    (spec section 7 items 2/4: coverage overall and separately for
    October 2026)."""
    all_rows: list[pd.DataFrame] = []
    for ladder_row in sorted(LADDER_ROW_NAMES):
        forecast = _oos_forecast_series(oos_store, ladder_row)
        if forecast.empty:
            logger.warning(
                "no OOS-store rows for ladder_row=%s with model_version=%s",
                ladder_row,
                OOS_MODEL_VERSION,
            )
            continue
        quantiles = compute_fhs_quantiles_for_row(
            forecast, price, window_days=window_days, config=CONFIG
        )
        valid_days = _fully_valid_days(quantiles)
        per_day = _metrics_per_day(quantiles, price, valid_days)
        per_day.insert(0, "ladder_row", ladder_row)
        per_day["is_october_2026"] = (per_day["day"] >= pd.Timestamp("2026-10-01").date()) & (
            per_day["day"] < pd.Timestamp("2026-11-01").date()
        )
        all_rows.append(per_day)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    t0 = time.monotonic()
    store.load_store(PROJECT_ROOT)
    oos_store = load_oos_store()
    price = read_quarterhourly_prices()["day_ahead_price"]

    chosen_window, window_report = select_window(oos_store, price)
    WINDOW_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    window_report.to_csv(WINDOW_SELECTION_PATH, index=False)

    backtest_report = full_backtest_report(oos_store, price, chosen_window)
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    backtest_report.to_csv(BACKTEST_PATH, index=False)

    runtime = time.monotonic() - t0
    logger.info(
        "compute_fhs_quantiles: chosen window=%d days, %d backtest row(s), %.1fs",
        chosen_window,
        len(backtest_report),
        runtime,
    )
    if not backtest_report.empty:
        overall = backtest_report.groupby("ladder_row")[
            ["coverage_50", "coverage_95", "wis"]
        ].mean()
        logger.info("overall coverage/WIS by ladder row:\n%s", overall.to_string())
        october = backtest_report.loc[backtest_report["is_october_2026"]]
        if not october.empty:
            october_summary = october.groupby("ladder_row")[
                ["coverage_50", "coverage_95", "wis"]
            ].mean()
            logger.info("October 2026 coverage/WIS by ladder row:\n%s", october_summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
