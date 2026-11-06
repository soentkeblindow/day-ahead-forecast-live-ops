"""Haltepunkt 1 (spec 7.4.1 section 3.6): recompute the OOS-Store's renewables
reconstruction with the historical capacity vintage each real live run would
actually have seen, and compare against the real submitted payloads.

Reproduces the OLD (pre-7.4.1) live run's own behavior exactly: for a given
target day, ONE capacity vintage is used for the ENTIRE walk-forward window
(training history and target day alike) -- the vintage valid on that day's
own Lauftag (``capacity_vintages.vintage_for_target_day``, spec section
3.2). This is deliberately NOT Teil B's per-day-in-store, grouped-by-vintage
approach (spec section 3.6 excludes it explicitly here) -- grouping several
target days under one shared vintage call would apply that group's vintage
to training rows that, as their own historical target day, might have used
an earlier one; this script cares about reproducing exactly what happened,
not about the forward-looking store's efficiency.

Writes to a SEPARATE local file -- ``logs/oos_store.csv`` is never read for
comparison input and never written by this script (spec section 3.6: "Der
OOS-Backfill lief... live lief... logs/oos_store.csv wird nicht angefasst").
Reuses ``scripts/oos_store_parity_check.py``'s own comparison logic
(``compare_day``, ``real_submitted_rows``) rather than a second
implementation.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Final

import pandas as pd

from energy_price_forecast.arena.live_inputs import (
    assemble_price_model_inputs,
    read_quarterhourly_prices,
)
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.capacity import CapacityExtrapolation, CapacitySource
from energy_price_forecast.data.capacity_vintages import vintage_for_target_day
from energy_price_forecast.evaluation.renewables_walkforward import (
    TARGET_COLUMNS,
    run_renewables_backtest,
)
from energy_price_forecast.ops.oos_store import LADDER_ROW_NAMES, OosStoreRow, rows_to_frame
from energy_price_forecast.ops.protocol import read_submission_records

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.oos_store_job import _weather_for_range  # noqa: E402
from scripts.oos_store_parity_check import (  # noqa: E402
    ParityDayResult,
    compare_day,
    real_submitted_rows,
)
from scripts.run_daily_submission import (  # noqa: E402
    PAYLOADS_DIR,
    RENEWABLES_TRAIN_SPAN_DAYS,
    SUBMISSIONS_LOG,
    compute_oos_rows_for_day,
    renewables_window,
)

logger = logging.getLogger(__name__)

# Real payloads start 2026-09-18 (spec 7.4.1 section 3.6); the ladder itself
# only went live 2026-09-26 (oos_store_parity_check.py's own
# PARITY_CHECK_START_DAY), so days 2026-09-18..-25 are expected to report
# "no real payload file" or an un-ladder-matching candidate name, not a
# genuine comparison -- reported, not hidden.
HALTEPUNKT1_START_DAY: Final[dt.date] = dt.date(2026, 9, 18)

OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "logs" / "haltepunkt1_capacity_vintage_recompute.csv"


def _renewables_predictions_for_single_day(
    df: pd.DataFrame, target_day: dt.date, anchors_path: Path
) -> pd.DataFrame:
    """Exactly one capacity vintage for the WHOLE window this target day's
    own fold needs (spec section 3.6) -- never grouped with another day."""
    window_start, window_end = renewables_window(target_day)
    # Same weather-day-range helper oos_store_job.py itself uses (start_day
    # == end_day == target_day here, so a single day's own full window).
    weather = _weather_for_range(target_day, target_day)
    target_hourly = df.loc[window_start:window_end, list(TARGET_COLUMNS.values())]
    run_inits = pd.DatetimeIndex(weather.index.get_level_values("run_init_utc"))
    weather_window = weather.loc[(run_inits >= window_start) & (run_inits <= window_end)]

    return run_renewables_backtest(
        target_hourly,
        weather_window,
        window="rolling",
        train_span_days=RENEWABLES_TRAIN_SPAN_DAYS,
        refit_every=1,
        objective="l2",
        source=CapacitySource.PUBLIC_REGISTRY,
        method=CapacityExtrapolation.LAST_INCREMENT,
        anchors_path=anchors_path,
        require_baseline=False,
    )


def recompute_for_day(
    df: pd.DataFrame, prices_qh: pd.DataFrame, target_day: dt.date
) -> list[OosStoreRow]:
    vintage_date, anchors_path = vintage_for_target_day(target_day)
    try:
        renewables_predictions = _renewables_predictions_for_single_day(
            df, target_day, anchors_path
        )
    except ValueError as exc:
        logger.warning("%s: renewables recompute failed (%s) -- skipped", target_day, exc)
        return []
    logger.info("%s: recomputed with capacity vintage %s", target_day, vintage_date)
    return compute_oos_rows_for_day(
        df,
        renewables_predictions,
        prices_qh,
        target_day,
        source="backfill",
        only_rows=LADDER_ROW_NAMES,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = read_submission_records(SUBMISSIONS_LOG)
    submitted = real_submitted_rows(records, start_day=HALTEPUNKT1_START_DAY)
    target_days = sorted(submitted)
    logger.info("%d real submitted target day(s) since %s", len(target_days), HALTEPUNKT1_START_DAY)

    df = assemble_price_model_inputs()
    prices_qh = read_quarterhourly_prices()

    all_rows: list[OosStoreRow] = []
    for target_day in target_days:
        all_rows.extend(recompute_for_day(df, prices_qh, target_day))
    recomputed_store = rows_to_frame(all_rows) if all_rows else rows_to_frame([])

    results: list[ParityDayResult] = []
    for target_day in target_days:
        ladder_row = submitted[target_day]
        payload_path = PAYLOADS_DIR / f"{target_day.isoformat()}.json"
        if not payload_path.exists():
            results.append(
                ParityDayResult(target_day, ladder_row, 0, None, None, "no real payload file")
            )
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        results.append(compare_day(recomputed_store, payload, target_day, ladder_row))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(OUTPUT_PATH, index=False)

    for r in results:
        mad = f"{r.mean_abs_diff:.3f}" if r.mean_abs_diff is not None else "n/a"
        print(f"{r.target_day}  {r.ladder_row:<24}  mean_abs_diff={mad}  {r.note}")
    print(f"\nWrote {len(results)} row(s) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
