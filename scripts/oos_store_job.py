"""Backfill and daily top-up for the OOS-Store (spec 7.1, sections 2.2/2.3).

One job for both (spec section 2.3): ``--mode backfill`` runs once over a
date range, ``--mode daily`` re-attempts a short trailing window every day
so a day that failed its own viability check on a prior attempt (e.g. an
NWP outage) can still be filled in later. Both modes call the SAME
``scripts.run_daily_submission.compute_oos_rows_for_day`` per missing
(target_day, ladder_row) pair -- never a second walk-forward
implementation -- and merge the result into ``logs/oos_store.csv`` via
``energy_price_forecast.ops.oos_store``'s idempotent merge (a key already
stored is never overwritten, spec section 2.3's "ändert nichts").

This job is deliberately NOT part of ``maintain_store.yml``/``submit.yml``
and runs in its own GitHub Actions concurrency group (spec section 2.3:
"Ein Ausfall darf die Einreichung nicht verhindern") -- see
``.github/workflows/oos_store_job.yml``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Final, Literal

import pandas as pd

from energy_price_forecast.arena.live_inputs import (
    assemble_price_model_inputs,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.capacity import CapacityExtrapolation, CapacitySource
from energy_price_forecast.evaluation.renewables_walkforward import (
    TARGET_COLUMNS,
    run_renewables_backtest,
)
from energy_price_forecast.ops import store
from energy_price_forecast.ops.oos_store import (
    load_oos_store,
    merge_rows,
    missing_target_days,
    rows_to_frame,
    write_oos_store,
)
from energy_price_forecast.ops.windows import LOCAL_TZ

# python scripts/oos_store_job.py puts scripts/ itself on sys.path, not the
# repo root, so `scripts` isn't importable as a package from here without
# this -- same fix run_daily_submission.py/freeze_golden_fixture.py already
# need (docs/feedback_dev_workflow.md, 2026-09-24: direct script-path
# invocation of a scripts/ module that imports another scripts/ module
# needs either this, or `python -m scripts.oos_store_job`; both are applied
# here -- this guard for direct invocation, -m in the workflow YAML for
# belt-and-braces).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_daily_submission import (  # noqa: E402
    RENEWABLES_TRAIN_SPAN_DAYS,
    compute_oos_rows_for_day,
    renewables_window,
)

logger = logging.getLogger(__name__)

# Earliest target day the backfill can actually produce a forecast for.
# Native quarter-hourly prices start 2025-09-30 22:00 UTC == 2025-10-01
# 00:00 Europe/Berlin, but models/bridge.py::fit_shape_profile's 28-day
# window (ending target_day - 1) additionally needs >= 3 real
# quarter-hourly days inside that window (min_observations=3 per cell) --
# found empirically on the first real local run: 2025-10-01 raised
# ValueError("... 0 observation(s) ... need >= 3") since its own window
# ends 2025-09-30, one day BEFORE the archive even starts. The first day
# whose 28-day window contains 3 real quarter-hourly days is 2025-10-04
# (window ending 2025-10-03 then contains exactly 2025-10-01/-02/-03).
EARLIEST_BACKFILL_DAY: Final[dt.date] = dt.date(2025, 10, 4)

# How many trailing days the daily top-up re-attempts, not just "yesterday"
# (spec section 2.3) -- a day that failed Check A/B on a prior attempt (a
# real NWP/renewables outage) may become computable once later maintenance
# fills the gap; missing_target_days() keeps re-attempting an already-filled
# day cheap (one dict lookup, no recompute).
DAILY_TOPUP_LOOKBACK_DAYS: Final[int] = 7


@dataclass
class BackfillReport:
    """What one run_backfill()/run_daily_topup() call actually did -- the
    numbers spec section 7 item 1 ("Backfill: Zeitraum, Anzahl Tage je
    Zeile, Laufzeit") and section 9's runtime Rückfrage trigger need."""

    days_attempted: int = 0
    rows_added: int = 0
    fully_skipped_days: list[dt.date] = field(default_factory=list)
    runtime_seconds: float = 0.0


def _weather_for_range(start_day: dt.date, end_day: dt.date) -> pd.DataFrame:
    """Every weather run any target day in [start_day, end_day] could
    need -- read_weather_runs() called once for the whole requested
    range, not once per target day, since it is a pure disk read."""
    window_start, _ = renewables_window(start_day)
    _, window_end = renewables_window(end_day)
    local_days = pd.date_range(
        window_start.tz_convert(LOCAL_TZ).normalize(),
        window_end.tz_convert(LOCAL_TZ).normalize(),
        freq="D",
    )
    return read_weather_runs([d.date() for d in local_days])


def _renewables_predictions_for_range(
    df: pd.DataFrame, weather: pd.DataFrame, start_day: dt.date, end_day: dt.date
) -> pd.DataFrame:
    """One renewables walk-forward covering every target day in
    [start_day, end_day] at once -- the real optimization
    compute_oos_rows_for_day's own docstring describes: a first local
    backfill run measured ~1 minute per target day when the walk-forward
    was instead re-run independently per day (scripts.run_daily_
    submission.run_renewables_step's own single-day window), which
    multiplies into hours over the full backfill range for no benefit,
    since consecutive target days' 90+365-day lookback windows overlap
    almost completely. Calls run_renewables_backtest directly with the
    SAME parameters run_renewables_step itself uses (require_baseline=
    False, rolling/365-day/refit_every=1/l2/PUBLIC_REGISTRY/
    LAST_INCREMENT) -- the identical walk-forward, just windowed over the
    whole requested range instead of one day, never a second
    implementation.
    """
    window_start, _ = renewables_window(start_day)
    _, window_end = renewables_window(end_day)
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
        require_baseline=False,
    )


def _run(*, target_days: list[dt.date], source: Literal["backfill", "daily"]) -> BackfillReport:
    report = BackfillReport()
    t0 = time.monotonic()

    existing = load_oos_store()
    to_fill = missing_target_days(existing, target_days)
    if not to_fill:
        report.runtime_seconds = time.monotonic() - t0
        return report

    df = assemble_price_model_inputs()
    prices_qh = read_quarterhourly_prices()
    weather = _weather_for_range(min(to_fill), max(to_fill))
    renewables_predictions = _renewables_predictions_for_range(
        df, weather, min(to_fill), max(to_fill)
    )

    new_rows = []
    for target_day in sorted(to_fill):
        report.days_attempted += 1
        rows = compute_oos_rows_for_day(
            df,
            renewables_predictions,
            prices_qh,
            target_day,
            source=source,
            only_rows=to_fill[target_day],
        )
        if not rows:
            report.fully_skipped_days.append(target_day)
            continue
        new_rows.extend(rows)

    if new_rows:
        merged, n_added = merge_rows(existing, rows_to_frame(new_rows))
        write_oos_store(merged)
        report.rows_added = n_added

    report.runtime_seconds = time.monotonic() - t0
    return report


def run_backfill(*, start: dt.date = EARLIEST_BACKFILL_DAY, end: dt.date) -> BackfillReport:
    """One-time (re-runnable, idempotent) historical fill, spec section 2.3."""
    target_days = list(pd.date_range(start, end, freq="D").date)
    return _run(target_days=target_days, source="backfill")


def run_daily_topup(
    *, as_of: pd.Timestamp, lookback_days: int = DAILY_TOPUP_LOOKBACK_DAYS
) -> BackfillReport:
    """Daily maintenance pass, spec section 2.3 -- re-attempts the last
    ``lookback_days`` days, not just yesterday (see the module-level
    constant's own docstring for why)."""
    yesterday = (as_of.tz_convert(LOCAL_TZ) - pd.Timedelta(days=1)).date()
    start = yesterday - dt.timedelta(days=lookback_days - 1)
    target_days = list(pd.date_range(start, yesterday, freq="D").date)
    return _run(target_days=target_days, source="daily")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["backfill", "daily"], required=True)
    parser.add_argument(
        "--end",
        type=lambda s: dt.date.fromisoformat(s),
        default=None,
        help="backfill only; defaults to yesterday (Europe/Berlin)",
    )
    args = parser.parse_args()

    store.load_store(PROJECT_ROOT)

    if args.mode == "backfill":
        as_of = pd.Timestamp.now("UTC")
        end = args.end or (as_of.tz_convert(LOCAL_TZ) - pd.Timedelta(days=1)).date()
        report = run_backfill(end=end)
    else:
        report = run_daily_topup(as_of=pd.Timestamp.now("UTC"))

    logger.info(
        "OOS-store %s: %d day(s) attempted, %d row(s) added, %d day(s) fully skipped (%s), %.1fs",
        args.mode,
        report.days_attempted,
        report.rows_added,
        len(report.fully_skipped_days),
        ", ".join(d.isoformat() for d in report.fully_skipped_days[:10]),
        report.runtime_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
