"""Daily weather-run availability probe (spec 6.5.1, §5.4c).

Very small script, deliberately: determine the run the NEXT submission will
actually need, attempt one live fetch, log the result. Uses the real
fetch_run() with the full grid point set -- no scaled-down probe request, so
this never drifts from what the real consumers of this run actually need
(decision 10).

Measures the OPERATIONALLY RELEVANT run -- run_init_for_target_day(
next_delivery_day(as_of)), the same run scripts/sync_store.py::_sync_weather
fetches for its own newest (k=0) target -- not "today's own delivery day"
(docs/sprint6_fix_weather_run_offset.md §3.2). Before this fix, this probed
run_init_for_target_day(today), i.e. YESTERDAY's 00Z run: the run today's
(already-decided) delivery needed, not the run tomorrow's live submission
needs. That mismatch is also why docs/cron_jobs.md §4's measured
"availability offset" numbers describe hours since the PROBE's OWN calendar
day's midnight, not hours since the run actually being probed's own
run_init -- a different, still-useful quantity, but not what it was assumed
to be. This script still only measures; it never feeds the store (that is
_sync_weather's own job, spec §3.2 last line).

A WeatherRunUnavailable (the run isn't published yet) is the EXPECTED
outcome of an early-morning probe, not a script failure: log_availability_
attempt() catches it internally and this exits 0 either way. Any other
exception -- a structural fail-fast check from fetch_run, or a transport
retry exhausted -- propagates and this exits non-zero, which is meant to
redden the workflow job (the distinction the workflow itself must not
blur, spec §5.4c).
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from energy_price_forecast.data.weather_client import (
    log_availability_attempt,
    run_init_for_target_day,
)
from energy_price_forecast.ops.windows import next_delivery_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    as_of = pd.Timestamp.now(tz="UTC")
    run_init_utc = run_init_for_target_day(next_delivery_day(as_of))

    row = log_availability_attempt(run_init_utc)

    if row["available"] == "true":
        logger.info("Weather run available: %s", row)
    else:
        logger.info("Weather run not yet available: %s", row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
