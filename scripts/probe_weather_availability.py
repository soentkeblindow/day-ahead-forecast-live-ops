"""Daily weather-run availability probe (spec 6.5.1, §5.4c).

Very small script, deliberately: determine today's local target day, derive
the run that a forecast for it is allowed to use, attempt one live fetch,
log the result. Uses the real fetch_run() with the full grid point set --
no scaled-down probe request, so this never drifts from what the real
consumers of this run actually need (decision 10).

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
from energy_price_forecast.ops.windows import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    target_day = pd.Timestamp.now(tz="UTC").tz_convert(LOCAL_TZ).date()
    run_init_utc = run_init_for_target_day(target_day)

    row = log_availability_attempt(run_init_utc)

    if row["available"] == "true":
        logger.info("Weather run available: %s", row)
    else:
        logger.info("Weather run not yet available: %s", row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
