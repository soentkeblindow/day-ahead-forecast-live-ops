"""Weather-run arrival-time probe, thin wiring (spec 8.0a).

Very small script, deliberately: run_probe() already does everything,
this just supplies run_id/run_url and a fresh as_of. Never raises for an
ordinary probe outcome (not yet there, partial coverage, rate limited,
transport/server error) -- run_probe() logs all of those and returns
normally. Only a genuine structural failure (the log file not writable,
an unexpected exception) propagates and reddens the workflow run, same
red/green split as the two existing probe scripts.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from energy_price_forecast.data.weather_run_arrival_probe import run_probe
from energy_price_forecast.ops.store_sources import run_id_and_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    run_id, _ = run_id_and_url()
    rows = run_probe(pd.Timestamp.now(tz="UTC"), run_id=run_id)
    logger.info("probe run complete: %d row(s) written", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
