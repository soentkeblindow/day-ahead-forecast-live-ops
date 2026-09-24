"""Daily Energy-Charts knowledge-time probe
(docs/sprint6_auftrag_energy_charts_backup.md). Small script, deliberately:
determine the delivery day the NEXT submission actually needs, probe all
four series (load, solar, wind_onshore, wind_offshore), append one row each
to the log. Collects only -- no evaluation, no store/feature/submission-path
involvement (spec section 6).

Target day comes from ``next_delivery_day``, the same shared function
scripts/probe_weather_availability.py and scripts/run_daily_submission.py
already use -- never derived locally from the clock (spec section 2: this
project's weather-run-offset bug class has already struck four times).

A raised ``EnergyChartsProbeAnomalyError`` (deprecated=true, or a mismatched
echoed production_type/forecast_type) propagates uncaught and exits this
non-zero, reddening the workflow run -- every other outcome (not yet
published, rate limited, transport/server error, partial coverage) is
logged and this still exits 0 (spec section 5).
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.energy_charts import SERIES, append_probe_row, probe_series
from energy_price_forecast.ops.store_sources import run_id_and_url
from energy_price_forecast.ops.windows import next_delivery_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_LOG_PATH = Path("logs/energy_charts_forecast_probe.csv")
# Default rate limit is 2 requests/min per endpoint and client IP (spec
# section 4). Four series need pacing within one run; 30s keeps every call
# well inside budget without relying on the burst allowance holding under
# load (the docs' own "adjusted dynamically to current server load" note).
_INTER_REQUEST_DELAY_S = 30.0


def run_probe(
    as_of: pd.Timestamp,
    *,
    log_path: Path = _LOG_PATH,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Probe all four series for the delivery day ``as_of``'s next
    submission needs, appending one row per series as it completes (so a
    later series raising still leaves the earlier rows on disk). Returns
    the rows written, for tests to assert on without re-reading the CSV."""
    run_id, _ = run_id_and_url()
    target_day = next_delivery_day(as_of)

    rows: list[dict[str, object]] = []
    for i, production_type in enumerate(SERIES):
        if i > 0:
            sleep(_INTER_REQUEST_DELAY_S)
        probe_time = pd.Timestamp.now(tz="UTC")
        row = probe_series(production_type, target_day, run_id=run_id, as_of=probe_time)
        append_probe_row(row, log_path)
        rows.append(row)
        logger.info("probed %s for %s: %s", production_type, target_day, row)
    return rows


def main() -> int:
    run_probe(pd.Timestamp.now(tz="UTC"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
