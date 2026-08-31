"""Bulk historical weather fetch: one 00 UTC ECMWF IFS run per calendar day,
merged into the interim weather artefact (spec 6.5.1, §5.6).

Resumable: existing cache entries are skipped, so re-running after an
interruption only fetches what's still missing. Hard stop on the first
HTTP 429 or five consecutive failures (spec §2.6) -- no automatic retry
past that threshold and no silent rate reduction; the script reports how
far it got and the owner decides how to proceed.

    uv run python scripts/build_weather_artefact.py \\
        --start 2024-03-14 --end 2026-08-30 \\
        --rate-limit-s 1.0 \\
        --out data/interim/weather_ifs_run00.parquet

No test for this script (glue), consistent with project practice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from energy_price_forecast.data._weather_cache import cache_path, read_cached_run
from energy_price_forecast.data.weather_client import WeatherRunUnavailable, fetch_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 5


def _runs_for_range(start: dt.date, end: dt.date) -> list[pd.Timestamp]:
    """One 00 UTC run per calendar day in [start, end], inclusive. The count
    is derived from the calendar, never hardcoded (spec §5.6, step 1)."""
    n_days = (end - start).days + 1
    return [pd.Timestamp(start + dt.timedelta(days=i), tz="UTC") for i in range(n_days)]


def _fetch_all(runs: list[pd.Timestamp], model: str, rate_limit_s: float) -> list[pd.Timestamp]:
    """Fetch every run not already cached. Returns the runs that turned out
    to be missing (WeatherRunUnavailable, not a hard stop). Raises
    RuntimeError on a hard stop (429, or five consecutive failures)."""
    missing: list[pd.Timestamp] = []
    consecutive_failures = 0
    n_skipped = 0

    for i, run in enumerate(runs, start=1):
        path = cache_path(run, model)
        if read_cached_run(path) is not None:
            n_skipped += 1
            logger.info(
                "%d of %d (%d skipped): cache hit for %s", i, len(runs), n_skipped, run.date()
            )
            continue

        try:
            fetch_run(run, model=model)  # writes to cache on success
        except WeatherRunUnavailable as exc:
            missing.append(run)
            consecutive_failures += 1
            logger.warning("%d of %d: run missing for %s (%s)", i, len(runs), run.date(), exc)
            if exc.http_status == 429:
                raise RuntimeError(
                    f"HTTP 429 at run {run.date()} ({i} of {len(runs)} attempted) -- "
                    "hard stop per spec §2.6"
                ) from exc
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{_MAX_CONSECUTIVE_FAILURES} consecutive failures, most recently at run "
                    f"{run.date()} ({i} of {len(runs)} attempted) -- hard stop per spec §2.6"
                ) from exc
        else:
            consecutive_failures = 0
            logger.info("%d of %d (%d skipped): fetched %s", i, len(runs), n_skipped, run.date())

        time.sleep(rate_limit_s)  # only reached after a real HTTP attempt, not a cache hit

    return missing


def _merge_cache(runs: list[pd.Timestamp], model: str, missing: list[pd.Timestamp]) -> pd.DataFrame:
    """Merge every successfully cached run into one frame.

    No separate cross-file schema check is needed here: read_cached_run
    already validates each file against weather_grid.expected_columns() and
    raises on any deviation (spec §2.5b), so every frame reaching this
    concat is already known to share the identical column set.
    """
    missing_set = set(missing)
    frames = []
    for run in runs:
        if run in missing_set:
            continue
        df = read_cached_run(cache_path(run, model))
        assert df is not None, f"expected a cache hit for {run!r} after _fetch_all"
        frames.append(df)
    return pd.concat(frames).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--model", default="ecmwf_ifs")
    parser.add_argument("--rate-limit-s", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("data/interim/weather_ifs_run00.parquet"))
    args = parser.parse_args()

    runs = _runs_for_range(args.start, args.end)
    logger.info(
        "Fetching %d runs (%s to %s), rate limit %.1fs",
        len(runs),
        args.start,
        args.end,
        args.rate_limit_s,
    )

    missing = _fetch_all(runs, args.model, args.rate_limit_s)

    if missing:
        logger.warning("Gap report: %d of %d calendar days have no run:", len(missing), len(runs))
        for run in missing:
            logger.warning("  missing: %s", run.date())
    else:
        logger.info("Gap report: no missing days.")

    merged = _merge_cache(runs, args.model, missing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out)
    logger.info("Wrote %d rows to %s", len(merged), args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
