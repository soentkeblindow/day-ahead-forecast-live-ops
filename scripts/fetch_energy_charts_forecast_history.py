"""Historical bulk fetch of the four Energy-Charts ``public_power_forecast``
series (docs/sprint6_auftrag_energy_charts_backup_2.md, Teil 1).

Pulls ``load``/``solar``/``wind_onshore``/``wind_offshore`` (day-ahead,
Germany) from the ECMWF IFS archive start (2024-03-14) through today,
range-wise (not day-wise, obligation 1) in per-calendar-year chunks --
verified live 2026-09-15 that a single call can return a multi-month range
without truncation, so year chunks comfortably respect the endpoint's rate
limit (default 2 req/min, burst 4) while keeping each response a bounded,
independently-retryable size. Writes one Parquet file per series per UTC
calendar month under ``data/raw/energy_charts/public_power_forecast/``
(same bucketing convention as ``data/_entsoe_cache.py``'s raw ENTSO-E
files) -- a local artifact only, never the store or the maintenance job
(section 5).

The time grid is measured per local calendar day, not assumed (obligation
4): every day whose point count differs from the DST-aware expectation, and
every DST-transition day regardless of whether it matches, is logged
individually and included in the CSV report
(``outputs/results/energy_charts_forecast_grid_report.csv``). A response
that answers but doesn't reach the requested archive start aborts loudly
(backup_2.md section 7, Rueckfrage 1) rather than silently narrowing the
evaluable window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.energy_charts_probe import (
    SERIES,
    EnergyChartsRateLimitedError,
    fetch_series_range,
)
from energy_price_forecast.ops.windows import LOCAL_TZ, expected_timestamp_count, local_day_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ECMWF IFS archive start (backup_2.md section 1: "Warum das jetzt geht") --
# matches data/weather_grid.py's own historical coverage, the reason a
# residual-load-forecast backup is retroactively buildable at all.
_ARCHIVE_START: Final = dt.date(2024, 3, 14)
_OUTPUT_DIR: Final = DATA_RAW / "energy_charts" / "public_power_forecast"
_REPORT_PATH: Final = (
    PROJECT_ROOT / "outputs" / "results" / "energy_charts_forecast_grid_report.csv"
)
_RESOLUTION_MINUTES: Final = 15
# Default rate limit is 2 req/min, burst 4, shared across all four series
# (spec obligation 2) -- 30s between requests mirrors
# scripts/probe_energy_charts_forecast.py's own established pacing.
_INTER_REQUEST_DELAY_S: Final = 30.0
_MAX_RATE_LIMIT_RETRIES: Final = 5


class _RequestCounter:
    """Mutable request tally shared across all series/chunks -- the actual
    number of HTTP requests driven (including 429 retries, each of which is
    a real request), reported per obligation 1."""

    def __init__(self) -> None:
        self.count = 0

    def add(self, n: int) -> None:
        self.count += n


def _year_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """[start, end] split at calendar-year boundaries -- range-wise per
    obligation 1, not day-wise, while keeping each response a bounded,
    independently-retryable size."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    chunks: list[tuple[dt.date, dt.date]] = []
    year = start.year
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(dt.date(year, 12, 31), end)
        chunks.append((chunk_start, chunk_end))
        year += 1
        chunk_start = dt.date(year, 1, 1)
    return chunks


def _fetch_with_retry(
    production_type: str,
    start: dt.date,
    end: dt.date,
    *,
    sleep: Callable[[float], None],
) -> tuple[pd.Series, int]:
    """One chunk fetch, retrying on 429 per the server's own Retry-After
    (obligation 2) up to ``_MAX_RATE_LIMIT_RETRIES`` times. Returns
    (series, n_requests_made_for_this_chunk)."""
    requests_made = 0
    while True:
        requests_made += 1
        try:
            return fetch_series_range(production_type, start, end), requests_made
        except EnergyChartsRateLimitedError as exc:
            if requests_made > _MAX_RATE_LIMIT_RETRIES:
                raise
            logger.warning(
                "%s [%s..%s]: rate limited, waiting %.1fs (attempt %d/%d)",
                production_type,
                start,
                end,
                exc.retry_after_s,
                requests_made,
                _MAX_RATE_LIMIT_RETRIES,
            )
            sleep(exc.retry_after_s)


def fetch_series_history(
    production_type: str,
    *,
    archive_start: dt.date,
    as_of_date: dt.date,
    sleep: Callable[[float], None],
    counter: _RequestCounter,
) -> pd.Series:
    """Full history for one series, concatenated across year chunks.

    Raises if chunks disagree on any shared timestamp (never happens for
    clean calendar-year boundaries, but a silent overlap is worse than a
    loud one), or if the earliest returned point is later than
    ``archive_start`` -- the documented Rueckfrage-1 trigger (section 7):
    the archive not reaching back far enough for this series is not an
    implementer decision.
    """
    chunks = _year_chunks(archive_start, as_of_date)
    parts: list[pd.Series] = []
    for chunk_start, chunk_end in chunks:
        if counter.count > 0:
            sleep(_INTER_REQUEST_DELAY_S)
        part, n_requests = _fetch_with_retry(production_type, chunk_start, chunk_end, sleep=sleep)
        counter.add(n_requests)
        parts.append(part)
        logger.info(
            "%s [%s..%s]: %d point(s) fetched (%d request(s))",
            production_type,
            chunk_start,
            chunk_end,
            len(part),
            n_requests,
        )

    series = pd.concat(parts).sort_index()
    series.name = production_type
    if series.index.has_duplicates:
        dupe_count = int(series.index.duplicated().sum())
        raise ValueError(
            f"{production_type}: {dupe_count} duplicate timestamp(s) across year-chunk "
            "boundaries -- refusing to guess which value to keep"
        )

    expected_start_utc = local_day_bounds(archive_start)[0].tz_convert("UTC")
    if series.empty or series.index.min() > expected_start_utc:
        earliest = series.index.min() if not series.empty else "n/a (empty response)"
        raise ValueError(
            f"{production_type}: earliest returned timestamp {earliest} is after the "
            f"requested archive start {expected_start_utc} -- the Energy-Charts archive does "
            f"not reach back to {archive_start} for this series (backup_2.md section 7, "
            "Rueckfrage 1); stopping rather than silently narrowing the evaluable window"
        )
    return series


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    """Temp file in the same directory, then os.replace -- mirrors
    data/_entsoe_cache.py / scripts/backfill_day_ahead_price_gap.py's
    identical atomic-write pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)


def write_monthly_files(series: pd.Series, *, output_dir: Path) -> list[Path]:
    """One Parquet file per UTC calendar month, matching the naming scheme
    of every other raw source in data/raw/ (obligation: "Namensschema wie
    bei den übrigen Rohquellen")."""
    frame = series.to_frame()
    index = pd.DatetimeIndex(frame.index)
    periods = index.tz_convert("UTC").tz_localize(None).to_period("M")
    written: list[Path] = []
    for period, group in frame.groupby(periods):
        path = output_dir / f"{series.name}_{period}.parquet"
        _write_parquet_atomically(group, path)
        written.append(path)
    return written


def build_grid_report(series: pd.Series, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """One row per local calendar day in [start_date, end_date]: measured
    point count vs. the DST-aware expectation (obligation 4). Days with
    zero returned points still get a row (a full-day gap is the most
    important deviation to see, not the easiest to miss)."""
    local_index = pd.DatetimeIndex(series.index).tz_convert(LOCAL_TZ)
    local_dates = local_index.date
    by_date = pd.DataFrame({"calendar_date": local_dates, "notna": series.notna().to_numpy()})
    counts = by_date.groupby("calendar_date").size()
    non_null_counts = by_date.groupby("calendar_date")["notna"].sum()

    rows: list[dict[str, object]] = []
    d = start_date
    while d <= end_date:
        day_start_utc, day_end_utc = (b.tz_convert("UTC") for b in local_day_bounds(d))
        expected = expected_timestamp_count(day_start_utc, day_end_utc, _RESOLUTION_MINUTES)
        n_points = int(counts.get(d, 0))
        rows.append(
            {
                "production_type": series.name,
                "calendar_date": d.isoformat(),
                "n_points": n_points,
                "expected_points": expected,
                "deviation": n_points - expected,
                "n_non_null": int(non_null_counts.get(d, 0)),
                "is_dst_transition_day": expected != 96,
            }
        )
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def _log_anomalies(report: pd.DataFrame) -> None:
    anomalies = report.loc[(report["deviation"] != 0) | report["is_dst_transition_day"]]
    for _, row in anomalies.iterrows():
        logger.info(
            "grid: %s %s n_points=%d expected=%d deviation=%+d dst_transition=%s",
            row["production_type"],
            row["calendar_date"],
            row["n_points"],
            row["expected_points"],
            row["deviation"],
            row["is_dst_transition_day"],
        )


def build_history(
    *,
    archive_start: dt.date = _ARCHIVE_START,
    as_of_date: dt.date | None = None,
    output_dir: Path = _OUTPUT_DIR,
    report_path: Path = _REPORT_PATH,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    resolved_as_of = as_of_date or pd.Timestamp.now(tz=LOCAL_TZ).date()

    counter = _RequestCounter()
    reports: list[pd.DataFrame] = []
    files_written: list[Path] = []
    for production_type in SERIES:
        series = fetch_series_history(
            production_type,
            archive_start=archive_start,
            as_of_date=resolved_as_of,
            sleep=sleep,
            counter=counter,
        )
        files_written.extend(write_monthly_files(series, output_dir=output_dir))
        reports.append(build_grid_report(series, start_date=archive_start, end_date=resolved_as_of))

    full_report = pd.concat(reports, ignore_index=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report.to_csv(report_path, index=False)
    _log_anomalies(full_report)

    summary: dict[str, object] = {
        "total_requests": counter.count,
        "archive_start": archive_start.isoformat(),
        "as_of_date": resolved_as_of.isoformat(),
        "n_files_written": len(files_written),
        "n_anomaly_or_dst_rows": int(
            ((full_report["deviation"] != 0) | full_report["is_dst_transition_day"]).sum()
        ),
    }
    logger.info("summary: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="local (Europe/Berlin) calendar date to fetch through (default: today)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    build_history(as_of_date=args.as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
