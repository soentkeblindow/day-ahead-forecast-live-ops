"""Sprint 6.8, Schritt 0: extend the Energy-Charts ``load`` day-ahead
forecast artifact (docs/sprint6_step6_8_spec.md section 3) from wherever
scripts/fetch_energy_charts_forecast_history.py last left it through today.

Only the ``load`` series -- 6.8 needs no new solar/wind_onshore/wind_offshore
history (section 9: no EC-Renewables use case in this spec), and re-running
the historical-pull script itself would re-fetch its *entire* archive_start..
today range unconditionally (it has no incremental mode), which would both
violate obligation "kein Neuabruf des Vorhandenen" and needlessly touch the
three series this step doesn't need.

Never overwrites an existing on-disk value (mirrors ops/store.py::heal_recent
and scripts/backfill_day_ahead_price_gap.py's identical discipline): merges
the newly fetched range into whichever monthly file(s) it lands in via
combine_first, existing wins, and verifies every pre-existing non-NaN cell
survives the merge byte-identical.

Re-measures the grid (obligation 4) over the FULL archive_start..today range
for the ``load`` series only, and replaces only the ``load`` rows in the
shared grid report CSV -- the solar/wind_onshore/wind_offshore rows already
in that file (from the Teil 1/2/3 historical pull) are left untouched, since
this script never touches those series.
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
    EnergyChartsRateLimitedError,
    fetch_series_range,
)
from energy_price_forecast.ops.windows import LOCAL_TZ, expected_timestamp_count, local_day_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SERIES: Final = "load"
_INTER_REQUEST_DELAY_S: Final = 30.0
_MAX_RATE_LIMIT_RETRIES: Final = 5

# Same paths scripts/fetch_energy_charts_forecast_history.py uses -- not
# imported from there (scripts/ is not a package; every script here that
# needs another script's small constant/helper reimplements it locally, the
# same precedent scripts/backfill_day_ahead_price_gap.py's atomic-write
# helper already set).
_ARCHIVE_START: Final = dt.date(2024, 3, 14)
_OUTPUT_DIR: Final = DATA_RAW / "energy_charts" / "public_power_forecast"
_REPORT_PATH: Final = (
    PROJECT_ROOT / "outputs" / "results" / "energy_charts_forecast_grid_report.csv"
)
_RESOLUTION_MINUTES: Final = 15


def build_grid_report(series: pd.Series, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """One row per local calendar day in [start_date, end_date] -- identical
    logic to fetch_energy_charts_forecast_history.py::build_grid_report,
    reimplemented rather than cross-script-imported (see module docstring)."""
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


def _existing_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob(f"{_SERIES}_*.parquet"))


def _read_existing(output_dir: Path) -> pd.DataFrame:
    files = _existing_files(output_dir)
    if not files:
        raise FileNotFoundError(
            f"no existing {_SERIES!r} files under {output_dir} -- run "
            "scripts/fetch_energy_charts_forecast_history.py first, this script only extends"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(f"{_SERIES}: {dupe_count} duplicate timestamp(s) across existing files")
    return frame


def _fetch_with_retry(
    start: dt.date, end: dt.date, *, sleep: Callable[[float], None]
) -> tuple[pd.Series, int]:
    requests_made = 0
    while True:
        requests_made += 1
        try:
            return fetch_series_range(_SERIES, start, end), requests_made
        except EnergyChartsRateLimitedError as exc:
            if requests_made > _MAX_RATE_LIMIT_RETRIES:
                raise
            logger.warning(
                "%s [%s..%s]: rate limited, waiting %.1fs (attempt %d/%d)",
                _SERIES,
                start,
                end,
                exc.retry_after_s,
                requests_made,
                _MAX_RATE_LIMIT_RETRIES,
            )
            sleep(exc.retry_after_s)


def _month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """[start, end] split at calendar-month boundaries (spec section 3
    obligation 1: "bereichsweise abrufen (monatsweise)")."""
    if start > end:
        return []
    chunks: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur <= end:
        month_end = pd.Timestamp(cur).to_period("M").end_time.date()
        chunk_end = min(month_end, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
    return chunks


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)


def _merge_fresh(existing_full: pd.DataFrame, fresh: pd.Series) -> pd.DataFrame:
    """Merge ``fresh`` into ``existing_full`` (the whole on-disk series),
    existing wins, verified byte-identical for every pre-existing non-NaN
    cell -- same discipline as backfill_day_ahead_price_gap.py::backfill_gap."""
    fresh_frame = fresh.to_frame()
    overlap = existing_full.index.intersection(fresh_frame.index)
    real_overlap = existing_full.loc[overlap, _SERIES].dropna()
    if not real_overlap.empty:
        disagreeing = ~fresh_frame.loc[real_overlap.index, _SERIES].eq(real_overlap)
        if disagreeing.any():
            first = disagreeing[disagreeing].index[0]
            raise ValueError(
                f"{int(disagreeing.sum())} timestamp(s) already have a real value that "
                f"disagrees with the freshly fetched one -- refusing to overwrite; first: {first}"
            )

    merged = existing_full.combine_first(fresh_frame)
    existing_notna = existing_full[_SERIES].dropna()
    if not merged.loc[existing_notna.index, _SERIES].equals(existing_notna):
        raise RuntimeError(
            "an existing non-NaN value changed during the merge -- this must never happen, "
            "aborting without writing"
        )
    return merged


def _write_monthly(merged_full: pd.DataFrame, *, output_dir: Path) -> list[Path]:
    """Rewrite every month file the merge touched (existing months affected
    by the extension plus any brand-new month) -- always the whole month's
    worth of the merged, byte-verified frame, never a partial-month write."""
    index = pd.DatetimeIndex(merged_full.index)
    periods = index.tz_convert("UTC").tz_localize(None).to_period("M")
    written: list[Path] = []
    for period, group in merged_full.groupby(periods):
        path = output_dir / f"{_SERIES}_{period}.parquet"
        _write_parquet_atomically(group, path)
        written.append(path)
    return written


def _update_report(full_series: pd.Series, *, report_path: Path) -> pd.DataFrame:
    new_rows = build_grid_report(
        full_series, start_date=_ARCHIVE_START, end_date=pd.Timestamp.now(tz=LOCAL_TZ).date()
    )
    if report_path.exists():
        existing_report = pd.read_csv(report_path)
        other_series = existing_report.loc[existing_report["production_type"] != _SERIES]
        combined = pd.concat([other_series, new_rows], ignore_index=True)
    else:
        combined = new_rows
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(report_path, index=False)
    return new_rows


def extend_load_forecast(
    *,
    as_of_date: dt.date | None = None,
    output_dir: Path = _OUTPUT_DIR,
    report_path: Path = _REPORT_PATH,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    resolved_as_of = as_of_date or pd.Timestamp.now(tz=LOCAL_TZ).date()

    existing_full = _read_existing(output_dir)
    last_cached_utc = existing_full.index.max()
    # Re-fetch from the start of the last cached day (not last_cached + 15min):
    # the last on-disk month file may itself be a partial calendar month, and
    # re-requesting its own last local day is cheap, always safe (existing-wins
    # merge, verified), and avoids an off-by-one at a DST/day boundary.
    fetch_start = pd.Timestamp(last_cached_utc).tz_convert(LOCAL_TZ).date()

    if fetch_start > resolved_as_of:
        logger.info("nothing to do -- already extended through %s", last_cached_utc)
        return {"total_requests": 0, "fetch_start": fetch_start.isoformat(), "n_files_written": 0}

    counter = 0
    parts: list[pd.Series] = []
    for chunk_start, chunk_end in _month_chunks(fetch_start, resolved_as_of):
        if counter > 0:
            sleep(_INTER_REQUEST_DELAY_S)
        part, n_requests = _fetch_with_retry(chunk_start, chunk_end, sleep=sleep)
        counter += n_requests
        parts.append(part)
        logger.info(
            "%s [%s..%s]: %d point(s) fetched (%d request(s))",
            _SERIES,
            chunk_start,
            chunk_end,
            len(part),
            n_requests,
        )

    fresh = pd.concat(parts).sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="first")]
    fresh.name = _SERIES

    merged_full = _merge_fresh(existing_full, fresh)
    files_written = _write_monthly(merged_full, output_dir=output_dir)
    new_rows = _update_report(merged_full[_SERIES], report_path=report_path)

    anomalies = new_rows.loc[(new_rows["deviation"] != 0) | new_rows["is_dst_transition_day"]]
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

    summary: dict[str, object] = {
        "total_requests": counter,
        "fetch_start": fetch_start.isoformat(),
        "as_of_date": resolved_as_of.isoformat(),
        "n_files_written": len(files_written),
        "new_max_timestamp": merged_full.index.max().isoformat(),
        "n_anomaly_or_dst_rows": int(len(anomalies)),
    }
    logger.info("summary: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="local (Europe/Berlin) calendar date to extend through (default: today)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    extend_load_forecast(as_of_date=args.as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
