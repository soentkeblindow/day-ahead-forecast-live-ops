"""Sprint 6.8, Schritt 0: historical bulk fetch of Energy-Charts day-ahead
prices (``/price``, bzn=DE-LU) over the same range as the store's own
``day_ahead_price`` history (docs/sprint6_step6_8_spec.md section 3).

New artifact -- unlike the ``public_power_forecast`` series, no prior EC
price history has been bulk-fetched (only single named-gap-day pulls via
scripts/backfill_day_ahead_price_gap.py). Range determined dynamically from
the store's own cached min/max (never hardcoded), so a re-run naturally
picks up whatever the store has grown to since the last run.

Obligations from docs/data_sources_for_live_model_use.md section 3.2, as
named again in the 6.8 spec:
1. Range-wise, **monthly** (the 6.8 spec's own explicit tightening of the
   general "range-wise, not day-wise" rule -- verified live that a
   multi-year single request also works, but the spec asks for monthly
   chunks specifically, so that is what this does; not an optimization
   opportunity to second-guess).
2. HTTP 429 -> honour ``Retry-After``, never a fixed wait.
3. ``deprecated`` checked on every response -- ``true`` aborts loudly.
4. "Echo" check: this endpoint has no production_type/forecast_type-style
   echo field to compare against the request (confirmed live -- the payload
   is only ``{license_info, unix_seconds, price, unit, deprecated}``). The
   closest equivalent is validating ``unit == "EUR / MWh"`` on every
   response, which this does; there is nothing else to echo-check here.
5. Grid measured, not assumed, per calendar day -- resolution-aware: hourly
   before ``data/quarterhourly.py::QUARTERHOUR_START`` (2025-09-30 22:00
   UTC), quarter-hourly from then on (confirmed live for an old date: a
   2021 request returned 24 hourly points/day, not 96 quarter-hourly with
   repeats).

Local artifact only, under ``data/raw/energy_charts/price/`` -- never the
store or a maintenance job (section 3, "Ablage").
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
from typing import Any, Final

import pandas as pd
import requests

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.quarterhourly import QUARTERHOUR_START
from energy_price_forecast.ops.store import read_cached_range
from energy_price_forecast.ops.store_sources import ENTSOE_SOURCES
from energy_price_forecast.ops.windows import LOCAL_TZ, expected_timestamp_count, local_day_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ENDPOINT: Final = "https://api.energy-charts.info/price"
_BZN: Final = "DE-LU"
_TIMEOUT_S: Final = 30.0
_EXPECTED_UNIT: Final = "EUR / MWh"
_COLUMN: Final = "day_ahead_price_ec"

_OUTPUT_DIR: Final = DATA_RAW / "energy_charts" / "price"
_REPORT_PATH: Final = PROJECT_ROOT / "outputs" / "results" / "energy_charts_price_grid_report.csv"
_FILE_PREFIX: Final = "DE_LU"

_INTER_REQUEST_DELAY_S: Final = 30.0
_MAX_RATE_LIMIT_RETRIES: Final = 5

_PRICE_SOURCE = next(s for s in ENTSOE_SOURCES if s.name == "day_ahead_price")


class EnergyChartsRateLimitedError(RuntimeError):
    def __init__(self, retry_after_s: float) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(f"rate limited by {_ENDPOINT}, Retry-After={retry_after_s}s")


class EnergyChartsPriceAnomalyError(RuntimeError):
    """A genuine API-behavior anomaly: ``deprecated=true`` or an unexpected
    unit -- the same anomaly class energy_charts_probe.py raises for the
    forecast endpoint, applied to this endpoint's own, smaller echo surface
    (module docstring, obligation 4)."""


def _parse_retry_after(response: requests.Response) -> float:
    header = response.headers.get("Retry-After")
    if header is None:
        return 30.0
    try:
        return float(header)
    except ValueError:
        return 30.0


def fetch_price_range(start: dt.date, end: dt.date, *, bzn: str = _BZN) -> pd.Series:
    """Raw day-ahead prices over local calendar-date range [start, end]
    (both ends inclusive), UTC-indexed, ``None`` kept as NaN. Raises
    ``EnergyChartsRateLimitedError`` on 429, ``requests.HTTPError`` on any
    other non-200, ``EnergyChartsPriceAnomalyError`` on ``deprecated=true``
    or an unexpected unit."""
    params = {"bzn": bzn, "start": start.isoformat(), "end": end.isoformat()}
    response = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_S)

    if response.status_code == 429:
        raise EnergyChartsRateLimitedError(_parse_retry_after(response))
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    if payload.get("deprecated"):
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: endpoint reports deprecated=true"
        )
    unit = payload.get("unit")
    if unit != _EXPECTED_UNIT:
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: unexpected unit {unit!r}, expected {_EXPECTED_UNIT!r}"
        )

    unix_seconds = payload.get("unix_seconds", [])
    prices = payload.get("price", [])
    if len(unix_seconds) != len(prices):
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: unix_seconds length {len(unix_seconds)} != "
            f"price length {len(prices)}"
        )

    index = pd.DatetimeIndex(pd.to_datetime(unix_seconds, unit="s", utc=True), name="timestamp")
    values = [float(v) if v is not None else float("nan") for v in prices]
    return pd.Series(values, index=index, name=_COLUMN, dtype="float64")


def _month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
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


def _fetch_with_retry(
    start: dt.date, end: dt.date, *, sleep: Callable[[float], None]
) -> tuple[pd.Series, int]:
    requests_made = 0
    while True:
        requests_made += 1
        try:
            return fetch_price_range(start, end), requests_made
        except EnergyChartsRateLimitedError as exc:
            if requests_made > _MAX_RATE_LIMIT_RETRIES:
                raise
            logger.warning(
                "price [%s..%s]: rate limited, waiting %.1fs (attempt %d/%d)",
                start,
                end,
                exc.retry_after_s,
                requests_made,
                _MAX_RATE_LIMIT_RETRIES,
            )
            sleep(exc.retry_after_s)


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)


def write_monthly_files(series: pd.Series, *, output_dir: Path) -> list[Path]:
    frame = series.to_frame()
    index = pd.DatetimeIndex(frame.index)
    periods = index.tz_convert("UTC").tz_localize(None).to_period("M")
    written: list[Path] = []
    for period, group in frame.groupby(periods):
        path = output_dir / f"{_FILE_PREFIX}_{period}.parquet"
        _write_parquet_atomically(group, path)
        written.append(path)
    return written


def _expected_points_for_day(d: dt.date) -> tuple[int, int]:
    """(expected_points, resolution_minutes) for local calendar day ``d`` --
    resolution-aware around the real quarter-hourly transition (module
    docstring, obligation 5)."""
    day_start, day_end = local_day_bounds(d)
    day_start_utc, day_end_utc = day_start.tz_convert("UTC"), day_end.tz_convert("UTC")
    resolution_minutes = 15 if day_start_utc >= QUARTERHOUR_START else 60
    expected = expected_timestamp_count(day_start_utc, day_end_utc, resolution_minutes)
    return expected, resolution_minutes


def build_grid_report(series: pd.Series, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """One row per local calendar day -- resolution-aware expectation
    (hourly pre-transition, quarter-hourly post), DST-transition days and
    the resolution-transition day itself always flagged individually."""
    local_index = pd.DatetimeIndex(series.index).tz_convert(LOCAL_TZ)
    by_date = pd.DataFrame({"calendar_date": local_index.date, "notna": series.notna().to_numpy()})
    counts = by_date.groupby("calendar_date").size()
    non_null_counts = by_date.groupby("calendar_date")["notna"].sum()

    rows: list[dict[str, object]] = []
    d = start_date
    while d <= end_date:
        expected, resolution_minutes = _expected_points_for_day(d)
        full_day_expected = 96 if resolution_minutes == 15 else 24
        n_points = int(counts.get(d, 0))
        rows.append(
            {
                "calendar_date": d.isoformat(),
                "resolution_minutes": resolution_minutes,
                "n_points": n_points,
                "expected_points": expected,
                "deviation": n_points - expected,
                "n_non_null": int(non_null_counts.get(d, 0)),
                "is_dst_transition_day": expected not in (full_day_expected,),
            }
        )
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def _log_anomalies(report: pd.DataFrame) -> None:
    anomalies = report.loc[(report["deviation"] != 0) | report["is_dst_transition_day"]]
    for _, row in anomalies.iterrows():
        logger.info(
            "grid: %s res=%dmin n_points=%d expected=%d deviation=%+d dst_transition=%s",
            row["calendar_date"],
            row["resolution_minutes"],
            row["n_points"],
            row["expected_points"],
            row["deviation"],
            row["is_dst_transition_day"],
        )


def _existing_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob(f"{_FILE_PREFIX}_*.parquet"))


def _read_existing(output_dir: Path) -> pd.DataFrame | None:
    files = _existing_files(output_dir)
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(f"price: {dupe_count} duplicate timestamp(s) across existing files")
    return frame


def _merge_fresh(existing_full: pd.DataFrame | None, fresh: pd.Series) -> pd.DataFrame:
    fresh_frame = fresh.to_frame()
    if existing_full is None:
        return fresh_frame

    overlap = existing_full.index.intersection(fresh_frame.index)
    real_overlap = existing_full.loc[overlap, _COLUMN].dropna()
    if not real_overlap.empty:
        disagreeing = ~fresh_frame.loc[real_overlap.index, _COLUMN].eq(real_overlap)
        if disagreeing.any():
            first = disagreeing[disagreeing].index[0]
            raise ValueError(
                f"{int(disagreeing.sum())} timestamp(s) already have a real value that "
                f"disagrees with the freshly fetched one -- refusing to overwrite; first: {first}"
            )

    merged = existing_full.combine_first(fresh_frame)
    existing_notna = existing_full[_COLUMN].dropna()
    if not merged.loc[existing_notna.index, _COLUMN].equals(existing_notna):
        raise RuntimeError(
            "an existing non-NaN value changed during the merge -- this must never happen, "
            "aborting without writing"
        )
    return merged


def fetch_price_history(
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    output_dir: Path = _OUTPUT_DIR,
    report_path: Path = _REPORT_PATH,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Full range determined from the store's own cached day_ahead_price
    min/max unless explicitly overridden (section 3: "über denselben
    Zeitraum wie die Preishistorie im Store")."""
    if start_date is None or end_date is None:
        store_series = read_cached_range(_PRICE_SOURCE.cache_dir)["day_ahead_price"]
        resolved_start = (
            start_date or pd.Timestamp(store_series.index.min()).tz_convert(LOCAL_TZ).date()
        )
        resolved_end = (
            end_date or pd.Timestamp(store_series.index.max()).tz_convert(LOCAL_TZ).date()
        )
    else:
        resolved_start, resolved_end = start_date, end_date

    existing_full = _read_existing(output_dir)

    # Skip a chunk only if it is FULLY covered already -- checked per chunk
    # against the actual expected point count (resolution-aware), not a
    # single "resume from the max timestamp" cursor. A single cursor would
    # be wrong here: unlike the load-forecast extension (which only ever
    # grows forward), this artifact is brand new and may already have a
    # partial, non-contiguous set of months on disk (e.g. from an earlier
    # small-range test run) with real gaps both before and after them.
    all_chunks = _month_chunks(resolved_start, resolved_end)
    chunks_to_fetch: list[tuple[dt.date, dt.date]] = []
    for chunk_start, chunk_end in all_chunks:
        if existing_full is None:
            chunks_to_fetch.append((chunk_start, chunk_end))
            continue
        expected = sum(
            _expected_points_for_day(d)[0]
            for d in pd.date_range(chunk_start, chunk_end, freq="D").date
        )
        day_start_utc = local_day_bounds(chunk_start)[0].tz_convert("UTC")
        day_end_utc = local_day_bounds(chunk_end)[1].tz_convert("UTC")
        actual = int(existing_full.loc[day_start_utc:day_end_utc, _COLUMN].notna().sum())
        if actual < expected:
            chunks_to_fetch.append((chunk_start, chunk_end))

    counter = 0
    parts: list[pd.Series] = []
    for chunk_start, chunk_end in chunks_to_fetch:
        if counter > 0:
            sleep(_INTER_REQUEST_DELAY_S)
        part, n_requests = _fetch_with_retry(chunk_start, chunk_end, sleep=sleep)
        counter += n_requests
        parts.append(part)
        logger.info(
            "price [%s..%s]: %d point(s) fetched (%d request(s), %d/%d chunks)",
            chunk_start,
            chunk_end,
            len(part),
            n_requests,
            len(parts),
            len(chunks_to_fetch),
        )
    logger.info(
        "%d of %d month chunk(s) already fully covered on disk, skipped",
        len(all_chunks) - len(chunks_to_fetch),
        len(all_chunks),
    )

    if not parts:
        logger.info("nothing to do -- already covers %s..%s", resolved_start, resolved_end)
        full_series = (
            existing_full[_COLUMN] if existing_full is not None else pd.Series(dtype="float64")
        )
    else:
        fresh = pd.concat(parts).sort_index()
        fresh = fresh[~fresh.index.duplicated(keep="first")]
        fresh.name = _COLUMN
        merged_full = _merge_fresh(existing_full, fresh)
        write_monthly_files(merged_full[_COLUMN], output_dir=output_dir)
        full_series = merged_full[_COLUMN]

    report = build_grid_report(full_series, start_date=resolved_start, end_date=resolved_end)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    _log_anomalies(report)

    summary: dict[str, object] = {
        "total_requests": counter,
        "start_date": resolved_start.isoformat(),
        "end_date": resolved_end.isoformat(),
        "n_chunks_fetched": len(chunks_to_fetch),
        "n_chunks_total": len(all_chunks),
        "n_anomaly_or_dst_rows": int(
            ((report["deviation"] != 0) | report["is_dst_transition_day"]).sum()
        ),
    }
    logger.info("summary: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=dt.date.fromisoformat, default=None)
    p.add_argument("--end", type=dt.date.fromisoformat, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    fetch_price_history(start_date=args.start, end_date=args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
