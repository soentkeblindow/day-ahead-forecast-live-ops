"""One-time, deliberate backfill for a specific day_ahead_price gap that
ENTSO-E never published (docs/sprint6_step6_7_2_log.md, "Sitzung 9" -- the
4th finding). Confirmed a genuine one-off incident, not a structural
pattern, via a real owner check of logs/availability.csv (63/63 prior
daily audits had the price; this was the first miss); the automated
maintenance heal (ops/store.py::heal_recent, 10-day lookback) never filled
it, and the raw ENTSO-E series has a real, permanent hole for that one
local day (zero rows, not NaN rows -- the auction result itself was
apparently never published to the Transparency Platform).

Energy-Charts' own /price endpoint (Fraunhofer ISE) has the full 96 real
quarter-hourly values for that day, sourced from Bundesnetzagentur /
SMARD.de per the response's own ``license_info`` field -- confirmed live
2026-09-13 to cover exactly the expected local-day span. This script
fetches exactly the named gap and merges it into the real cache file,
never overwriting an existing value (mirrors ops/store.py::heal_recent's
own rule 3) and verifying every pre-existing non-NaN cell survives
byte-identical (mirrors heal_recent's rule 4: "the proof is executed, not
claimed").

Deliberately NOT part of scripts/sync_store.py's automated path: this is
an owner-invoked, one-off correction for a named, already-investigated
incident using a genuinely different data source, not a general
gap-filling mechanism that should run unattended.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import uuid
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops.store_sources import ENTSOE_SOURCES
from energy_price_forecast.ops.windows import local_day_bounds

logger = logging.getLogger(__name__)

_ENDPOINT: Final = "https://api.energy-charts.info/price"
_BZN: Final = "DE-LU"
_TIMEOUT_S: Final = 30
_EXPECTED_UNIT: Final = "EUR / MWh"

# Matches data/entsoe_client.py::AREA_DE_LU / fetch_day_ahead_prices' own
# file_prefix -- read from the shared registry rather than hardcoded again,
# so a future rename of either can't silently drift the two apart.
_PRICE_SOURCE = next(s for s in ENTSOE_SOURCES if s.name == "day_ahead_price")
_FILE_PREFIX: Final = "DE_LU"

_BACKFILL_LOG_PATH: Final = PROJECT_ROOT / "logs" / "day_ahead_price_backfills.csv"


def _cache_filepath(gap_date: dt.date) -> Path:
    return _PRICE_SOURCE.cache_dir / f"{_FILE_PREFIX}_{gap_date:%Y-%m}.parquet"


def _request_url(gap_date: dt.date, *, bzn: str) -> str:
    params = {"bzn": bzn, "start": gap_date.isoformat(), "end": gap_date.isoformat()}
    prepared = requests.Request("GET", _ENDPOINT, params=params).prepare()
    assert prepared.url is not None
    return prepared.url


def fetch_energy_charts_price(gap_date: dt.date, *, bzn: str = _BZN) -> pd.Series:
    """Real quarter-hourly day-ahead prices for local calendar date
    ``gap_date``, straight from Energy-Charts -- no cache, since this
    script is itself the one-off cache-filling action.

    Raises if the response's unit isn't EUR/MWh, has a null value
    anywhere, or doesn't cover exactly the expected local-day span (via
    ops/windows.py::local_day_bounds, the same boundary convention used
    everywhere else in this project) -- a partial or differently-shaped
    response is a reason to stop and look, never to silently accept.
    """
    params = {"bzn": bzn, "start": gap_date.isoformat(), "end": gap_date.isoformat()}
    resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()

    if payload["unit"] != _EXPECTED_UNIT:
        raise ValueError(f"unexpected unit {payload['unit']!r}, expected {_EXPECTED_UNIT!r}")

    values = payload["price"]
    if any(v is None for v in values):
        raise ValueError(f"Energy-Charts response has a null price value for {gap_date}")

    index = pd.DatetimeIndex(pd.to_datetime(payload["unix_seconds"], unit="s", utc=True))
    series = pd.Series([float(v) for v in values], index=index, name="day_ahead_price")

    day_start, day_end = local_day_bounds(gap_date)
    expected_index = pd.date_range(
        day_start.tz_convert("UTC"), day_end.tz_convert("UTC"), freq="15min", inclusive="left"
    )
    if not series.index.equals(expected_index):
        raise ValueError(
            f"Energy-Charts response for {gap_date} does not cover exactly the expected "
            f"local-day span ({len(expected_index)} quarter-hours) -- got {len(series)} "
            "point(s); refusing to guess which ones to use"
        )
    return series


def backfill_gap(existing: pd.DataFrame, fresh: pd.Series) -> tuple[pd.DataFrame, int]:
    """Merge ``fresh`` into ``existing`` (the real on-disk frame), filling
    only cells genuinely absent on disk. Returns (merged_frame,
    cells_actually_filled).

    Raises if any of ``fresh``'s timestamps already has a real, differing
    value on disk (a genuine conflict between two real sources is a reason
    to stop, never to silently pick a winner), or if the merge would have
    changed any pre-existing non-NaN value (rule 4 -- executed, not
    claimed).
    """
    overlap = existing.index.intersection(fresh.index)
    real_overlap = existing.loc[overlap, "day_ahead_price"].dropna()
    if not real_overlap.empty:
        disagreeing = ~fresh.loc[real_overlap.index].eq(real_overlap)
        if disagreeing.any():
            first = disagreeing[disagreeing].index[0]
            raise ValueError(
                f"{int(disagreeing.sum())} timestamp(s) already have a real ENTSO-E value "
                f"that disagrees with Energy-Charts -- refusing to overwrite; first: {first}"
            )

    merged = existing.combine_first(fresh.to_frame())
    if len(existing.columns):
        merged = merged[existing.columns]

    existing_notna = existing["day_ahead_price"].dropna()
    if not merged.loc[existing_notna.index, "day_ahead_price"].equals(existing_notna):
        raise RuntimeError(
            "an existing non-NaN day_ahead_price value changed during the merge -- "
            "this must never happen, aborting without writing"
        )

    was_missing = existing.reindex(fresh.index)["day_ahead_price"].isna()
    return merged, int(was_missing.sum())


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    """Temp file in the same directory, then os.replace -- atomic on
    Windows too, mirrors data/_entsoe_cache.py::_write_parquet /
    data/_weather_cache.py::write_cached_run's identical pattern.
    Reimplemented here (three lines) rather than importing that module's
    private helper, matching the 6.7.1a precedent for a loop this small.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)


def _append_backfill_log(gap_date: dt.date, cells_filled: int, *, as_of: pd.Timestamp) -> None:
    """Durable provenance record, one row per backfill run -- plain
    pandas to_csv append, the same idiom scripts/sync_store.py::_write_log_row
    already uses for its own operational log, rather than stdlib
    csv.DictWriter (no header-migration concern here: this file's schema
    is fixed, unlike store_sync.csv's own growing per-source columns)."""
    row: dict[str, object] = {
        "filled_at_utc": as_of.isoformat(),
        "gap_local_date": gap_date.isoformat(),
        "cells_filled": cells_filled,
        "bzn": _BZN,
        "source": "Energy-Charts (Fraunhofer ISE) -- Bundesnetzagentur/SMARD.de",
        "source_url": _request_url(gap_date, bzn=_BZN),
        "license": "CC BY 4.0",
    }
    _BACKFILL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    write_header = not _BACKFILL_LOG_PATH.exists()
    frame.to_csv(_BACKFILL_LOG_PATH, mode="a", header=write_header, index=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--date",
        required=True,
        type=dt.date.fromisoformat,
        help="local (Europe/Berlin) calendar date of the known day_ahead_price gap, "
        "e.g. 2026-09-13",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="fetch and validate, but do not write to disk"
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    gap_date: dt.date = args.date

    cache_path = _cache_filepath(gap_date)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"expected an existing month file at {cache_path} -- this script fills a gap "
            "inside an already-synced month, it does not create one from scratch"
        )

    fresh = fetch_energy_charts_price(gap_date)
    existing = pd.read_parquet(cache_path)
    if isinstance(existing.index, pd.DatetimeIndex) and existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")

    merged, cells_filled = backfill_gap(existing, fresh)
    logger.info(
        "%s: %d of %d quarter-hour(s) genuinely filled from Energy-Charts "
        "(bzn=%s, source: Bundesnetzagentur/SMARD.de) -- the rest already had a real, "
        "matching ENTSO-E value",
        gap_date,
        cells_filled,
        len(fresh),
        _BZN,
    )

    if cells_filled == 0:
        logger.info("nothing to do -- every quarter-hour of %s already has a real value", gap_date)
        return 0

    if args.dry_run:
        logger.info("--dry-run: not writing")
        return 0

    _write_parquet_atomically(merged, cache_path)
    _append_backfill_log(gap_date, cells_filled, as_of=pd.Timestamp.now("UTC"))
    logger.info("wrote %s; logged to %s", cache_path, _BACKFILL_LOG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
