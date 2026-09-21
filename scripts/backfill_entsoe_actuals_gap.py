"""Deliberate, re-runnable backfill for a real, still-open ENTSO-E gap in
`load_actual`/`gen_wind_onshore`/`gen_solar` that began 2026-09-19 evening
and was still ongoing 2026-09-20 (docs/data_sources_for_live_model_use.md
Sec. 1.3, "erstes reales Schweigen im Live-Betrieb" + its 13:16/13:37 UTC
update). Confirmed selective by series (gen_wind_offshore/gen_hard_coal/
gen_lignite unaffected) and, unlike the settled 2026-09-13 day_ahead_price
gap this mirrors, partially self-healing while still open -- ENTSO-E backfilled
some older hours but not the live tail. gas/biomass carry the same gap but
are "carried", not "checked" (6.7.1a split, nothing reads them) -- out of
scope here.

Energy-Charts' `public_power` endpoint (Fraunhofer ISE) has the same three
series (`Load`, `Wind onshore`, `Solar`) with no gap, confirmed live against
the overlapping pre-gap hours (~0.5-1.2% systematic offset vs. ENTSO-E, same
shape -- not a unit/scaling mismatch). Energy-Charts frames a "day" in local
Europe/Berlin time and only has data up to its own publication lag (~2h
observed) -- a request for a still-open day naturally returns fewer than 96
rows, not null-padded ones. That is expected here, unlike the settled,
single full-day price gap: this script does not require full-day coverage,
and re-running it later (or the next Berlin calendar day, to pick up the
UTC tail that spills across the local-day boundary) fills in more as it
becomes available.

Same fill-only, verify-after-write discipline as
backfill_day_ahead_price_gap.py: never overwrites an existing non-NaN
ENTSO-E cell outright. Unlike that script, exact equality is not the bar
for a conflict (Energy-Charts and ENTSO-E are independent measurements with
a small, real, systematic offset) -- only a disagreement beyond
_MAX_DISAGREEMENT_FRACTION raises. Every pre-existing non-NaN value is
checked byte-identical after the merge regardless.

Deliberately NOT part of scripts/sync_store.py: an owner-invoked correction
for a named, still-open incident, not a general gap-filling mechanism that
should run unattended.
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

logger = logging.getLogger(__name__)

_ENDPOINT: Final = "https://api.energy-charts.info/public_power"
_COUNTRY: Final = "de"
_TIMEOUT_S: Final = 30
_FILE_PREFIX: Final = "DE_LU"

# Generous margin above the ~1.2% systematic offset observed live
# (docs/data_sources_for_live_model_use.md) -- exact equality is the wrong
# bar for two independent real sources; a large disagreement is the thing
# actually worth stopping for.
_MAX_DISAGREEMENT_FRACTION: Final = 0.05

# (Energy-Charts production_type name, ops.store_sources.ENTSOE_SOURCES
# name, our column name). gas/biomass share the same gap but are carried,
# not checked (6.7.1a) -- deliberately excluded.
_TARGETS: Final = (
    ("Load", "load", "load_actual"),
    ("Wind onshore", "generation", "gen_wind_onshore"),
    ("Solar", "generation", "gen_solar"),
)

_CACHE_DIR: Final = {s.name: s.cache_dir for s in ENTSOE_SOURCES}
_BACKFILL_LOG_PATH: Final = PROJECT_ROOT / "logs" / "entsoe_actuals_backfills.csv"


def _cache_filepath(source_name: str, month: pd.Period) -> Path:
    return _CACHE_DIR[source_name] / f"{_FILE_PREFIX}_{month.strftime('%Y-%m')}.parquet"


def _request_url(local_date: dt.date) -> str:
    params = {"country": _COUNTRY, "start": local_date.isoformat(), "end": local_date.isoformat()}
    prepared = requests.Request("GET", _ENDPOINT, params=params).prepare()
    assert prepared.url is not None
    return prepared.url


def fetch_energy_charts_public_power(
    local_date: dt.date, *, country: str = _COUNTRY
) -> pd.DataFrame:
    """Real, already-realised load/generation for local (Europe/Berlin)
    calendar date ``local_date`` -- no cache, this script is itself the
    one-off fill action.

    Returns a frame indexed by UTC timestamp with columns
    ``load_actual``/``gen_wind_onshore``/``gen_solar`` (the three targets
    this gap actually blocks Check B on). May be shorter than a full
    96-row day if ``local_date`` hasn't fully elapsed/published yet -- not
    an error, see module docstring.

    Raises on a deprecated response, a null value inside rows Energy-Charts
    DID return (it truncates rather than null-pads -- a None here would be
    new, suspicious behaviour), or a missing expected series.
    """
    params = {"country": country, "start": local_date.isoformat(), "end": local_date.isoformat()}
    resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("deprecated"):
        raise ValueError(f"Energy-Charts marked the {local_date} public_power response deprecated")

    index = pd.DatetimeIndex(pd.to_datetime(payload["unix_seconds"], unit="s", utc=True))
    series_by_name = {s["name"]: s["data"] for s in payload["production_types"]}

    columns: dict[str, pd.Series] = {}
    for ec_name, _source_name, our_col in _TARGETS:
        if ec_name not in series_by_name:
            raise ValueError(
                f"Energy-Charts response for {local_date} is missing series {ec_name!r}"
            )
        values = series_by_name[ec_name]
        if any(v is None for v in values):
            raise ValueError(
                f"Energy-Charts response for {local_date} has a null {ec_name!r} value inside "
                "the returned range -- refusing to guess"
            )
        columns[our_col] = pd.Series([float(v) for v in values], index=index)

    return pd.DataFrame(columns)


def backfill_column(existing: pd.DataFrame, column: str, fresh: pd.Series) -> tuple[pd.Series, int]:
    """Merge ``fresh`` into ``existing[column]``, filling only cells
    genuinely absent on disk. Returns (merged_column, cells_actually_filled).

    Raises if a timestamp already has a real ENTSO-E value that disagrees
    with Energy-Charts by more than _MAX_DISAGREEMENT_FRACTION (a genuine,
    large conflict between two real sources is a reason to stop, never to
    silently pick a winner), or if the merge would have changed any
    pre-existing non-NaN value (the proof is executed, not claimed).
    """
    overlap = existing.index.intersection(fresh.index)
    real_overlap = existing.loc[overlap, column].dropna()
    if not real_overlap.empty:
        ec_overlap = fresh.loc[real_overlap.index]
        rel_diff = (ec_overlap - real_overlap).abs() / real_overlap.abs().clip(lower=1.0)
        bad = rel_diff[rel_diff > _MAX_DISAGREEMENT_FRACTION]
        if not bad.empty:
            first = bad.index[0]
            raise ValueError(
                f"{column}: {len(bad)} timestamp(s) disagree with the existing ENTSO-E value by "
                f"more than {_MAX_DISAGREEMENT_FRACTION:.0%} -- refusing to overwrite; first: "
                f"{first} (ENTSO-E={real_overlap[first]:.2f}, Energy-Charts={ec_overlap[first]:.2f})"
            )

    merged = existing[column].combine_first(fresh)
    if not merged.loc[real_overlap.index].equals(real_overlap):
        raise RuntimeError(
            f"an existing non-NaN {column!r} value changed during the merge -- "
            "this must never happen, aborting without writing"
        )

    was_missing = existing[column].reindex(fresh.index).isna()
    return merged, int(was_missing.sum())


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    """Temp file in the same directory, then os.replace -- mirrors
    backfill_day_ahead_price_gap.py's identical helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)


def _append_backfill_log(local_date: dt.date, cells_filled: int, *, as_of: pd.Timestamp) -> None:
    row: dict[str, object] = {
        "filled_at_utc": as_of.isoformat(),
        "gap_local_date": local_date.isoformat(),
        "cells_filled": cells_filled,
        "columns": ";".join(col for _, _, col in _TARGETS),
        "source": "Energy-Charts (Fraunhofer ISE) public_power",
        "source_url": _request_url(local_date),
        "license": "CC BY 4.0 (Energy-Charts general license; no per-response license_info "
        "field on this endpoint, unlike /price)",
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
        help="local (Europe/Berlin) calendar date to pull from Energy-Charts, e.g. 2026-09-20 "
        "-- safe to re-run for the same date as more of it gets published",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="fetch and validate, but do not write to disk"
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    local_date: dt.date = args.date

    fresh = fetch_energy_charts_public_power(local_date)
    if fresh.empty:
        logger.info("%s: Energy-Charts returned no rows yet -- nothing to do", local_date)
        return 0
    logger.info(
        "%s: Energy-Charts returned %d quarter-hour(s), %s .. %s",
        local_date,
        len(fresh),
        fresh.index.min(),
        fresh.index.max(),
    )

    total_filled = 0
    source_names = sorted({source_name for _, source_name, _ in _TARGETS})
    fresh_index = pd.DatetimeIndex(fresh.index)
    fresh_periods = fresh_index.to_period("M")
    for source_name in source_names:
        columns = [col for _, s, col in _TARGETS if s == source_name]
        for month in fresh_periods.unique():
            month_index = fresh_index[fresh_periods == month]
            cache_path = _cache_filepath(source_name, month)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"expected an existing month file at {cache_path} -- this script fills "
                    "gaps inside an already-synced month, it does not create one from scratch"
                )
            existing = pd.read_parquet(cache_path)
            if isinstance(existing.index, pd.DatetimeIndex) and existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")

            file_changed = False
            for column in columns:
                merged_col, cells_filled = backfill_column(
                    existing, column, fresh.loc[month_index, column]
                )
                logger.info(
                    "%s / %s (%s): %d cell(s) filled", source_name, column, month, cells_filled
                )
                total_filled += cells_filled
                if cells_filled:
                    existing[column] = merged_col
                    file_changed = True

            if file_changed and not args.dry_run:
                _write_parquet_atomically(existing, cache_path)
                logger.info("wrote %s", cache_path)

    if total_filled == 0:
        logger.info("nothing to do -- every returned quarter-hour already has a real value")
        return 0

    if args.dry_run:
        logger.info("--dry-run: not writing, not logging")
        return 0

    _append_backfill_log(local_date, total_filled, as_of=pd.Timestamp.now("UTC"))
    logger.info("%d cell(s) filled total; logged to %s", total_filled, _BACKFILL_LOG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
