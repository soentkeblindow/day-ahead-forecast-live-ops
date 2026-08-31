"""Per-run cache for Open-Meteo Single Runs API responses.

One file per run (spec 6.5.1, §2.5): no interval-boundary arithmetic
anywhere in this module, unlike data/_entsoe_cache.py's month chunks -- that
kind of arithmetic (_entsoe_cache.py::_month_bounds, "first of next month
minus one hour") is exactly what caused the DST/month-boundary bug fixed in
sprint 6.4 (commit cb1f63d). A cache that only ever knows individual runs,
never date ranges, cannot repeat that bug by construction. This module
reuses _entsoe_cache.py's general read/write shape, not its interval logic.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.weather_grid import CACHE_KEY, expected_columns

CACHE_ROOT: Path = Path("data/cache/weather_single_runs")


def cache_path(run_init_utc: pd.Timestamp, model: str, root: Path = CACHE_ROOT) -> Path:
    """Path for one cached run.

    ``{root}/{model}/{CACHE_KEY}/{YYYY}/{MM}/{YYYY-MM-DD}THHZ.parquet``

    No colon anywhere in the path (spec §2.5a) -- Windows forbids it in
    filenames, and this is developed on Windows. ``CACHE_KEY`` rather than
    just ``GRID_VERSION`` is part of the path (spec §2.5b): it already covers
    both the point set and the variable tuple, so a schema change in either
    invalidates the cache location, not just silently mixes into it.
    """
    run_init_utc = run_init_utc.tz_convert("UTC")
    filename = f"{run_init_utc.strftime('%Y-%m-%d')}T{run_init_utc.strftime('%H')}Z.parquet"
    return (
        root
        / model
        / CACHE_KEY
        / f"{run_init_utc.year:04d}"
        / f"{run_init_utc.month:02d}"
        / filename
    )


def check_columns(df: pd.DataFrame) -> None:
    """Raise if df's columns don't exactly match weather_grid.expected_columns().

    A schema mismatch on a cached file is a finding, not an operating state
    (spec §5.5) -- callers must not treat this as a cache miss and silently
    re-fetch, which is why this is a separate, explicit check rather than
    folded into a try/except around a re-fetch.
    """
    expected = set(expected_columns())
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            "cached weather file has an unexpected column set -- "
            f"missing: {sorted(missing)}, extra: {sorted(extra)}"
        )


def read_cached_run(path: Path) -> pd.DataFrame | None:
    """Read a cached run if the file exists; None if it doesn't (a real
    cache miss, safe to fetch and write). Raises via check_columns if the
    file exists but its column set doesn't match expected_columns()."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    check_columns(df)
    return df


def write_cached_run(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomically.

    Writes to a temp file in the same directory, then ``os.replace`` (atomic
    on Windows too, unlike a plain rename across filesystems). With ~880
    consecutive writes in the bulk fetch, a mid-write crash is realistic; a
    half-written file at the real path would look like a completed run on
    the next run (spec §5.5).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, path)
