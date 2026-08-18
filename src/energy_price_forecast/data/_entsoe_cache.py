import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    next_month = month % 12 + 1
    next_year = year + (1 if month == 12 else 0)
    last = pd.Timestamp(year=next_year, month=next_month, day=1, tz="UTC") - pd.Timedelta(hours=1)
    return first, last


def _is_complete_month(year: int, month: int) -> bool:
    # One-day buffer: ENTSO-E publishes day-ahead prices the day before, so a
    # cache written close to month-end could be missing the final hours.
    _, last = _month_bounds(year, month)
    return last < pd.Timestamp.now("UTC") - pd.Timedelta(days=1)


def _is_sufficiently_complete(df: pd.DataFrame, year: int, month: int) -> bool:
    first, last = _month_bounds(year, month)
    expected_hours = int((last - first).total_seconds() / 3600) + 1
    return len(df) >= 0.9 * expected_hours


def _cache_filepath(cache_dir: Path, file_prefix: str, year: int, month: int) -> Path:
    return cache_dir / f"{file_prefix}_{year:04d}-{month:02d}.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")


def cached_fetch(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    file_prefix: str,
    fetch_fn: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    start = _to_utc(start)
    end = _to_utc(end)

    chunks: list[pd.DataFrame] = []
    for year, month in _month_range(start, end):
        path = _cache_filepath(cache_dir, file_prefix, year, month)

        cached_df: pd.DataFrame | None = None
        if path.exists() and _is_complete_month(year, month):
            candidate = _read_parquet(path)
            if _is_sufficiently_complete(candidate, year, month):
                logger.info("Cache hit: %s", path.name)
                cached_df = candidate
            else:
                logger.warning(
                    "Cache incomplete (%d rows, <90%% of expected): %s — re-fetching",
                    len(candidate),
                    path.name,
                )

        if cached_df is not None:
            chunks.append(cached_df)
        else:
            logger.info("Cache miss: %s — fetching from API", path.name)
            month_start, month_end = _month_bounds(year, month)
            df = fetch_fn(month_start, month_end)
            if not df.empty:
                _write_parquet(df, path)
            chunks.append(df)

    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        # Preserve column schema from the first chunk that has columns defined so
        # callers receive a well-formed empty DataFrame even when every API call
        # returned no data.
        template = next((c for c in chunks if len(c.columns) > 0), None)
        return pd.DataFrame(columns=template.columns) if template is not None else pd.DataFrame()
    return pd.concat(non_empty).sort_index().loc[start:end]
