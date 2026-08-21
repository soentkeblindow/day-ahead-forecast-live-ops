"""Native quarter-hourly day-ahead price artefact.

data/normalize.py::to_hourly resamples day-ahead prices to hourly and only
that resample is persisted in hourly.parquet -- the native quarter-hourly
values are never kept. This module builds the separate, unresampled series
that the Arena baseline replica and the shape profile need (spec 6.4,
section 5.1 a2). No resampling is applied here; that is the point of the
artefact.
"""

import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.entsoe_client import AREA_DE_LU, fetch_day_ahead_prices
from energy_price_forecast.ops.windows import expected_timestamp_count, local_day_bounds

logger = logging.getLogger(__name__)

_PATH = Path("data/interim/quarterhourly_prices.parquet")
_TZ = "Europe/Berlin"
_RESOLUTION_MIN = 15

# Start of native quarter-hourly resolution for DE_LU day-ahead prices.
# Mirrors data/loaders.py::_BREAK_TS -- kept consistent by test, not by
# sharing code, following this project's existing convention for
# copied-vs-new module boundaries (see ops/windows.py's module docstring).
QUARTERHOUR_START = pd.Timestamp("2025-09-30 22:00", tz="UTC")


def build_quarterhourly_prices(
    end: pd.Timestamp,
    path: Path = _PATH,
    start: pd.Timestamp = QUARTERHOUR_START,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    """Fetch/reconstruct and persist the native quarter-hourly price series.

    Delegates to fetch_day_ahead_prices()/cached_fetch() unchanged: months
    already cached (e.g. by the 6.2 audit workflow's training-window pulls)
    are reconstructed from the cache without a new API call; the rest is
    fetched fresh -- explicitly allowed by spec 6.4 section 2.4 (regime 2),
    since this artefact does not exist yet in any form.

    ``end`` may be later than "now": day-ahead prices are auction-clearing
    results published a day ahead of delivery, so at any moment prices for
    (local) tomorrow are typically already known. Passing an end date that
    only reaches "now" would silently truncate already-published prices.
    """
    df = fetch_day_ahead_prices(start, end, area)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(
        "wrote %d quarter-hourly rows to %s (%s -> %s)",
        len(df),
        path,
        df.index.min() if not df.empty else None,
        df.index.max() if not df.empty else None,
    )
    return df


def validate_delivery_day_slot_counts(index: pd.DatetimeIndex, tz: str = _TZ) -> None:
    """Raise if any local delivery day lacks its calendar-expected slot count.

    Expected count (92/96/100) is derived from the calendar via
    ops.windows.expected_timestamp_count -- never hard-coded to 96 -- so
    both DST directions are handled correctly. Gaps are never interpolated
    or silently skipped (spec 6.4 section 2.6): the first offending day
    raises, naming the day and the count actually found.
    """
    local_days = pd.DatetimeIndex(index).tz_convert(tz).normalize().unique().sort_values()
    for local_midnight in local_days:
        date = local_midnight.date()
        day_start, day_end = local_day_bounds(date)
        expected = expected_timestamp_count(day_start, day_end, _RESOLUTION_MIN)
        actual = int(((index >= day_start) & (index < day_end)).sum())
        if actual != expected:
            raise ValueError(
                f"Delivery day {date} has {actual} quarter-hourly prices, expected {expected}."
            )
