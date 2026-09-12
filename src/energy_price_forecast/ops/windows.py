"""Local (Europe/Berlin) calendar-day boundaries for the availability audit.

This module exists solely because market_time.py -- byte-identical copied,
leakage-critical code -- is not touched in 6.2 (spec §3, Entscheidung 5):
its only day-boundary helper, _local_day, floors *value* timestamps to their
local calendar day for the knowledge-time contract, which is a different
question from "give me the local midnight bounds of calendar date D, D-1,
D-90". The two are kept consistent by test, not by sharing code -- see
tests/test_ops_windows.py's consistency test against market_time._local_day.

Nothing here hardcodes a DST rule: every boundary is produced by localizing
a bare calendar date into LOCAL_TZ, so the UTC offset is resolved fresh for
that specific date by the tz database, not carried over via arithmetic on an
already-tz-aware timestamp (see arena/payload.py's expected_value_count for
why that distinction matters -- DateOffset on a *fixed*-offset timestamp
does not re-resolve DST, only on a named-zone one does).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

LOCAL_TZ = "Europe/Berlin"


def local_day_bounds(date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Local midnight of date and of the following day (end exclusive)."""
    start = pd.Timestamp(date, tz=LOCAL_TZ)
    end = pd.Timestamp(date + dt.timedelta(days=1), tz=LOCAL_TZ)
    return start, end


def next_delivery_day(as_of: pd.Timestamp) -> dt.date:
    """Which local delivery day the next submission, made at ``as_of``, is
    for -- one calendar day after ``as_of``'s own local date (fix for
    sync_store.py's weather-run offset bug, docs/sprint6_fix_weather_run_offset.md).

    The single shared answer to "which delivery day does the next
    submission build for": scripts/run_daily_submission.py (the consumer,
    needing run_init_for_target_day(next_delivery_day(as_of))'s OWN run)
    and scripts/sync_store.py::_sync_weather (the producer, which must
    fetch that same run ahead of time) both call this instead of each
    re-deriving the answer independently -- which is exactly how the two
    drifted a full day apart before this fix.

    Date-only arithmetic (``dt.timedelta`` on a bare ``date``, never
    ``pd.Timedelta``/``pd.DateOffset`` on a tz-aware ``Timestamp``) --
    the same "Kalender ja, Uhr nein" discipline as run_init_for_target_day
    itself, spec 6.5.1 §11. This bug class has already struck four times
    in Sprint 6.
    """
    return as_of.tz_convert(LOCAL_TZ).date() + dt.timedelta(days=1)


def local_window_bounds(
    date: dt.date, *, days_back_start: int, days_back_end: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Local midnight bounds of a multi-day window anchored on date.

    Window is [date - days_back_start, date - days_back_end) in calendar
    days, both ends localized independently -- not via Timedelta(days=N) or
    DateOffset chaining from a single point, so a window spanning a DST
    transition still lands on the correct local midnight at each end
    (Timestamp - Timedelta(hours=24*N) would be off by an hour whenever the
    window crosses a transition).

    days_back_end may be negative to reach a day *after* date (the
    DA_FORECAST critical window extension, spec §5.2, uses days_back_end=-1
    to reach D+1).
    """
    start_date = date - dt.timedelta(days=days_back_start)
    end_date = date - dt.timedelta(days=days_back_end)
    start = pd.Timestamp(start_date, tz=LOCAL_TZ)
    end = pd.Timestamp(end_date, tz=LOCAL_TZ)
    return start, end


def expected_timestamp_count(
    start: pd.Timestamp, end: pd.Timestamp, resolution_minutes: int
) -> int:
    """Number of resolution_minutes-spaced timestamps in [start, end).

    Safe to call with any tz-aware start/end (e.g. from local_day_bounds or
    local_window_bounds): Timestamp subtraction always yields the actual
    elapsed real time regardless of how each endpoint's tzinfo was
    constructed, so a DST-transition window's 23h/25h falls out automatically.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be tz-aware")
    elapsed = end - start
    return int(elapsed / pd.Timedelta(minutes=resolution_minutes))
