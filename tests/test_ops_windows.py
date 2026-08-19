"""Unit tests for ops/windows.py -- local calendar-day boundaries for the audit.

The consistency test at the bottom is this file's actual purpose (spec §5.8,
§7): it is the only thing keeping ops/windows.py and market_time.py's private
_local_day from silently drifting apart, since 6.2 deliberately does not
import market_time.py from production code (Entscheidung 5).
"""

import datetime as dt

import pandas as pd
import pytest

from energy_price_forecast.market_time import _local_day
from energy_price_forecast.ops.windows import (
    expected_timestamp_count,
    local_day_bounds,
    local_window_bounds,
)

# ---------------------------------------------------------------------------
# local_day_bounds / expected_timestamp_count
# ---------------------------------------------------------------------------


def test_local_day_bounds_normal_day() -> None:
    start, end = local_day_bounds(dt.date(2026, 8, 20))
    assert start.isoformat() == "2026-08-20T00:00:00+02:00"
    assert end.isoformat() == "2026-08-21T00:00:00+02:00"
    assert expected_timestamp_count(start, end, 15) == 96
    assert expected_timestamp_count(start, end, 60) == 24


def test_local_day_bounds_spring_forward_dst() -> None:
    # 2026-03-29: 02:00 -> 03:00 CEST, 23h day.
    start, end = local_day_bounds(dt.date(2026, 3, 29))
    assert start.isoformat() == "2026-03-29T00:00:00+01:00"
    assert end.isoformat() == "2026-03-30T00:00:00+02:00"
    assert expected_timestamp_count(start, end, 15) == 92
    assert expected_timestamp_count(start, end, 60) == 23


def test_local_day_bounds_fall_back_dst() -> None:
    # 2026-10-25: 03:00 -> 02:00 CET, 25h day.
    start, end = local_day_bounds(dt.date(2026, 10, 25))
    assert start.isoformat() == "2026-10-25T00:00:00+02:00"
    assert end.isoformat() == "2026-10-26T00:00:00+01:00"
    assert expected_timestamp_count(start, end, 15) == 100
    assert expected_timestamp_count(start, end, 60) == 25


def test_expected_timestamp_count_rejects_naive_bounds() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        expected_timestamp_count(pd.Timestamp("2026-08-20"), pd.Timestamp("2026-08-21"), 60)


# ---------------------------------------------------------------------------
# local_window_bounds
# ---------------------------------------------------------------------------


def test_local_window_bounds_matches_two_independent_day_bounds() -> None:
    date = dt.date(2026, 8, 20)
    start, end = local_window_bounds(date, days_back_start=2, days_back_end=1)
    assert start == local_day_bounds(dt.date(2026, 8, 18))[0]
    assert end == local_day_bounds(dt.date(2026, 8, 19))[0]


def test_local_window_bounds_supports_negative_days_back_end_for_forward_reach() -> None:
    # DA_FORECAST critical-window extension (spec §5.2): D-1 to D+1.
    date = dt.date(2026, 8, 20)
    start, end = local_window_bounds(date, days_back_start=1, days_back_end=-1)
    assert start == local_day_bounds(dt.date(2026, 8, 19))[0]
    assert end == local_day_bounds(dt.date(2026, 8, 21))[0]


def test_local_window_bounds_90_days_across_dst_lands_on_local_midnight() -> None:
    # Window ending 2026-11-01 (CET, after the fall-back transition), starting
    # 2026-08-03 (CEST, before it): the window genuinely straddles the
    # transition, so naive Timestamp(days=90) subtraction from `end` is off
    # by an hour; per-endpoint localization is not.
    date = dt.date(2026, 11, 1)
    start, end = local_window_bounds(date, days_back_start=90, days_back_end=0)

    assert start == pd.Timestamp(dt.date(2026, 8, 3), tz="Europe/Berlin")
    assert end == pd.Timestamp(date, tz="Europe/Berlin")

    naive_start = end - pd.Timedelta(days=90)
    assert naive_start != start
    assert naive_start.isoformat() == "2026-08-03T01:00:00+02:00"  # off by 1h, not local midnight


# ---------------------------------------------------------------------------
# Consistency against market_time._local_day (the point of this file)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date",
    [
        dt.date(2026, 8, 20),  # normal day
        dt.date(2026, 3, 29),  # spring-forward
        dt.date(2026, 10, 25),  # fall-back
    ],
)
def test_local_day_bounds_agrees_with_market_time_local_day(date: dt.date) -> None:
    """ops/windows.py's local midnight for `date` must equal what
    market_time._local_day floors any UTC timestamp within that day down to.

    This is the guard against Entscheidung 5's two implementations silently
    diverging (spec §13): if it ever fails, ops/windows.py and market_time.py
    disagree on what "local day" means and 6.2's window arithmetic is no
    longer trustworthy.
    """
    expected_start, _ = local_day_bounds(date)

    # Any UTC instant within the local day must floor back to expected_start,
    # not just the boundary itself -- exercise midnight and mid-day.
    for offset_hours in (0, 12):
        probe_utc = expected_start.tz_convert("UTC") + pd.Timedelta(hours=offset_hours)
        floored = _local_day(pd.DatetimeIndex([probe_utc]))[0]
        assert floored == expected_start
