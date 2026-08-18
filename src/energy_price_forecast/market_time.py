from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Berlin"
GATE_CLOSURE_LOCAL_HOUR = 12  # EPEX day-ahead gate closure: 12:00 local time on D-1


def _local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local (Europe/Berlin) calendar-day midnight for each UTC timestamp."""
    return index.tz_convert(LOCAL_TZ).normalize()


def gate_closure_for_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Gate closure (UTC) for each target timestamp: 12:00 Europe/Berlin on the
    calendar day before the timestamp's local delivery day.

    Mirrors Fold.gate_closure from walkforward.py and coincides with it on the
    gapless hourly grid produced in step 2.1 (cross-checked by test). DST-safe
    via DateOffset; 12:00 is never in the ambiguous DST window.
    """
    prev_local_day = _local_day(index) - pd.DateOffset(days=1)
    return (prev_local_day + pd.Timedelta(hours=GATE_CLOSURE_LOCAL_HOUR)).tz_convert("UTC")
