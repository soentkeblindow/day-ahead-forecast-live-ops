"""Similar-Day-Patch for the ENTSO-E day-ahead load forecast (spec 6.9,
section 2.2 / 5.5).

Row 2 of the fallback ladder (``core_gas_ec``, wired in step 11) replaces a
missing/incomplete ``load_forecast_day_ahead`` for the target day with the
same forecast from a reference day, chosen by a fixed weekday rule with a
weekly-escalating fallback (Owner, 2026-09-24) if the first reference itself
turns out to be unusable. Both functions here are pure -- no I/O, no wall
clock -- so the same code path serves Messung E's own re-measurement (spec
2.2) and the live submission path (spec 3.2: "Einzige Implementierung").

The holiday source is deliberately not built here: ``is_holiday`` is passed
in by the caller and must be backed by the same nationwide-DE-only check
``features/calendar.py``'s own ``is_holiday`` feature uses
(``holidays.country_holidays("DE", ...)``, not the two additional regional
holidays ``is_regional_holiday`` tracks separately) -- spec 2.2: "Die
Feiertagsquelle ist dieselbe wie für die Kalender-Features."
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from energy_price_forecast.ops.windows import local_day_bounds

MAX_LOAD_PATCH_WEEKS_BACK: Final[int] = 4  # not measured, see spec 2.2

_DST_HOUR = 2  # the only local hour ever duplicated/missing on a DST transition day


@dataclass(frozen=True)
class LoadPatchReference:
    """The reference day the patch settled on, and the escalation trail
    that led there (spec 5.5/5.7: ``load_patch_weeks_back``/``_skipped``)."""

    reference_day: dt.date
    weeks_back: int  # 0 = first reference of the rule, 1 = one week further, ...
    skipped: tuple[tuple[dt.date, str], ...]  # (day, "holiday" | "incomplete")


def _last_sunday_before(day: dt.date) -> dt.date:
    """Most recent Sunday strictly before ``day`` -- never ``day`` itself,
    even if ``day`` happens to be a Sunday ("letzter Sonntag" reads as past
    tense, spec 2.2)."""
    days_back = (day.weekday() - 6) % 7
    if days_back == 0:
        days_back = 7
    return day - dt.timedelta(days=days_back)


def _candidate_for(target_day: dt.date, *, is_holiday_target: bool, weeks_back: int) -> dt.date:
    """The single candidate day for one escalation step (spec 2.2's three
    lookback chains). ``weeks_back`` is the step index, not literally
    "weeks before target_day" for the Tue-Fri chain's own first step."""
    if is_holiday_target:
        return _last_sunday_before(target_day) - dt.timedelta(weeks=weeks_back)

    weekday = target_day.weekday()  # Mon=0 .. Sun=6
    if weekday in (5, 6, 0):  # Sat, Sun, Mon: D-7, D-14, ...
        return target_day - dt.timedelta(weeks=weeks_back + 1)

    # Tue-Fri: D-1, then D-7, D-14, ...
    if weeks_back == 0:
        return target_day - dt.timedelta(days=1)
    return target_day - dt.timedelta(weeks=weeks_back)


def choose_reference_day(
    target_day: dt.date,
    *,
    is_holiday: Callable[[dt.date], bool],
    is_complete: Callable[[dt.date], bool],
) -> LoadPatchReference | None:
    """Spec 2.2: Tue-Fri D-1, then D-7, D-14 ...; Mon/Sat/Sun D-7, D-14 ...;
    holiday target: last Sunday, then the Sunday before ... A holiday
    reference is skipped unless it is a Sunday; an incomplete reference is
    always skipped. None if nothing usable within
    MAX_LOAD_PATCH_WEEKS_BACK."""
    is_holiday_target = is_holiday(target_day)
    skipped: list[tuple[dt.date, str]] = []

    for weeks_back in range(MAX_LOAD_PATCH_WEEKS_BACK + 1):
        day = _candidate_for(target_day, is_holiday_target=is_holiday_target, weeks_back=weeks_back)

        if is_holiday(day) and day.weekday() != 6:  # a Sunday is always usable, even if a holiday
            skipped.append((day, "holiday"))
            continue
        if not is_complete(day):
            skipped.append((day, "incomplete"))
            continue

        return LoadPatchReference(reference_day=day, weeks_back=weeks_back, skipped=tuple(skipped))

    return None


def _local_hour_index(day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(day)
    return pd.date_range(start, end, freq="h", inclusive="left")


def _canonical_24h(local_index: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Normalises one local day's values (23, 24, or 25 of them) onto a
    fixed 24-slot array indexed by local hour-of-day, resolving the
    reference day's own DST oddity (spec 2.2's "Referenz 25h"/"Referenz
    23h" rows) -- the only local hour ever affected is 2 o'clock."""
    local_hours = local_index.hour.to_numpy()
    canonical = np.empty(24, dtype=float)
    for hour in range(24):
        positions = np.flatnonzero(local_hours == hour)
        if hour == _DST_HOUR and len(positions) == 0:
            # Reference is a 23h day: the missing 2 o'clock value comes from 1 o'clock.
            fallback = np.flatnonzero(local_hours == _DST_HOUR - 1)
            canonical[hour] = values[fallback[0]]
        else:
            # Normal day: exactly one match. Reference is a 25h day: two matches
            # at hour 2 (the repeated wall-clock hour) -- the first one wins.
            canonical[hour] = values[positions[0]]
    return canonical


def build_patched_load(
    reference_load: pd.Series,
    reference_day: dt.date,
    target_day: dt.date,
) -> pd.Series:
    """Map the reference day's hourly load forecast onto the target day by
    local wall-clock hour (spec 2.2 DST table). Output has exactly the
    target day's number of local hours."""
    ref_index = _local_hour_index(reference_day)
    ref_values = reference_load.reindex(ref_index.tz_convert("UTC")).to_numpy()
    canonical = _canonical_24h(ref_index, ref_values)

    target_index = _local_hour_index(target_day)
    target_hours = target_index.hour.to_numpy()
    out_values = canonical[target_hours]

    return pd.Series(out_values, index=target_index.tz_convert("UTC"), name=reference_load.name)
