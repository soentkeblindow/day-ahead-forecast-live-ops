"""Replica of the Energy-Arena's persistence baseline.

Separate from models/baseline.py on purpose: that module holds this
project's own benchmarks, built for this project's own evaluation logic.
This is a replica of a *platform's* specification -- code that has to match
someone else's definition, even where a different choice would have been
made -- so it gets its own file and its own docstring naming the source of
the definition (the Arena's stated persistence rule, spec 6.4 section 2.1).
"""

import datetime as dt

import pandas as pd


def persistence_forecast(
    prices_qh: pd.Series,
    target_day: pd.Timestamp,
    *,
    tz: str = "Europe/Berlin",
) -> pd.Series:
    """Replicate the Energy-Arena baseline for one delivery day.

    The platform substitutes a missing submission with the performance of a
    persistence forecast: the realised quarter-hourly day-ahead price of the
    preceding delivery day, aligned by LOCAL WALL-CLOCK TIME (see spec 2.1).

    Availability: at gate closure of D (12:00 local on D-1) the day-ahead
    prices for delivery day D-1 are known -- they cleared in the auction on
    D-2. The baseline therefore uses no information the forecaster does not
    have, and needs no separate leakage guard beyond this contract.

    DST handling, both directions, is explicit rather than incidental:

    - Spring forward (D has 92 quarter-hours): D-1 is a normal 96-slot day
      and supplies every wall-clock slot D needs. D-1's 02:00 slots are
      simply unused.
    - Fall back (D has 100 quarter-hours, D-1 has 96): the wall-clock hour
      02:00 occurs twice on D but only once on D-1. Both occurrences take
      D-1's single 02:00 value.

    Returns a Series indexed like the realised prices of ``target_day``, i.e.
    92, 96 or 100 entries. Raises if the required D-1 values are missing --
    the baseline is never silently partial.
    """
    target_date = pd.Timestamp(target_day).date()
    prev_date = target_date - dt.timedelta(days=1)

    target_index = _local_day_index(target_date, tz)
    prev_index = _local_day_index(prev_date, tz)

    prev_values = prices_qh.reindex(prev_index)
    if prev_values.isna().any():
        missing = prev_index[prev_values.isna()]
        raise ValueError(
            f"Missing realised price(s) for {len(missing)} quarter-hour(s) on {prev_date} "
            f"-- cannot build the persistence baseline for {target_date}. "
            f"First missing timestamp: {missing[0]}."
        )

    # D-1 is never itself a DST-transition day (transitions occur on exactly
    # one calendar day, and D-1 immediately precedes D), so its wall-clock
    # (hour, minute) pairs are always distinct -- a plain dict lookup is safe.
    prev_local = prev_index.tz_convert(tz)
    lookup = dict(
        zip(
            zip(prev_local.hour, prev_local.minute, strict=True),
            prev_values.to_numpy(),
            strict=True,
        )
    )

    target_local = target_index.tz_convert(tz)
    keys = list(zip(target_local.hour, target_local.minute, strict=True))
    missing_keys = [k for k in keys if k not in lookup]
    if missing_keys:
        raise ValueError(
            f"No matching D-1 wall-clock slot for target day {target_date}: "
            f"{missing_keys[0][0]:02d}:{missing_keys[0][1]:02d}."
        )

    values = [lookup[k] for k in keys]
    return pd.Series(values, index=target_index, name="baseline")


def _local_day_index(date: dt.date, tz: str) -> pd.DatetimeIndex:
    """Quarter-hourly UTC index covering local calendar day ``date``.

    Bounds are built by localizing bare calendar dates into ``tz`` -- so the
    UTC offset is resolved fresh by the tz database for that specific date,
    never carried over via arithmetic on an already-tz-aware timestamp (the
    distinction spec 6.4 section 11 warns about, after two real bugs found
    this way in 6.2). Length is 92/96/100 depending on DST, derived from the
    calendar, never hard-coded to 96.
    """
    start = pd.Timestamp(date, tz=tz)
    end = pd.Timestamp(date + dt.timedelta(days=1), tz=tz)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
