"""Additive intra-hour shape profile and hourly-to-quarter-hourly expansion.

See spec 6.4 section 5.3 for the exact computation this implements:

  m[d,h]   = mean(p[d,h,0..3])            hourly mean of a known delivery day
  delta[d,h,q] = p[d,h,q] - m[d,h]        deviation from that hourly mean
  S[h,q]   = mean over d in window of delta[d,h,q]

Because every (day, hour) group is centered on its own mean before being
averaged into a cell, sum_q S[h,q] == 0 for every h -- exactly, up to float
rounding -- regardless of how many groups feed a cell. That covers the
fall-back day's doubled local hour too: both passes are separate (day, hour)
groups, each centered independently, and both feed the same cell.
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

_QUARTER_OFFSETS = pd.to_timedelta([0, 15, 30, 45], unit="min")


@dataclass(frozen=True)
class ShapeProfile:
    """Additive intra-hour shape, one value per (hour_of_day, quarter) cell.

    Values are deviations from the hourly mean in EUR/MWh. By construction the
    four quarters of any hour sum to zero, so expanding an hourly forecast with
    this profile leaves the hourly mean of the expanded series exactly equal to
    the hourly forecast. That invariant is asserted on construction.
    """

    values: pd.Series  # MultiIndex (hour, quarter), 24 x 4 = 96 cells
    n_days: int  # window length actually used
    window_end_day: pd.Timestamp  # last delivery day that fed the profile
    counts: pd.Series  # observations per cell, for diagnostics

    def __post_init__(self) -> None:
        by_hour = self.values.groupby(level="hour").sum()
        bad = by_hour[by_hour.abs() > 1e-6]
        if not bad.empty:
            hour = bad.index[0]
            raise ValueError(
                f"Shape profile zero-sum invariant violated at hour={hour}: "
                f"sum={bad.iloc[0]!r} (expected ~0)."
            )


def fit_shape_profile(
    prices_qh: pd.Series,
    *,
    end_day: pd.Timestamp,
    n_days: int,
    tz: str = "Europe/Berlin",
    min_observations: int = 3,
) -> ShapeProfile:
    """Estimate the additive intra-hour shape from the last n_days.

    KNOWLEDGE-TIME CONTRACT: ``end_day`` is the last delivery day whose prices
    are known at gate closure of the target day, i.e. target_day - 1. The
    window is [end_day - n_days + 1, end_day], inclusive on both sides. The
    function must never read a price with a delivery timestamp after end_day;
    this is asserted, not merely intended (see the leakage test in section 7).

    Additive rather than multiplicative because day-ahead prices reach and
    cross zero; a ratio profile would divide by near-zero values and invert
    the shape on sign changes (spec 2.2).

    DST days contribute what they have: the missing 02:00 hour of a
    spring-forward day contributes no observations to those cells, and the
    doubled hour of a fall-back day contributes two. Cells with fewer than
    ``min_observations`` raise -- a shape estimated from one or two days is
    noise wearing a profile's clothing.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")

    end_date = pd.Timestamp(end_day).date()
    start_date = end_date - dt.timedelta(days=n_days - 1)

    # Exclusive upper bound at local midnight after end_day -- the leakage
    # guard: nothing at or after this instant is ever read.
    window_start = pd.Timestamp(start_date, tz=tz)
    window_end = pd.Timestamp(end_date + dt.timedelta(days=1), tz=tz)

    idx = pd.DatetimeIndex(prices_qh.index)
    mask = (idx >= window_start) & (idx < window_end)
    window = prices_qh.loc[mask]
    window_idx = pd.DatetimeIndex(window.index)

    # Physical (UTC) hour bucket -- separates the two passes of a fall-back
    # day's doubled local hour into distinct groups, each with its own mean.
    hour_bucket = window_idx.floor("h")
    group_mean = window.groupby(hour_bucket).transform("mean")
    delta = window - group_mean

    local = window_idx.tz_convert(tz)
    cell_index = pd.MultiIndex.from_arrays(
        [local.hour, local.minute // 15], names=["hour", "quarter"]
    )
    delta_by_cell = pd.Series(delta.to_numpy(), index=cell_index).groupby(level=["hour", "quarter"])

    full_index = pd.MultiIndex.from_product([range(24), range(4)], names=["hour", "quarter"])
    counts = delta_by_cell.count().reindex(full_index, fill_value=0).astype(int)
    values = delta_by_cell.mean().reindex(full_index)

    sparse = counts[counts < min_observations]
    if not sparse.empty:
        hour, quarter = sparse.index[0]
        raise ValueError(
            f"Shape profile cell (hour={hour}, quarter={quarter}) has only "
            f"{sparse.iloc[0]} observation(s) in the {n_days}-day window ending "
            f"{end_date}, need >= {min_observations}."
        )

    return ShapeProfile(
        values=values.sort_index(),
        n_days=n_days,
        window_end_day=pd.Timestamp(end_date),
        counts=counts.sort_index(),
    )


def expand_to_quarterhour(
    hourly_forecast: pd.Series,
    profile: ShapeProfile | None,
    *,
    target_day: pd.Timestamp,
    tz: str = "Europe/Berlin",
) -> pd.Series:
    """Expand 24 hourly forecasts into 92/96/100 quarter-hourly forecasts.

    ``profile=None`` performs the flat expansion (each hourly value repeated
    four times), which is the ablation this step measures the shape profile
    against.

    The output index is the local calendar day ``target_day`` at 15-minute
    resolution, converted to the repo's canonical UTC index. Its length is
    92, 96 or 100 depending on DST -- derived from the calendar, never
    hard-coded to 96.

    ``hourly_forecast`` must be indexed by the UTC hourly timestamps of the
    local delivery day (23/24/25 rows across DST, the same convention
    evaluation/walkforward.py's Fold.test_index uses) -- checked, not
    assumed.
    """
    target_date = pd.Timestamp(target_day).date()
    day_start = pd.Timestamp(target_date, tz=tz)
    day_end = pd.Timestamp(target_date + dt.timedelta(days=1), tz=tz)
    expected_hours = pd.date_range(day_start, day_end, freq="h", inclusive="left").tz_convert("UTC")

    actual_hours = pd.DatetimeIndex(hourly_forecast.index)
    if not actual_hours.sort_values().equals(expected_hours):
        raise ValueError(
            f"hourly_forecast index does not match the expected {len(expected_hours)} "
            f"UTC hours for local delivery day {target_date} (tz={tz}); "
            f"got {len(actual_hours)} timestamps."
        )
    hourly_forecast = hourly_forecast.reindex(expected_hours)

    local_hours = expected_hours.tz_convert(tz).hour

    parts = []
    for hour_ts, level, hour_of_day in zip(
        expected_hours, hourly_forecast.to_numpy(), local_hours, strict=True
    ):
        deltas = np.zeros(4) if profile is None else profile.values.loc[hour_of_day].to_numpy()
        qh_index = hour_ts + _QUARTER_OFFSETS
        parts.append(pd.Series(level + deltas, index=qh_index))

    result = pd.concat(parts)
    result.name = "y_pred"
    return result
