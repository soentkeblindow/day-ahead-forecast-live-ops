from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd

from ..market_time import gate_closure_for_index

# Publication lag of real-time actuals on the ENTSO-E Transparency Platform (~1 h).
# Conservative bumping is allowed; a larger lag can only reject more, never accept.
RT_ACTUAL_LAG = pd.Timedelta(hours=1)

# "Known since forever" sentinel for deterministic (calendar) features.
_ALWAYS_KNOWN = pd.Timestamp("1900-01-01", tz="UTC")


class Availability(Enum):
    """How early a raw quantity becomes known, relative to its value time."""

    DETERMINISTIC = auto()  # calendar: known arbitrarily far in advance
    DA_FIXED = auto()  # day-ahead auction result (price, scheduled flows)
    DA_FORECAST = auto()  # day-ahead forecast for the value day (load/RES fc)
    RT_ACTUAL = auto()  # realised in real time, ~1 h publication lag
    COMMODITY = auto()  # daily settlement, known from the next day


def knowledge_time(cls: Availability, value_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC instant at which each value in value_index becomes known."""
    if cls is Availability.DETERMINISTIC:
        return pd.DatetimeIndex(
            np.full(len(value_index), _ALWAYS_KNOWN.value, dtype="datetime64[ns]"), tz="UTC"
        )
    if cls is Availability.RT_ACTUAL:
        return value_index + RT_ACTUAL_LAG
    if cls is Availability.DA_FIXED:
        # UTC midnight of the source's UTC date. Anchoring in UTC (not local time)
        # avoids the fall DST trap: in CEST a 24h lag can land on local delivery-day
        # midnight (22:00 UTC), which is 12 h after gate closure (10:00 UTC). UTC
        # midnight (00:00) is always before gate closure (10-11 UTC) on any UTC day.
        return value_index.normalize()
    if cls is Availability.DA_FORECAST:
        return gate_closure_for_index(value_index)
    if cls is Availability.COMMODITY:
        # UTC midnight of the day after the source's UTC date. Anchoring in UTC
        # (not local time) avoids the fall DST trap: in CEST, local midnight is
        # 22:00 UTC, which can land after gate closure (10:00 UTC) on transition
        # days. Daily settlements close ~16:00 UTC, so next UTC midnight is safe.
        return value_index.normalize() + pd.Timedelta(days=1)
    raise ValueError(f"unknown availability class {cls!r}")


# Raw-column -> availability class. Source of truth: sprint2_raw_feature_availability.md.
# Cross-border columns are matched by prefix (variable neighbour suffixes).
_RAW_AVAILABILITY: dict[str, Availability] = {
    "day_ahead_price": Availability.DA_FIXED,
    "load_forecast_day_ahead": Availability.DA_FORECAST,
    "wind_onshore_forecast": Availability.DA_FORECAST,
    "wind_offshore_forecast": Availability.DA_FORECAST,
    "solar_forecast": Availability.DA_FORECAST,
    "load_actual": Availability.RT_ACTUAL,
    "gen_nuclear": Availability.RT_ACTUAL,
    "gen_lignite": Availability.RT_ACTUAL,
    "gen_hard_coal": Availability.RT_ACTUAL,
    "gen_gas": Availability.RT_ACTUAL,
    "gen_oil": Availability.RT_ACTUAL,
    "gen_biomass": Availability.RT_ACTUAL,
    "gen_hydro": Availability.RT_ACTUAL,
    "gen_wind_onshore": Availability.RT_ACTUAL,
    "gen_wind_offshore": Availability.RT_ACTUAL,
    "gen_solar": Availability.RT_ACTUAL,
    "gen_other": Availability.RT_ACTUAL,
    "ttf_gas_eur_per_mwh": Availability.COMMODITY,
    "eua_co2_eur_per_t": Availability.COMMODITY,
}


def availability_of(column: str) -> Availability:
    """Availability class of a raw column. Raises KeyError if none is registered.

    The raise is the invariant at the raw level: an unregistered column has no
    knowledge-time rule and therefore cannot be turned into a feature.
    """
    if column in _RAW_AVAILABILITY:
        return _RAW_AVAILABILITY[column]
    if column.startswith("scheduled_net_de_to_"):
        return Availability.DA_FIXED
    if column.startswith("physical_net_de_to_"):
        return Availability.RT_ACTUAL
    raise KeyError(f"no availability rule registered for raw column {column!r}")


@dataclass(frozen=True)
class Feature:
    """A model input column plus the instant each value became known.

    values         : the feature column, indexed by target (UTC) timestamps.
    knowledge_time : same index; the latest instant any input to that value was
                     known. There is NO default -- a Feature cannot exist without
                     its knowledge time (the structural invariant).
    """

    name: str
    values: pd.Series
    knowledge_time: pd.Series  # UTC instants, same index as values


def calendar_feature(name: str, values: pd.Series) -> Feature:
    """A deterministic calendar feature (hour, weekday, holiday, ...): always known."""
    dti = pd.DatetimeIndex(values.index)
    kt = pd.Series(knowledge_time(Availability.DETERMINISTIC, dti), index=values.index)
    return Feature(name, values.rename(name), kt)


def lag(
    name: str,
    raw: pd.Series,
    column: str,
    *,
    hours: int,
    target_index: pd.DatetimeIndex,
) -> Feature:
    """Lag a raw column by hours (>= 1) and align it to target_index.

    Value for target t is raw(t - hours); knowledge time is that of the raw
    column's class at t - hours.
    """
    if hours < 1:
        raise ValueError("lag hours must be >= 1")
    cls = availability_of(column)
    source_index = target_index - pd.Timedelta(hours=hours)
    values = pd.Series(raw.reindex(source_index).to_numpy(), index=target_index, name=name)
    kt = pd.Series(knowledge_time(cls, source_index), index=target_index)
    return Feature(name, values, kt)


def forecast_for_target(
    name: str,
    raw: pd.Series,
    column: str,
    *,
    target_index: pd.DatetimeIndex,
) -> Feature:
    """A day-ahead forecast used as the input for the very day it forecasts.

    Value for target t = forecast(t); knowledge time = gate closure of t's delivery
    day. Only valid for DA_FORECAST columns; passes the leakage check by equality.
    """
    cls = availability_of(column)
    if cls is not Availability.DA_FORECAST:
        raise ValueError(f"forecast_for_target expects DA_FORECAST, got {column!r} ({cls})")
    values = pd.Series(raw.reindex(target_index).to_numpy(), index=target_index, name=name)
    kt = pd.Series(knowledge_time(cls, target_index), index=target_index)
    return Feature(name, values, kt)


def combine(
    name: str,
    parts: Sequence[Feature],
    fn: Callable[..., pd.Series],
) -> Feature:
    """Derive a feature from several others.

    Values = fn(part.values...). Knowledge time = elementwise MAX over the parts
    (the latest input gates availability).
    """
    if not parts:
        raise ValueError("combine needs at least one input feature")
    values = fn(*[p.values for p in parts]).rename(name)
    kt = pd.concat([p.knowledge_time for p in parts], axis=1).max(axis=1)
    return Feature(name, values, kt)


class LeakageError(AssertionError):
    """Raised when a feature value would be known only after its gate closure."""


def assert_no_leakage(features: Sequence[Feature]) -> None:
    """Assert every feature value is known by the gate closure of its target day.

    Runs over the full index (train and test rows alike). Raises LeakageError
    naming the first offending feature and the affected row count.
    """
    for f in features:
        idx = pd.DatetimeIndex(f.values.index)
        gc = pd.Series(gate_closure_for_index(idx), index=f.values.index)
        offending = f.knowledge_time > gc
        if bool(offending.any()):
            n = int(offending.sum())
            first = f.knowledge_time.index[int(offending.to_numpy().argmax())]
            raise LeakageError(
                f"feature {f.name!r}: {n} row(s) known after gate closure (first at {first})"
            )


def rolling_mean(
    name: str,
    raw: pd.Series,
    column: str,
    *,
    window_hours: int,
    lag_hours: int,
    target_index: pd.DatetimeIndex,
) -> Feature:
    """Trailing rolling mean of a raw column, aligned to a target index.

    The window is a backward-looking mean over `window_hours` whose leading
    (most recent) edge sits `lag_hours` before each target timestamp:

        value(t) = mean( raw[t - lag_hours - window_hours + 1h : t - lag_hours] )

    Knowledge time = that of the raw column's availability class at the leading
    edge (t - lag_hours). The leading edge is the latest input in the window, so
    by the max-composition rule it gates the whole mean (all earlier points in
    the window are known no later than the leading edge).

    Assumes a regular hourly grid (guaranteed by step 2.1); the window is given
    in whole hours and used as an integer period count.
    """
    if lag_hours < 1:
        raise ValueError("lag_hours must be >= 1")
    if window_hours < 1:
        raise ValueError("window_hours must be >= 1")
    cls = availability_of(column)
    # Rolling mean at value time first (min_periods = full window -> warm-up = NaN),
    # then shift the leading edge `lag_hours` back relative to each target.
    rolled = raw.rolling(window=window_hours, min_periods=window_hours).mean()
    leading_edge = target_index - pd.Timedelta(hours=lag_hours)
    values = pd.Series(rolled.reindex(leading_edge).to_numpy(), index=target_index, name=name)
    kt = pd.Series(knowledge_time(cls, leading_edge), index=target_index)
    return Feature(name, values, kt)


def build_matrix(features: Sequence[Feature]) -> pd.DataFrame:
    """Check leakage for every feature, then assemble the value columns into X."""
    assert_no_leakage(features)
    return pd.concat([f.values for f in features], axis=1)
