"""Installed generation capacity: interpolation, extrapolation, swappable source.

The installed capacity of a production type is needed at two points -- as the
label denominator during training and as the factor when converting a
predicted capacity factor back to MW at inference time (spec 6.5.2, section
2.5). Both call sites use `installed_capacity_at`, the single place in the
repo that resolves an installed-capacity figure. Two rules, or two data
sources reached through different code paths, would not cancel a shared
systematic error the way one rule and one code path does (decision 10,
train/serve consistency).

Source (spec 6.5.2, section 2.11): the ENTSO-E Transparency Platform (which
would supply 14.1.A, the regulatory series this project would otherwise use)
has been unreachable since 2026-08-30 with no known recovery date. Rather
than block on it, `installed_capacity_at` takes a `source` parameter with two
values behind one interface: `CapacitySource.PUBLIC_REGISTRY` (implemented
now) and `CapacitySource.ENTSOE_14_1_A` (deferred, raises NotImplementedError
until the platform is reachable again -- see docs/sprint6_step6_5_2_log.md).
The interpolation/extrapolation rule itself does not know or care which
source supplied the anchors.

PUBLIC_REGISTRY anchor table provenance (spec 6.5.2, section 2.11 -- full
detail also in docs/sprint6_step6_5_2_log.md, section "Schritt 2 (neu)"):

- Source: Energy-Charts API, Fraunhofer-Institut fuer Solare Energiesysteme
  (ISE) -- a curated aggregation over German capacity register data, not raw
  Marktstammdatenregister records themselves.
- URL: https://api.energy-charts.info/installed_power?country=de&time_step=monthly&installation_decommission=false
- Fetched: 2026-09-01 (the source's own `last_update` field: 2026-08-31T14:49:27 UTC).
- Definition: monthly, installed power *at the end of* each calendar month
  (confirmed against the endpoint's own OpenAPI description, not assumed),
  originally in GW, converted to MW in `data/capacity_anchors_public_registry.csv`.
  The `solar` series is the source's "Solar DC" column -- module/nameplate DC
  capacity, matching the plausibility figures this project already cites
  (e.g. Fraunhofer ISE's own December-2025 press release: "116.8 gigawatts of
  module capacity (DC)") -- not "Solar AC" (inverter-limited), a different
  quantity. `wind_onshore` and `wind_offshore` are reported as separate
  series by the source, never summed.
- A source switch to ENTSOE_14_1_A is a training-affecting event, not a
  detail: it changes the label denominator, so it means retraining, and the
  switch is to be logged when it happens (spec 6.5.2, section 2.11).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT

PUBLIC_REGISTRY_ANCHORS_PATH: Final = PROJECT_ROOT / "data" / "capacity_anchors_public_registry.csv"

# The public registry revises its most recent one to two monthly anchors upward
# as later reports arrive (spec 6.5.2, section 2.11) -- the newest anchor is
# therefore the least reliable one, and it is exactly the one that matters for
# the extrapolation edge. Discarding it and bridging the gap with the
# extrapolation rule instead is more honest than trusting a value known to be
# revised. Named so it can be revisited once the extrapolation diagnostic
# (section 5.8) shows whether one or two anchors is the right number.
TRAILING_ANCHORS_DISCARDED: Final[int] = 2

# An anchor series with a median spacing below this is treated as monthly
# (and subject to TRAILING_ANCHORS_DISCARDED); at or above, as yearly (spec
# 6.5.2, section 2.11: "bei jaehrlicher Aufloesung entfaellt die Regel").
_MONTHLY_CADENCE_THRESHOLD_DAYS: Final[float] = 60.0

_EPOCH: Final = pd.Timestamp("1970-01-01", tz="UTC")


class ProductionType(StrEnum):
    SOLAR = "solar"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"


class CapacitySource(StrEnum):
    PUBLIC_REGISTRY = "public_registry"
    ENTSOE_14_1_A = "entsoe_14_1_a"


class CapacityExtrapolation(StrEnum):
    LAST_INCREMENT = "last_increment"
    MEAN_LAST_THREE_INCREMENTS = "mean_last_three_increments"
    LINEAR_REGRESSION_LAST_THREE = "linear_regression_last_three"


def _validate_anchors(anchors: pd.Series) -> None:
    if anchors.index.has_duplicates:
        raise ValueError("capacity anchors contain duplicate timestamps")
    if not anchors.index.is_monotonic_increasing:
        raise ValueError("capacity anchors are not monotonically increasing")


def _load_public_registry_anchors(production_type: ProductionType) -> pd.Series:
    table = pd.read_csv(PUBLIC_REGISTRY_ANCHORS_PATH, comment="#")
    subset = table.loc[table["production_type"] == production_type.value]
    if subset.empty:
        raise ValueError(
            f"no public-registry anchors for production_type={production_type.value!r}"
        )
    index = pd.DatetimeIndex(pd.to_datetime(subset["anchor_date"]), tz="UTC")
    return pd.Series(subset["capacity_mw"].to_numpy(dtype="float64"), index=index).sort_index()


def _load_entsoe_anchors(production_type: ProductionType) -> pd.Series:
    raise NotImplementedError(
        "CapacitySource.ENTSOE_14_1_A is deferred (spec 6.5.2, section 2.11) -- the "
        "ENTSO-E Transparency Platform has been unreachable since 2026-08-30. Use "
        "CapacitySource.PUBLIC_REGISTRY, or see docs/sprint6_step6_5_2_log.md for status."
    )


_ANCHOR_LOADERS: Final = {
    CapacitySource.PUBLIC_REGISTRY: _load_public_registry_anchors,
    CapacitySource.ENTSOE_14_1_A: _load_entsoe_anchors,
}


def _days_since_epoch(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index - _EPOCH) / pd.Timedelta(days=1), dtype="float64")


def _is_monthly_cadence(anchors: pd.Series) -> bool:
    gaps_days = np.diff(_days_since_epoch(pd.DatetimeIndex(anchors.index)))
    return bool(np.median(gaps_days) < _MONTHLY_CADENCE_THRESHOLD_DAYS)


def _discard_trailing_anchors(anchors: pd.Series) -> pd.Series:
    if not _is_monthly_cadence(anchors):
        return anchors
    if TRAILING_ANCHORS_DISCARDED == 0:
        return anchors
    if len(anchors) <= TRAILING_ANCHORS_DISCARDED:
        raise ValueError(
            f"only {len(anchors)} anchors available, cannot discard "
            f"{TRAILING_ANCHORS_DISCARDED} trailing ones"
        )
    return anchors.iloc[:-TRAILING_ANCHORS_DISCARDED]


def _extrapolation_rate_per_day(anchors: pd.Series, method: CapacityExtrapolation) -> float:
    """MW/day rate used to extend the series beyond the last anchor."""
    days = _days_since_epoch(pd.DatetimeIndex(anchors.index))
    values = anchors.to_numpy(dtype="float64")

    if method == CapacityExtrapolation.LAST_INCREMENT:
        if len(anchors) < 2:
            raise ValueError("LAST_INCREMENT needs at least 2 anchors")
        return (values[-1] - values[-2]) / (days[-1] - days[-2])

    if method == CapacityExtrapolation.MEAN_LAST_THREE_INCREMENTS:
        if len(anchors) < 4:
            raise ValueError("MEAN_LAST_THREE_INCREMENTS needs at least 4 anchors")
        rates = np.diff(values[-4:]) / np.diff(days[-4:])
        return float(np.mean(rates))

    if method == CapacityExtrapolation.LINEAR_REGRESSION_LAST_THREE:
        if len(anchors) < 3:
            raise ValueError("LINEAR_REGRESSION_LAST_THREE needs at least 3 anchors")
        slope, _intercept = np.polyfit(days[-3:], values[-3:], deg=1)
        return float(slope)

    raise ValueError(f"unknown extrapolation method: {method!r}")  # pragma: no cover


def installed_capacity_at(
    production_type: ProductionType,
    timestamps: pd.DatetimeIndex,
    *,
    source: CapacitySource = CapacitySource.PUBLIC_REGISTRY,
    method: CapacityExtrapolation = CapacityExtrapolation.LAST_INCREMENT,
) -> pd.Series:
    """Installed capacity in MW at each timestamp.

    Linear interpolation between known anchor points; extrapolation beyond
    the last anchor by the most recently observed increment (spec 2.5).
    Raises if asked for a point more than one anchor interval beyond the
    last known one -- silently carrying a stale capacity forward is the
    error this function exists to prevent.

    The interpolation rule is independent of the source (spec 2.11): both
    sources feed the same code path, they are not two implementations.

    The same function serves both the label denominator and the conversion
    of a predicted capacity factor back to MW. Two rules would not cancel;
    one rule largely does.
    """
    if not isinstance(production_type, ProductionType):
        raise ValueError(f"unknown production_type: {production_type!r}")
    if source not in _ANCHOR_LOADERS:
        raise ValueError(f"unknown capacity source: {source!r}")
    if len(timestamps) == 0:
        return pd.Series(dtype="float64", index=timestamps)
    if timestamps.tz is None:
        raise ValueError("timestamps must be tz-aware")

    anchors = _ANCHOR_LOADERS[source](production_type)
    _validate_anchors(anchors)
    anchors = _discard_trailing_anchors(anchors)

    anchor_days = _days_since_epoch(pd.DatetimeIndex(anchors.index))
    first_anchor = anchors.index[0]
    last_anchor = anchors.index[-1]
    last_interval_days = anchor_days[-1] - anchor_days[-2]
    max_extrapolation = last_anchor + pd.Timedelta(days=last_interval_days)

    ts_min, ts_max = timestamps.min(), timestamps.max()
    if ts_min < first_anchor:
        raise ValueError(f"timestamp {ts_min} precedes the first known anchor {first_anchor}")
    if ts_max > max_extrapolation:
        raise ValueError(
            f"timestamp {ts_max} is more than one anchor interval beyond the last "
            f"known anchor {last_anchor}"
        )

    in_range = timestamps <= last_anchor
    result = np.empty(len(timestamps), dtype="float64")

    if in_range.any():
        y_anchors = anchors.to_numpy(dtype="float64")
        query_days = _days_since_epoch(timestamps[in_range])
        result[in_range] = np.interp(query_days, anchor_days, y_anchors)

    beyond = ~in_range
    if beyond.any():
        rate_per_day = _extrapolation_rate_per_day(anchors, method)
        delta_days = _days_since_epoch(timestamps[beyond]) - anchor_days[-1]
        result[beyond] = anchors.iloc[-1] + rate_per_day * delta_days

    return pd.Series(result, index=timestamps)
