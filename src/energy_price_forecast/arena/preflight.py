"""Preflight checks for the daily submission job (spec 6.7.2, sections
2.1-2.6, 5.2, 5.5; restructured by spec 6.9, sections 2.3, 2.6, 2.7).

Every function here is pure: no network, no file I/O (spec section 3.2 --
"Kein Netz, keine Dateien" -- fully testable without a store or a live
API). Callers (scripts/run_daily_submission.py) do the actual reading
(the store, a weather run file, the capacity anchor table) and pass
already-loaded values in; a short signature sketch in the spec itself
takes a StoreState directly, but this project's own precedent (6.7.1 A6:
"die Kurzsignaturen der Spec sind Skizzen, keine woertlichen Vertraege")
is to implement the fully testable, I/O-free version when the two are in
tension, not the literal pseudocode. The same precedent applies to spec
6.9 section 5.4's check_training_window sketch, which takes a full
Candidate: this module accepts ``required_features: frozenset[str]``
instead, since arena/candidates.py already imports from this module --
taking the dataclass itself would be a circular import for no benefit
check_target_row does not already need.

Two stages, and they cannot be fully upfront (spec section 2.1):

- Check A -- cheap, before any fit. Can the renewables reconstruction for
  D even be built? Spec 6.9 section 2.3 splits this into a **global**
  question (check_holiday_calendar, spec section 5.1 step 3: a failure
  here means every candidate row is unreachable, so the run is silent
  before anything else runs) and a **per-row** question that only rows
  needing the NWP reconstruction care about (check_weather_run,
  check_capacity_anchor, spec section 5.1 step 4: their combined result is
  what the caller calls ``nwp_available``). A row without an NWP
  dependency (the still-unbuilt gasfrei fallback row, spec 6.9 section
  2.1) is not gated by the per-row half at all.
- Check B (check_target_row) -- after the feature matrix is built. Is the
  built row for D complete? This is the derived, not described, freshness
  check (spec section 2.2/2.3): staleness shows up as NaN exactly where
  it bites, not against a guessed per-source age table.

check_capacity_anchor deliberately separates an expired anchor table from
every other Check A failure (spec 6.9 section 2.3): its own expiry
(currently 2026-10-28, data.capacity.anchor_table_valid_until) is a
maintenance lapse, not a data outage, and must stay visible
(``CAPACITY_ANCHOR_WARN_DAYS`` warning window) well before it can ever
block a run -- callers should treat its ``warning``/expiry state as
something to surface regardless of whether Check A as a whole passes.

check_training_window guards the one failure mode neither check sees: a
frozen source that silently shortens the training window's recent end
with no NaN anywhere (spec section 2.4 -- happened for real on
2026-09-11, see docs/sprint6_step6_7_1a_log.md). It replaces the former
check_training_extent's exact row-count band with measured tolerances
(spec 6.9 section 2.6): a training day counts only if every required
feature cell AND the (post-coalesce) price label are present for all of
that day's local hours, and the two thresholds
(``MIN_TRAINING_DAYS``/``MAX_TRAINING_EDGE_AGE_DAYS``) are checked
against that day-level count rather than an hour-count band. The former
known_defect_tolerance_hours() and its recursive chain-explanation logic
are removed rather than patched -- that recursion is the real, root
cause of the 2026-09-22 training-extent chain bug (docs/bugs_in_live_
system.md section 1, closed by this change): it broke whenever a chain's
own root defect day rolled out of the training window while a day
depending on it was still inside. data.weather_grid.KNOWN_WEATHER_DEFECTS
still has a place here, but purely as a reporting cross-reference
(``TrainingWindowReport.known_weather_defect_days``, a simple
intersection with the missing days found) -- it grants no tolerance of
its own any more.

check_renewables_label_edge_age (spec 6.9 section 2.7) is a second,
narrower edge-age guard, not a generalisation of check_training_window:
it asks whether the renewables reconstruction's own 14.1.D training
labels (evaluation.renewables_walkforward.TARGET_COLUMNS -- read, not
imported for modification, so this stays inside spec section 3.3's "models/,
evaluation/ unverändert" boundary) reach close enough to be usable at all.
Deliberately NOT an extension of run_renewables_backtest's own 365-day
rolling window check (there isn't one -- see this spec step's own log,
Schritt 1 point 4): that gap is explicitly out of this spec's scope.
MAX_RNW_LABEL_EDGE_AGE_DAYS is NOT measured (spec section 2.7 says so
explicitly), unlike check_training_window's two constants.

The remaining functions implement the three-layer defence against a
plausible-looking, wrong result (spec section 2.6): capacity-factor
bounds and a daily-sum band for the renewables step, and a payload
plausibility band for the price step (spec section 5.6).

check_commodity_staleness is the one named exception to "measured where it
bites, not per source" (spec section 2.3): commodities are forward-filled
(data/loaders.py::COMMODITY_FFILL_LIMIT), so a stale-but-still-within-limit
value never shows up as NaN in Check B at all -- this is the separate,
non-blocking measurement spec section 2.3 requires purely for the
protocol/run-summary ("hier hätten wir früher geschwiegen"), never a skip
reason.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from energy_price_forecast.data.loaders import COMMODITY_STALENESS_WARN_DAYS
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import KNOWN_WEATHER_DEFECTS
from energy_price_forecast.ops.store import validate_weather_run
from energy_price_forecast.ops.windows import LOCAL_TZ, local_day_bounds

# spec 6.9 section 2.6: replaces the former check_training_extent's exact
# row-count band. Not derived from KNOWN_WEATHER_DEFECTS or any other
# measurement -- MIN_TRAINING_DAYS is simply "83 of the nominal 90",
# MAX_TRAINING_EDGE_AGE_DAYS mirrors the k=7 already used informally
# before this step (Messung B).
MIN_TRAINING_DAYS: Final[int] = 83
MAX_TRAINING_EDGE_AGE_DAYS: Final[int] = 7

# spec 6.9 section 2.3: a Pflegeversäumnis warning window, not a measurement.
CAPACITY_ANCHOR_WARN_DAYS: Final[float] = 14

# spec 6.9 section 2.7: explicitly NOT measured (unlike the two constants above).
MAX_RNW_LABEL_EDGE_AGE_DAYS: Final[int] = 7


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of one preflight check.

    ``missing_features`` is populated only by check_target_row -- a
    separate field from ``reasons`` (not embedded in a prose string) so a
    caller can copy it straight into
    ``ops.protocol.SubmissionRecord.missing_features`` (spec section 2.5:
    named individually, the input 6.8 needs).
    """

    ok: bool
    reasons: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()


def check_holiday_calendar(
    *, target_day: dt.date, holiday_calendar_covers_target: bool
) -> PreflightResult:
    """spec 6.9 section 2.3, row 1 -- the one Check A question that applies
    to every candidate row alike (spec section 5.1 step 3). A failure here
    means total silence before anything else runs: no row can recover from
    a holiday-calendar gap, since every candidate's calendar feature
    depends on it. The one calendar input that can go wrong silently: a
    table that ends turns a real holiday into an ordinary weekday with no
    NaN and no error.
    """
    if not holiday_calendar_covers_target:
        return PreflightResult(
            ok=False, reasons=(f"holiday calendar does not cover target day {target_day}",)
        )
    return PreflightResult(ok=True)


def check_weather_run(weather_run: pd.DataFrame | None) -> PreflightResult:
    """spec 6.9 section 2.3, row 2 -- only candidate rows that need the NWP
    reconstruction are gated by this (today, the only row; spec section
    2.1's future gasfrei row will not be).

    Reuses ops.store.validate_weather_run() (spec section 2.6 layer 1) --
    the same function that would have caught the real
    HTTP-200-but-all-NaN weather file found in 6.7.1 A9.
    ``weather_run=None`` means the caller could not load it at all
    (missing file, or a raised WeatherRunUnavailable) -- itself a failure
    here, not a separate case to special-case.
    """
    if weather_run is None:
        return PreflightResult(
            ok=False, reasons=("weather run for D-1 00 UTC could not be loaded",)
        )
    weather_result = validate_weather_run(weather_run)
    if not weather_result.ok:
        return PreflightResult(
            ok=False,
            reasons=(
                f"weather run for D-1 00 UTC failed validation: {'; '.join(weather_result.reasons)}",
            ),
        )
    return PreflightResult(ok=True)


@dataclass(frozen=True)
class CapacityAnchorReport:
    """spec 6.9 section 2.3: split out from PreflightResult because an
    expired anchor table is a maintenance lapse, not a data outage -- its
    ``warning``/``days_until_expiry`` state should stay visible to a
    caller well before ``ok`` ever turns False, and unlike an ordinary
    Check A failure it must not quietly disappear into whatever fallback
    a later row provides (see run_daily_submission.py's own handling)."""

    warning: bool
    days_until_expiry: float


def check_capacity_anchor(
    target_day: dt.date,
    anchor_valid_until: pd.Timestamp,
    *,
    warn_days: float = CAPACITY_ANCHOR_WARN_DAYS,
) -> tuple[PreflightResult, CapacityAnchorReport]:
    """spec 6.9 section 2.3, row 3 -- is the capacity anchor table still
    valid for this run_init, i.e. is ``anchor_valid_until`` beyond it?
    Past that boundary the capacity denominator is extrapolated beyond its
    documented limit (currently 2026-10-28,
    data.capacity.anchor_table_valid_until).

    Returns both a PreflightResult (``ok=False`` only once actually
    expired) and a CapacityAnchorReport carrying the warning state, so a
    caller can log the approaching-expiry warning independently of
    whether this check currently blocks anything.
    """
    run_init = run_init_for_target_day(target_day)
    days_until_expiry = (anchor_valid_until - run_init) / pd.Timedelta(days=1)
    expired = run_init >= anchor_valid_until
    warning = (not expired) and days_until_expiry <= warn_days
    reasons: tuple[str, ...] = ()
    if expired:
        reasons = (
            f"capacity anchor table not valid for run_init {run_init} "
            f"(valid until {anchor_valid_until})",
        )
    return (
        PreflightResult(ok=not expired, reasons=reasons),
        CapacityAnchorReport(warning=warning, days_until_expiry=days_until_expiry),
    )


def check_target_row(
    features: pd.DataFrame, target_day: dt.date, required: frozenset[str]
) -> PreflightResult:
    """Every feature the candidate declares must be present and non-NaN in
    the built row(s) for the target day (spec sections 2.2, 2.5).

    This is the freshness check, expressed where staleness actually
    bites: a price history that only reaches D-3 leaves the 24h/48h lags
    empty for D, and the row fails. No per-source max-age table is needed
    or wanted (spec section 2.3).

    ``missing_features`` names every offending column individually and in
    a deterministic (sorted) order, never just "incomplete".
    """
    missing: list[str] = []
    for column in sorted(required):
        if column not in features.columns or features[column].isna().any():
            missing.append(column)

    if missing:
        return PreflightResult(
            ok=False,
            reasons=(f"{len(missing)} required feature(s) missing or NaN for {target_day}",),
            missing_features=tuple(missing),
        )
    return PreflightResult(ok=True)


def _local_hours(day: dt.date) -> pd.DatetimeIndex:
    """The exact UTC hourly index for one Europe/Berlin calendar day --
    23/24/25 hours across DST, the same boundary convention
    ops.windows.local_day_bounds/features.build already use (spec 6.9
    section 2.6: a training day's completeness is judged hour by hour, so
    a 23-hour spring-forward day must not look one hour short)."""
    start, end = local_day_bounds(day)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")


def _is_day_complete(day: dt.date, frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    """Every one of ``columns`` is present and non-NaN for every local
    hour of ``day`` -- a day missing from ``frame``'s index entirely
    reindexes to all-NaN rows here, so it fails the same way a day with a
    genuine NaN gap does (spec 6.9 section 2.6/2.7: the whole-day-out
    policy and a frozen source both look identical from this function's
    point of view, which is the point)."""
    hours = _local_hours(day)
    if not columns:
        return True
    return not frame.reindex(index=hours, columns=list(columns)).isna().any().any()


@dataclass(frozen=True)
class TrainingWindowReport:
    """spec 6.9 section 2.6/5.4: returned alongside a PreflightResult on
    both pass and failure (unlike PreflightResult's own reasons-only-on-
    failure convention), since the protocol/run-summary need these
    numbers regardless of outcome (spec section 5.7)."""

    n_training_days: int
    age_of_last_complete_day: int
    missing_days: tuple[dt.date, ...]
    known_weather_defect_days: tuple[dt.date, ...]
    tolerance_used: tuple[int, int]


def check_training_window(
    features: pd.DataFrame,
    labels: pd.Series,
    required_features: frozenset[str],
    *,
    must_reach: dt.date,
    window_days: int,
    min_training_days: int = MIN_TRAINING_DAYS,
    max_edge_age_days: int = MAX_TRAINING_EDGE_AGE_DAYS,
) -> tuple[PreflightResult, TrainingWindowReport]:
    """Measured tolerances instead of an exact row count (spec 6.9 section
    2.6), replacing the former check_training_extent.

    A calendar day (Europe/Berlin) in the ``window_days``-day window
    ending at ``must_reach`` counts as a complete training day only if
    every cell of every column in ``required_features`` AND ``labels``
    (the post-coalesce price series, spec section 2.6: "das Preis-Label
    nach Zusammenführung") are present for all of that day's local hours.
    A column in ``features`` that is not in ``required_features`` is never
    consulted -- a NaN there does not count against the day.

    ``n_training_days`` is the count of such complete days.
    ``age_of_last_complete_day`` is 0 when ``must_reach`` itself is
    complete, otherwise the number of days back from ``must_reach`` to the
    first complete day found (capped at the window's own start if none
    is found inside it). The rule (spec section 2.6): ``n_training_days >=
    min_training_days`` AND ``age_of_last_complete_day <=
    max_edge_age_days``; a violation fails this candidate.

    ``known_weather_defect_days`` is a simple intersection of the missing
    days with data.weather_grid.KNOWN_WEATHER_DEFECTS (matched on each
    missing day's own D-1 00 UTC run_init) -- reporting only, grants no
    tolerance (spec section 2.6: "nur im Bericht").
    """
    window_start = must_reach - dt.timedelta(days=window_days - 1)
    all_days = [window_start + dt.timedelta(days=i) for i in range(window_days)]
    required_sorted = sorted(required_features)

    def _complete(day: dt.date) -> bool:
        if not _is_day_complete(day, features, required_sorted):
            return False
        return not labels.reindex(_local_hours(day)).isna().any()

    complete = {day: _complete(day) for day in all_days}
    n_training_days = sum(complete.values())
    missing_days = tuple(day for day in all_days if not complete[day])

    age = 0
    cursor = must_reach
    while cursor >= window_start and not complete.get(cursor, False):
        age += 1
        cursor -= dt.timedelta(days=1)
    if cursor < window_start:
        age = (must_reach - window_start).days + 1

    known_weather_defect_days = tuple(
        day
        for day in missing_days
        if pd.Timestamp(day - dt.timedelta(days=1), tz="UTC") in KNOWN_WEATHER_DEFECTS
    )

    reasons: list[str] = []
    if n_training_days < min_training_days:
        reasons.append(
            f"only {n_training_days} complete training day(s) in the {window_days}-day window "
            f"ending {must_reach}, need >= {min_training_days}"
        )
    if age > max_edge_age_days:
        reasons.append(
            f"last complete training day is {age} day(s) before {must_reach}, "
            f"exceeds {max_edge_age_days}"
        )

    report = TrainingWindowReport(
        n_training_days=n_training_days,
        age_of_last_complete_day=age,
        missing_days=missing_days,
        known_weather_defect_days=known_weather_defect_days,
        tolerance_used=(min_training_days, max_edge_age_days),
    )
    return PreflightResult(ok=not reasons, reasons=tuple(reasons)), report


def check_renewables_label_edge_age(
    target_hourly: pd.DataFrame,
    *,
    must_reach: dt.date,
    max_edge_age_days: int = MAX_RNW_LABEL_EDGE_AGE_DAYS,
) -> tuple[PreflightResult, int]:
    """spec 6.9 section 2.7: whether the renewables reconstruction's own
    14.1.D training labels (``target_hourly``'s columns -- the caller
    passes evaluation.renewables_walkforward.TARGET_COLUMNS's values,
    read not imported for modification) reach close enough to
    ``must_reach`` for the walk-forward to have anything current to train
    on. Cheap (only reads already-in-memory actuals), so it runs before
    the walk-forward itself is attempted (spec section 5.1 step 5), the
    same "gate before the expensive work" shape as check_weather_run/
    check_capacity_anchor.

    A day is a complete label day when every one of ``target_hourly``'s
    columns is non-NaN for all of that day's local hours -- the same
    day-completeness definition as check_training_window's, applied to
    the raw actuals instead of a built feature matrix. Scans backward
    from ``must_reach``.

    ``MAX_RNW_LABEL_EDGE_AGE_DAYS`` is explicitly NOT measured (spec
    section 2.7), unlike check_training_window's two constants.
    """
    columns = list(target_hourly.columns)
    earliest_day = (
        pd.DatetimeIndex(target_hourly.index).tz_convert(LOCAL_TZ).normalize().min().date()
    )

    age = 0
    cursor = must_reach
    while cursor >= earliest_day and not _is_day_complete(cursor, target_hourly, columns):
        age += 1
        cursor -= dt.timedelta(days=1)
    if cursor < earliest_day:
        age = (must_reach - earliest_day).days + 1

    ok = age <= max_edge_age_days
    reasons: tuple[str, ...] = ()
    if not ok:
        reasons = (
            f"last complete renewables label day is {age} day(s) before {must_reach}, "
            f"exceeds MAX_RNW_LABEL_EDGE_AGE_DAYS={max_edge_age_days}",
        )
    return PreflightResult(ok=ok, reasons=reasons), age


def check_capacity_factor_bounds(capacity_factor: pd.Series) -> PreflightResult:
    """Capacity factor per target must lie in [0, 1] -- structural, a
    violation is a bug, not weather (spec section 2.6 layer 3). Sperrt.
    """
    out_of_bounds = capacity_factor[(capacity_factor < 0) | (capacity_factor > 1)]
    if not out_of_bounds.empty:
        first_idx = out_of_bounds.index[0]
        return PreflightResult(
            ok=False,
            reasons=(
                f"capacity factor out of [0, 1] at {len(out_of_bounds)} point(s), "
                f"e.g. {first_idx}={out_of_bounds.iloc[0]:.4f}",
            ),
        )
    return PreflightResult(ok=True)


def check_daily_sum_plausibility(
    daily_sum_mw: float,
    *,
    historical_month_sums_mw: pd.Series,
    margin: float = 0.5,
) -> PreflightResult:
    """Today's daily-sum output must lie within the historical spread for
    this calendar month, with a generous multiplicative margin (spec
    section 2.6 layer 3: "weit ist hier eine Eigenschaft, kein
    Kompromiss" -- a narrow band would second-guess the model, and a
    storm is not a bug). Sperrt.

    ``historical_month_sums_mw`` is the already-computed set of daily
    sums (MW) for this calendar month across the available history --
    computed by the caller from the real store data, never a hardcoded
    constant.
    """
    if historical_month_sums_mw.empty:
        return PreflightResult(
            ok=False, reasons=("no historical daily sums available to judge plausibility",)
        )
    lo = float(historical_month_sums_mw.min()) * (1 - margin)
    hi = float(historical_month_sums_mw.max()) * (1 + margin)
    if not (lo <= daily_sum_mw <= hi):
        return PreflightResult(
            ok=False,
            reasons=(
                f"daily sum {daily_sum_mw:.1f} MW outside the plausible band "
                f"[{lo:.1f}, {hi:.1f}] MW ({margin:.0%} margin over the historical "
                "range for this calendar month)",
            ),
        )
    return PreflightResult(ok=True)


def check_payload_plausibility(
    values: Sequence[float],
    *,
    historical_min: float,
    historical_max: float,
    margin_eur_mwh: float = 200.0,
) -> PreflightResult:
    """Wide plausibility band against the historical price distribution --
    catches "96x the same value" and "5000 EUR/MWh", not a real price
    spike (spec section 5.6, same logic as the renewables layer-3 checks
    above). Sperrt.

    Two independent failure modes, both checked: a degenerate (constant)
    payload -- the shape a wiring fault produces, not a real forecast --
    and a value outside the historical range plus a generous fixed
    margin.
    """
    reasons: list[str] = []
    unique_values = {round(v, 6) for v in values}
    if len(values) > 1 and len(unique_values) == 1:
        reasons.append(
            f"all {len(values)} value(s) identical ({values[0]:.2f} EUR/MWh) -- "
            "likely a wiring fault, not a real forecast"
        )

    lo = historical_min - margin_eur_mwh
    hi = historical_max + margin_eur_mwh
    out_of_band = [v for v in values if not (lo <= v <= hi)]
    if out_of_band:
        reasons.append(
            f"{len(out_of_band)} value(s) outside the plausibility band "
            f"[{lo:.1f}, {hi:.1f}] EUR/MWh, e.g. {out_of_band[0]:.1f}"
        )

    return PreflightResult(ok=not reasons, reasons=tuple(reasons))


def check_commodity_staleness(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    columns: tuple[str, ...] = ("ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"),
    warn_days: float = COMMODITY_STALENESS_WARN_DAYS,
) -> dict[str, float]:
    """Per-column age (in days, since the last non-NaN value) for every
    commodity column whose age is at or beyond ``warn_days`` (spec section
    2.3) -- never blocking, purely a named entry for the protocol and the
    run summary (spec section 5.7: "protokolliert, nicht sperrend").

    ``warn_days`` defaults to the ORIGINAL ffill limit (COMMODITY_STALENESS_
    WARN_DAYS = 4, deliberately the value data/loaders.py::COMMODITY_
    FFILL_LIMIT used before it was first raised to 7) -- this marks exactly
    the cases that would have been a silent day before that first change,
    not an arbitrarily chosen new threshold. Left unchanged by the second
    raise (7 to 14, spec 6.9 section 2.11): the gap between this warning and
    an actual silent day is now 10 days instead of 3, still deliberately
    the same warn_days value, not re-derived from the new ffill limit.

    Columns absent from ``df`` or entirely NaN are silently skipped (nothing
    to measure an age from) -- a genuinely dead feed already shows up as NaN
    in Check B once it exceeds the 14-day ffill limit (spec 6.9 section
    2.11, raised from 7), which is where it actually blocks.
    """
    warnings: dict[str, float] = {}
    for column in columns:
        if column not in df.columns:
            continue
        non_na = df[column].dropna()
        if non_na.empty:
            continue
        age_days = (as_of - non_na.index.max()) / pd.Timedelta(days=1)
        if age_days >= warn_days:
            warnings[column] = age_days
    return warnings
