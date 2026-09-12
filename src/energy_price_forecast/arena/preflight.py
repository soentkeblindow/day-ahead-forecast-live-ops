"""Preflight checks for the daily submission job (spec 6.7.2, sections
2.1-2.6, 5.2, 5.5).

Every function here is pure: no network, no file I/O (spec section 3.2 --
"Kein Netz, keine Dateien" -- fully testable without a store or a live
API). Callers (scripts/run_daily_submission.py) do the actual reading
(the store, a weather run file, the capacity anchor table) and pass
already-loaded values in; a short signature sketch in the spec itself
takes a StoreState directly, but this project's own precedent (6.7.1 A6:
"die Kurzsignaturen der Spec sind Skizzen, keine woertlichen Vertraege")
is to implement the fully testable, I/O-free version when the two are in
tension, not the literal pseudocode.

Two stages, and they cannot be fully upfront (spec section 2.1):

- Check A (check_reconstruction_inputs) -- cheap, before any fit. Can the
  renewables reconstruction for D even be built?
- Check B (check_target_row) -- after the feature matrix is built. Is the
  built row for D complete? This is the derived, not described, freshness
  check (spec section 2.2/2.3): staleness shows up as NaN exactly where
  it bites, not against a guessed per-source age table.

check_training_extent guards the one failure mode neither check sees: a
frozen source that silently shortens the training window's recent end
with no NaN anywhere (spec section 2.4 -- happened for real on
2026-09-11, see docs/sprint6_step6_7_1a_log.md). Its own zero-tolerance
row-count band would otherwise permanently block every training window
touching a documented, permanent gap (data.weather_grid.KNOWN_WEATHER_DEFECTS)
-- found live during the 6.7.2 wiring probe (2026-09-12): the 90-day window
for 2026-09-11 failed here purely because of the already-known 2026-06-23
corrupt weather run and its 2026-06-25 persistence-lag knock-on, both
already tracked, neither a new anomaly. known_defect_tolerance_hours()
extends the band by exactly the hours a documented defect (or a knock-on
chain rooted in one) explains -- never a blanket allowance for any
excluded day, which would mask a genuinely new, undocumented gap the same
way this check's zero-tolerance design exists to catch (the 2025-06-13
local-cache gap found the same session is exactly that kind of case, and
must still block).

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

import pandas as pd

from energy_price_forecast.data.loaders import COMMODITY_STALENESS_WARN_DAYS
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import KNOWN_WEATHER_DEFECTS
from energy_price_forecast.ops.store import validate_weather_run


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


def check_reconstruction_inputs(
    *,
    weather_run: pd.DataFrame | None,
    target_day: dt.date,
    anchor_valid_until: pd.Timestamp,
    holiday_calendar_covers_target: bool,
) -> PreflightResult:
    """Cheap gate before any fitting happens (spec section 2.1).

    Three binary questions, no guessed thresholds:

    1. Is the 00 UTC run of D-1 present AND does it pass
       ops.store.validate_weather_run()? Reusing that function is the
       point (spec section 2.6 layer 1) -- it is what would have caught
       the real HTTP-200-but-all-NaN weather file found in 6.7.1 A9.
       ``weather_run=None`` means the caller could not load it at all
       (missing file, or a raised WeatherRunUnavailable) -- itself a
       failure here, not a separate case to special-case.
    2. Is the capacity anchor table still valid for this run_init, i.e.
       is ``anchor_valid_until`` beyond it? Past that boundary the
       capacity denominator is extrapolated beyond its documented limit
       (currently 2026-10-28, data.capacity.anchor_table_valid_until).
    3. Does the holiday source cover the target day (spec section 2.11)?
       The one calendar input that can go wrong silently: a table that
       ends turns a real holiday into an ordinary weekday with no NaN and
       no error.

    Returns a result rather than raising: a failure here is a silent day,
    an expected operating state, not an error (spec section 2.7).
    """
    reasons: list[str] = []

    if weather_run is None:
        reasons.append("weather run for D-1 00 UTC could not be loaded")
    else:
        weather_result = validate_weather_run(weather_run)
        if not weather_result.ok:
            reasons.append(
                f"weather run for D-1 00 UTC failed validation: {'; '.join(weather_result.reasons)}"
            )

    run_init = run_init_for_target_day(target_day)
    if run_init >= anchor_valid_until:
        reasons.append(
            f"capacity anchor table not valid for run_init {run_init} "
            f"(valid until {anchor_valid_until})"
        )

    if not holiday_calendar_covers_target:
        reasons.append(f"holiday calendar does not cover target day {target_day}")

    return PreflightResult(ok=not reasons, reasons=tuple(reasons))


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


def known_defect_tolerance_hours(excluded_days: Sequence[dt.date] | set[dt.date]) -> int:
    """Hours of extra tolerance check_training_extent's row-count band
    should grant for training days already excluded by the whole-day-out
    policy AND already explained by a documented, permanent gap (spec
    2.4's own extent check would otherwise block every window touching
    one of these forever -- found live 2026-09-12, see this module's
    check_training_extent docstring).

    A day is explained if either its own D-1 run_init is a
    data.weather_grid.KNOWN_WEATHER_DEFECTS entry, or it is a knock-on of
    an immediately preceding day that is itself explained (the observed
    real chain: the 2026-06-23 corrupt run excludes delivery day
    2026-06-24, which then starves 2026-06-25's own persistence-lag
    feature of a D-1 value, excluding it too, with no weather defect of
    its own).

    Deliberately NOT a blanket allowance for any excluded day -- an
    excluded day whose chain never reaches a documented entry (e.g. the
    2025-06-13 local-cache gap found the same session, an operational
    hole, not a provider defect) contributes zero hours here and still
    trips check_training_extent, exactly as intended.
    """
    excluded = set(excluded_days)

    def _explained(day: dt.date) -> bool:
        run_init = pd.Timestamp(day - dt.timedelta(days=1), tz="UTC")
        if run_init in KNOWN_WEATHER_DEFECTS:
            return True
        previous_day = day - dt.timedelta(days=1)
        return previous_day in excluded and _explained(previous_day)

    return sum(24 for day in excluded if _explained(day))


def check_training_extent(
    features: pd.DataFrame,
    *,
    expected_days: int,
    must_reach: dt.date,
    tolerated_missing_hours: int = 0,
    tz: str = "Europe/Berlin",
) -> PreflightResult:
    """The training window must reach as far as it should and carry the
    expected number of rows (spec section 2.4).

    Guards the one failure mode check_target_row cannot see: a frozen
    source silently shortens the window's recent end with no NaN
    anywhere. Happened for real on 2026-09-11 (a carried-column
    validation failure froze `generation`, pre-6.7.1a) -- a 90-day window
    that quietly became 85 days would have trained without a single NaN
    to show for it.

    Two numbers, not a second full inspection (spec section 2.4): does
    the window's last local day reach ``must_reach``, and is the row
    count within a DST-tolerant band of ``expected_days * 24`` (a
    difference of more than 2 hours cannot be explained by a single DST
    transition inside the window and means real rows are missing) --
    widened by ``tolerated_missing_hours`` (the caller's own
    known_defect_tolerance_hours() result) for training days already
    excluded and already explained by a documented, permanent gap.

    ``expected_days``/``must_reach`` are read from the feature
    configuration and the run's own calendar by the caller, never set
    here (spec section 2.4: "aus der Konfiguration abzulesen").
    """
    index = pd.DatetimeIndex(features.index)
    local_days = index.tz_convert(tz).normalize().unique()
    max_day = local_days.max().date()
    n_rows = len(features)
    expected_rows_nominal = expected_days * 24

    reasons: list[str] = []
    if max_day < must_reach:
        reasons.append(
            f"training window ends at {max_day}, but must reach {must_reach} -- "
            "a source may have silently frozen (the real 2026-09-11 incident this "
            "check exists for)"
        )
    if abs(n_rows - expected_rows_nominal) > 2 + tolerated_missing_hours:
        extra = (
            f", +{tolerated_missing_hours}h for known weather defects"
            if tolerated_missing_hours
            else ""
        )
        reasons.append(
            f"training window has {n_rows} row(s), expected ~{expected_rows_nominal} "
            f"for a {expected_days}-day window (tolerance: 2h, one DST transition{extra})"
        )
    return PreflightResult(ok=not reasons, reasons=tuple(reasons))


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

    ``warn_days`` defaults to the OLD ffill limit (COMMODITY_STALENESS_WARN_DAYS
    = 4, deliberately the value data/loaders.py::COMMODITY_FFILL_LIMIT used
    before it was raised to 7) -- this marks exactly the cases that would
    have been a silent day before that change, not an arbitrarily chosen
    new threshold.

    Columns absent from ``df`` or entirely NaN are silently skipped (nothing
    to measure an age from) -- a genuinely dead feed already shows up as NaN
    in Check B once it exceeds the 7-day ffill limit, which is where it
    actually blocks.
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
