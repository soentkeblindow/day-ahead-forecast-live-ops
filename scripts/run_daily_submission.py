"""Daily submission job entry point (spec 6.7.2 section 5.1, spec 6.7.3
sections 2.4/5.4-5.6) -- thin caller over
check/live_inputs/renewables/features/models/payload/protocol.

The live switch (arena/config.py::is_live_enabled()) decides whether
submit() actually POSTs; main() is the only caller that reads it from the
real environment. The idempotency lock from 6.7.2 is gone (spec section
2.4): every run that produces a complete payload submits, and the Arena's
own latest_before_deadline selection policy means a later run's submission
simply overwrites an earlier one for the same target_day.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import holidays
import pandas as pd

from energy_price_forecast.arena.candidates import (
    Candidate,
    best_accepted_rank,
    may_submit,
    three_row_ladder,
)
from energy_price_forecast.arena.catalog import ChallengeSpec, get_challenge
from energy_price_forecast.arena.config import ARENA_CHALLENGE_ID, is_live_enabled
from energy_price_forecast.arena.live_inputs import (
    DEFAULT_WEATHER_MODEL,
    assemble_price_model_inputs,
    price_provenance_report,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.arena.load_patch import (
    LoadPatchReference,
    build_patched_load,
    choose_reference_day,
)
from energy_price_forecast.arena.payload import build_payload, validate_payload
from energy_price_forecast.arena.preflight import (
    CapacityAnchorReport,
    PreflightResult,
    check_capacity_anchor,
    check_commodity_staleness,
    check_holiday_calendar,
    check_payload_plausibility,
    check_renewables_label_edge_age,
    check_training_window,
    check_weather_run,
)
from energy_price_forecast.arena.submit import SubmissionResult, submit
from energy_price_forecast.arena.target_day_fill import (
    GroupFill,
    apply_forward_fill,
    apply_partial_fill,
    plan_target_day_fill,
)
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data._weather_cache import CACHE_ROOT, cache_path, read_cached_run
from energy_price_forecast.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    anchor_table_valid_until,
)
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.evaluation.renewables_walkforward import (
    TARGET_COLUMNS,
    run_renewables_backtest,
)
from energy_price_forecast.evaluation.walkforward import Fold, walk_forward_splits
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig
from energy_price_forecast.features.fundamentals import build_commodity_features
from energy_price_forecast.features.lags import build_price_lags
from energy_price_forecast.features.nwp_fundamentals import (
    IncompleteReconstructionError,
    build_nwp_forecast_fundamentals,
)
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import LGBMForecaster
from energy_price_forecast.ops import store
from energy_price_forecast.ops.protocol import (
    SubmissionRecord,
    append_submission_record,
    read_submission_records,
)
from energy_price_forecast.ops.store_sources import code_sha, run_id_and_url
from energy_price_forecast.ops.windows import LOCAL_TZ, local_day_bounds, next_delivery_day

# `python scripts/run_daily_submission.py` (the invocation submit.yml uses) puts
# scripts/ itself on sys.path, not the repo root, so `scripts` isn't importable
# as a package from here without this -- same fix as freeze_golden_fixture.py.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.measurement_a_candidate_intake import build_floor_for_day  # noqa: E402

logger = logging.getLogger(__name__)

SUBMISSIONS_LOG = PROJECT_ROOT / "logs" / "submissions.jsonl"
PAYLOADS_DIR = PROJECT_ROOT / "logs" / "submission_payloads"

# The price model's own rolling training window (spec section 5.6, ops/availability_audit.py's
# existing TRAINING_WINDOW_DAYS).
PRICE_TRAIN_SPAN_DAYS = 90

# The renewables model's own rolling training window (spec section 5.3, identical to 6.5.2/6.6).
RENEWABLES_TRAIN_SPAN_DAYS = 365

# Preregistered shape-profile window (spec 6.4) -- always passed explicitly by every existing
# caller (scripts/backtest_arena.py's --shape-window-days has no default), so this is the one
# place the live path pins it, rather than inventing a new default elsewhere.
SHAPE_WINDOW_DAYS = 28

# Restarbeit Teil C.3: named so a future policy change is a config/date
# change (spec section 2.7), not a silent code edit. Today's schedule only
# ever has at most one submission attempt remaining after "now" (0 once
# is_last_slot_of_day is True, 1 otherwise -- there is no real cross-run
# streak counter, spec section 2.7 doesn't ask for one). threshold=1 means
# that single remaining attempt is what turns a silent run red: an earlier
# silent slot still has a later slot that might recover, so it must not.
# _silence_turns_run_red reads this constant rather than the raw 1, so the
# constant genuinely decides the outcome.
SILENCE_STREAK_THRESHOLD: Final[int] = 1


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def renewables_window(target_day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The [start, end] UTC window target_hourly/weather must cover for
    run_renewables_backtest to produce exactly the folds spec section 2.9
    measures: the price model's PRICE_TRAIN_SPAN_DAYS-day training window
    plus target_day itself, each needing its own RENEWABLES_TRAIN_SPAN_DAYS
    days of prior training history.

    ``end`` is target_day's own LAST UTC hour, not UTC midnight of the same
    calendar date -- a real bug found live during the wiring probe
    (2026-09-12): Europe/Berlin local midnight is 22:00 UTC the day before
    (CEST) or 23:00 UTC the day before (CET), so ``pd.Timestamp(target_day,
    tz="UTC")`` cut the window off 2-3 hours into target_day's own local
    day. ``target_hourly``/``weather_window`` in run_renewables_step then
    silently lacked target_day's own remaining ~21-22 hours, which made
    features/nwp_fundamentals.py's whole-day-out policy correctly (from its
    own point of view) treat target_day as an incomplete reconstruction and
    exclude it -- observed on the real store as target_day itself showing
    up in build_price_feature_matrix's own ``excluded`` set, which should
    structurally never happen for the one day this whole run exists to
    predict. Fixed via ops/windows.py::local_day_bounds, the same boundary
    convention features/build.py::_hourly_utc_index_for_local_day already
    uses for exactly this reason.

    Deliberately NOT the full store history that
    arena.live_inputs.assemble_price_model_inputs()/read_weather_runs()
    return -- passing the whole history into run_renewables_backtest would
    silently multiply its walk-forward fold count (and runtime) far beyond
    what spec section 2.9 requires to be measured, not estimated.
    """
    _, next_local_day_start = local_day_bounds(target_day)
    end = next_local_day_start.tz_convert("UTC") - pd.Timedelta(hours=1)
    price_window_start = end - pd.Timedelta(days=PRICE_TRAIN_SPAN_DAYS)
    start = price_window_start - pd.Timedelta(days=RENEWABLES_TRAIN_SPAN_DAYS)
    return start, end


def run_renewables_step(
    df: pd.DataFrame, weather: pd.DataFrame, target_day: dt.date
) -> tuple[pd.DataFrame, float]:
    """Walk-forward renewables reconstruction for exactly the window the
    price model's training needs (spec section 5.3: refit_every=1, rolling
    365-day window, rolling365_l2 -- identical to 6.5.2/6.6), timed (spec
    section 2.9: the measured runtime, not the ~2.5-minute extrapolation, is
    what the operational log must carry).

    ``target_hourly`` is deliberately not read separately from the store
    (see arena/live_inputs.py's own module docstring): it is derived here as
    the TARGET_COLUMNS slice of the same merged, to_hourly()-normalised
    ``df`` the price model itself trains on, so both models see identically
    aggregated data.

    Returns the raw run_renewables_backtest output (unsliced to target_day
    -- the caller's feature-build step needs the whole windowed frame, not
    just the target day) and the measured wall-clock seconds.

    ``require_baseline=False`` (spec 6.9 section 2.13): the live path never
    scores a backtest baseline, so a fold whose D-1 label is missing (a
    real ENTSO-E outage) must not discard the prediction itself -- only
    the backtest caller (``scripts/train_renewables.py`` and friends) needs
    the default ``True`` behaviour, and does not go through this function.
    """
    start, end = renewables_window(target_day)
    target_hourly = df.loc[start:end, list(TARGET_COLUMNS.values())]
    run_inits = pd.DatetimeIndex(weather.index.get_level_values("run_init_utc"))
    weather_window = weather.loc[(run_inits >= start) & (run_inits <= end)]

    t0 = time.monotonic()
    predictions = run_renewables_backtest(
        target_hourly,
        weather_window,
        window="rolling",
        train_span_days=RENEWABLES_TRAIN_SPAN_DAYS,
        refit_every=1,
        objective="l2",
        source=CapacitySource.PUBLIC_REGISTRY,
        method=CapacityExtrapolation.LAST_INCREMENT,
        keep_rows_for=target_day,
        require_baseline=False,
    )
    runtime_seconds = time.monotonic() - t0
    return predictions, runtime_seconds


def _hourly_index_for_local_day(target_day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def target_day_fold(df: pd.DataFrame, target_day: dt.date) -> Fold:
    """The single Fold for target_day (train_index/test_index/gate_closure),
    obtained from evaluation/walkforward.py::walk_forward_splits (spec
    section 5.4) -- purely calendar/DST-derived from ``df``'s own hourly
    index, independent of which row's features get fit on it (spec 6.9
    section 2.6: "Zeile 2 hat dasselbe Trainingsfenster wie Zeile 1"; row 3
    shares the identical fold too, only its own required columns/builder
    differ). Computed once per run and shared by every row, rather than
    re-derived per candidate.

    Raises ValueError if target_day has no complete rolling training window
    at all -- a failure every row shares alike, so a caller may treat it as
    silence before ever reaching row-specific evaluation.
    """
    hourly_index = pd.DatetimeIndex(df.index)
    target_str = target_day.isoformat()
    folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=target_str,
            test_end=target_str,
            window="rolling",
            train_span_days=PRICE_TRAIN_SPAN_DAYS,
        )
    )
    if not folds:
        raise ValueError(
            f"no evaluable fold for {target_day} -- incomplete {PRICE_TRAIN_SPAN_DAYS}-day "
            "training window or missing test-day data"
        )
    return folds[0]


def build_price_feature_matrix(
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    target_day: dt.date,
    fold: Fold,
    *,
    builder: Callable[[dt.date, pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, set[dt.date]]:
    """Build one row's feature matrix for target_day's rolling
    PRICE_TRAIN_SPAN_DAYS-day training window plus target_day itself (spec
    section 5.4). ``builder`` defaults to features.build.build_feature_set_
    for_day -- the one codepath backtest and rows 1/2 both call (Entscheidung
    10) -- resolved at call time (``None`` sentinel, not a bound default
    argument) so a test patching the module-level name still takes effect
    without passing ``builder=`` explicitly. Row 3 (spec 6.9 section 2.1, no
    NWP dependency) passes an adapter around scripts.measurement_a_
    candidate_intake.build_floor_for_day instead (same signature shape,
    ``renewables_predictions`` ignored), reusing the exact function Schritt
    10's Parity Check already proved is a byte-identical column subset --
    never a second, hand-rewritten builder.

    Days where ``builder`` raises IncompleteReconstructionError are
    excluded, not filled (spec 6.5.3 section 3.3's whole-day-out policy) --
    ``build_floor_for_day`` never raises this (no NWP dependency at all), so
    row 3's own ``excluded`` set is effectively always empty in practice.

    Raises ValueError if every day in the needed range was excluded (target_
    day's own presence/absence in the returned matrix is the caller's job to
    check via the returned ``excluded`` set, not this function's).
    """
    build = builder if builder is not None else build_feature_set_for_day
    first_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().min()
    all_days_needed = pd.date_range(
        first_train_day, pd.Timestamp(target_day, tz=LOCAL_TZ), freq="D", tz=LOCAL_TZ
    )

    frames = []
    excluded: set[dt.date] = set()
    for day in all_days_needed:
        try:
            frames.append(build(day.date(), df, renewables_predictions))
        except IncompleteReconstructionError:
            excluded.add(day.date())
    if not frames:
        raise ValueError(f"no day produced a usable feature row for {target_day}")

    return pd.concat(frames).sort_index(), excluded


def fit_predict_expand(
    df: pd.DataFrame, matrix: pd.DataFrame, fold: Fold, prices_qh: pd.DataFrame
) -> tuple[pd.Series, int, int]:
    """Fit the price model on fold.train_index, predict fold.test_index's 24
    hourly values, then expand to quarter-hourly via the shape profile (spec
    section 5.6).

    Returns the quarter-hourly forecast plus ``(n_training_rows,
    n_training_labels)`` (Restarbeit Teil A) -- read off the exact
    ``x_train``/``y_train`` objects passed to ``model.fit()`` below, not a
    separately re-derived count, so the protocol can never report a number
    other than what was actually fitted (Restarbeit Teil A, P1).

    objective="quantile"/alpha=0.5 (the conditional median) stays the
    default per the 6.3 finding (CLAUDE.md: no robust case for switching);
    random_state=0/n_jobs=1 match scripts/backtest_arena.py's own CLI
    defaults, both already LGBMForecaster's own class defaults (spec section
    4 rule 2: fixed seed, no run-to-run variation).

    The whole-day-out policy means x_train can have fewer rows than
    fold.train_index (a training day excluded upstream reindexes to an
    all-NaN row here, dropped by dropna()) -- consistent with never silently
    filling incomplete NWP reconstruction, matching
    evaluation/arena_walkforward.py::run_live_gate_backtest's identical
    handling.

    A training-window hour whose own price is missing (docs/sprint6_fix_partial_today.md
    section 3.1: arena/live_inputs.py::assemble_price_model_inputs no longer
    drops such rows upstream, since other consumers may still need them) can
    now reach this function -- handled without any change needed here:
    models/lgbm.py::LGBMForecaster.fit already masks out any row with a
    NaN target before ever calling LightGBM (its own "missing features are
    LightGBM's job" comment), so a priceless training hour is silently
    excluded from training, never fabricated, never forward-filled, and
    never fed to LightGBM as a NaN target.
    """
    y_hourly = df["day_ahead_price"]
    x_train = matrix.reindex(fold.train_index).dropna()
    y_train = y_hourly.reindex(fold.train_index).loc[x_train.index]
    n_training_rows = len(x_train)
    n_training_labels = int(y_train.notna().sum())

    model = LGBMForecaster(objective="quantile", alpha=0.5, random_state=0, n_jobs=1)
    model.fit(y_train, x_train)

    x_test = matrix.loc[fold.test_index]
    history = y_hourly.loc[fold.train_index]
    pred_hourly = model.predict(fold.test_index, history=history, x_test=x_test)

    end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
    profile = fit_shape_profile(
        prices_qh["day_ahead_price"],
        end_day=end_day_minus_1,
        n_days=SHAPE_WINDOW_DAYS,
        tz=LOCAL_TZ,
    )
    forecast = expand_to_quarterhour(
        pred_hourly, profile, target_day=fold.delivery_day, tz=LOCAL_TZ
    )
    return forecast, n_training_rows, n_training_labels


def build_arena_payload(
    quarterhourly_forecast: pd.Series, target_day: dt.date
) -> tuple[dict[str, Any], ChallengeSpec]:
    """Fetch the challenge spec and build + structurally validate the dense
    payload for target_day (spec section 5.6).

    get_challenge() is a real, read-only Arena API call -- owner-approved
    (2026-09-12): matches ops/availability_audit.py's own daily
    ARENA_CHALLENGE_ID + get_challenge() call, already an established,
    accepted pattern. The guarantee this step actually needs is not "zero
    Arena calls" but two narrower, separately-tested properties: no
    ENTSO-E/weather fetch client is ever called (spec section 4 rule 4), and
    submit() is never invoked with live=True (arena/submit.py already makes
    that structurally impossible to reach _post -- see
    tests/test_arena_submit.py). Both tests belong to the full wiring probe,
    not here.

    Values are read off quarterhourly_forecast's own sorted output, never
    recomputed or re-rounded separately -- build_payload's own rounding to
    challenge.precision_decimals is the only rounding that happens, so the
    payload_min/mean/max the operational log eventually records (spec
    section 5.7) are always the numbers actually built, per that section's
    own "not what the code intended" rule.

    Raises PayloadValidationError (arena/payload.py) if the payload is
    structurally invalid. The wide plausibility band (spec section 5.6, same
    layer-3 logic as the renewables checks in arena/preflight.py) is a
    separate, non-raising check (check_payload_plausibility) left to the
    caller, since it needs the historical price range from ``df``, which
    this function does not take.
    """
    challenge = get_challenge(ARENA_CHALLENGE_ID)
    target_start = pd.Timestamp(target_day, tz=challenge.timezone)
    values = quarterhourly_forecast.sort_index().to_list()
    payload = build_payload(challenge, target_start, values)
    validate_payload(payload, challenge)
    return payload, challenge


def holiday_calendar_covers_target(target_day: dt.date) -> bool:
    """Whether the ``holidays`` package still produces holidays for
    target_day's year (spec section 2.11, Check A's third question).

    Empirically, the DE ruleset is computed (Easter offset + fixed dates,
    features/calendar.py::build_calendar_features), not a bounded lookup
    table with a hard cutoff -- verified directly: ``holidays.country_holidays
    ("DE", years=[2100])`` still returns a full, correct holiday set. There
    is therefore no realistic "table that ends" date for this source the way
    there is for the capacity anchor table or the weather archive.

    What this DOES catch, verified the same way: an implausible year (e.g.
    long before the modern German public-holiday calendar existed) returns
    an EMPTY result rather than raising -- exactly the silent-zero failure
    mode spec section 2.11 warns about ("kein NaN, kein Fehler, ein falscher
    Wert"). A real target_day for this job is always in the near future, so
    an empty result here would mean the ``holidays`` package itself
    regressed, not that this particular year is uncovered.
    """
    de = holidays.country_holidays("DE", years=[target_day.year])
    return len(de) > 0


def check_a_global(target_day: dt.date) -> PreflightResult:
    """spec 6.9 section 2.3/5.1 step 3 -- the one Check A question every
    candidate row alike depends on. A failure here is total silence before
    anything else runs (no renewables step, no fit): no future row can
    recover from a holiday-calendar gap, since every row's calendar
    feature depends on it.
    """
    return check_holiday_calendar(
        target_day=target_day,
        holiday_calendar_covers_target=holiday_calendar_covers_target(target_day),
    )


@dataclass(frozen=True)
class NwpAvailability:
    """spec 6.9 section 2.3/5.1 step 4: whether the renewables
    reconstruction can be built for ``target_day`` at all -- the combined
    result of the weather-run and capacity-anchor questions. Only
    candidate rows that need the NWP reconstruction are gated by
    ``ok`` (today, the only row; spec section 2.1's future gasfrei row will
    not be). ``anchor`` is carried separately so a caller can act on its
    ``warning``/expiry state even when ``ok`` is True.
    """

    ok: bool
    reasons: tuple[str, ...]
    anchor: CapacityAnchorReport


def check_a_nwp(
    target_day: dt.date,
    *,
    weather_root: Path = CACHE_ROOT,
    anchor_valid_until: pd.Timestamp | None = None,
) -> NwpAvailability:
    """Assemble Check A's NWP-dependent inputs and run them (spec section
    5.2) -- the cheap gate before any fit happens (spec section 2.1).

    Reads exactly one weather run (D-1's 00 UTC run, via the same pure disk
    reader arena.live_inputs.read_weather_runs uses internally) and the
    capacity anchor table's validity boundary. No fetch client, no network
    (spec section 4 rule 4) -- a missing or unreadable weather run surfaces
    as ``weather_run=None``, itself a failure, not an exception.

    ``weather_root`` defaults to the real local cache (module-level
    ``CACHE_ROOT``, unchanged behaviour for every existing caller) --
    overridable so scripts/outage_drill.py (spec 6.9 section 2.12) can
    point this at a copy of the published store instead. Found live
    2026-09-23: this was the one place in the whole read path that was NOT
    already parameterized like arena.live_inputs.assemble_price_model_inputs/
    read_weather_runs/read_quarterhourly_prices, so an early outage-drill
    run silently checked this machine's own (stale) local weather cache
    instead of the store copy it had just downloaded.

    ``anchor_valid_until`` (spec 6.9 section 6.3, Schritt 14): ``None``
    (every existing caller) computes the real boundary exactly as before --
    the anchor CSV itself is a committed repo file, not part of the
    downloaded store, so it has no store-copy equivalent for
    scripts/outage_drill.py to corrupt the way every other drill scenario
    does. An explicit override lets the ``anchor_expired`` drill scenario
    force an already-past boundary instead.
    """
    run_init = run_init_for_target_day(target_day)
    weather_run = read_cached_run(cache_path(run_init, DEFAULT_WEATHER_MODEL, root=weather_root))
    if anchor_valid_until is None:
        anchor_valid_until = anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)

    weather_result = check_weather_run(weather_run)
    anchor_result, anchor_report = check_capacity_anchor(target_day, anchor_valid_until)

    return NwpAvailability(
        ok=weather_result.ok and anchor_result.ok,
        reasons=weather_result.reasons + anchor_result.reasons,
        anchor=anchor_report,
    )


def check_a_renewables_labels(df: pd.DataFrame, target_day: dt.date) -> tuple[PreflightResult, int]:
    """spec 6.9 section 2.7: slices ``df`` to the renewables window's own
    TARGET_COLUMNS and runs check_renewables_label_edge_age against it -- a
    separate, named function (not inlined into run_submission_for_day) so
    tests can mock this one unit the same way they already mock
    check_a_nwp, rather than needing every ``df`` fixture to carry the
    three NWP-target columns whenever the label edge itself isn't what is
    under test.

    ``must_reach`` mirrors check_training_window's own Soll-Rand convention
    (target_day - 1): the renewables model's own most recent required
    training day.
    """
    window_start, window_end = renewables_window(target_day)
    target_hourly = df.loc[window_start:window_end, list(TARGET_COLUMNS.values())]
    return check_renewables_label_edge_age(
        target_hourly, must_reach=target_day - dt.timedelta(days=1)
    )


@dataclass(frozen=True)
class SubmissionOutcome:
    """Outcome of one run_submission_for_day() call (spec section 5.1,
    steps 4-10) -- the fields ops.protocol.SubmissionRecord needs from the
    prediction pipeline itself. Run metadata (run_id/run_url/code_sha,
    nominal_slot, gate_closure_ok, total runtime, the source-age dict) is
    the caller's job, not this function's: it needs real wall-clock time
    and environment variables this function deliberately never touches, so
    it can be tested end-to-end with synthetic frames alone (spec section
    7, "Kein Netz").

    Exactly one of ``candidate_selected``/``skip_reason`` is non-None,
    mirroring ops.protocol.SubmissionRecord's own contract. A run that
    failed Check A never reaches candidate selection, so it reports its
    reason here as ``skip_reason`` too (Check A's own reasons, not a
    separate field) -- consistent with spec section 5.7's "Check A's own
    reason belongs in skip_reason too".
    """

    candidate_selected: str | None
    skip_reason: str | None
    missing_features: tuple[str, ...] = ()
    payload: dict[str, Any] | None = None
    submission_result: SubmissionResult | None = None
    renewables_runtime_seconds: float | None = None
    excluded_training_days: frozenset[dt.date] = field(default_factory=frozenset)
    commodity_staleness_warnings: dict[str, float] = field(default_factory=dict)
    # Restarbeit Teil A: both None for any outcome returned before
    # fit_predict_expand runs (Check A/extent/Check B skip) -- fitting never
    # happened, so there is nothing real to report yet.
    n_training_rows: int | None = None
    n_training_labels: int | None = None
    # spec 6.7.3 section 2.5: set only by run_smoke_submission_for_day. candidate_selected
    # stays None for a smoke outcome -- it never goes through Check B/candidate selection,
    # so there is no real candidate name to report; submission_mode="smoke" (derived from
    # is_smoke in build_submission_record) is what actually marks the row, not this field.
    is_smoke: bool = False
    smoke_baseline_source_day: dt.date | None = None
    # spec 6.9 section 5.7, protocol_version 4 fields this step (Schritt 8) computes -- all
    # None/empty for any outcome returned before the corresponding check actually ran (spec
    # section 2.3: nwp_available/capacity_anchor_days_left only once check_a_nwp ran;
    # rnw_label_edge_age_days only once check_a_renewables_labels ran; the training-window
    # fields only once check_training_window ran).
    nwp_available: bool | None = None
    nwp_unavailable_reason: str | None = None
    capacity_anchor_days_left: float | None = None
    rnw_label_edge_age_days: int | None = None
    n_training_days: int | None = None
    age_of_last_complete_day: int | None = None
    training_tolerance_used: bool | None = None
    training_missing_days: tuple[dt.date, ...] = ()
    known_weather_defect_days: tuple[dt.date, ...] = ()
    # spec 6.9 section 5.7 (Protokoll v5) / Leitprinzip 3 ("jede Rückfall-Zeile ...
    # sichtbar"): one entry per row actually evaluated, rank order, {"name", "rank", "ok",
    # "reasons"} -- populated even on total silence, so a silent day still names every row's
    # own reason (spec section 5.1 step 7: "Trägt keine: Schweigen mit allen Gründen").
    candidates_evaluated: tuple[dict[str, Any], ...] = ()
    candidate_rank: int | None = None
    load_forecast_source: str | None = None
    load_patch_reference_day: dt.date | None = None
    load_patch_weeks_back: int | None = None
    load_patch_skipped: tuple[tuple[dt.date, str], ...] = ()
    # spec 6.9 section 2.8/5.7 (Schritt 12): one entry per feature group actually
    # forward-filled or partial-filled for the selected row's own target-day gap, empty
    # when target_day had no gaps at all -- ops.protocol.SubmissionRecord's own shape.
    target_day_fills: tuple[dict[str, Any], ...] = ()
    # spec 6.9 section 2.3: set True only when the capacity anchor table itself has expired
    # (a Pflegeversäumnis, not a data outage) -- run_daily_submission() reddens the run on this
    # alone, even if a lower row still submitted successfully (spec: "Zeile 3 reicht ein, aber
    # der Lauf ist rot").
    capacity_anchor_expired: bool = False
    # spec 6.9 section 2.9 (Downgrade-Schutz): set True when a row would have been selected but
    # a better-or-equal rank was already accepted live for target_day earlier today -- the run
    # stays green (spec: "endet der Lauf grün"), candidate_selected/payload are still populated
    # (the payload is archived regardless), but no submit() call is made.
    downgrade_blocked: bool = False
    best_accepted_rank_before: int | None = None


def determine_load_patch_reference(
    target_day: dt.date, df: pd.DataFrame
) -> LoadPatchReference | None:
    """Spec 6.9 section 5.1 step 7c -- only reached for row 2
    (core_gas_loadpatch). ``is_holiday``/``is_complete`` are injected per
    arena.load_patch's own contract (spec section 5.5): the holiday source
    is the identical nationwide-DE-only check features/calendar.py's
    is_holiday feature and holiday_calendar_covers_target above both use
    (spec section 2.2: "Die Feiertagsquelle ist dieselbe wie für die
    Kalender-Features"); ``is_complete`` reads the raw, un-coalesced ENTSO-E
    load_forecast_day_ahead column directly (spec section 2.2 point 2: "im
    Speicher"), not any candidate's own built feature matrix.
    """
    return choose_reference_day(
        target_day,
        is_holiday=lambda day: day in holidays.country_holidays("DE", years=[day.year]),
        is_complete=lambda day: bool(
            df["load_forecast_day_ahead"].reindex(_hourly_index_for_local_day(day)).notna().all()
        ),
    )


def build_row2_target_row(
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    target_day: dt.date,
    reference_day: dt.date,
) -> pd.DataFrame:
    """Spec 6.9 section 5.1 step 8 (the Last-Patch): patches ONLY
    target_day's own load_forecast_day_ahead cells via
    arena.load_patch.build_patched_load (the package's one implementation,
    spec section 3.2), then rebuilds target_day's row through the
    unmodified, unchanged build_feature_set_for_day -- "gleiche Features,
    gleiches Training" (spec section 2.1: row 2 is literally the same model
    as row 1). The patch is applied to a local copy of ``df`` that is
    discarded after this call returns -- no training row is ever touched
    (spec section 2.2: "Der Patch schreibt nie in Trainingszeilen").
    """
    patched_load = build_patched_load(df["load_forecast_day_ahead"], reference_day, target_day)
    patched_df = df.copy()
    patched_df.loc[patched_load.index, "load_forecast_day_ahead"] = patched_load.to_numpy()
    return build_feature_set_for_day(target_day, patched_df, renewables_predictions)


def nwp_group_columns_for_day(
    target_day: dt.date, df: pd.DataFrame, renewables_predictions: pd.DataFrame
) -> tuple[frozenset[str], frozenset[str]]:
    """The (load_forecast, nwp_residual) column-name split (spec section
    2.1/5.2) for rows 1/2, read off build_nwp_forecast_fundamentals's own
    six-feature output for target_day -- safe to call here since this is
    only reached once row1_row2_matrix's own build already succeeded for
    target_day (build_ladder's own docstring). A separate, named function
    (not inlined into run_submission_for_day) purely so tests can mock this
    one unit without needing a real renewables_predictions frame shaped
    exactly as build_nwp_forecast_fundamentals expects.
    """
    target_index = _hourly_index_for_local_day(target_day)
    fundamentals = build_nwp_forecast_fundamentals(df, renewables_predictions, target_index)
    return frozenset({fundamentals[0].name}), frozenset(f.name for f in fundamentals[1:])


def build_ladder(
    target_day: dt.date,
    df: pd.DataFrame,
    *,
    nwp_group_columns: tuple[frozenset[str], frozenset[str]] | None,
    cfg: FeatureConfig | None = None,
) -> tuple[Candidate, ...]:
    """Assembles arena.candidates.three_row_ladder with target_day's own
    real per-group column names -- never hand-enumerated (spec section
    2.1). calendar/price_lags/gas never touch NWP/renewables data at all
    (safe to derive regardless of NWP availability, the same independence
    row 3 itself relies on); ``nwp_group_columns`` (load_forecast,
    nwp_residual) is None when NWP is unavailable for target_day -- rows
    1/2 are always gated on ``requires_nwp`` before their (then empty,
    meaningless) required_columns() is ever consulted, so this is harmless.
    """
    cfg = cfg or FeatureConfig()
    target_index = _hourly_index_for_local_day(target_day)
    calendar_columns = frozenset(f.name for f in build_calendar_features(target_index, cfg))
    price_lag_columns = frozenset(f.name for f in build_price_lags(df, target_index, cfg))
    gas_columns = frozenset(f.name for f in build_commodity_features(df, target_index, cfg)[:1])
    load_forecast_columns, nwp_residual_columns = nwp_group_columns or (frozenset(), frozenset())
    return three_row_ladder(
        calendar_columns=calendar_columns,
        nwp_residual_columns=nwp_residual_columns,
        load_forecast_columns=load_forecast_columns,
        price_lag_columns=price_lag_columns,
        gas_columns=gas_columns,
    )


def _base_builder(day: dt.date, df: pd.DataFrame, _renewables: pd.DataFrame) -> pd.DataFrame:
    """Row 3's own builder adapter for build_price_feature_matrix's
    ``builder`` parameter (spec section 2.1: gas-free, no NWP dependency at
    all) -- reuses scripts.measurement_a_candidate_intake.build_floor_for_day
    unchanged (Schritt 10's Parity Check already proved it is a byte-
    identical column subset of build_feature_set_for_day), never a second,
    hand-rewritten builder. The ``_renewables`` argument only exists to
    match build_feature_set_for_day's own call shape -- build_floor_for_day
    takes no renewables data at all.
    """
    return build_floor_for_day(day, df, FeatureConfig(), commodities="none")


def run_submission_for_day(
    df: pd.DataFrame,
    weather: pd.DataFrame,
    prices_qh: pd.DataFrame,
    target_day: dt.date,
    *,
    as_of: pd.Timestamp,
    live: bool = False,
    now: Callable[[], pd.Timestamp] = _utcnow,
    weather_root: Path = CACHE_ROOT,
    anchor_valid_until: pd.Timestamp | None = None,
    submitted_records: Iterable[dict[str, object]] = (),
) -> SubmissionOutcome:
    """The prediction pipeline for one target day (spec 6.9 section 5.1) --
    Check A, the three-row fallback ladder (rank order: core_gas ->
    core_gas_loadpatch -> base), price fit/predict/expand, payload build/
    validate/plausibility, a second gate-closure check, downgrade
    protection, and submit() -- ``live`` is threaded straight through to it
    (arena/submit.py's own ``if not live: return`` is what actually keeps a
    dry run from sending, not a guard here).

    A failure at any check point (including the second gate-closure check)
    returns immediately with ``skip_reason`` set and ``candidate_selected``
    left as it was found -- a silent day is an expected operating state
    (spec section 2.7), never an exception from this function for an
    expected failure mode. An *unexpected* exception (a genuine bug, an
    LGBMForecaster failure) is deliberately NOT caught here -- it
    propagates to the caller, which is where "unerwartete Ausnahme" (spec
    section 2.7's second red condition, distinct from silence) belongs.

    ``as_of``/``now``/``weather_root`` are unchanged from before this step
    -- see the prior revision's own docstring for their reasoning.
    ``anchor_valid_until`` (Schritt 14): ``None`` for every real caller,
    forwarded straight to ``check_a_nwp`` -- see that function's own
    docstring for why the anchor table needs this instead of a
    ``weather_root``-style directory override.
    ``submitted_records`` is the already-read protocol log (spec section
    2.9's downgrade protection, arena.candidates.best_accepted_rank) -- read
    once by the caller (run_daily_submission), not by this function, so
    tests can pass a synthetic list without touching a real log file.
    """
    check_global = check_a_global(target_day)
    if not check_global.ok:
        return SubmissionOutcome(
            candidate_selected=None, skip_reason="Check A: " + "; ".join(check_global.reasons)
        )

    commodity_staleness_warnings = check_commodity_staleness(df, as_of)

    nwp = check_a_nwp(target_day, weather_root=weather_root, anchor_valid_until=anchor_valid_until)
    # spec 6.9 section 2.3: the anchor's own approaching-expiry warning is logged
    # independently of whether Check A as a whole passes -- a Pflegeversäumnis must stay
    # visible well before it can ever block anything.
    if nwp.anchor.warning:
        logger.warning(
            "capacity anchor table expires in %.1f day(s) (CAPACITY_ANCHOR_WARN_DAYS)",
            nwp.anchor.days_until_expiry,
        )
    # spec section 2.3: an expired anchor is its own, separately-visible red condition
    # (checked later against whichever row is finally selected, if any), distinct from an
    # ordinary weather-run failure -- both still make rows 1/2 unusable the same way.
    anchor_expired = not nwp.ok and any("capacity anchor table not valid" in r for r in nwp.reasons)

    common_fields: dict[str, Any] = {
        "commodity_staleness_warnings": commodity_staleness_warnings,
        "nwp_available": nwp.ok,
        "capacity_anchor_days_left": nwp.anchor.days_until_expiry,
    }
    if not nwp.ok:
        common_fields["nwp_unavailable_reason"] = "; ".join(nwp.reasons)

    # spec 6.9 section 5.1 step 6 ("Feature-Matrix bauen"): the fold (train_index/test_index)
    # is shared by every row, including row 3 -- computed here, after Check A, not before it
    # (spec's own step order), so a Check A failure never needs df to have a usable rolling
    # window at all.
    try:
        fold = target_day_fold(df, target_day)
    except ValueError as exc:
        return SubmissionOutcome(
            candidate_selected=None,
            skip_reason=str(exc),
            **common_fields,
        )

    nwp_usable = nwp.ok
    row1_row2_matrix: pd.DataFrame | None = None
    renewables_predictions: pd.DataFrame | None = None
    renewables_runtime_seconds: float | None = None
    excluded_training_days: frozenset[dt.date] = frozenset()
    nwp_group_columns: tuple[frozenset[str], frozenset[str]] | None = None
    nwp_unusable_reason = "; ".join(nwp.reasons) if not nwp.ok else None

    if nwp_usable:
        # spec 6.9 section 2.7: a cheap guard before the walk-forward itself.
        label_result, rnw_label_edge_age_days = check_a_renewables_labels(df, target_day)
        common_fields["rnw_label_edge_age_days"] = rnw_label_edge_age_days
        if not label_result.ok:
            nwp_usable = False
            nwp_unusable_reason = "; ".join(label_result.reasons)
        else:
            renewables_predictions, renewables_runtime_seconds = run_renewables_step(
                df, weather, target_day
            )
            try:
                row1_row2_matrix, excluded = build_price_feature_matrix(
                    df, renewables_predictions, target_day, fold
                )
            except ValueError as exc:
                nwp_usable = False
                nwp_unusable_reason = str(exc)
            else:
                excluded_training_days = frozenset(excluded)
                if target_day in excluded:
                    nwp_usable = False
                    nwp_unusable_reason = f"NWP reconstruction incomplete for {target_day} itself"
                else:
                    nwp_group_columns = nwp_group_columns_for_day(
                        target_day, df, renewables_predictions
                    )

    ladder = build_ladder(target_day, df, nwp_group_columns=nwp_group_columns)
    last_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().max().date()

    candidates_evaluated: list[dict[str, Any]] = []
    selected: Candidate | None = None
    selected_matrix: pd.DataFrame | None = None
    selected_fills: tuple[GroupFill, ...] = ()
    load_patch_reference: LoadPatchReference | None = None
    training_report = None

    for candidate in ladder:
        if candidate.requires_nwp and not nwp_usable:
            candidates_evaluated.append(
                {
                    "name": candidate.name,
                    "rank": candidate.rank,
                    "ok": False,
                    "reasons": (f"NWP unavailable: {nwp_unusable_reason}",),
                }
            )
            continue

        reference: LoadPatchReference | None = None
        if candidate.load_source == "similar_day":
            reference = determine_load_patch_reference(target_day, df)
            if reference is None:
                candidates_evaluated.append(
                    {
                        "name": candidate.name,
                        "rank": candidate.rank,
                        "ok": False,
                        "reasons": ("no usable Similar-Day-Patch reference day found",),
                    }
                )
                continue
            assert row1_row2_matrix is not None and renewables_predictions is not None
            row2_target_row = build_row2_target_row(
                df, renewables_predictions, target_day, reference.reference_day
            )
            row_matrix = row1_row2_matrix.copy()
            row_matrix.loc[fold.test_index, row2_target_row.columns] = row2_target_row.reindex(
                fold.test_index
            )
        elif candidate.rank == 3:
            try:
                row_matrix, base_excluded = build_price_feature_matrix(
                    df, pd.DataFrame(), target_day, fold, builder=_base_builder
                )
            except ValueError as exc:
                candidates_evaluated.append(
                    {
                        "name": candidate.name,
                        "rank": candidate.rank,
                        "ok": False,
                        "reasons": (str(exc),),
                    }
                )
                continue
            if target_day in base_excluded:
                candidates_evaluated.append(
                    {
                        "name": candidate.name,
                        "rank": candidate.rank,
                        "ok": False,
                        "reasons": (f"row 3's own build excluded {target_day}",),
                    }
                )
                continue
        else:
            assert row1_row2_matrix is not None
            row_matrix = row1_row2_matrix

        required = candidate.required_columns()
        row_matrix = row_matrix[sorted(required)]
        training_matrix = row_matrix.loc[row_matrix.index.isin(fold.train_index)]
        window_result, window_report = check_training_window(
            training_matrix,
            df["day_ahead_price"],
            required,
            must_reach=last_train_day,
            window_days=PRICE_TRAIN_SPAN_DAYS,
        )
        if not window_result.ok:
            candidates_evaluated.append(
                {
                    "name": candidate.name,
                    "rank": candidate.rank,
                    "ok": False,
                    "reasons": window_result.reasons,
                }
            )
            continue

        fill_plan = plan_target_day_fill(row_matrix.loc[fold.test_index], candidate)
        if isinstance(fill_plan, PreflightResult):
            candidates_evaluated.append(
                {
                    "name": candidate.name,
                    "rank": candidate.rank,
                    "ok": False,
                    "reasons": fill_plan.reasons,
                    "missing_features": fill_plan.missing_features,
                }
            )
            continue
        row_matrix = apply_forward_fill(row_matrix, fill_plan)

        selected = candidate
        selected_matrix = row_matrix
        selected_fills = fill_plan
        training_report = window_report
        load_patch_reference = reference
        candidates_evaluated.append(
            {"name": candidate.name, "rank": candidate.rank, "ok": True, "reasons": ()}
        )
        break

    training_window_fields: dict[str, Any] = {}
    if training_report is not None:
        training_window_fields = {
            "n_training_days": training_report.n_training_days,
            "age_of_last_complete_day": training_report.age_of_last_complete_day,
            "training_tolerance_used": bool(training_report.missing_days),
            "training_missing_days": training_report.missing_days,
            "known_weather_defect_days": training_report.known_weather_defect_days,
        }
    load_patch_fields: dict[str, Any] = {}
    if load_patch_reference is not None:
        load_patch_fields = {
            "load_patch_reference_day": load_patch_reference.reference_day,
            "load_patch_weeks_back": load_patch_reference.weeks_back,
            "load_patch_skipped": load_patch_reference.skipped,
        }
    target_day_fills = tuple(
        {
            "group": fill.group,
            "n_hours": len(fill.hours),
            "hours": [h.isoformat() for h in fill.hours],
            "action": fill.action,
        }
        for fill in selected_fills
    )

    if selected is None or selected_matrix is None:
        # spec section 2.5: every offending column named individually, across every row
        # tried (deduplicated, first-seen order) -- not just the last one, since a silent
        # day may be silent for three different reasons at once.
        missing_features: list[str] = []
        for e in candidates_evaluated:
            for column in e.get("missing_features", ()):
                if column not in missing_features:
                    missing_features.append(column)
        return SubmissionOutcome(
            candidate_selected=None,
            skip_reason="Check B: "
            + "; ".join(f"{e['name']}: {'; '.join(e['reasons'])}" for e in candidates_evaluated),
            missing_features=tuple(missing_features),
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=excluded_training_days,
            candidates_evaluated=tuple(candidates_evaluated),
            capacity_anchor_expired=anchor_expired,
            **common_fields,
            **training_window_fields,
        )

    quarterhourly_forecast, n_training_rows, n_training_labels = fit_predict_expand(
        df, selected_matrix, fold, prices_qh
    )
    if any(fill.action == "partial_fill" for fill in selected_fills):
        persistence = persistence_forecast(
            prices_qh["day_ahead_price"], pd.Timestamp(target_day, tz=LOCAL_TZ)
        )
        quarterhourly_forecast = apply_partial_fill(
            quarterhourly_forecast, persistence, selected_fills
        )
    payload, challenge = build_arena_payload(quarterhourly_forecast, target_day)

    plausibility = check_payload_plausibility(
        payload["values"],
        historical_min=float(df["day_ahead_price"].min()),
        historical_max=float(df["day_ahead_price"].max()),
    )
    if not plausibility.ok:
        return SubmissionOutcome(
            candidate_selected=selected.name,
            candidate_rank=selected.rank,
            load_forecast_source=selected.load_source,
            skip_reason="payload plausibility: " + "; ".join(plausibility.reasons),
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=excluded_training_days,
            candidates_evaluated=tuple(candidates_evaluated),
            n_training_rows=n_training_rows,
            n_training_labels=n_training_labels,
            capacity_anchor_expired=anchor_expired,
            target_day_fills=target_day_fills,
            **common_fields,
            **training_window_fields,
            **load_patch_fields,
        )

    # spec 6.7.3 section 2.4: re-check the gate closure immediately before the POST -- a
    # run whose fit/predict/build pass took long enough to cross the deadline must not
    # send late. A skip here still carries candidate_selected AND the built payload (spec
    # section 5.4 step 11: the payload archive runs regardless of whether anything sent).
    if is_past_gate_closure(target_day, now()):
        return SubmissionOutcome(
            candidate_selected=selected.name,
            candidate_rank=selected.rank,
            load_forecast_source=selected.load_source,
            skip_reason="gate closure passed before POST",
            payload=payload,
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=excluded_training_days,
            candidates_evaluated=tuple(candidates_evaluated),
            n_training_rows=n_training_rows,
            n_training_labels=n_training_labels,
            capacity_anchor_expired=anchor_expired,
            target_day_fills=target_day_fills,
            **common_fields,
            **training_window_fields,
            **load_patch_fields,
        )

    # spec 6.9 section 2.9 (Downgrade-Schutz): a later slot must not overwrite an
    # already-accepted, better-or-equal-rank live submission for target_day.
    best_rank = best_accepted_rank(submitted_records, target_day, ladder)
    if not may_submit(selected.rank, best_rank):
        return SubmissionOutcome(
            candidate_selected=selected.name,
            candidate_rank=selected.rank,
            load_forecast_source=selected.load_source,
            skip_reason=f"KEPT -- rank {best_rank} already accepted, this run: rank {selected.rank}",
            payload=payload,
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=excluded_training_days,
            candidates_evaluated=tuple(candidates_evaluated),
            n_training_rows=n_training_rows,
            n_training_labels=n_training_labels,
            capacity_anchor_expired=anchor_expired,
            downgrade_blocked=True,
            best_accepted_rank_before=best_rank,
            target_day_fills=target_day_fills,
            **common_fields,
            **training_window_fields,
            **load_patch_fields,
        )

    submission_result = submit(challenge, payload, live=live)

    return SubmissionOutcome(
        candidate_selected=selected.name,
        candidate_rank=selected.rank,
        load_forecast_source=selected.load_source,
        skip_reason=None,
        payload=payload,
        submission_result=submission_result,
        renewables_runtime_seconds=renewables_runtime_seconds,
        excluded_training_days=excluded_training_days,
        candidates_evaluated=tuple(candidates_evaluated),
        n_training_rows=n_training_rows,
        n_training_labels=n_training_labels,
        capacity_anchor_expired=anchor_expired,
        best_accepted_rank_before=best_rank,
        target_day_fills=target_day_fills,
        **common_fields,
        **training_window_fields,
        **load_patch_fields,
    )


def is_past_gate_closure(target_day: dt.date, as_of: pd.Timestamp) -> bool:
    """Spec section 5.1 step 1: a run at or after target_day's gate closure
    (12:00 Europe/Berlin on target_day - 1) must exit cleanly without
    attempting anything -- no run may assume it fires on schedule.
    """
    target_start_local, _ = local_day_bounds(target_day)
    gate_closure_utc = gate_closure_for_index(pd.DatetimeIndex([target_start_local]))[0]
    return bool(as_of >= gate_closure_utc)


def accepted_earlier_today(log_path: Path, target_day: dt.date) -> bool:
    """Spec 6.7.3 section 2.4: whether some earlier run already got a
    submission accepted for target_day. Read only for the last slot's own
    day-coloring decision in run_daily_submission, never as a lock -- the
    idempotency lock 6.7.2 used to apply here is gone (every run that
    produces a complete payload submits; a later run's submission simply
    overwrites an earlier one on the Arena side via
    latest_before_deadline). A smoke-mode line never satisfies this: it
    targets D+2, never the same target_day as a regular run (spec section
    2.5), so no special-casing is needed here.
    """
    for record in read_submission_records(log_path):
        if record.get("target_day") == target_day.isoformat() and record.get("submitted") is True:
            return True
    return False


def source_ages_from_manifest(
    manifest: store.Manifest, as_of: pd.Timestamp
) -> dict[str, float | None]:
    """Per-manifest-source age in hours since ``covered_end_utc``, at
    ``as_of`` (spec section 5.7's ``source_ages`` field) -- logged
    unconditionally, never gates a run (spec section 2.3).
    """
    ages: dict[str, float | None] = {}
    for name, entry in manifest.sources.items():
        if entry.covered_end_utc is None:
            ages[name] = None
            continue
        covered_end = pd.Timestamp(entry.covered_end_utc)
        ages[name] = (as_of - covered_end) / pd.Timedelta(hours=1)
    return ages


def _payload_stats(
    payload: dict[str, Any] | None,
) -> tuple[int | None, float | None, float | None, float | None]:
    """Read n/min/mean/max off the payload actually built, never off what
    the code intended to build (spec section 5.7 -- the generalised lesson
    from 6.6's _log_gate_verdict resolution-mixup finding).
    """
    if payload is None:
        return None, None, None, None
    values = payload["values"]
    if not values:
        return 0, None, None, None
    return len(values), min(values), sum(values) / len(values), max(values)


def _submission_fields(
    result: SubmissionResult | None,
) -> tuple[bool, str | None, str | None, str | None, int | None, str | None, bool | None]:
    """Map a SubmissionResult onto the protocol's submitted/api_status/
    api_message/submission_mode/submission_id/api_response_received_utc/
    confirmed_via_query fields (spec 6.7.3 sections 5.4-5.6).

    Returns (submitted, api_status, api_message, submission_mode,
    submission_id, api_response_received_utc, confirmed_via_query). All
    None/False if no submit() call happened at all -- an early skip, or the
    second gate-closure check firing right before the POST.
    """
    if result is None:
        return False, None, None, None, None, None, None
    if not result.sent:
        return False, "dry_run", None, "dry_run", None, None, None
    if result.accepted:
        return (
            True,
            result.status or "accepted",
            result.message,
            "live",
            result.submission_id,
            result.response_received_utc,
            result.confirmed_via_query,
        )
    return (
        False,
        result.error_kind,
        result.message,
        "live",
        result.submission_id,
        result.response_received_utc,
        result.confirmed_via_query,
    )


def _jsonable_candidate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One SubmissionOutcome.candidates_evaluated entry, JSON-line-safe
    (spec section 5.7): ``reasons``/``missing_features`` are tuples inside
    run_submission_for_day (matching arena.preflight.PreflightResult's own
    convention) but must be lists for json.dumps."""
    out = dict(entry)
    out["reasons"] = list(out.get("reasons", ()))
    if "missing_features" in out:
        out["missing_features"] = list(out["missing_features"])
    return out


def build_submission_record(
    outcome: SubmissionOutcome,
    manifest: store.Manifest,
    *,
    target_day: dt.date,
    nominal_slot: str,
    gate_closure_ok: bool,
    as_of: pd.Timestamp,
    runtime_seconds: float,
    price_provenance: dict[str, str] | None = None,
    price_source_conflicts: int | None = None,
) -> SubmissionRecord:
    """Assemble one run's ops.protocol.SubmissionRecord (spec section 5.7)
    from this run's SubmissionOutcome plus run/store metadata. Commodity
    staleness warnings are folded into source_ages under their own column
    name -- both measure "how old is this data", just from two different
    vantage points (the store manifest's own covered_end_utc vs. the
    assembled df's actual last non-NaN value); the more precise, data-level
    measurement wins for the two commodity columns it covers.

    ``price_provenance``/``price_source_conflicts`` (spec 6.9 section 5.3/
    5.7, Schritt 7) are optional, additive keywords -- omitted, the record
    keeps SubmissionRecord's own defaults ({}/None), same as any run that
    predates this step. The caller computes them (arena.live_inputs.py::
    price_provenance_report) rather than this function reading raw price
    data itself, keeping this assembly function a pure read of already-
    computed facts, same discipline as every other field here.
    """
    run_id, run_url = run_id_and_url()
    source_ages = source_ages_from_manifest(manifest, as_of)
    source_ages.update(
        {column: age_days * 24 for column, age_days in outcome.commodity_staleness_warnings.items()}
    )
    n_values, payload_min, payload_mean, payload_max = _payload_stats(outcome.payload)
    (
        submitted,
        api_status,
        api_message,
        submission_mode,
        submission_id,
        api_response_received_utc,
        confirmed_via_query,
    ) = _submission_fields(outcome.submission_result)
    # spec 6.7.3 section 2.5: submission_mode="smoke" overrides whatever _submission_fields
    # derived from the SubmissionResult alone (a smoke run always calls submit(live=True), so
    # it would otherwise read "live" -- is_smoke is the one authoritative signal for this).
    if outcome.is_smoke:
        submission_mode = "smoke"
    smoke_baseline_source_day = (
        outcome.smoke_baseline_source_day.isoformat()
        if outcome.smoke_baseline_source_day is not None
        else None
    )
    return SubmissionRecord(
        run_timestamp_utc=as_of.isoformat(),
        run_id=run_id,
        run_url=run_url,
        code_sha=code_sha(),
        target_day=target_day.isoformat(),
        nominal_slot=nominal_slot,
        gate_closure_ok=gate_closure_ok,
        source_ages=source_ages,
        candidate_selected=outcome.candidate_selected,
        skip_reason=outcome.skip_reason,
        missing_features=list(outcome.missing_features),
        n_values=n_values,
        payload_min=payload_min,
        payload_mean=payload_mean,
        payload_max=payload_max,
        submitted=submitted,
        api_status=api_status,
        api_message=api_message,
        runtime_seconds=runtime_seconds,
        n_training_rows=outcome.n_training_rows,
        n_training_labels=outcome.n_training_labels,
        submission_mode=submission_mode,
        submission_id=submission_id,
        api_response_received_utc=api_response_received_utc,
        confirmed_via_query=confirmed_via_query,
        smoke_baseline_source_day=smoke_baseline_source_day,
        price_provenance=price_provenance or {},
        price_source_conflicts=price_source_conflicts,
        nwp_available=outcome.nwp_available,
        nwp_unavailable_reason=outcome.nwp_unavailable_reason,
        capacity_anchor_days_left=outcome.capacity_anchor_days_left,
        rnw_label_edge_age_days=outcome.rnw_label_edge_age_days,
        n_training_days=outcome.n_training_days,
        age_of_last_complete_day=outcome.age_of_last_complete_day,
        training_tolerance_used=outcome.training_tolerance_used,
        training_missing_days=[d.isoformat() for d in outcome.training_missing_days],
        known_weather_defect_days=[d.isoformat() for d in outcome.known_weather_defect_days],
        candidate_rank=outcome.candidate_rank,
        candidates_evaluated=[_jsonable_candidate_entry(e) for e in outcome.candidates_evaluated],
        load_forecast_source=outcome.load_forecast_source,
        downgrade_blocked=outcome.downgrade_blocked,
        best_accepted_rank_before=outcome.best_accepted_rank_before,
        load_patch_reference_day=(
            outcome.load_patch_reference_day.isoformat()
            if outcome.load_patch_reference_day is not None
            else None
        ),
        load_patch_weeks_back=outcome.load_patch_weeks_back,
        load_patch_skipped=[[d.isoformat(), reason] for d, reason in outcome.load_patch_skipped],
        target_day_fills=list(outcome.target_day_fills),
    )


def write_payload_to_repo(payload: dict[str, Any], target_day: dt.date) -> Path:
    """Dry-run: write the built payload into the repo instead of sending it
    (spec section 5.1 step 11). One archived JSON file per target_day --
    later runs for the same day overwrite it, since latest_before_deadline
    means only the latest run's payload for a day is the one that would
    have counted.
    """
    PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = PAYLOADS_DIR / f"{target_day.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _silence_turns_run_red(*, is_last_slot_of_day: bool) -> bool:
    """Whether a silent outcome should turn the whole workflow run red
    (spec section 2.7) -- reads SILENCE_STREAK_THRESHOLD rather than
    returning is_last_slot_of_day directly, so the constant is what
    genuinely decides this, not documentation next to an unrelated check.
    remaining_attempts is 0 on the day's last slot, 1 otherwise. threshold=1
    (default) reddens exactly the last slot's own silence, as before this
    was named; threshold=2 (>= the max remaining_attempts this schedule can
    ever report) would redden EVERY silent run, including non-final ones;
    threshold<=0 would never redden any silent run at all -- both directions
    provable by patching the module constant in a test, not just asserted.
    """
    remaining_attempts = 0 if is_last_slot_of_day else 1
    return remaining_attempts < SILENCE_STREAK_THRESHOLD


def run_daily_submission(
    as_of: pd.Timestamp, *, nominal_slot: str, is_last_slot_of_day: bool, live: bool = False
) -> int:
    """The full run sequence (spec section 5.1; spec 6.7.3 section 2.4),
    taking the run's own wall-clock time as an explicit argument rather
    than reading it internally -- the same testability discipline as
    run_submission_for_day itself, so this can be driven by synthetic
    timestamps in tests without a real clock or a real store. main() is the
    only caller that reads the real clock, environment, and live switch.

    No idempotency lock (spec section 2.4 -- see accepted_earlier_today's
    own docstring): every run that reaches candidate selection attempts a
    submission, live or dry-run.
    """
    target_day = next_delivery_day(as_of)

    if is_past_gate_closure(target_day, as_of):
        logger.info(
            "gate closure for %s already passed at %s -- exiting cleanly", target_day, as_of
        )
        return 0

    t0 = time.monotonic()
    store_state = store.load_store(PROJECT_ROOT)

    df = assemble_price_model_inputs()
    window_start, window_end = renewables_window(target_day)
    local_days = pd.date_range(
        window_start.tz_convert(LOCAL_TZ).normalize(),
        window_end.tz_convert(LOCAL_TZ).normalize(),
        freq="D",
    )
    weather = read_weather_runs([d.date() for d in local_days])
    prices_qh = read_quarterhourly_prices()
    # spec 6.9 section 2.9 (Downgrade-Schutz): read once here, not inside
    # run_submission_for_day itself, so tests can pass a synthetic list.
    submitted_records = read_submission_records(SUBMISSIONS_LOG)

    # Restarbeit Teil C.4: an unexpected exception here (spec section 2.7's
    # second red condition, distinct from silence) is deliberately NOT
    # swallowed -- run_submission_for_day's own docstring already commits to
    # letting it propagate, since that's where a genuine bug belongs, not a
    # handled outcome. This only adds a summary line distinguishable from
    # "SILENT -- ..." (a config/coding-not-input-state failure looks
    # different in the Actions log without reading the full traceback) and
    # re-raises unchanged -- exit code, traceback, and CI redness are
    # exactly as before this addition.
    try:
        outcome = run_submission_for_day(
            df,
            weather,
            prices_qh,
            target_day,
            as_of=as_of,
            live=live,
            submitted_records=submitted_records,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised unchanged below
        print(f"EXCEPTION — {type(exc).__name__}: {exc}")
        raise
    runtime_seconds = time.monotonic() - t0
    price_provenance, price_source_conflicts = price_provenance_report(as_of=as_of)

    record = build_submission_record(
        outcome,
        store_state.manifest,
        target_day=target_day,
        nominal_slot=nominal_slot,
        gate_closure_ok=True,
        as_of=as_of,
        runtime_seconds=runtime_seconds,
        price_provenance=price_provenance,
        price_source_conflicts=price_source_conflicts,
    )
    append_submission_record(SUBMISSIONS_LOG, record)

    if outcome.payload is not None:
        write_payload_to_repo(outcome.payload, target_day)

    # Restarbeit Teil A.4: only surface this when it's actually nonzero -- a
    # "0 missing labels" line every single day is noise nobody reads after a
    # week, and the full numbers are already in logs/submissions.jsonl for
    # 6.8 regardless of whether this line printed.
    if (
        outcome.n_training_rows is not None
        and outcome.n_training_labels is not None
        and outcome.n_training_rows > outcome.n_training_labels
    ):
        missing = outcome.n_training_rows - outcome.n_training_labels
        print(
            f"training labels: {missing} of {outcome.n_training_rows} "
            "training row(s) have no real day_ahead_price label"
        )

    # spec 6.9 section 2.9 (Downgrade-Schutz): a later, worse-or-would-be-equal-but-blocked
    # rank never reddens the run -- an earlier, better-or-equal rank is already accepted and
    # archived for target_day (spec: "endet der Lauf grün").
    if outcome.downgrade_blocked:
        print(f"KEPT — {outcome.skip_reason}")
        return 0

    if outcome.skip_reason is not None:
        print(f"SILENT — {outcome.skip_reason}")
        # spec 6.9 section 2.3: an expired capacity anchor table is a Pflegeversäumnis, not
        # an ordinary data outage -- it reddens the run even on a day with no complete rows.
        if outcome.capacity_anchor_expired:
            return 1
        # spec 6.7.3 section 2.4: the last slot reads the protocol (never a lock) --
        # a day with an already-accepted submission from an earlier run stays green
        # even if this, the final, run itself stayed silent.
        if is_last_slot_of_day and accepted_earlier_today(SUBMISSIONS_LOG, target_day):
            return 0
        # Red only on the day's last slot otherwise (spec section 2.7): an earlier run
        # without complete data is an expected operating state, not a failure.
        return 1 if _silence_turns_run_red(is_last_slot_of_day=is_last_slot_of_day) else 0

    response_exit = _print_response_summary(outcome, target_day)
    if response_exit is not None:
        # spec 6.9 section 2.3: "Zeile 3 reicht ein, aber der Lauf ist rot" -- an expired
        # anchor table reddens even an otherwise-successful submission (never downgrades an
        # already-red response_exit, e.g. a rejected live POST).
        if outcome.capacity_anchor_expired and response_exit == 0:
            print("FALLBACK — capacity anchor table expired")
            return 1
        return response_exit

    print(
        f"submitted (dry-run): candidate={outcome.candidate_selected}"
        f"{_candidate_summary_suffix(outcome)}, target_day={target_day}"
    )
    if outcome.capacity_anchor_expired:
        print("FALLBACK — capacity anchor table expired")
        return 1
    return 0


def _print_response_summary(outcome: SubmissionOutcome, target_day: dt.date) -> int | None:
    """Spec 6.7.3 section 5.5's response-evaluation print/exit-code logic --
    shared between the regular (run_daily_submission) and smoke
    (run_daily_submission_smoke) run sequences. Returns None if no submit()
    call was made at all (an early skip, or a dry run), leaving the
    dry-run message to the caller.
    """
    result = outcome.submission_result
    if result is None or not result.sent:
        return None

    if not result.accepted:
        # A rejected or transport-failed live POST makes this run red immediately, on
        # every slot -- distinct from "SILENT", a genuine attempt that failed, not data
        # that wasn't ready yet.
        print(
            f"{(result.error_kind or 'error').upper()} — target {target_day}, "
            f"status {result.http_status}, message: {result.message}"
        )
        return 1

    n_values = len(outcome.payload["values"]) if outcome.payload is not None else None
    print(
        f"SUBMITTED — target {target_day}, {outcome.candidate_selected}"
        f"{_candidate_summary_suffix(outcome)}, {n_values} values, "
        f"status {result.status}, id {result.submission_id}"
    )
    return 0


def _candidate_summary_suffix(outcome: SubmissionOutcome) -> str:
    """Spec 6.9 section 5.7's SUBMITTED example: ``core_gas_loadpatch (rank
    2, FALLBACK: load_forecast missing, reference 2026-10-01)`` -- names why
    every higher-ranked row failed and, for row 2, which reference day the
    patch used. Rank 1 (no fallback happened) gets no FALLBACK clause.
    Returns "" for a smoke outcome (candidate_rank is always None there).
    """
    if outcome.candidate_rank is None:
        return ""
    if outcome.candidate_rank == 1:
        return f" (rank {outcome.candidate_rank})"
    prior_reasons = "; ".join(
        f"{e['name']}: {'; '.join(e['reasons'])}"
        for e in outcome.candidates_evaluated
        if not e["ok"] and e["rank"] < outcome.candidate_rank
    )
    reference = (
        f", reference {outcome.load_patch_reference_day}"
        if outcome.load_patch_reference_day is not None
        else ""
    )
    return f" (rank {outcome.candidate_rank}, FALLBACK: {prior_reasons}{reference})"


def smoke_target_day(as_of: pd.Timestamp) -> dt.date:
    """Spec 6.7.3 section 2.5: the smoke test always targets D+2, never
    D+1 -- a smoke dispatch only makes sense after 12:00 local (D+1's own
    gate closure has already passed by then, spec section 2.5's own
    reasoning), and D+2's window opens 3 days before its deadline.
    """
    return next_delivery_day(as_of) + dt.timedelta(days=1)


def run_smoke_submission_for_day(target_day: dt.date) -> SubmissionOutcome:
    """Spec 6.7.3 sections 2.5/5.4: the smoke test's own prediction step.
    Steps 4-10 of the regular run (spec section 5.1) do not apply here --
    no renewables walk-forward, no feature build, no model fit. The
    payload is the existing 6.4 baseline replica (persistence_forecast),
    built from target_day's own D-1 (target_day - 1) realised
    quarter-hourly prices -- the day whose day-ahead auction has already
    cleared "today" by the time a smoke dispatch can run.

    Sends unconditionally: live=True regardless of ARENA_LIVE (spec
    section 2.5 -- the manual mode=smoke dispatch IS the deliberate human
    authorization, the whole point of a smoke test).
    """
    source_day = target_day - dt.timedelta(days=1)
    challenge = get_challenge(ARENA_CHALLENGE_ID)
    prices_qh = read_quarterhourly_prices()
    baseline = persistence_forecast(
        prices_qh["day_ahead_price"], pd.Timestamp(target_day), tz=challenge.timezone
    )
    target_start = pd.Timestamp(target_day, tz=challenge.timezone)
    payload = build_payload(challenge, target_start, baseline.sort_index().to_list())
    validate_payload(payload, challenge)

    submission_result = submit(challenge, payload, live=True)

    return SubmissionOutcome(
        candidate_selected=None,
        skip_reason=None,
        payload=payload,
        submission_result=submission_result,
        is_smoke=True,
        smoke_baseline_source_day=source_day,
    )


def run_daily_submission_smoke(as_of: pd.Timestamp, *, nominal_slot: str = "smoke") -> int:
    """The smoke test's own full run sequence (spec 6.7.3 sections
    2.5/5.4) -- the gate-closure check still runs first (step 1,
    unchanged in shape, just against smoke_target_day instead of
    next_delivery_day), then straight to baseline/submit/protocol. No
    idempotency/day-coloring concept applies here (spec section 2.5: a
    smoke dispatch is a one-off manual action, not one of the day's
    regular slots). main() is the only caller that reads MODE from the
    real environment.
    """
    target_day = smoke_target_day(as_of)

    if is_past_gate_closure(target_day, as_of):
        logger.info(
            "gate closure for %s already passed at %s -- exiting cleanly", target_day, as_of
        )
        return 0

    t0 = time.monotonic()
    store_state = store.load_store(PROJECT_ROOT)

    try:
        outcome = run_smoke_submission_for_day(target_day)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised unchanged below
        print(f"EXCEPTION — {type(exc).__name__}: {exc}")
        raise
    runtime_seconds = time.monotonic() - t0

    record = build_submission_record(
        outcome,
        store_state.manifest,
        target_day=target_day,
        nominal_slot=nominal_slot,
        gate_closure_ok=True,
        as_of=as_of,
        runtime_seconds=runtime_seconds,
    )
    append_submission_record(SUBMISSIONS_LOG, record)

    if outcome.payload is not None:
        write_payload_to_repo(outcome.payload, target_day)

    response_exit = _print_response_summary(outcome, target_day)
    if response_exit is not None:
        return response_exit

    # Should not happen -- run_smoke_submission_for_day always calls submit(live=True).
    print(f"submitted (dry-run): candidate={outcome.candidate_selected}, target_day={target_day}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = pd.Timestamp.now("UTC")
    mode = os.environ.get("MODE", "scheduled")
    if mode == "smoke":
        return run_daily_submission_smoke(as_of)

    nominal_slot = os.environ.get("NOMINAL_SLOT", "manual")
    # Wired to the real slot schedule once ops/trigger/'s SLOTS table gains the three
    # submission slots (spec section 3.2) -- conservative default (False) means an
    # under-configured run never wrongly marks a day red on its own.
    is_last_slot_of_day = os.environ.get("IS_LAST_SLOT_OF_DAY", "false").lower() == "true"
    live = is_live_enabled(os.environ)
    return run_daily_submission(
        as_of, nominal_slot=nominal_slot, is_last_slot_of_day=is_last_slot_of_day, live=live
    )


if __name__ == "__main__":
    sys.exit(main())
