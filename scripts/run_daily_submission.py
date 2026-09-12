"""Daily submission job entry point (spec 6.7.2, section 5.1) -- thin
caller over check/live_inputs/renewables/features/models/payload/protocol.
POST stays off in this step: submit() is never called with live=True (that
flip is 6.7.3's job).

Only the load-bearing pieces built so far live here; the full run sequence
(spec 5.1's twelve steps) is still under construction across this and
following sessions -- see docs/sprint6_step6_7_2_log.md.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import pandas as pd

from energy_price_forecast.arena.candidates import full_live_set_candidate_table, select_candidate
from energy_price_forecast.arena.catalog import ChallengeSpec, get_challenge
from energy_price_forecast.arena.live_inputs import (
    DEFAULT_WEATHER_MODEL,
    assemble_price_model_inputs,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.arena.payload import build_payload, validate_payload
from energy_price_forecast.arena.preflight import (
    PreflightResult,
    check_commodity_staleness,
    check_payload_plausibility,
    check_reconstruction_inputs,
    check_training_extent,
    known_defect_tolerance_hours,
)
from energy_price_forecast.arena.submit import SubmissionResult, submit
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
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.market_time import gate_closure_for_index
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

logger = logging.getLogger(__name__)

SUBMISSIONS_LOG = PROJECT_ROOT / "logs" / "submissions.jsonl"
PAYLOADS_DIR = PROJECT_ROOT / "logs" / "submission_payloads"

# "Day-Ahead Prices | Germany-Luxembourg | Point Forecast" -- matches
# ops/availability_audit.py's own ARENA_CHALLENGE_ID (spec section 5.4 there).
ARENA_CHALLENGE_ID = "2"

# The price model's own rolling training window (spec section 5.6, ops/availability_audit.py's
# existing TRAINING_WINDOW_DAYS).
PRICE_TRAIN_SPAN_DAYS = 90

# The renewables model's own rolling training window (spec section 5.3, identical to 6.5.2/6.6).
RENEWABLES_TRAIN_SPAN_DAYS = 365

# Preregistered shape-profile window (spec 6.4) -- always passed explicitly by every existing
# caller (scripts/backtest_arena.py's --shape-window-days has no default), so this is the one
# place the live path pins it, rather than inventing a new default elsewhere.
SHAPE_WINDOW_DAYS = 28


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
    )
    runtime_seconds = time.monotonic() - t0
    return predictions, runtime_seconds


def build_price_feature_matrix(
    df: pd.DataFrame, renewables_predictions: pd.DataFrame, target_day: dt.date
) -> tuple[pd.DataFrame, Fold, set[dt.date]]:
    """Build the price-model feature matrix for target_day's rolling
    PRICE_TRAIN_SPAN_DAYS-day training window plus target_day itself (spec
    section 5.4) -- features.build.build_feature_set_for_day is the one
    codepath backtest and live both call (Entscheidung 10), never a second
    live-only builder.

    The single Fold for target_day (train_index/test_index/gate_closure) is
    obtained from evaluation/walkforward.py::walk_forward_splits, the same
    function evaluation/arena_walkforward.py::run_live_gate_backtest uses,
    rather than re-deriving local-day/DST boundaries here.

    Days where build_feature_set_for_day raises IncompleteReconstructionError
    are excluded, not filled (spec 6.5.3 section 3.3's whole-day-out policy,
    the same handling evaluation/arena_walkforward.py::_build_matrix_over_days
    applies -- reimplemented here in three lines rather than importing that
    module's private helper, since evaluation/ stays untouched and this loop
    is too small to be worth reaching into another module's internals for).

    Raises ValueError if target_day itself has no complete rolling training
    window yet, or if every day in the needed range was excluded.
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
    fold = folds[0]

    first_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().min()
    all_days_needed = pd.date_range(
        first_train_day, pd.Timestamp(target_day, tz=LOCAL_TZ), freq="D", tz=LOCAL_TZ
    )

    frames = []
    excluded: set[dt.date] = set()
    for day in all_days_needed:
        try:
            frames.append(build_feature_set_for_day(day.date(), df, renewables_predictions))
        except IncompleteReconstructionError:
            excluded.add(day.date())
    if not frames:
        raise ValueError(f"no day produced a usable feature row for {target_day}")

    return pd.concat(frames).sort_index(), fold, excluded


def fit_predict_expand(
    df: pd.DataFrame, matrix: pd.DataFrame, fold: Fold, prices_qh: pd.DataFrame
) -> pd.Series:
    """Fit the price model on fold.train_index, predict fold.test_index's 24
    hourly values, then expand to quarter-hourly via the shape profile (spec
    section 5.6).

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
    """
    y_hourly = df["day_ahead_price"]
    x_train = matrix.reindex(fold.train_index).dropna()
    y_train = y_hourly.reindex(fold.train_index).loc[x_train.index]

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
    return expand_to_quarterhour(pred_hourly, profile, target_day=fold.delivery_day, tz=LOCAL_TZ)


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


def check_a_inputs(target_day: dt.date) -> PreflightResult:
    """Assemble Check A's already-loaded inputs and run it (spec section
    5.2) -- the cheap gate before any fit happens (spec section 2.1).

    Reads exactly one weather run (D-1's 00 UTC run, via the same pure disk
    reader arena.live_inputs.read_weather_runs uses internally) and the
    capacity anchor table's validity boundary. No fetch client, no network
    (spec section 4 rule 4) -- a missing or unreadable weather run surfaces
    as ``weather_run=None``, itself a Check A failure, not an exception.
    """
    run_init = run_init_for_target_day(target_day)
    weather_run = read_cached_run(cache_path(run_init, DEFAULT_WEATHER_MODEL, root=CACHE_ROOT))
    anchor_valid_until = anchor_table_valid_until(CapacitySource.PUBLIC_REGISTRY)
    return check_reconstruction_inputs(
        weather_run=weather_run,
        target_day=target_day,
        anchor_valid_until=anchor_valid_until,
        holiday_calendar_covers_target=holiday_calendar_covers_target(target_day),
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


def run_submission_for_day(
    df: pd.DataFrame,
    weather: pd.DataFrame,
    prices_qh: pd.DataFrame,
    target_day: dt.date,
    *,
    as_of: pd.Timestamp,
) -> SubmissionOutcome:
    """The prediction pipeline for one target day (spec section 5.1, steps
    4-10): Check A, renewables walk-forward, feature build, Check B plus the
    training-extent guard, candidate selection, price fit/predict/expand,
    payload build/validate/plausibility, and a dry-run submit() call
    (live=False always -- 6.7.3's job to flip).

    A failure at any check point returns immediately with ``skip_reason``
    set and ``candidate_selected=None`` -- a silent day is an expected
    operating state (spec section 2.7), never an exception from this
    function for an expected failure mode. An *unexpected* exception (a
    genuine bug, a raised IncompleteReconstructionError escaping past the
    per-day catch in build_price_feature_matrix, an LGBMForecaster failure)
    is deliberately NOT caught here -- it propagates to the caller, which
    is where "unerwartete Ausnahme" (spec section 2.7's second red
    condition, distinct from silence) belongs.

    ``as_of`` (the run's own wall-clock time) is used only for the
    non-blocking commodity-staleness measurement (spec section 2.3) --
    computed once, right after Check A, and carried in every returned
    SubmissionOutcome (including early skips), since it is diagnostic
    information about ``df``'s own freshness, independent of how far the
    run otherwise got.
    """
    check_a = check_a_inputs(target_day)
    if not check_a.ok:
        return SubmissionOutcome(
            candidate_selected=None, skip_reason="Check A: " + "; ".join(check_a.reasons)
        )

    commodity_staleness_warnings = check_commodity_staleness(df, as_of)

    renewables_predictions, renewables_runtime_seconds = run_renewables_step(
        df, weather, target_day
    )
    matrix, fold, excluded = build_price_feature_matrix(df, renewables_predictions, target_day)

    # Checked against the TRAINING portion only (target_day's own row is excluded here) --
    # must_reach is the training window's own last day, target_day - 1, not target_day itself.
    # Using the full matrix (with target_day's row always present) would make this check
    # unable to ever see a frozen training window, exactly the failure mode it exists to catch
    # (spec section 2.4).
    training_matrix = matrix.loc[matrix.index.isin(fold.train_index)]
    last_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().max().date()
    excluded_in_training_window = {day for day in excluded if day <= last_train_day}
    tolerance_hours = known_defect_tolerance_hours(excluded_in_training_window)
    extent_result = check_training_extent(
        training_matrix,
        expected_days=PRICE_TRAIN_SPAN_DAYS,
        must_reach=last_train_day,
        tolerated_missing_hours=tolerance_hours,
    )
    if not extent_result.ok:
        return SubmissionOutcome(
            candidate_selected=None,
            skip_reason="training extent: " + "; ".join(extent_result.reasons),
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=frozenset(excluded),
            commodity_staleness_warnings=commodity_staleness_warnings,
        )

    # Check B / candidate selection look only at target_day's own row(s) -- required by
    # check_target_row's own contract (NaN-checks the frame it's given, not a specific day
    # within it), confirmed against tests/test_candidates.py's fixtures, which pass single-row
    # frames. Passing the full training-plus-target matrix here would wrongly fail on any NaN
    # anywhere in 90 days of training history, not target_day's own freshness.
    candidates = full_live_set_candidate_table(frozenset(matrix.columns))
    selection = select_candidate(candidates, matrix.loc[fold.test_index], target_day)
    if selection.candidate is None:
        return SubmissionOutcome(
            candidate_selected=None,
            skip_reason="Check B: " + "; ".join(selection.result.reasons),
            missing_features=selection.result.missing_features,
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=frozenset(excluded),
            commodity_staleness_warnings=commodity_staleness_warnings,
        )

    quarterhourly_forecast = fit_predict_expand(df, matrix, fold, prices_qh)
    payload, challenge = build_arena_payload(quarterhourly_forecast, target_day)

    plausibility = check_payload_plausibility(
        payload["values"],
        historical_min=float(df["day_ahead_price"].min()),
        historical_max=float(df["day_ahead_price"].max()),
    )
    if not plausibility.ok:
        return SubmissionOutcome(
            candidate_selected=selection.candidate.name,
            skip_reason="payload plausibility: " + "; ".join(plausibility.reasons),
            renewables_runtime_seconds=renewables_runtime_seconds,
            excluded_training_days=frozenset(excluded),
            commodity_staleness_warnings=commodity_staleness_warnings,
        )

    # live=False always in 6.7.2 -- structurally never reaches submit.py's transport
    # (tests/test_arena_submit.py already proves this), so no dry-run branch is needed here.
    submission_result = submit(challenge, payload, live=False)

    return SubmissionOutcome(
        candidate_selected=selection.candidate.name,
        skip_reason=None,
        payload=payload,
        submission_result=submission_result,
        renewables_runtime_seconds=renewables_runtime_seconds,
        excluded_training_days=frozenset(excluded),
        commodity_staleness_warnings=commodity_staleness_warnings,
    )


def is_past_gate_closure(target_day: dt.date, as_of: pd.Timestamp) -> bool:
    """Spec section 5.1 step 1: a run at or after target_day's gate closure
    (12:00 Europe/Berlin on target_day - 1) must exit cleanly without
    attempting anything -- no run may assume it fires on schedule.
    """
    target_start_local, _ = local_day_bounds(target_day)
    gate_closure_utc = gate_closure_for_index(pd.DatetimeIndex([target_start_local]))[0]
    return bool(as_of >= gate_closure_utc)


def already_submitted(log_path: Path, target_day: dt.date) -> bool:
    """Spec section 5.1 step 2: idempotency. True if an earlier run already
    selected a candidate for target_day (a completed, non-skipped run) --
    spares the remaining minutes of an already-successful day, but is not
    itself a correctness guarantee (that is latest_before_deadline's job on
    the Arena side, spec section 5.1).
    """
    for record in read_submission_records(log_path):
        if record.get("target_day") == target_day.isoformat() and record.get("candidate_selected"):
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


def build_submission_record(
    outcome: SubmissionOutcome,
    manifest: store.Manifest,
    *,
    target_day: dt.date,
    nominal_slot: str,
    gate_closure_ok: bool,
    as_of: pd.Timestamp,
    runtime_seconds: float,
) -> SubmissionRecord:
    """Assemble one run's ops.protocol.SubmissionRecord (spec section 5.7)
    from this run's SubmissionOutcome plus run/store metadata. Commodity
    staleness warnings are folded into source_ages under their own column
    name -- both measure "how old is this data", just from two different
    vantage points (the store manifest's own covered_end_utc vs. the
    assembled df's actual last non-NaN value); the more precise, data-level
    measurement wins for the two commodity columns it covers.
    """
    run_id, run_url = run_id_and_url()
    source_ages = source_ages_from_manifest(manifest, as_of)
    source_ages.update(
        {column: age_days * 24 for column, age_days in outcome.commodity_staleness_warnings.items()}
    )
    n_values, payload_min, payload_mean, payload_max = _payload_stats(outcome.payload)
    submitted = outcome.submission_result.sent if outcome.submission_result is not None else False
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
        api_status="dry_run" if outcome.payload is not None else None,
        api_message=None,
        runtime_seconds=runtime_seconds,
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


def run_daily_submission(
    as_of: pd.Timestamp, *, nominal_slot: str, is_last_slot_of_day: bool
) -> int:
    """The full run sequence (spec section 5.1), taking the run's own
    wall-clock time as an explicit argument rather than reading it
    internally -- the same testability discipline as run_submission_for_day
    itself, so this can be driven by synthetic timestamps in tests without
    a real clock or a real store. main() is the only caller that reads the
    real clock and environment.
    """
    target_day = next_delivery_day(as_of)

    if is_past_gate_closure(target_day, as_of):
        logger.info(
            "gate closure for %s already passed at %s -- exiting cleanly", target_day, as_of
        )
        return 0

    if already_submitted(SUBMISSIONS_LOG, target_day):
        logger.info(
            "%s already has a submitted candidate from an earlier run today -- exiting cleanly",
            target_day,
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

    outcome = run_submission_for_day(df, weather, prices_qh, target_day, as_of=as_of)
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

    if outcome.skip_reason is not None:
        print(f"SILENT — {outcome.skip_reason}")
        # Red only on the day's last slot (spec section 2.7): an earlier run without
        # complete data is an expected operating state, not a failure.
        return 1 if is_last_slot_of_day else 0

    print(f"submitted (dry-run): candidate={outcome.candidate_selected}, target_day={target_day}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = pd.Timestamp.now("UTC")
    nominal_slot = os.environ.get("NOMINAL_SLOT", "manual")
    # Wired to the real slot schedule once ops/trigger/'s SLOTS table gains the three
    # submission slots (spec section 3.2) -- conservative default (False) means an
    # under-configured run never wrongly marks a day red on its own.
    is_last_slot_of_day = os.environ.get("IS_LAST_SLOT_OF_DAY", "false").lower() == "true"
    return run_daily_submission(
        as_of, nominal_slot=nominal_slot, is_last_slot_of_day=is_last_slot_of_day
    )


if __name__ == "__main__":
    sys.exit(main())
