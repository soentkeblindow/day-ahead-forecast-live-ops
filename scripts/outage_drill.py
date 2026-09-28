"""Outage drill / as-of reproduction harness (spec 6.9, sections 2.12 and
6.3) -- runs the real submission pipeline against a COPY of the published
store, with an injected ``as_of``, ``live=False``, and the protocol record
written to a temporary file, never the real ``logs/submissions.jsonl``.

Two roles, one tool (deliberately -- section 2.12: "Gate-Werkzeug zuerst"):

- The "As-of-Lauf" gate every build step in this spec's step-by-step live
  rollout uses (section 2.12): ``--scenario none``, ``as_of`` pinned to
  11:45 Berlin today (or yesterday if run before 10:00 -- resolve_as_of),
  target_day derived the same way the real submission job derives it
  (ops.windows.next_delivery_day). Passes if a payload clears the
  validator, the expected candidate is selected, and -- for every build
  step that is not meant to change the prediction (every step except the
  ones marked B* in the spec's own section 10 table) -- the resulting
  payload is bit-identical to the archived live payload for the same
  target_day.
- The vehicle for section 6.3's outage drills (added in step 13): a
  ``scenario`` corrupts/removes files in the store COPY before the
  pipeline reads it, so the ``SCENARIOS`` registry is the one place a
  later step adds a new named fault, never the harness itself.

Never calls a fetch client (spec section 4 rule 4): the "copy" step is
``ops.store.load_store()`` into a fresh temp directory -- the same
read-only download this project's own maintenance jobs already do, not a
new kind of network access. Nothing here writes back to the published
store (no ``publish_store()`` call anywhere in this module) and nothing
here can reach ``arena.submit.submit(live=True)`` (``run_submission_for_day``
is always called with ``live=False``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pandas as pd

from energy_price_forecast.arena.live_inputs import (
    DEFAULT_WEATHER_MODEL,
    assemble_price_model_inputs,
    price_provenance_report,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data._weather_cache import CACHE_ROOT, cache_path
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.ops import store
from energy_price_forecast.ops.protocol import append_submission_record
from energy_price_forecast.ops.store_sources import (
    COMMODITIES_DIR,
    ENERGY_CHARTS_DIR,
    ENTSOE_SOURCES,
    EntsoeSource,
)
from energy_price_forecast.ops.windows import LOCAL_TZ, local_day_bounds, next_delivery_day
from scripts.run_daily_submission import (
    PAYLOADS_DIR,
    build_submission_record,
    renewables_window,
    run_submission_for_day,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("outage_drill")


@dataclasses.dataclass
class CopiedStore:
    """The store copy's own source bindings -- same shape as
    ops.store_sources.ENTSOE_SOURCES/COMMODITIES_DIR/data._weather_cache.CACHE_ROOT,
    just re-rooted under a temp workdir instead of PROJECT_ROOT. Every
    reader in arena.live_inputs already accepts these as explicit keyword
    overrides (the established wiring-probe pattern, e.g.
    scripts/backtest_arena.py's own path overrides) -- no monkeypatching of
    module-level constants anywhere.

    ``anchor_valid_until`` (Schritt 14, not frozen any more so a scenario can
    set it): the capacity anchor table has no store-copy file to corrupt --
    it's a committed repo file, not part of the downloaded store -- so
    ``_scenario_anchor_expired`` is the one scenario that signals its fault
    through this field instead of mutating a file under ``workdir``.
    """

    workdir: Path
    entsoe_sources: tuple[EntsoeSource, ...]
    commodities_dir: Path
    weather_root: Path
    energy_charts_dir: Path
    anchor_valid_until: pd.Timestamp | None = None


def _copy_sources_for_workdir(workdir: Path) -> CopiedStore:
    """Re-root ENTSOE_SOURCES/COMMODITIES_DIR/CACHE_ROOT/ENERGY_CHARTS_DIR
    under ``workdir`` by relativizing against PROJECT_ROOT -- derives the
    paths rather than hard-coding them a second time, so a future new
    source in ops/store_sources.py is picked up automatically."""
    entsoe_sources = tuple(
        dataclasses.replace(s, cache_dir=workdir / s.cache_dir.relative_to(PROJECT_ROOT))
        for s in ENTSOE_SOURCES
    )
    commodities_dir = workdir / COMMODITIES_DIR.relative_to(PROJECT_ROOT)
    weather_root = workdir / CACHE_ROOT
    energy_charts_dir = workdir / ENERGY_CHARTS_DIR.relative_to(PROJECT_ROOT)
    return CopiedStore(
        workdir=workdir,
        entsoe_sources=entsoe_sources,
        commodities_dir=commodities_dir,
        weather_root=weather_root,
        energy_charts_dir=energy_charts_dir,
    )


ScenarioFn = Callable[[CopiedStore, dt.date], None]


def _scenario_none(copied: CopiedStore, target_day: dt.date) -> None:
    """No fault injected -- the baseline drill and the As-of-Lauf gate
    every build step in this spec's rollout uses (spec section 2.12). Most
    of spec section 6.3's eleven named fault scenarios (ENTSO-E cutoffs, a
    removed weather run, a frozen source, an expired anchor table, ...) are
    step 13's own job -- adding a new entry to SCENARIOS below, never a
    change to run_outage_drill itself."""


def _scenario_load_missing(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 10, Schritt 11's own As-of-Lauf requirement ("Zusätzlich
    die Übung 'Last für D fehlt' -> Zeile 2", one of spec section 6.3's
    eleven drills, pulled forward to this step since Schritt 11's own gate
    needs it now rather than waiting for step 13's full suite): NaNs out
    target_day's own local hours of ``load_forecast_day_ahead`` in the
    copied "load" source's monthly cache file -- the whole-day gap that
    makes row 1 (core_gas) fail Check B and row 2 (core_gas_loadpatch) take
    over via the Similar-Day-Patch (arena/load_patch.py).
    """
    load_source = next(s for s in copied.entsoe_sources if s.name == "load")
    load_cache_path = (
        load_source.cache_dir / f"DE_LU_{target_day.year:04d}-{target_day.month:02d}.parquet"
    )
    df = pd.read_parquet(load_cache_path)
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
        df.index = index
    start, end = local_day_bounds(target_day)
    mask = (index >= start.tz_convert("UTC")) & (index < end.tz_convert("UTC"))
    df.loc[mask, "load_forecast_day_ahead"] = float("nan")
    df.to_parquet(load_cache_path)


def _scenario_load_gap(copied: CopiedStore, target_day: dt.date, hours: int) -> None:
    """Shared implementation for the two partial-load-gap drills below: NaNs
    the trailing ``hours`` local hours of target_day's own
    ``load_forecast_day_ahead`` (spec section 2.8: "gemessen sind nur
    zusammenhängende Randlücken am Tagesende") -- a partial, not whole-day,
    gap, unlike ``_scenario_load_missing``.
    """
    load_source = next(s for s in copied.entsoe_sources if s.name == "load")
    load_cache_path = (
        load_source.cache_dir / f"DE_LU_{target_day.year:04d}-{target_day.month:02d}.parquet"
    )
    df = pd.read_parquet(load_cache_path)
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
        df.index = index
    start, end = local_day_bounds(target_day)
    local_hours = pd.date_range(start, end, freq="h", inclusive="left")
    gap_start = local_hours[-hours].tz_convert("UTC")
    mask = (index >= gap_start) & (index < end.tz_convert("UTC"))
    df.loc[mask, "load_forecast_day_ahead"] = float("nan")
    df.to_parquet(load_cache_path)


def _scenario_load_gap_3h(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 10, Schritt 12's own As-of-Lauf requirement (spec section
    6.3 table: "3-Stunden-Lücke ENTSO-E-Last -> Fortschreiben, Zeile 1"): a
    3-hour trailing gap is within the ``load_forecast`` group's
    ``FFILL_MAX_HOURS`` (3), so ``arena.target_day_fill`` forward-fills it
    and row 1 (core_gas) stays selected -- unlike ``_scenario_load_missing``'s
    whole-day gap, which always routes to row 2.
    """
    _scenario_load_gap(copied, target_day, 3)


def _scenario_load_gap_4h(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 10, Schritt 12's own As-of-Lauf requirement (spec section
    6.3 table: "4-Stunden-Lücke ENTSO-E-Last -> Zeile 1 scheitert, Zeile 2"):
    one hour more than ``_scenario_load_gap_3h``'s own gap, past
    ``FFILL_MAX_HOURS``, so the ``load_forecast`` group fails row 1 and row 2
    (core_gas_loadpatch) takes over via the Similar-Day-Patch -- the same
    row-2 outcome as ``_scenario_load_missing``, reached via a partial rather
    than whole-day gap.
    """
    _scenario_load_gap(copied, target_day, 4)


def _scenario_weather_missing(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3's "Wetterlauf für D entfernt" drill (pulled forward
    to this step for the same reason _scenario_load_missing was, see its
    own docstring): deletes the D-1 00 UTC weather run file from the copied
    store -- read_cached_run returns None for a missing file, check_a_nwp
    treats that as a Check A NWP failure, so both row 1 (core_gas) and row
    2 (core_gas_loadpatch) are unreachable (both require_nwp) and row 3
    (base, no NWP dependency at all) is the only one left standing. Exists
    to prove row 3's own build/fit/predict/payload path -- never exercised
    by the "none" or "load_missing" scenarios, both of which resolve at
    rank 1 or 2 -- actually works end to end against real data.
    """
    run_init = run_init_for_target_day(target_day)
    path = cache_path(run_init, DEFAULT_WEATHER_MODEL, root=copied.weather_root)
    path.unlink()


def _year_months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Every (year, month) touched by [start, end], inclusive -- cache files
    are monthly-partitioned, so a gap/cutoff window crossing a month
    boundary needs every file in range, not just the one ``start`` falls
    into."""
    months: list[tuple[int, int]] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_marker = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= end_marker:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def _blank_rows(df: pd.DataFrame, mask: pd.Series | Any, column: str | None) -> pd.DataFrame:
    """NaNs either one named column or every column (``column=None``, an
    entire-source cutoff) for the rows ``mask`` selects -- shared by every
    scenario below that mutates a cached parquet file in place."""
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
        df = df.copy()
        df.index = index
    if column is None:
        df.loc[mask, :] = float("nan")
    else:
        df.loc[mask, column] = float("nan")
    return df


def _blank_monthly_files(
    cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp, *, column: str | None = None
) -> None:
    """Blanks every monthly-partitioned cache file under ``cache_dir``
    (globbed by its trailing ``_YYYY-MM.parquet`` suffix -- works for both
    the single-file-per-month sources and the per-neighbor multi-file
    sources like cross_border_flows/scheduled_exchanges, which never share a
    file naming assumption with this function) for rows in ``[start, end)``.
    """
    for year, month in _year_months_between(start, end):
        for path in sorted(cache_dir.glob(f"*_{year:04d}-{month:02d}.parquet")):
            df = pd.read_parquet(path)
            index = pd.DatetimeIndex(df.index)
            if index.tz is None:
                index = index.tz_localize("UTC")
                df.index = index
            mask = (index >= start) & (index < end)
            if not mask.any():
                continue
            df = _blank_rows(df, mask, column)
            df.to_parquet(path)


def _blank_single_file(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, *, column: str | None = None
) -> None:
    """Same as ``_blank_monthly_files`` for a source stored as one file
    covering its whole history (the two commodities, the two Energy-Charts
    sources) rather than monthly partitions."""
    df = pd.read_parquet(path)
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
        df.index = index
    mask = (index >= start) & (index < end)
    if not mask.any():
        return
    df = _blank_rows(df, mask, column)
    df.to_parquet(path)


def _scenario_entsoe_cutoff(
    copied: CopiedStore, target_day: dt.date, days_before_target: int
) -> None:
    """Shared implementation for the three "ENTSO-E ab D-N abgeschnitten"
    drills (spec section 6.3): blanks every column of five of the six
    ENTSO-E sources from ``target_day - days_before_target`` onward, through
    a couple of days past target_day (a real outage doesn't self-heal inside
    this drill's own observation window) -- simulates a total ENTSO-E outage
    starting at that point, except the price itself.

    ``day_ahead_price`` is deliberately excluded (owner decision, this
    session): a real ENTSO-E outage still leaves the price recoverable via
    Energy-Charts' own independent feed (coalesce_price, spec section 5.3),
    so cutting it here would conflate two different failure modes. Cutting
    it would also mask this drill's own point -- the spec table expects row
    2 for a D-1 cutoff, which needs the *renewables* reconstruction to keep
    working even though its own ENTSO-E training labels (``generation``/
    ``wind_solar``) are gone; a first version of this scenario cut all six
    sources including the price and landed on row 3 instead, for a reason
    unrelated to what this drill is supposed to exercise.
    """
    cutoff = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") - pd.Timedelta(
        days=days_before_target
    )
    end = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=2)
    for source in copied.entsoe_sources:
        if source.name == "day_ahead_price":
            continue
        _blank_monthly_files(source.cache_dir, cutoff, end, column=None)


def _scenario_entsoe_cutoff_d1(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "ENTSO-E (alle Quellen) ab D-1 abgeschnitten ->
    Preise aus EC, Zeile 2 mit Referenz eine Woche zurück, Trainingsalter 1".
    """
    _scenario_entsoe_cutoff(copied, target_day, 1)


def _scenario_entsoe_cutoff_d3(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "ENTSO-E ab D-3 abgeschnitten -> Zeile 2,
    Renewables-Fit läuft (rnw_label_edge_age_days = 3)"."""
    _scenario_entsoe_cutoff(copied, target_day, 3)


def _scenario_entsoe_cutoff_d9(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "ENTSO-E ab D-9 abgeschnitten -> Grenzen
    überschritten, Zeile 3" -- 9 days exceeds both MAX_TRAINING_EDGE_AGE_DAYS
    (7) and MAX_RNW_LABEL_EDGE_AGE_DAYS (7)."""
    _scenario_entsoe_cutoff(copied, target_day, 9)


def _scenario_ttf_dead_15d(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "TTF 15 Tage tot -> Zeilen 1 und 2 scheitern, Zeile
    3" -- 15 days exceeds COMMODITY_FFILL_LIMIT (14, spec section 6.9 §2.11),
    so the gas STRICT group in rows 1/2 stays NaN for target_day; row 3
    (gas-free per Fassung 2) is unaffected.
    """
    cutoff = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") - pd.Timedelta(days=15)
    end = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=2)
    _blank_single_file(copied.commodities_dir / "ttf_gas.parquet", cutoff, end, column=None)


def _scenario_price_gap(copied: CopiedStore, target_day: dt.date, hours: int | None) -> None:
    """Shared implementation for the two "Preis-Lücke in beiden Quellen"
    drills: gaps the trailing ``hours`` local hours of D-1 (``hours=None`` ->
    the whole day) in BOTH the ENTSO-E ``day_ahead_price`` cache and the EC
    ``day_ahead_price_ec`` cache. Both sources must be gapped at the
    identical hours -- coalesce_price only falls back to EC when ENTSO-E is
    missing, so gapping only one source would leave the merged series (what
    price_lags actually reads) fully covered.
    """
    d_minus_1 = target_day - dt.timedelta(days=1)
    start, end = local_day_bounds(d_minus_1)
    if hours is None:
        gap_start = start.tz_convert("UTC")
    else:
        local_hours = pd.date_range(start, end, freq="h", inclusive="left")
        gap_start = local_hours[-hours].tz_convert("UTC")
    gap_end = end.tz_convert("UTC")

    entsoe_price = next(s for s in copied.entsoe_sources if s.name == "day_ahead_price")
    _blank_monthly_files(entsoe_price.cache_dir, gap_start, gap_end, column="day_ahead_price")
    _blank_single_file(
        copied.energy_charts_dir / "day_ahead_price_ec.parquet",
        gap_start,
        gap_end,
        column="day_ahead_price_ec",
    )


def _scenario_price_lags_gap_5h(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "5-Stunden-Lücke Preis-Lags (beide Quellen) ->
    Fortschreiben, Zeile 1" -- 5h is within price_lags' own FILL_ONLY
    ceiling (MAX_GAP_HOURS=6), so the group forward-fills and row 1 stays
    selected.
    """
    _scenario_price_gap(copied, target_day, 5)


def _scenario_price_d1_removed_both(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "Preis D-1 in beiden Quellen entfernt -> alle
    Zeilen scheitern, Schweigen" -- the whole of D-1 (23/24/25 local hours)
    exceeds price_lags' MAX_GAP_HOURS=6 in every row, including row 3 (which
    also carries a price_lags group).
    """
    _scenario_price_gap(copied, target_day, None)


def _scenario_source_frozen_8d(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "Quelle eingefroren, letzter vollständiger Tag 8
    Tage alt -> Altersgrenze greift" -- target_day_fold's own training
    window ends at target_day - 1 (must_reach), so blanking "load" from
    target_day - 8 onward leaves target_day - 9 as its last complete day,
    exactly MAX_TRAINING_EDGE_AGE_DAYS (7) + 1 days behind must_reach.
    """
    cutoff = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") - pd.Timedelta(days=8)
    end = pd.Timestamp(target_day, tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=2)
    load_source = next(s for s in copied.entsoe_sources if s.name == "load")
    _blank_monthly_files(load_source.cache_dir, cutoff, end, column=None)


def _scenario_anchor_expired(copied: CopiedStore, target_day: dt.date) -> None:
    """Spec section 6.3: "Ankertabelle abgelaufen -> Zeile 3, Lauf rot" --
    the anchor table is a committed repo file, not part of the downloaded
    store, so there is no file under the copy to corrupt (see CopiedStore's
    own docstring). Signals the fault through ``copied.anchor_valid_until``
    instead, which run_outage_drill forwards to
    run_submission_for_day(..., anchor_valid_until=...).
    """
    run_init = run_init_for_target_day(target_day)
    copied.anchor_valid_until = run_init - pd.Timedelta(days=1)


SCENARIOS: Final[dict[str, ScenarioFn]] = {
    "none": _scenario_none,
    "load_missing": _scenario_load_missing,
    "load_gap_3h": _scenario_load_gap_3h,
    "load_gap_4h": _scenario_load_gap_4h,
    "weather_missing": _scenario_weather_missing,
    "entsoe_cutoff_d1": _scenario_entsoe_cutoff_d1,
    "entsoe_cutoff_d3": _scenario_entsoe_cutoff_d3,
    "entsoe_cutoff_d9": _scenario_entsoe_cutoff_d9,
    "ttf_dead_15d": _scenario_ttf_dead_15d,
    "price_lags_gap_5h": _scenario_price_lags_gap_5h,
    "price_d1_removed_both": _scenario_price_d1_removed_both,
    "source_frozen_8d": _scenario_source_frozen_8d,
    "anchor_expired": _scenario_anchor_expired,
}


def resolve_as_of(real_now: pd.Timestamp) -> tuple[pd.Timestamp, dt.date]:
    """Spec section 2.12: ``as_of`` = 11:45 Berlin today, target_day =
    tomorrow -- unless ``real_now`` is before 10:00 Berlin today, in which
    case yesterday's 11:45 submission (which targeted today) is the one
    still "current" from a live-system point of view: ``as_of`` = yesterday
    11:45, target_day = today.

    ``target_day`` is never computed independently here -- it is always
    ``ops.windows.next_delivery_day(as_of)``, the same function the real
    submission job uses, so this cannot silently drift from the production
    definition of "which day does this as_of target" the way two
    independent derivations already have in this project's history
    (docs/sprint6_fix_weather_run_offset.md).
    """
    local_now = real_now.tz_convert(LOCAL_TZ)
    today_local_midnight = local_now.normalize()
    if local_now.time() < dt.time(10, 0):
        as_of_local_day = today_local_midnight - pd.DateOffset(days=1)
    else:
        as_of_local_day = today_local_midnight
    as_of = as_of_local_day.replace(hour=11, minute=45).tz_convert("UTC")
    return as_of, next_delivery_day(as_of)


@dataclasses.dataclass(frozen=True)
class RegressionCheck:
    """Spec section 2.12's bit-identity regression check against the
    archived live payload for the same target_day -- only meaningful when
    an archived payload actually exists (a brand-new target_day, or one no
    real run has reached yet, has nothing to compare against)."""

    archived_payload_found: bool
    identical: bool | None  # None if no archived payload to compare against
    max_abs_diff: float | None
    note: str


def _compare_against_archived_payload(
    payload: dict[str, Any] | None, target_day: dt.date
) -> RegressionCheck:
    archived_path = PAYLOADS_DIR / f"{target_day.isoformat()}.json"
    if not archived_path.exists():
        return RegressionCheck(
            archived_payload_found=False,
            identical=None,
            max_abs_diff=None,
            note=f"no archived payload at {archived_path} -- nothing to regress against",
        )
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    if payload is None:
        return RegressionCheck(
            archived_payload_found=True,
            identical=False,
            max_abs_diff=None,
            note="drill produced no payload (silent run) but an archived payload exists",
        )
    archived_values = archived["values"]
    drill_values = payload["values"]
    if len(archived_values) != len(drill_values):
        return RegressionCheck(
            archived_payload_found=True,
            identical=False,
            max_abs_diff=None,
            note=f"value count differs: archived={len(archived_values)} drill={len(drill_values)}",
        )
    max_abs_diff = max(abs(a - b) for a, b in zip(archived_values, drill_values, strict=True))
    identical = max_abs_diff == 0.0
    return RegressionCheck(
        archived_payload_found=True,
        identical=identical,
        max_abs_diff=max_abs_diff,
        note="bitgleich" if identical else f"deviates, max abs diff={max_abs_diff:.6g}",
    )


@dataclasses.dataclass(frozen=True)
class OutageDrillResult:
    scenario: str
    as_of: pd.Timestamp
    target_day: dt.date
    candidate_selected: str | None
    skip_reason: str | None
    payload: dict[str, Any] | None
    regression: RegressionCheck
    protocol_path: Path
    runtime_seconds: float


def run_outage_drill(
    scenario: str,
    *,
    now: pd.Timestamp | None = None,
    protocol_path: Path | None = None,
) -> OutageDrillResult:
    """Spec section 2.12/6.3: copy the published store, optionally inject a
    scenario's fault into the copy, run the real dry-run pipeline
    (``live=False`` throughout -- ``run_submission_for_day``'s own
    ``submit(challenge, payload, live=live)`` call can never reach a real
    POST here), write the protocol record to ``protocol_path`` (a fresh
    temp file if not given -- never ``logs/submissions.jsonl``), and
    compare the resulting payload against the real archived one.
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r} -- available: {sorted(SCENARIOS)} "
            "(most of spec section 6.3's eleven named outage drills are added in step 13)"
        )

    real_now = now if now is not None else pd.Timestamp.now("UTC")
    as_of, target_day = resolve_as_of(real_now)
    log.info("scenario=%r as_of=%s target_day=%s", scenario, as_of, target_day)

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="outage_drill_") as tmp:
        workdir = Path(tmp)
        log.info("loading a copy of the published store into %s ...", workdir)
        store_state = store.load_store(workdir)
        copied = _copy_sources_for_workdir(workdir)
        SCENARIOS[scenario](copied, target_day)

        df = assemble_price_model_inputs(
            entsoe_sources=copied.entsoe_sources,
            commodities_dir=copied.commodities_dir,
            energy_charts_dir=copied.energy_charts_dir,
        )
        window_start, window_end = renewables_window(target_day)
        local_days = pd.date_range(
            window_start.tz_convert(LOCAL_TZ).normalize(),
            window_end.tz_convert(LOCAL_TZ).normalize(),
            freq="D",
        )
        weather = read_weather_runs([d.date() for d in local_days], root=copied.weather_root)
        prices_qh = read_quarterhourly_prices(
            entsoe_sources=copied.entsoe_sources, energy_charts_dir=copied.energy_charts_dir
        )

        outcome = run_submission_for_day(
            df,
            weather,
            prices_qh,
            target_day,
            as_of=as_of,
            live=False,
            now=lambda: as_of,
            weather_root=copied.weather_root,
            anchor_valid_until=copied.anchor_valid_until,
        )
        runtime_seconds = time.monotonic() - t0
        price_provenance, price_source_conflicts = price_provenance_report(
            entsoe_sources=copied.entsoe_sources,
            energy_charts_dir=copied.energy_charts_dir,
            as_of=as_of,
        )

        record = build_submission_record(
            outcome,
            store_state.manifest,
            target_day=target_day,
            nominal_slot="outage_drill",
            gate_closure_ok=True,
            as_of=as_of,
            runtime_seconds=runtime_seconds,
            price_provenance=price_provenance,
            price_source_conflicts=price_source_conflicts,
        )
        out_path = protocol_path or (
            Path(tempfile.gettempdir()) / f"outage_drill_{scenario}_protocol.jsonl"
        )
        append_submission_record(out_path, record)

    regression = _compare_against_archived_payload(outcome.payload, target_day)

    return OutageDrillResult(
        scenario=scenario,
        as_of=as_of,
        target_day=target_day,
        candidate_selected=outcome.candidate_selected,
        skip_reason=outcome.skip_reason,
        payload=outcome.payload,
        regression=regression,
        protocol_path=out_path,
        runtime_seconds=runtime_seconds,
    )


def _print_result(result: OutageDrillResult) -> None:
    print(f"scenario={result.scenario}  as_of={result.as_of}  target_day={result.target_day}")
    print(f"candidate_selected={result.candidate_selected}  skip_reason={result.skip_reason}")
    print(f"payload built: {result.payload is not None}")
    print(f"regression check: {result.regression.note}")
    print(f"protocol written to: {result.protocol_path}")
    print(f"runtime: {result.runtime_seconds:.1f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        default="none",
        choices=sorted(SCENARIOS),
        help="which fault to inject into the store copy (spec section 6.3); "
        "'none' is the As-of-Lauf gate used after every build step (spec section 2.12)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="protocol output path (default: a fresh temp file, never logs/submissions.jsonl)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_outage_drill(args.scenario, protocol_path=args.out)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
