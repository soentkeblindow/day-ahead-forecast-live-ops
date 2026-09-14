"""Maintenance-job entry point for the persistent raw-data store (spec
6.7.1, section 5.4). Thin wiring only -- every decision (what counts as
valid, how a gap gets healed, how a store version gets published) lives in
energy_price_forecast.ops.store; this script just calls those functions
with real time, real files and the real ENTSO-E/weather/commodities
clients, in the order the spec prescribes.

Per-source loop, in order: determine the gap against the store's own
manifest, fetch just that gap via the existing (unmodified) client
functions, validate the result independently, and only then let it become
part of the version that gets packed and published. A source that fails
validation is reverted to the pre-fetch on-disk state (spec section 4,
Leitprinzip rule 1: "In den Speicher kommt nur, was eine eigenständige
Kontrolle bestanden hat") -- the existing cache-write functions
(data/_entsoe_cache.py::cached_fetch) write to disk unconditionally as a
side effect of fetching, with no validation gate of their own (section 3.3
keeps that module untouched), so this script is what actually makes "only
validated data enters the published store" true in practice.

One failing source does not stop another (spec section 2.7): the loop
never returns early, and the overall exit code is built from all results
only at the very end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data._weather_cache import cache_path as weather_cache_path
from energy_price_forecast.data.weather_client import (
    WeatherRunUnavailable,
    fetch_run,
    run_init_for_target_day,
)
from energy_price_forecast.ops import store
from energy_price_forecast.ops.store_sources import (
    COMMODITIES_DIR,
    COMMODITY_SOURCES,
    ENTSOE_SOURCES,
    EntsoeFetchFn,
    EntsoeSource,
    RowFetchFn,
    code_sha,
    run_id_and_url,
)
from energy_price_forecast.ops.windows import next_delivery_day

logger = logging.getLogger(__name__)

LOGS_DIR = PROJECT_ROOT / "logs"
STORE_SYNC_LOG = LOGS_DIR / "store_sync.csv"

# Default lookback for a source with no manifest history yet (first-ever
# sync, or a source that was never successfully written before). Generous
# rather than exact -- cached_fetch() only re-fetches whatever isn't
# already cached on disk, so an over-wide window costs a few extra (cheap,
# cache-hitting) months of gap computation, not extra API calls.
_ENTSOE_DEFAULT_LOOKBACK_DAYS = 400
_COMMODITY_DEFAULT_LOOKBACK_DAYS = 400

# How many delivery days behind next_delivery_day(as_of) _sync_weather also
# checks, in addition to next_delivery_day itself (docs/sprint6_fix_weather_run_offset.md
# section 3.1). Not just the newest run: a maintenance day that fully missed
# its own newest-run fetch (a red validation, an outage) would otherwise
# leave a permanent hole, since weather is immutable-per-run and heal_recent
# explicitly skips it. Costs nothing when nothing is missing -- an
# already-cached day is a path check, no network call (fetch_run's own
# use_cache=True cache-hit short-circuit).
WEATHER_BACKFILL_DAYS = 2


@dataclass
class SourceLogRow:
    fetched: bool = False
    rows_added: int = 0
    healed_cells: int = 0
    validation: str = "not_run"
    # Non-blocking findings from validate_source (spec 6.7.1a section 5.4:
    # "gehen ins Manifest und in die Protokollzeile"). Separate column, not
    # folded into `validation`, so a green run's log row still shows what
    # was found even though nothing blocked.
    hints: str = ""


@dataclass
class RunLog:
    sources: dict[str, SourceLogRow] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    any_failure: bool = False

    def get(self, name: str) -> SourceLogRow:
        return self.sources.setdefault(name, SourceLogRow())


def _backup_dir(cache_dir: Path) -> dict[Path, bytes]:
    """Snapshot of every file currently under cache_dir, for a possible
    revert if the freshly-fetched data fails validation (see module
    docstring: cached_fetch() writes unconditionally, before this script
    gets a chance to validate)."""
    if not cache_dir.exists():
        return {}
    return {p: p.read_bytes() for p in cache_dir.glob("*.parquet")}


def _restore_dir(cache_dir: Path, backup: dict[Path, bytes]) -> None:
    """Undo whatever cached_fetch() just wrote: restore every backed-up
    file to its exact prior bytes, and delete any file that didn't exist
    before (a brand-new, now-rejected month file)."""
    current = set(cache_dir.glob("*.parquet")) if cache_dir.exists() else set()
    for path, content in backup.items():
        path.write_bytes(content)
    for path in current - set(backup):
        path.unlink()


def _verify_heal_persisted(
    source_name: str,
    cache_dir: Path,
    healed: pd.DataFrame,
    window_start: pd.Timestamp,
    as_of: pd.Timestamp,
) -> None:
    """Raise StoreError if any cell heal_recent reported as filled is not
    actually on disk (spec 6.7.1a, section 5.4 point 3 / section 8
    acceptance criterion: "der Beweis wird geführt, nicht behauptet" --
    the exact discipline heal_recent's own rule 4 already applies to the
    in-memory result, now extended to what's actually written).
    """
    on_disk = store.read_cached_range(cache_dir)
    healed_window = healed[(healed.index >= window_start) & (healed.index < as_of)]
    on_disk_window = (
        on_disk[(on_disk.index >= window_start) & (on_disk.index < as_of)]
        if len(on_disk)
        else on_disk
    )
    common = healed_window.index.intersection(on_disk_window.index)
    for column in healed_window.columns:
        if column not in on_disk_window.columns:
            raise store.StoreError(
                f"heal_recent for {source_name!r} reported filled cells in column "
                f"{column!r}, but that column is not on disk after the heal "
                "(spec 6.7.1a section 5.4)"
            )
        claimed = healed_window.loc[common, column]
        on_disk_values = on_disk_window.loc[common, column]
        mismatch = claimed.notna() & (on_disk_values.isna() | (on_disk_values != claimed))
        if mismatch.any():
            raise store.StoreError(
                f"heal_recent for {source_name!r} reported {int(mismatch.sum())} filled "
                f"cell(s) in column {column!r} that are not actually on disk after the "
                "heal -- the heal's own report must be verified, not trusted "
                "(spec 6.7.1a section 5.4)"
            )


def _previous_entry(manifest: store.Manifest, name: str) -> store.SourceManifestEntry | None:
    return manifest.sources.get(name)


def _gap_start(
    previous: store.SourceManifestEntry | None, default_days: int, as_of: pd.Timestamp
) -> pd.Timestamp:
    if previous is not None and previous.covered_end_utc is not None:
        return pd.Timestamp(previous.covered_end_utc)
    return as_of - pd.Timedelta(days=default_days)


def _sync_entsoe_source(
    source: EntsoeSource, manifest: store.Manifest, as_of: pd.Timestamp, log: RunLog
) -> store.SourceManifestEntry | None:
    row = log.get(source.name)
    previous = _previous_entry(manifest, source.name)
    period_start = _gap_start(previous, _ENTSOE_DEFAULT_LOOKBACK_DAYS, as_of).floor("D")
    period_end = as_of.floor("D") + pd.Timedelta(days=1)
    # Fetched from the source's true covered_start_utc, not just period_start
    # (the gap) -- cached_fetch() only hits the API for genuinely new
    # months, so this is cheap, but it's required for correctness: the
    # manifest-growth check below and the SourceManifestEntry built at the
    # end of this function both assume `frame` is the *cumulative* series
    # (matching how commodities' own combine_first merge already works),
    # not just the newly-fetched delta. Confirmed a real, not theoretical,
    # bug by A10's first live wiring probe (2026-09-10): every ENTSO-E
    # source failed with "row count shrank: 67 < previous 83495"-style
    # errors every single incremental sync, because a period_start-only
    # fetch's tiny row count was being compared against the full historical
    # previous.count. Also explains a second symptom from the same run --
    # `generation`'s column mismatch (`gen_nuclear` missing) -- a
    # gap-only fetch since 2026-09 never spans the pre-April-2023 window
    # gen_nuclear has any real (non-empty) values in.
    fetch_start = (
        pd.Timestamp(previous.covered_start_utc)
        if previous is not None and previous.covered_start_utc is not None
        else period_start
    )

    backup = _backup_dir(source.cache_dir)
    try:
        # period_end (midnight after today), not as_of (the real wall clock)
        # -- day_ahead_price/load_forecast_day_ahead/wind_solar/
        # scheduled_exchanges are all published D-1, so as_of's own "now"
        # artificially truncated an already-public window on every
        # incremental sync, which the shrink-protection then (correctly)
        # rejected as apparent data loss every time a wider previous existed
        # (session 2026-09-13, misdiagnosed then as an ENTSO-E-side
        # regression -- it was this line). Harmless no-op for actual-only
        # sources (generation, cross_border_flows): ENTSO-E simply returns
        # nothing for the still-future portion of the window regardless.
        frame = source.fetch(fetch_start, period_end)
        row.fetched = True
    except Exception as exc:  # noqa: BLE001 -- an unreachable source is a green outcome (spec 2.7)
        logger.warning("Source %r unreachable this run: %s", source.name, exc)
        row.validation = f"unreachable: {exc}"
        return previous

    expectation = store.EXPECTATION_TABLE[source.name]
    result = store.validate_source(
        source.name,
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=as_of,
        previous=previous,
        mode="live",
    )
    row.validation = "ok" if result.ok else "; ".join(result.reasons)
    row.hints = "; ".join(result.hints)

    if not result.ok:
        logger.error("Source %r failed validation, reverting: %s", source.name, result.reasons)
        _restore_dir(source.cache_dir, backup)
        log.any_failure = True
        return previous

    row.rows_added = len(frame) - (previous.count if previous is not None else 0)
    return store.SourceManifestEntry(
        covered_start_utc=(
            previous.covered_start_utc
            if previous is not None and previous.covered_start_utc is not None
            else frame.index.min().isoformat()
        ),
        covered_end_utc=frame.index.max().isoformat(),
        count=len(frame),
        last_success_utc=as_of.isoformat(),
        last_attempt_utc=as_of.isoformat(),
        live_nan_cell_counts=result.nan_cell_counts,
    )


def _sync_commodity_source(
    name: str,
    fetch: RowFetchFn,
    column: str,
    manifest: store.Manifest,
    as_of: pd.Timestamp,
    log: RunLog,
) -> store.SourceManifestEntry | None:
    """At most once per UTC calendar day (spec section 2.8): the four
    known Yahoo Finance deviations in the whole measured history all fell
    on the one day multiple audit.yml runs landed unusually close together
    (docs/cron_jobs.md section 5) -- this cadence rule prevents a repeat,
    independent of how many times a maintenance run happens to fire today.
    """
    row = log.get(name)
    previous = _previous_entry(manifest, name)
    if previous is not None:
        last_attempt = pd.Timestamp(previous.last_attempt_utc)
        if last_attempt.date() == as_of.date():
            row.validation = "skipped: already attempted today (spec section 2.8)"
            return previous

    path = COMMODITIES_DIR / f"{name}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=[column])
    if not isinstance(existing.index, pd.DatetimeIndex):
        existing.index = pd.DatetimeIndex(existing.index, tz="UTC")

    gap_start = _gap_start(previous, _COMMODITY_DEFAULT_LOOKBACK_DAYS, as_of)
    try:
        fresh = fetch(gap_start, as_of)
        row.fetched = True
    except Exception as exc:  # noqa: BLE001 -- unreachable is green (spec 2.7)
        logger.warning("Commodity source %r unreachable this run: %s", name, exc)
        row.validation = f"unreachable: {exc}"
        return dataclass_replace_attempt(previous, as_of)

    merged = existing.combine_first(fresh) if len(existing) else fresh
    expectation = store.EXPECTATION_TABLE[name]
    result = store.validate_source(
        name,
        merged,
        expectation,
        period_start=merged.index.min() if len(merged) else as_of,
        period_end=as_of,
        as_of=as_of,
        previous=previous,
        mode="live",
    )
    row.validation = "ok" if result.ok else "; ".join(result.reasons)
    row.hints = "; ".join(result.hints)

    if not result.ok:
        logger.error("Commodity source %r failed validation, not writing: %s", name, result.reasons)
        return dataclass_replace_attempt(previous, as_of)

    store.write_if_valid(path, merged, result)
    row.rows_added = len(merged) - (previous.count if previous is not None else 0)
    return store.SourceManifestEntry(
        covered_start_utc=merged.index.min().isoformat() if len(merged) else None,
        covered_end_utc=merged.index.max().isoformat() if len(merged) else None,
        count=len(merged),
        last_success_utc=as_of.isoformat(),
        last_attempt_utc=as_of.isoformat(),
        live_nan_cell_counts=result.nan_cell_counts,
    )


def dataclass_replace_attempt(
    previous: store.SourceManifestEntry | None, as_of: pd.Timestamp
) -> store.SourceManifestEntry | None:
    """Bump last_attempt_utc without touching last_success_utc -- the exact
    distinction the manifest exists to carry (spec section 5.3)."""
    if previous is None:
        return None
    return store.SourceManifestEntry(
        covered_start_utc=previous.covered_start_utc,
        covered_end_utc=previous.covered_end_utc,
        count=previous.count,
        last_success_utc=previous.last_success_utc,
        last_attempt_utc=as_of.isoformat(),
    )


def _sync_weather(
    manifest: store.Manifest, as_of: pd.Timestamp, log: RunLog
) -> store.SourceManifestEntry | None:
    """Fetch the run the NEXT submission actually needs, plus a short
    trailing backfill (docs/sprint6_fix_weather_run_offset.md).

    Before this fix, this fetched ``run_init_for_target_day(as_of's own
    local date)`` -- the run for TODAY's (already-decided) delivery day,
    i.e. YESTERDAY's 00Z run. The live submission job, run the same day,
    needs ``run_init_for_target_day(next_delivery_day(as_of))`` -- TODAY's
    own 00Z run, for TOMORROW's delivery. Those are two different runs, one
    calendar day apart: the run this function fetched was never the one
    the submission needed, structurally, every single day.

    ``next_delivery_day(as_of)`` is now the single shared answer both this
    function and scripts/run_daily_submission.py call, rather than each
    re-deriving it -- the drift between them is exactly what caused the bug.

    Checks WEATHER_BACKFILL_DAYS additional days behind next_delivery_day
    too, not just the newest one: this doubles as the self-heal weather
    never had (heal_recent explicitly skips it; the single-run version
    before this fix asked about exactly one day, so a day this function
    missed -- a red validation, a maintenance outage -- stayed missing
    forever). Already-cached days cost a path check only, no network call.
    """
    row = log.get("weather_single_runs")
    previous = manifest.sources.get("weather_single_runs")
    delivery_day = next_delivery_day(as_of)

    count = previous.count if previous is not None else 0
    covered_start = previous.covered_start_utc if previous is not None else None
    newest_run_init: pd.Timestamp | None = (
        pd.Timestamp(previous.covered_end_utc)
        if previous is not None and previous.covered_end_utc
        else None
    )
    oldest_new_run_init: pd.Timestamp | None = None
    any_new = False
    statuses: list[str] = []

    for k in range(WEATHER_BACKFILL_DAYS + 1):
        target_day = delivery_day - dt.timedelta(days=k)
        run_init_utc = run_init_for_target_day(target_day)

        # Reine Pfadprüfung (spec section 3.1): a cache hit costs nothing,
        # not even the call into fetch_run's own (equivalent) cache check.
        if weather_cache_path(run_init_utc, model="ecmwf_ifs").exists():
            continue

        try:
            frame = fetch_run(run_init_utc, use_cache=True)
        except WeatherRunUnavailable as exc:
            logger.info("Weather run %s not yet available: %s", run_init_utc, exc)
            statuses.append(f"{run_init_utc.date()}: not_yet_available")
            continue

        result = store.validate_weather_run(frame)
        if not result.ok:
            # data/_weather_cache.py writes atomically (temp file + os.replace,
            # spec-untouched module) -- no half-written file risk, but the
            # write happens before this validation runs, same as ENTSO-E's
            # cached_fetch. Unlike ENTSO-E there is no existing history to
            # protect (one file per run, immutable, never rewritten) -- just
            # this one newly-written bad file, which must be deleted so a
            # later publish_store's per-source glob packing doesn't sweep it up
            # alongside the still-valid history the manifest continues to
            # point to below. Confirmed a real gap, not theoretical: A9's
            # first real full-history validation (2026-09-10) found two
            # already-cached historical runs with this exact failure pattern
            # (near-100% NaN, HTTP 200) sitting on disk undetected until then.
            weather_cache_path(run_init_utc, model="ecmwf_ifs").unlink(missing_ok=True)
            logger.error(
                "Weather run %s failed validation, cache file removed: %s",
                run_init_utc,
                result.reasons,
            )
            log.any_failure = True
            statuses.append(f"{run_init_utc.date()}: {'; '.join(result.reasons)}")
            continue

        row.fetched = True
        any_new = True
        count += 1
        if oldest_new_run_init is None or run_init_utc < oldest_new_run_init:
            oldest_new_run_init = run_init_utc
        if newest_run_init is None or run_init_utc > newest_run_init:
            newest_run_init = run_init_utc
        statuses.append(f"{run_init_utc.date()}: ok")

    row.rows_added = count - (previous.count if previous is not None else 0)
    row.validation = "; ".join(statuses) if statuses else "already cached"

    if not any_new:
        return dataclass_replace_attempt(previous, as_of)

    if covered_start is None:
        assert oldest_new_run_init is not None  # any_new implies at least one new run
        covered_start = oldest_new_run_init.isoformat()
    assert newest_run_init is not None  # any_new implies newest_run_init was set above
    return store.SourceManifestEntry(
        covered_start_utc=covered_start,
        covered_end_utc=newest_run_init.isoformat(),
        count=count,
        last_success_utc=as_of.isoformat(),
        last_attempt_utc=as_of.isoformat(),
    )


def _write_log_row(
    as_of: pd.Timestamp,
    run_id: str,
    code_sha: str,
    log: RunLog,
    store_bytes: int | None,
    exit_status: str,
) -> None:
    row: dict[str, object] = {
        "run_timestamp_utc": as_of.isoformat(),
        "run_id": run_id,
        "code_sha": code_sha,
        "store_bytes": store_bytes if store_bytes is not None else "",
        "warnings": " | ".join(log.warnings),
        "exit_status": exit_status,
    }
    for name, source_row in sorted(log.sources.items()):
        row[f"{name}_fetched"] = source_row.fetched
        row[f"{name}_rows_added"] = source_row.rows_added
        row[f"{name}_healed_cells"] = source_row.healed_cells
        row[f"{name}_validation"] = source_row.validation
        row[f"{name}_hints"] = source_row.hints

    frame = pd.DataFrame([row])
    STORE_SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)

    if not STORE_SYNC_LOG.exists():
        frame.to_csv(STORE_SYNC_LOG, index=False)
        return

    existing = pd.read_csv(STORE_SYNC_LOG)
    if list(existing.columns) == list(frame.columns):
        frame.to_csv(STORE_SYNC_LOG, mode="a", header=False, index=False)
        return

    # The column set grows whenever RunLog.sources covers a source name
    # never seen before, or a new per-source field is added (real incident,
    # 2026-09-11: adding SourceLogRow.hints left the file's existing header
    # at its old, narrower width while the next row was written with more
    # fields than that header declared -- unparseable, pd.read_csv raised
    # ParserError. A plain mode="a" append can never fix this on its own,
    # since it never looks at what's already in the file.) Rewrite with the
    # union of old and new columns instead of silently appending a
    # mismatched row -- old rows get "" for a column they predate.
    union_columns = list(dict.fromkeys([*existing.columns, *frame.columns]))
    existing = existing.reindex(columns=union_columns)
    frame = frame.reindex(columns=union_columns)
    pd.concat([existing, frame], ignore_index=True).to_csv(STORE_SYNC_LOG, index=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute everything, upload nothing."
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated subset of source names, for manual debugging.",
    )
    args = parser.parse_args()

    as_of = pd.Timestamp.now(tz="UTC")
    log = RunLog()
    only = set(args.sources.split(",")) if args.sources else None

    try:
        state = store.load_store(PROJECT_ROOT)
        manifest = state.manifest
    except store.StoreError as exc:
        logger.error("No store to load: %s -- run scripts/rebuild_store.py first.", exc)
        return 2

    new_sources = dict(manifest.sources)

    for source in ENTSOE_SOURCES:
        if only and source.name not in only:
            continue
        entry = _sync_entsoe_source(source, manifest, as_of, log)
        if entry is not None:
            new_sources[source.name] = entry

    for name, fetch, column in COMMODITY_SOURCES:
        if only and name not in only:
            continue
        entry = _sync_commodity_source(name, fetch, column, manifest, as_of, log)
        if entry is not None:
            new_sources[name] = entry

    if not only or "weather_single_runs" in only:
        entry = _sync_weather(manifest, as_of, log)
        if entry is not None:
            new_sources["weather_single_runs"] = entry

    # Heal step: trailing HEAL_LOOKBACK_DAYS, every run, all ENTSO-E sources
    # (spec section 2.6). Skipped for weather (immutable per-run files, no
    # "gap inside an already-written range" concept) and commodities
    # (handled by their own combine_first merge above already).
    for source in ENTSOE_SOURCES:
        if only and source.name not in only:
            continue
        current_entry = new_sources.get(source.name)
        if current_entry is None or current_entry.covered_end_utc is None:
            continue
        try:
            existing_frame = source.fetch(
                as_of - pd.Timedelta(days=store.HEAL_LOOKBACK_DAYS + 5), as_of
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %r for healing: %s", source.name, exc)
            continue

        def _refetch(
            s: pd.Timestamp, e: pd.Timestamp, _fn: EntsoeFetchFn = source.fetch
        ) -> pd.DataFrame:
            # use_cache=False (6.7.1a, closes spec 6.7.1 section 2.1 finding
            # 1): a real bypass of the cache READ, so this is no longer the
            # same cached_fetch call `existing_frame` above already made,
            # seconds apart, comparing nothing. The cache WRITE still
            # happens, with fill semantics (_entsoe_cache.py::
            # _merge_and_write) -- that write is what actually persists a
            # healed cell to disk; see _verify_heal_persisted below for the
            # proof that it did.
            return _fn(s, e, use_cache=False)

        healed, heal_result = store.heal_recent(
            source.name, existing_frame, store.HEAL_LOOKBACK_DAYS, _refetch, as_of=as_of
        )
        log.get(source.name).healed_cells = heal_result.filled_cells
        if heal_result.still_missing_cells:
            log.warnings.append(
                f"{source.name}: {heal_result.still_missing_cells} cell(s) still missing "
                f"after heal in the last {store.HEAL_LOOKBACK_DAYS} days"
            )
        if heal_result.filled_cells:
            # `healed` is consumed here, not discarded (spec 6.7.1a section
            # 8 acceptance criterion) -- persistence itself happens as a
            # side effect of _refetch's use_cache=False write, so this is
            # independent verification that the write actually landed, not
            # a second place healed itself needs to be saved.
            _verify_heal_persisted(
                source.name,
                source.cache_dir,
                healed,
                as_of - pd.Timedelta(days=store.HEAL_LOOKBACK_DAYS),
                as_of,
            )

    run_id, run_url = run_id_and_url()
    sha = code_sha()
    manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc=as_of.isoformat(),
        run_id=run_id,
        run_url=run_url,
        code_sha=sha,
        sources=new_sources,
    )

    store_bytes: int | None = None
    if not args.dry_run:
        asset = store.publish_store(PROJECT_ROOT, manifest)
        store_bytes = asset.size_bytes
        size_warning = store.check_store_size(asset.size_bytes)
        if size_warning:
            log.warnings.append(size_warning)
    else:
        logger.info("--dry-run: not uploading, manifest computed only.")

    log.warnings.extend(store.check_deadlines(as_of))

    exit_status = "ok"
    if log.any_failure:
        exit_status = "partial_failure"
    if log.warnings:
        logger.warning("Maintenance run warnings: %s", log.warnings)

    _write_log_row(as_of, run_id, sha, log, store_bytes, exit_status)
    print(f"Store sync complete: {exit_status}, warnings={log.warnings}")

    # A partial-source failure is a documented, expected finding (spec
    # section 2.7: only a bad-data source turns the source itself rot, and
    # rot for a source must still let the run finish and log) -- but the
    # overall run should still redden so it's visible in the Actions tab
    # (spec section 2.7, "der rote Lauf ... ist der Ersatz [für
    # Melde-Infrastruktur]").
    return 1 if log.any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
