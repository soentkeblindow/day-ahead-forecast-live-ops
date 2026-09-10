"""Persistent raw-data store: packing, manifest, versioned publish, input
control and the heal step (spec 6.7.1, Entscheidungen 17-19, section 5.3).

Only raw, externally-fetched data ever enters the store (section 2.2) --
never anything derived (hourly.parquet, features.parquet,
quarterhourly_prices.parquet, the renewables artefact, any .bak file).
Packing walks an explicit POSITIVE list of paths (_SOURCE_GLOBS), not an
exclusion list: what isn't named doesn't travel, so a forgotten .bak file
or a newly-added derived artefact can't sneak in by omission the way an
exclusion list eventually would. That positive list is additionally keyed
by source name and restricted, at pack time, to sources actually present
in the manifest being packed -- a source validate_source rejected this run
has no manifest entry and is therefore not packed either, even though its
raw cache files are already sitting on disk (data/entsoe_client.py's
cached_fetch writes unconditionally on a successful fetch, independent of
this store's own validation). Confirmed missing in practice, not just in
theory: A9's first real publish (2026-09-10) packed all 6 ENTSO-E fetch
groups' cache files into the tar even though 4 of them failed validation
that run and were correctly left out of the manifest -- this benignly
turned out to be genuinely fine raw data in that instance, but the
"only validated data enters the store" principle this module's own design
rests on was not actually true until this per-source gating was added.

``root``/``workdir`` throughout this module means the repository root (the
directory containing ``data/``), not an arbitrary staging area:
data/entsoe_client.py's cache paths are anchored to
``energy_price_forecast.config.PROJECT_ROOT`` (the installed package's own
location, independent of CWD), while data/_weather_cache.py's
``CACHE_ROOT`` is a bare relative path (depends on CWD) -- an existing
inconsistency in code this spec must not touch (section 3.3). Passing the
real repo root as workdir/root satisfies both without changing either
cache module: the existing, unmodified fetch functions find their files
exactly where they already look.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd

from energy_price_forecast.data.entsoe_client import NEIGHBORS
from energy_price_forecast.data.weather_grid import HOURLY_VARIABLES, expected_columns
from energy_price_forecast.ops import release_assets
from energy_price_forecast.ops.release_assets import AssetRef

STORE_RELEASE_TAG: Final[str] = "data-store"
STORE_FORMAT_VERSION: Final[int] = 1
STORE_KEEP_VERSIONS: Final[int] = 5

# Positive list (spec section 5.3), keyed by manifest source name so pack_
# store can restrict packing to sources the manifest being packed actually
# covers. Sourced from data/entsoe_client.py's six DATA_RAW/"entsoe"/<group>
# cache directories, data/_weather_cache.py's CACHE_ROOT, and a new
# commodities cache (data/raw/commodities/, added by this spec --
# commodities_client.py had no on-disk cache before, spec section 3.1 point
# 3 / A1 finding). Commodities get one glob per file, not a shared
# directory glob, since ttf_gas/eua_co2 are two separate manifest sources
# sharing one directory.
_SOURCE_GLOBS: Final[dict[str, tuple[str, ...]]] = {
    "day_ahead_price": ("data/raw/entsoe/day_ahead_prices/*.parquet",),
    "load": ("data/raw/entsoe/load/*.parquet",),
    "wind_solar": ("data/raw/entsoe/wind_solar/*.parquet",),
    "generation": ("data/raw/entsoe/generation/*.parquet",),
    "scheduled_exchanges": ("data/raw/entsoe/scheduled_exchanges/*.parquet",),
    "cross_border_flows": ("data/raw/entsoe/cross_border_flows/*.parquet",),
    "weather_single_runs": ("data/cache/weather_single_runs/**/*.parquet",),
    "ttf_gas": ("data/raw/commodities/ttf_gas.parquet",),
    "eua_co2": ("data/raw/commodities/eua_co2.parquet",),
}

_MANIFEST_NAME: Final[str] = "manifest.json"


class StoreError(RuntimeError):
    """The store is missing, malformed, or a write to it was refused."""


@dataclass(frozen=True)
class SourceManifestEntry:
    """One source's row in the manifest (spec section 5.3 manifest table).

    ``count`` is rows for row-oriented sources (ENTSO-E, commodities) or
    files for the weather per-run cache -- deliberately untyped further
    than "how much is there", since the two source families are not
    directly comparable and the manifest's job is bookkeeping, not
    cross-source analytics.

    ``last_success_utc`` and ``last_attempt_utc`` are distinct on purpose
    (spec section 5.3, "trägt zwei Zwecke"): a source that keeps failing
    updates its last-attempt timestamp every run while last-success stays
    frozen -- the gap between the two is exactly what 6.7.2's freshness
    check and this spec's commodity cadence rule (section 2.8) both read.
    """

    covered_start_utc: str | None
    covered_end_utc: str | None
    count: int
    last_success_utc: str | None
    last_attempt_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_start_utc": self.covered_start_utc,
            "covered_end_utc": self.covered_end_utc,
            "count": self.count,
            "last_success_utc": self.last_success_utc,
            "last_attempt_utc": self.last_attempt_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceManifestEntry:
        return cls(
            covered_start_utc=data["covered_start_utc"],
            covered_end_utc=data["covered_end_utc"],
            count=data["count"],
            last_success_utc=data["last_success_utc"],
            last_attempt_utc=data["last_attempt_utc"],
        )


@dataclass(frozen=True)
class Manifest:
    """manifest.json, packed alongside the data at the archive root (spec
    section 5.3). Every field here is attributable: which run wrote this
    store, with which code, and what each source looked like at write
    time -- the direct answer to Leitprinzip rule 4 (section 4), "jeder
    Schreibzugriff ist zuordenbar"."""

    store_format_version: int
    created_at_utc: str
    run_id: str
    run_url: str
    code_sha: str
    sources: dict[str, SourceManifestEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_format_version": self.store_format_version,
            "created_at_utc": self.created_at_utc,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "code_sha": self.code_sha,
            "sources": {name: entry.to_dict() for name, entry in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            store_format_version=data["store_format_version"],
            created_at_utc=data["created_at_utc"],
            run_id=data["run_id"],
            run_url=data["run_url"],
            code_sha=data["code_sha"],
            sources={
                name: SourceManifestEntry.from_dict(entry)
                for name, entry in data.get("sources", {}).items()
            },
        )


@dataclass(frozen=True)
class StoreState:
    workdir: Path
    manifest: Manifest


# ---------------------------------------------------------------------------
# Packing (positive list) and manifest serialisation
# ---------------------------------------------------------------------------


def _matched_paths(root: Path, source_names: Iterable[str]) -> list[Path]:
    """Every file under root matching one of source_names' _SOURCE_GLOBS,
    relative to root, sorted and deduplicated (a file could in principle
    match two globs). A source name absent from source_names -- typically
    because it has no manifest entry, i.e. its own last fetch was never
    validated -- contributes nothing here, even if its cache files already
    exist on disk."""
    matched: set[Path] = set()
    for name in source_names:
        for pattern in _SOURCE_GLOBS.get(name, ()):
            matched.update(p.relative_to(root) for p in root.glob(pattern) if p.is_file())
    return sorted(matched)


def pack_store(root: Path, manifest: Manifest, out_path: Path) -> None:
    """Pack root's files for every source in manifest.sources, plus
    manifest.json itself, into a single uncompressed tar at out_path (spec
    section 5.3).

    No compression: the content is practically all parquet, which is
    already compressed -- a second layer costs time and buys nothing. One
    archive, not one asset per file, so a store version comes into
    existence atomically and a reader never sees a half-updated set.
    """
    manifest_bytes = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tf:
        info = tarfile.TarInfo(name=_MANIFEST_NAME)
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))
        for rel_path in _matched_paths(root, manifest.sources.keys()):
            tf.add(root / rel_path, arcname=str(rel_path))


def unpack_store(tar_path: Path, workdir: Path) -> Manifest:
    """Extract tar_path into workdir and return its manifest.

    Uses tarfile's built-in "data" extraction filter (PEP 706, stdlib since
    3.12 -- this project's minimum version) to reject path traversal or
    unexpected file types in the archive, since the archive's own contents
    ultimately come from a third-party API response chain (ENTSO-E,
    Open-Meteo, Yahoo Finance) by the time it reaches this function.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(workdir, filter="data")
    manifest_path = workdir / _MANIFEST_NAME
    if not manifest_path.exists():
        raise StoreError(f"Archive {tar_path.name!r} has no {_MANIFEST_NAME}")
    return Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Kernfunktionen (spec section 5.3)
# ---------------------------------------------------------------------------


def load_store(workdir: Path) -> StoreState:
    """Download the newest store asset and unpack it into workdir.

    Raises StoreError if no asset exists -- an empty store is not a valid
    operating state, it is a signal to run scripts/rebuild_store.py.
    """
    release = release_assets.ensure_release(STORE_RELEASE_TAG)
    assets = release_assets.list_assets(release)
    if not assets:
        raise StoreError(
            "No store asset exists yet -- run scripts/rebuild_store.py to create the first one."
        )
    newest = assets[0]
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / newest.name
        release_assets.download_asset(newest, tar_path)
        manifest = unpack_store(tar_path, workdir)
    return StoreState(workdir=workdir, manifest=manifest)


def publish_store(workdir: Path, manifest: Manifest) -> AssetRef:
    """Pack, upload as a new timestamped asset, then prune to the newest
    STORE_KEEP_VERSIONS. Never overwrites an existing asset (spec section
    2.5) -- the timestamp in manifest.created_at_utc is also the asset's
    file name, so two publishes in the same second would collide; callers
    are expected to call this at most once per run.
    """
    release = release_assets.ensure_release(STORE_RELEASE_TAG)
    ts = pd.Timestamp(manifest.created_at_utc).strftime("%Y%m%dT%H%M%SZ")
    tar_name = f"store-{ts}.tar"
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / tar_name
        pack_store(workdir, manifest, tar_path)
        asset = release_assets.upload_asset(release, tar_path)
    release_assets.prune_assets(release, STORE_KEEP_VERSIONS)
    return asset


# ---------------------------------------------------------------------------
# Input control (spec section 2.4) -- deliberately independent of
# data/_entsoe_cache.py::_is_sufficiently_complete(): that function waved
# through an incomplete August month file in 6.6 because it assumed hourly
# resolution while day_ahead_price has been 15-minute since 2025-09-30.
# Asking the same guard that already failed once is not a control.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    source: str
    ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceExpectation:
    """Comment-carrying expectation for one row-oriented (ENTSO-E-shaped)
    source (spec section 5.3, "Die Erwartungstabelle ist Pflicht, nicht
    Kür"). Every deviation from the "0% NaN, gapless" default MUST be named
    here with a reason in a comment -- otherwise the input control
    re-flags the same already-known, already-understood finding on every
    single run, and the whole point of a documented exception table is
    lost (spec section 2.4).

    ``grid_based=False`` turns off the coverage/NaN-in-period checks
    entirely for a source: commodities (TTF gas, EUA CO2) are daily,
    trading-days-only data -- weekends and holidays are absent BY DESIGN
    (data/commodities_client.py), so "gapless at the inferred resolution"
    is the wrong question to ask of them. Their actual freshness (how
    stale is the latest point) is 6.7.2's job (source-age selection), not
    this store's write-time gate.

    ``known_low_resolution_windows`` documents column, time-window pairs
    where the column is known to have genuinely reported at a coarser
    cadence than the source's own quarter-hourly grid (A9 finding,
    2026-09-10: individual ENTSO-E cross-border series moved from hourly to
    15-minute reporting at different, source-specific dates -- merging a
    still-hourly column against an already-15-minute one produces a
    structural 75% NaN for the hourly column, not a data-quality problem).
    Excluded from the NaN check for exactly that column and window, every
    other period is judged normally -- unlike max_nan_fraction, this can't
    be approximated by a single tolerance float without permanently waving
    through a genuine large-scale failure outside the known window.

    ``live_settling_columns`` names the columns that are TSO-reported
    *actuals* (measured after the fact, with a variable reporting lag) as
    opposed to forecasts/schedules (published once, in advance, for the
    whole delivery day -- no "still trickling in" dynamic to account for).
    Only these columns get LIVE_SETTLING_BUFFER excluded from the tail in
    ``mode="live"`` validation (see validate_source); everything else is
    judged over its full checked window in both modes. A real measurement
    (2026-09-10, A10) of the trailing contiguous NaN run right now across
    every actuals column found 0h for most, but 10h for gen_hard_coal in
    that one snapshot -- high enough variance that a short, precisely-fitted
    buffer is not defensible; see LIVE_SETTLING_BUFFER's own comment for
    the reasoning behind the chosen width.
    """

    expected_columns: frozenset[str]
    max_nan_fraction: dict[str, float] = field(default_factory=dict)
    grid_based: bool = True
    known_low_resolution_windows: dict[str, tuple[str, str]] = field(default_factory=dict)
    live_settling_columns: frozenset[str] = field(default_factory=frozenset)


# Six ENTSO-E fetch groups (data/entsoe_client.py's own _FETCH_FUNCTIONS/
# cache_subdir names) plus the two commodity series
# (data/commodities_client.py). Column names copied from
# ops/availability_audit.py's _BASE_FETCH_GROUP / build_checklist, the
# existing single source of truth for which raw column belongs to which
# fetch group -- not re-derived here to avoid a second, driftable copy of
# that mapping.
EXPECTATION_TABLE: Final[dict[str, SourceExpectation]] = {
    "day_ahead_price": SourceExpectation(expected_columns=frozenset({"day_ahead_price"})),
    "load": SourceExpectation(
        expected_columns=frozenset({"load_actual", "load_forecast_day_ahead"}),
        # A9's real full-history run (2026-09-10): load_actual had 4 live-tail
        # cells; load_forecast_day_ahead had two full missing calendar days
        # (2022-02-xx, 2022-03-xx, 96 quarter-hours each) plus known DST
        # fall-back hours -- same small/isolated/permanent-gap category as
        # generation's finding, same 0.001 tolerance for the same reason
        # (rebuild-mode only -- see validate_source's mode parameter).
        max_nan_fraction={"load_actual": 0.001, "load_forecast_day_ahead": 0.001},
        # load_actual is a TSO actual (measured after the fact, variable
        # reporting lag) -- load_forecast_day_ahead is published once, in
        # advance, for the whole day; no live-tail dynamic to account for.
        live_settling_columns=frozenset({"load_actual"}),
    ),
    "wind_solar": SourceExpectation(
        expected_columns=frozenset(
            {"wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"}
        ),
        # A9's real full-history run found exactly the DST fall-back gap
        # already documented in 6.5.3 (2023-10-28, 2024-10-26, 2025-10-25,
        # 22:00-22:45 UTC, solar_forecast/wind_onshore_forecast only) -- now
        # visible for the first time in a full-history validation. Same
        # 0.001 tolerance, applied to all three columns for consistency.
        max_nan_fraction={
            "wind_onshore_forecast": 0.001,
            "wind_offshore_forecast": 0.001,
            "solar_forecast": 0.001,
        },
    ),
    "generation": SourceExpectation(
        expected_columns=frozenset(
            {
                "gen_nuclear",
                "gen_lignite",
                "gen_hard_coal",
                "gen_gas",
                "gen_oil",
                "gen_biomass",
                "gen_hydro",
                "gen_wind_onshore",
                "gen_wind_offshore",
                "gen_solar",
                "gen_other",
            }
        ),
        max_nan_fraction={
            # Germany has had no operating nuclear plants since April 2023 --
            # permanently and correctly 100% NaN, not a gap (the exact
            # example spec section 2.4 itself names).
            "gen_nuclear": 1.0,
            # A9's first-ever full 2020-2026 history fetch found 3 isolated
            # single-interval gaps, not reproducible as a fetch/network
            # issue (unrelated to the as_of tail, both well in the past):
            # gen_biomass/gen_gas/gen_solar/gen_wind_onshore each missing
            # exactly one row at 2025-07-09 18:00-18:45 UTC, gen_oil missing
            # one row at 2026-02-20 06:30 UTC. Worst observed fraction
            # 2/234625 ~= 0.000085 -- genuine small permanent gaps in
            # ENTSO-E's own published generation-by-type series, the same
            # kind of finding as 6.5.1's 4 known weather provider gaps, not
            # a code bug. 0.001 gives ~12x headroom over the observed rate
            # without waving through a real large-scale failure (owner
            # decision 2026-09-10, applied to every non-exempt column since
            # the same reporting-gap phenomenon could equally hit any of
            # them, not just the ones observed so far).
            "gen_lignite": 0.001,
            "gen_hard_coal": 0.001,
            "gen_gas": 0.001,
            "gen_oil": 0.001,
            "gen_biomass": 0.001,
            "gen_hydro": 0.001,
            "gen_wind_onshore": 0.001,
            "gen_wind_offshore": 0.001,
            "gen_solar": 0.001,
            "gen_other": 0.001,
        },
        # generation-by-type is a TSO actual, reported after the fact --
        # gen_nuclear excluded since it is permanently ~100% NaN anyway
        # (max_nan_fraction=1.0 above already covers it regardless of mode).
        live_settling_columns=frozenset(
            {
                "gen_lignite",
                "gen_hard_coal",
                "gen_gas",
                "gen_oil",
                "gen_biomass",
                "gen_hydro",
                "gen_wind_onshore",
                "gen_wind_offshore",
                "gen_solar",
                "gen_other",
            }
        ),
    ),
    "scheduled_exchanges": SourceExpectation(
        expected_columns=frozenset(f"scheduled_net_de_to_{n.lower()}" for n in NEIGHBORS),
        # Same 0.001 backstop tolerance as "generation", for isolated
        # settled gaps -- applied to every neighbor column since the
        # mechanism applies to all of them, not just the ones observed so
        # far. No live_settling_columns here: scheduled exchanges are a
        # day-ahead schedule, published once in advance for the whole
        # delivery day, not a TSO actual with a variable after-the-fact
        # reporting lag -- there is no live tail to exclude.
        max_nan_fraction={f"scheduled_net_de_to_{n.lower()}": 0.001 for n in NEIGHBORS},
        # A9's real full-history rebuild (2026-09-10) found scheduled_net_
        # de_to_{at,ch,nl,pl,dk_1} genuinely reported hourly, not
        # quarter-hourly, for a shared multi-month stretch (scheduled_net_
        # de_to_fr stayed quarter-hourly throughout, hence no entry here) --
        # merging an hourly column against an already-15-minute one via
        # pd.concat's outer join produces a structural 75% NaN for the
        # hourly column, not a data-quality problem. Windows are the exact
        # observed first/last affected calendar month per neighbor
        # (start/end of month, not day/hour-precise) -- deliberately
        # generous over the measured data to fully cover each transition's
        # partial-month edge (e.g. scheduled_net_de_to_at measured 28.6% in
        # 2024-06 and 53.0% in 2025-07, both transition months, both fully
        # inside the window below) rather than cut exactly at the observed
        # fraction.
        known_low_resolution_windows={
            "scheduled_net_de_to_at": ("2024-06-01", "2025-08-01"),
            "scheduled_net_de_to_ch": ("2024-06-01", "2025-07-01"),
            "scheduled_net_de_to_nl": ("2024-06-01", "2025-07-01"),
            "scheduled_net_de_to_pl": ("2024-06-01", "2025-04-01"),
            "scheduled_net_de_to_dk_1": ("2024-06-01", "2025-04-01"),
        },
    ),
    "cross_border_flows": SourceExpectation(
        expected_columns=frozenset(f"physical_net_de_to_{n.lower()}" for n in NEIGHBORS),
        # Same 0.001 backstop as scheduled_exchanges above -- A9's probe run
        # (2026-09-10, 75-day window) separately found an isolated settled
        # gap in physical_net_de_to_pl at 2026-09-02 03:00 UTC, the same
        # small/permanent-gap category as generation's finding, independent
        # of the resolution-transition finding below.
        max_nan_fraction={f"physical_net_de_to_{n.lower()}": 0.001 for n in NEIGHBORS},
        # physical cross-border flows are TSO actuals (realised, measured
        # after the fact) -- unlike scheduled_exchanges, which are a
        # day-ahead schedule. Every neighbor column gets the live-tail
        # settling exclusion.
        live_settling_columns=frozenset(f"physical_net_de_to_{n.lower()}" for n in NEIGHBORS),
        # Same hourly-vs-quarter-hourly resolution-transition phenomenon as
        # scheduled_exchanges, found in the same A9 real full-history
        # rebuild (2026-09-10): physical_net_de_to_{fr,pl} were genuinely
        # hourly for years (physical_net_de_to_dk_1 stayed quarter-hourly
        # throughout, hence no entry here); physical_net_de_to_{nl,at,ch}
        # additionally show one brief shared blip in 2021-08 (~2.2% that
        # month) at the exact same month FR/PL's multi-year transition
        # begins -- almost certainly the same network-wide 15-minute
        # settlement rollout, just already complete for these three borders
        # within that one month. Windows again generous to the calendar
        # month, not fitted to the exact observed fraction.
        known_low_resolution_windows={
            "physical_net_de_to_fr": ("2021-08-01", "2025-05-01"),
            "physical_net_de_to_pl": ("2021-08-01", "2024-07-01"),
            "physical_net_de_to_nl": ("2021-08-01", "2021-09-01"),
            "physical_net_de_to_at": ("2021-08-01", "2021-09-01"),
            "physical_net_de_to_ch": ("2021-08-01", "2021-09-01"),
        },
    ),
    "ttf_gas": SourceExpectation(
        expected_columns=frozenset({"ttf_gas_eur_per_mwh"}), grid_based=False
    ),
    "eua_co2": SourceExpectation(
        expected_columns=frozenset({"eua_co2_eur_per_t"}), grid_based=False
    ),
}

# A month is only judged for gaplessness once it is definitely over --
# mirrors the one-day publication buffer data/_entsoe_cache.py's own
# _is_complete_month() uses, re-derived independently rather than imported
# (spec section 2.4: the input control must not depend on the code it is
# there to double-check).
_COMPLETENESS_BUFFER: Final[pd.Timedelta] = pd.Timedelta(days=1)

# --- Two validation contexts, deliberately not one shared tolerance ---
#
# A9/A10 (2026-09-10) found that a single NaN tolerance cannot serve both
# of this store's real use cases at once:
#
# 1. "rebuild" -- scripts/rebuild_store.py, a one-shot or occasional
#    from-zero pass over years of history. max_nan_fraction (EXPECTATION_
#    TABLE, e.g. 0.001) is sized against that huge denominator -- a handful
#    of genuine isolated permanent gaps (6.5.1's weather gaps, A9's ENTSO-E
#    equivalents) stay negligible as a fraction. No settling buffer is
#    needed here: even a full day of live-tail lag is a tiny fraction of a
#    multi-year window.
# 2. "live" -- scripts/sync_store.py, a small incremental window (often
#    under a day). The exact same 0.001 fraction is meaningless there: one
#    isolated permanent cell in a ~20-row window is already ~5%, and
#    A10's first real wiring-probe run against production hit exactly
#    this ("row count shrank"-adjacent NaN-fraction failures on load/
#    generation from a single already-known cell, confirmed via
#    docs/sprint6_step6_7_1_log.md). A live sync needs its own, much
#    coarser tolerance, sized like a classic data-quality gate rather than
#    a permanent-gap allowance.
#
# LIVE_MAX_NAN_FRACTION (~5%) is that live-mode default; max_nan_fraction
# entries still act as a floor under it (a full 1.0 exemption like
# gen_nuclear's must never be *lowered* by the live default -- see
# validate_source's max() below).
LIVE_MAX_NAN_FRACTION: Final[float] = 0.05

# LIVE_SETTLING_BUFFER only matters for live_settling_columns (TSO
# actuals -- see SourceExpectation's own docstring for why forecasts/
# schedules don't need it at all). Sized from a real measurement, not
# fitted to one or two anecdotes: on 2026-09-10, the trailing contiguous
# NaN run right now was measured across every actuals column at once
# (tests/test_store.py-adjacent scratch check, not committed) -- most
# showed 0h, but gen_hard_coal showed 10h in that single snapshot. That
# range (0-10h in one sample) is wide enough that a short, precisely-fit
# buffer would misrepresent how settled the underlying signal really is.
# 24h is chosen deliberately round, not fitted: it comfortably covers the
# one high observation with real margin, and costs nothing beyond a day's
# apparent freshness in the manifest, since heal_recent's own
# HEAL_LOOKBACK_DAYS=10 re-examines and backfills the same trailing window
# on every later run regardless of what this buffer excluded today.
LIVE_SETTLING_BUFFER: Final[pd.Timedelta] = pd.Timedelta(hours=24)


def _infer_resolution_seconds(index: pd.DatetimeIndex) -> float | None:
    """Median spacing of index, independently of
    data/_entsoe_cache.py::_infer_resolution_seconds -- same idea (derive,
    don't hardcode), separate implementation, so a bug in one cannot
    silently validate the other's blind spot. None for <2 points: nothing
    to infer a spacing from, and a resolution-dependent check must not
    guess (spec section 2.4)."""
    if len(index) < 2:
        return None
    diffs = index.to_series().sort_values().diff().dropna()
    if diffs.empty:
        return None
    return diffs.median().total_seconds()


def validate_source(
    name: str,
    frame: pd.DataFrame,
    expectation: SourceExpectation,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    as_of: pd.Timestamp,
    previous: SourceManifestEntry | None = None,
    mode: Literal["rebuild", "live"] = "rebuild",
) -> ValidationResult:
    """Independent input control for one row-oriented source (spec section
    2.4). Checks, in order: exact column set; a sane tz-aware UTC index
    with no duplicate timestamps; for ``grid_based`` sources whose
    ``[period_start, period_end)`` is definitely over (``as_of`` past
    ``period_end`` plus a one-day buffer), gaplessness at the
    independently-inferred resolution; per-column NaN fraction against a
    tolerance and checked window that both depend on ``mode`` (see below);
    and, if ``previous`` is given, that the covered range and row count
    only ever grow.

    ``mode`` picks which of this store's two real use cases is being
    validated (A9/A10 finding, 2026-09-10 -- see LIVE_MAX_NAN_FRACTION's
    own comment for the full reasoning): ``"rebuild"`` (the default) uses
    ``expectation.max_nan_fraction`` as-is over the full
    ``[period_start, period_end)`` window, sized for scripts/
    rebuild_store.py's multi-year denominator. ``"live"`` is for scripts/
    sync_store.py's small incremental windows: the per-column allowance
    becomes ``max(expectation.max_nan_fraction.get(column, 0.0),
    LIVE_MAX_NAN_FRACTION)`` (a permanent exemption like gen_nuclear's 1.0
    is never *lowered* by the live default), and columns in
    ``expectation.live_settling_columns`` additionally get the trailing
    ``LIVE_SETTLING_BUFFER`` excluded from the checked window (TSO actuals
    only -- forecasts/schedules have no live-tail dynamic to exclude).
    ``expectation.known_low_resolution_windows`` are excluded from the NaN
    judgment in both modes, since those are real, dated historical facts,
    not a live-vs-rebuild distinction.

    Returns a result object rather than raising, so one bad source does
    not stop another source of the same run from being written (spec
    section 2.7) -- the caller (write_if_valid, scripts/sync_store.py)
    decides what a failing result means for the overall run.
    """
    reasons: list[str] = []

    actual_columns = set(frame.columns)
    missing = expectation.expected_columns - actual_columns
    extra = actual_columns - expectation.expected_columns
    if missing or extra:
        reasons.append(f"column mismatch -- missing: {sorted(missing)}, extra: {sorted(extra)}")

    index_usable = isinstance(frame.index, pd.DatetimeIndex)
    index_is_utc = False
    if not index_usable:
        reasons.append(f"index is not a DatetimeIndex (got {type(frame.index).__name__})")
    else:
        dt_index = cast(pd.DatetimeIndex, frame.index)
        index_is_utc = dt_index.tz is not None and str(dt_index.tz) == "UTC"
        if not index_is_utc:
            reasons.append(f"index is not tz-aware UTC (got {dt_index.tz!r})")
        if dt_index.has_duplicates:
            reasons.append(f"index has {int(dt_index.duplicated().sum())} duplicate timestamp(s)")

    # Period-bound checks need a tz-aware UTC index to compare against
    # period_start/period_end (both always tz-aware) without raising --
    # skipped, not silently coerced, when the index itself already failed
    # the check above.
    if index_usable and index_is_utc and not missing and not extra:
        dt_index = cast(pd.DatetimeIndex, frame.index)
        in_period = frame[(dt_index >= period_start) & (dt_index < period_end)]

        if expectation.grid_based:
            is_complete_period = period_end < as_of - _COMPLETENESS_BUFFER
            if is_complete_period:
                resolution_seconds = _infer_resolution_seconds(pd.DatetimeIndex(in_period.index))
                if resolution_seconds is None or resolution_seconds <= 0:
                    reasons.append(
                        f"cannot infer a resolution for the completed period "
                        f"{period_start}..{period_end} (too few points to measure spacing)"
                    )
                else:
                    expected_count = int(
                        (period_end - period_start).total_seconds() / resolution_seconds
                    )
                    present_count = len(in_period)
                    if present_count < expected_count:
                        reasons.append(
                            f"coverage gap: {present_count}/{expected_count} periods present "
                            f"for completed range {period_start}..{period_end} "
                            f"(inferred resolution {resolution_seconds:.0f}s)"
                        )

            nan_check_period = frame[(dt_index >= period_start) & (dt_index < period_end)]
            if len(nan_check_period):
                for column in sorted(expectation.expected_columns):
                    base_allowed = expectation.max_nan_fraction.get(column, 0.0)
                    allowed = (
                        max(base_allowed, LIVE_MAX_NAN_FRACTION) if mode == "live" else base_allowed
                    )
                    column_frame = nan_check_period
                    column_period_end = period_end

                    if mode == "live" and column in expectation.live_settling_columns:
                        column_period_end = min(period_end, as_of - LIVE_SETTLING_BUFFER)
                        column_index = cast(pd.DatetimeIndex, column_frame.index)
                        column_frame = column_frame[column_index < column_period_end]

                    window = expectation.known_low_resolution_windows.get(column)
                    if window is not None:
                        win_start = pd.Timestamp(window[0], tz="UTC")
                        win_end = pd.Timestamp(window[1], tz="UTC")
                        column_index = cast(pd.DatetimeIndex, column_frame.index)
                        column_frame = column_frame[
                            (column_index < win_start) | (column_index >= win_end)
                        ]
                    if not len(column_frame):
                        continue
                    nan_fraction = float(column_frame[column].isna().mean())
                    if nan_fraction > allowed:
                        reasons.append(
                            f"column {column!r}: NaN fraction {nan_fraction:.3f} "
                            f"exceeds allowed {allowed:.3f} in {period_start}..{column_period_end} "
                            f"(mode={mode!r}, outside any known_low_resolution_windows exclusion)"
                        )

    if previous is not None and index_usable and index_is_utc and len(frame.index):
        if previous.covered_end_utc is not None:
            prev_end = pd.Timestamp(previous.covered_end_utc)
            new_max = pd.Timestamp(frame.index.max())
            if new_max < prev_end:
                reasons.append(f"covered range shrank: new max {new_max} < previous {prev_end}")
        if len(frame) < previous.count:
            reasons.append(f"row count shrank: {len(frame)} < previous {previous.count}")

    return ValidationResult(source=name, ok=not reasons, reasons=tuple(reasons))


def write_if_valid(path: Path, frame: pd.DataFrame, result: ValidationResult) -> None:
    """Write frame to path (parquet, snappy) only if result.ok.

    A separate, minimal function rather than folding the write into
    validate_source: validate_source stays pure (no I/O, trivially
    testable), while this one-line gate is what actually makes "a failing
    source does not block another source of the same run" true in
    practice (spec section 2.7) -- each (validate, write_if_valid) pair is
    independent of every other source's outcome.
    """
    if not result.ok:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression="snappy")


# ---------------------------------------------------------------------------
# Heal step (spec section 2.6) -- refills gaps in the trailing
# HEAL_LOOKBACK_DAYS from the source itself, every maintenance run.
# extend_interim.py only ever appends forward from the last existing row
# and never re-checks already-written rows -- a source's one-off hiccup
# stays in the store forever under that design. This is what actually fixed
# that in 6.7.1.
# ---------------------------------------------------------------------------

HEAL_LOOKBACK_DAYS: Final[int] = 10  # EU Reg. 543/2013 allows up to 5 days of revision for actuals

RefetchFn = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class HealResult:
    source: str
    filled_cells: int
    still_missing_cells: int


def heal_recent(
    name: str,
    existing: pd.DataFrame,
    lookback_days: int,
    refetch: RefetchFn,
    *,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, HealResult]:
    """Refill gaps in the last ``lookback_days`` days (relative to
    ``as_of``) from the source itself.

    Four binding rules (spec section 2.6):

    1. Never invents a value -- no ffill, no interpolation, no zero. The
       only accepted replacement for a missing cell is a fresh value from
       the same source.
    2. ``refetch`` MUST bypass the cache -- in 6.6 the cache itself was the
       cause of the incident. Callers pass the underlying client query
       function directly (e.g. entsoe-py's raw query, or
       ``weather_client.fetch_run(..., use_cache=False)``), never
       ``data/_entsoe_cache.py::cached_fetch``.
    3. Only missing cells are filled (``combine_first`` semantics -- a pure
       pandas primitive, reused rather than reimplemented, since it was
       never implicated in the 6.6 bug). An existing non-NaN value is never
       overwritten: ENTSO-E revises actuals continuously, and those
       revisions must not seep into rows this store already wrote.
    4. The proof is executed, not claimed: every pre-existing non-NaN cell
       is verified byte-identical after the heal (raises StoreError if not
       -- this must never silently happen), and the number of filled cells
       is reported. A cell the refetch still can't fill stays as NaN and is
       reported, never interpolated.
    """
    window_start = as_of - pd.Timedelta(days=lookback_days)
    fresh = refetch(window_start, as_of)

    healed = existing.combine_first(fresh)
    if len(existing.columns):
        healed = healed[existing.columns]

    before_window = existing[(existing.index >= window_start) & (existing.index < as_of)]
    after_window = healed[(healed.index >= window_start) & (healed.index < as_of)]

    common_index = before_window.index.intersection(after_window.index)
    for column in existing.columns:
        prior = before_window.loc[common_index, column]
        prior_present = prior.notna()
        if not prior_present.any():
            continue
        healed_vals = after_window.loc[common_index, column]
        if not healed_vals[prior_present].equals(prior[prior_present]):
            raise StoreError(
                f"heal_recent for {name!r} would have changed an existing value in column "
                f"{column!r} -- refusing (spec section 2.6, rule 3: existing values are never "
                "overwritten)"
            )

    before_reindexed = before_window.reindex(after_window.index)
    filled_mask = before_reindexed.isna() & after_window.notna()
    filled_cells = int(filled_mask.to_numpy().sum())
    still_missing_cells = int(after_window.isna().to_numpy().sum())

    return healed, HealResult(
        source=name, filled_cells=filled_cells, still_missing_cells=still_missing_cells
    )


# ---------------------------------------------------------------------------
# Weather input control -- a separate, lighter path from validate_source
# above: the per-run weather cache (data/_weather_cache.py) is not a
# growing monthly series, it's one immutable file per (model, run_init_utc)
# that either exists complete or does not exist at all (weather_client.py's
# fetch_run raises WeatherRunUnavailable rather than ever writing a partial
# file) -- "coverage"/"monotonic growth" are the wrong questions to ask of
# a single snapshot. Column-set and per-variable NaN checks still apply.
# ---------------------------------------------------------------------------

# The two radiation variables are None at night by construction (measured,
# not assumed -- spike 6.5.0 F7/F9, data/weather_grid.py's own
# ConventionProvenance doc). Every other variable is expected fully
# present at every point/hour of a valid run.
_WEATHER_NAN_ALLOWED_VARIABLES: Final[frozenset[str]] = frozenset(
    {"shortwave_radiation", "direct_normal_irradiance"}
)


def validate_weather_run(df: pd.DataFrame) -> ValidationResult:
    """Independent input control for one cached weather run file."""
    reasons: list[str] = []

    actual_columns = set(df.columns)
    expected = set(expected_columns())
    missing = expected - actual_columns
    extra = actual_columns - expected
    if missing or extra:
        reasons.append(f"column mismatch -- missing: {sorted(missing)}, extra: {sorted(extra)}")

    if len(df) == 0:
        reasons.append("run file has zero rows")

    if not missing and not extra and len(df):
        for variable in HOURLY_VARIABLES:
            if variable in _WEATHER_NAN_ALLOWED_VARIABLES:
                continue
            variable_columns = [c for c in df.columns if c.endswith(f"__{variable}")]
            nan_fraction = float(df[variable_columns].isna().mean().mean())
            if nan_fraction > 0.0:
                reasons.append(
                    f"variable {variable!r}: {nan_fraction:.3f} NaN fraction, "
                    "expected fully present (not a radiation variable)"
                )

    return ValidationResult(source="weather_single_runs", ok=not reasons, reasons=tuple(reasons))


def validate_historical_weather_runs(
    root: Path, as_of: pd.Timestamp
) -> tuple[SourceManifestEntry | None, tuple[str, ...]]:
    """Validate every already-cached weather run file under root (spec:
    rebuild_store.py does not re-fetch weather, section 5.6 -- it only
    packs whatever data/_weather_cache.py already wrote). Without this,
    "weather_single_runs" never gets a manifest entry from a rebuild run,
    which after pack_store's per-source gating (see _matched_paths) meant
    weather was silently dropped from the packed tar entirely -- caught in
    practice, not just in theory: A9's second real rebuild (2026-09-10)
    produced a store missing all weather data because of exactly this gap.

    All-or-nothing across every cached file found, matching the
    per-run-file granularity spec §2.5 already uses elsewhere: one silently
    bad run file mixed in with hundreds of good ones is a finding that
    should block packing weather entirely for this run, not be papered
    over. Returns (None, reasons) if any file fails or none exist; the
    caller decides what a None entry means for the overall run (mirrors
    validate_source/write_if_valid's own split of concerns).
    """
    paths = _matched_paths(root, ("weather_single_runs",))
    if not paths:
        return None, ("no cached weather run files found",)

    run_inits: list[pd.Timestamp] = []
    for rel_path in paths:
        df = pd.read_parquet(root / rel_path)
        result = validate_weather_run(df)
        if not result.ok:
            return None, (f"{rel_path}: {'; '.join(result.reasons)}",)
        # Filename is "{YYYY-MM-DD}THHZ.parquet" (data/_weather_cache.py::
        # cache_path) -- the run's own init time, not read from file content
        # since HOURLY_VARIABLES columns hold forecast valid-times, not the
        # run init itself.
        date_part, hour_part = rel_path.stem.split("T")
        run_inits.append(pd.Timestamp(f"{date_part}T{hour_part.rstrip('Z')}:00:00", tz="UTC"))

    return (
        SourceManifestEntry(
            covered_start_utc=min(run_inits).isoformat(),
            covered_end_utc=max(run_inits).isoformat(),
            count=len(run_inits),
            last_success_utc=as_of.isoformat(),
            last_attempt_utc=as_of.isoformat(),
        ),
        (),
    )


# ---------------------------------------------------------------------------
# Size and deadline guards (spec section 2.3/5.5) -- both are "grün mit
# Warnung" by construction: neither function raises. A run must never turn
# red over a size or deadline finding (spec section 2.7's rot/grün table
# lists this explicitly), so making that impossible structurally (no
# exception path at all) is stronger than trusting every caller to catch
# one.
# ---------------------------------------------------------------------------

# 500 MB: comfortably above the "niedriger dreistelliger MB-Bereich" the
# spec itself estimates for the whole store (section 2.3) and comfortably
# below the 2 GB per-asset limit -- a warning here means the store is
# growing in a way nobody decided, not that it's about to break.
STORE_SIZE_WARN_BYTES: Final[int] = 500 * 1024 * 1024

DEADLINE_WARN_DAYS: Final[int] = 30


@dataclass(frozen=True)
class Deadline:
    name: str
    date_utc: str  # ISO date, e.g. "2026-12-07"
    note: str


# Exactly one entry for 6.7.1 (spec section 5.5): the external trigger's
# fine-grained PAT (ops/trigger/, GITHUB_DISPATCH_TOKEN). If it expires
# unnoticed, nothing fires and nothing errors -- the worst failure mode in
# a system whose only monitoring is "a run turns red" (spec section 5.5).
# 6.7.2 adds the capacity-anchor extrapolation boundary (2026-10-28, per
# the 6.6 log) as a second entry -- the mechanism is built generically here
# specifically so that addition is a tuple entry, not new code.
DEADLINE_TABLE: Final[tuple[Deadline, ...]] = (
    Deadline(
        name="trigger_pat_expiry",
        date_utc="2026-12-07",
        note=(
            "Fine-grained PAT for ops/trigger/ (secret GITHUB_DISPATCH_TOKEN) expires -- "
            "rotate per ops/trigger/README.md, section 'Rotating the token'."
        ),
    ),
)


def check_store_size(size_bytes: int) -> str | None:
    """A warning string if size_bytes exceeds STORE_SIZE_WARN_BYTES, else
    None. Never raises -- a run must not go red for this (spec sections
    2.3, 2.7)."""
    if size_bytes > STORE_SIZE_WARN_BYTES:
        return (
            f"store size {size_bytes:,} bytes exceeds the warn threshold "
            f"{STORE_SIZE_WARN_BYTES:,} bytes -- check that no derived artefact is "
            "leaking into the positive list (spec section 2.3)"
        )
    return None


def check_deadlines(
    as_of: pd.Timestamp,
    table: tuple[Deadline, ...] = DEADLINE_TABLE,
    warn_days: int = DEADLINE_WARN_DAYS,
) -> tuple[str, ...]:
    """A warning string for every deadline in table within warn_days of
    as_of (or already past). Never raises."""
    warnings: list[str] = []
    for deadline in table:
        deadline_ts = pd.Timestamp(deadline.date_utc, tz="UTC")
        days_left = (deadline_ts - as_of).days
        if days_left <= warn_days:
            status = (
                f"OVERDUE by {-days_left} day(s)" if days_left < 0 else f"in {days_left} day(s)"
            )
            warnings.append(f"{deadline.name} ({deadline.date_utc}): {status} -- {deadline.note}")
    return tuple(warnings)
