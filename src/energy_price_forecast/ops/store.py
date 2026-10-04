"""Persistent raw-data store: packing, manifest, versioned publish, input
control and the heal step (spec 6.7.1, Entscheidungen 17-19, section 5.3).
6.7.1a (2026-09-11) reworked validate_source's live-mode gate (a fixed
NaN window, a gross-corruption-only block, non-blocking per-column hints)
and split EXPECTATION_TABLE into CHECKED vs. CARRIED columns -- see
validate_source's and SourceExpectation's own docstrings.

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
import logging
import os
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

from energy_price_forecast.data.entsoe_client import NEIGHBORS
from energy_price_forecast.data.weather_grid import (
    HOURLY_VARIABLES,
    KNOWN_WEATHER_DEFECTS,
    KnownWeatherDefectCategory,
    expected_columns,
)
from energy_price_forecast.ops import release_assets
from energy_price_forecast.ops.release_assets import AssetRef

logger = logging.getLogger(__name__)

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
    "day_ahead_price_ec": ("data/raw/energy_charts/store/day_ahead_price_ec.parquet",),
    "load_forecast_day_ahead_ec": (
        "data/raw/energy_charts/store/load_forecast_day_ahead_ec.parquet",
    ),
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
    # Absolute NaN cell count per CHECKED column over validate_source's
    # live-mode fixed window (6.7.1a, spec section 5.3) -- empty for
    # rebuild-produced entries and for any manifest written before this
    # field existed (from_dict tolerates a missing key, spec section 4.1
    # point 4 / section 8). Not decoration: this is the direct input to
    # 6.7.2's own per-column freshness check (spec section 11 point 2),
    # which reads it rather than re-deriving it.
    live_nan_cell_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_start_utc": self.covered_start_utc,
            "covered_end_utc": self.covered_end_utc,
            "count": self.count,
            "last_success_utc": self.last_success_utc,
            "last_attempt_utc": self.last_attempt_utc,
            "live_nan_cell_counts": self.live_nan_cell_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceManifestEntry:
        return cls(
            covered_start_utc=data["covered_start_utc"],
            covered_end_utc=data["covered_end_utc"],
            count=data["count"],
            last_success_utc=data["last_success_utc"],
            last_attempt_utc=data["last_attempt_utc"],
            live_nan_cell_counts=data.get("live_nan_cell_counts", {}),
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
# Encryption (publication spec, section 2.1) -- the store contains
# third-party market data (TTF via Yahoo Finance) that may not be
# redistributed, so the packed archive is encrypted before it ever leaves
# this process and decrypted right after it arrives, nowhere else. Fernet
# (``cryptography``) is symmetric, authenticated encryption: a wrong key or
# a damaged file fails loudly (InvalidToken) rather than silently handing
# back garbage. Code and every other data source in this repo are public;
# only the packed archive as a whole is not.
# ---------------------------------------------------------------------------

STORE_ENCRYPTION_KEY_ENV_VAR: Final[str] = "STORE_ENCRYPTION_KEY"

# A Fernet token is urlsafe-base64 of a fixed-format binary blob starting
# with a version byte 0x80 -- base64-encoding that single byte always
# produces the literal ASCII prefix "gAAAAA". Used below to reject a
# plaintext archive with a clear error before ever attempting to decrypt
# it (publication spec section 8: the transitional read path that used to
# accept plaintext here was removed once the observation phase cleared).
_FERNET_PREFIX: Final[bytes] = b"gAAAAA"


class StoreEncryptionError(StoreError):
    """STORE_ENCRYPTION_KEY is missing/invalid, or an archive could not be
    decrypted with it."""


def _fernet() -> Fernet:
    key = os.environ.get(STORE_ENCRYPTION_KEY_ENV_VAR)
    if not key:
        raise StoreEncryptionError(
            f"{STORE_ENCRYPTION_KEY_ENV_VAR} is not set -- the store cannot be read or "
            "written without it (README 'Data store and scheduling' section). If the key "
            "itself is lost, scripts/rebuild_store.py can rebuild the store from source "
            "instead of decrypting an existing archive."
        )
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise StoreEncryptionError(
            f"{STORE_ENCRYPTION_KEY_ENV_VAR} is set but is not a valid Fernet key."
        ) from exc


def _encrypt_store_archive(path: Path) -> None:
    """Encrypt path's bytes in place with Fernet. The only call site is
    publish_store, directly before upload_asset -- this is what makes
    "the store never uploads plaintext" true: _fernet() raises before
    upload_asset is ever reached if the key is missing or invalid."""
    plaintext = path.read_bytes()
    token = _fernet().encrypt(plaintext)
    path.write_bytes(token)


def _decrypt_store_archive(path: Path) -> None:
    """Decrypt path's bytes in place. Raises StoreEncryptionError if the
    archive is not a Fernet token -- plaintext store archives are no longer
    supported (publication spec section 8, the transitional read path from
    section 2.1 was removed once the observation phase cleared)."""
    raw = path.read_bytes()
    if not raw.startswith(_FERNET_PREFIX):
        raise StoreEncryptionError(
            f"{path.name!r} is not a Fernet-encrypted archive -- plaintext store "
            "archives are no longer supported."
        )
    try:
        plaintext = _fernet().decrypt(raw)
    except InvalidToken as exc:
        raise StoreEncryptionError(
            f"Could not decrypt {path.name!r} with {STORE_ENCRYPTION_KEY_ENV_VAR} -- wrong "
            "or corrupted key, or the file is damaged."
        ) from exc
    path.write_bytes(plaintext)


# ---------------------------------------------------------------------------
# Kernfunktionen (spec section 5.3)
# ---------------------------------------------------------------------------


def load_store(workdir: Path) -> StoreState:
    """Download the newest store asset, decrypt it, and unpack it into
    workdir.

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
        _decrypt_store_archive(tar_path)
        logger.info("Store archive %s decrypted", newest.name)
        manifest = unpack_store(tar_path, workdir)
    return StoreState(workdir=workdir, manifest=manifest)


def publish_store(workdir: Path, manifest: Manifest) -> AssetRef:
    """Pack, encrypt, upload as a new timestamped asset, then prune to the
    newest STORE_KEEP_VERSIONS. Never overwrites an existing asset (spec
    section 2.5) -- the timestamp in manifest.created_at_utc is also the
    asset's file name, so two publishes in the same second would collide;
    callers are expected to call this at most once per run.
    """
    release = release_assets.ensure_release(STORE_RELEASE_TAG)
    ts = pd.Timestamp(manifest.created_at_utc).strftime("%Y%m%dT%H%M%SZ")
    tar_name = f"store-{ts}.tar"
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / tar_name
        pack_store(workdir, manifest, tar_path)
        _encrypt_store_archive(tar_path)
        asset = release_assets.upload_asset(release, tar_path)
    release_assets.prune_assets(release, STORE_KEEP_VERSIONS)
    return asset


# ---------------------------------------------------------------------------
# Pure on-disk cache reader (spec 6.7.2, section 5.8)
# ---------------------------------------------------------------------------


def _group_frames_by_shared_columns(frames: list[pd.DataFrame]) -> list[list[pd.DataFrame]]:
    """Partition frames into connected components where two frames are
    connected if they share at least one column name (transitively) --
    see read_cached_range's own docstring for why exact column-set
    equality is too strict a grouping key.
    """
    n = len(frames)
    column_sets = [set(f.columns) for f in frames]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(n):
        for j in range(i + 1, n):
            if column_sets[i] & column_sets[j]:
                union(i, j)

    components: dict[int, list[pd.DataFrame]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(frames[i])
    return list(components.values())


def read_cached_range(
    cache_dir: Path, *, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Read every month/neighbour parquet under cache_dir straight from
    disk, concatenated and sorted. Pure I/O: no client, no network, no write.

    Moved here from scripts/sync_store.py::_read_cache_dir (6.7.1a), where
    it was introduced for the heal read-back verification. The submission
    job needs exactly the same thing, and a second copy under arena/ would
    be the duplication this project keeps refusing.

    Why it must exist at all: cached_fetch() never treats the *current*
    calendar month as a cache hit (_is_complete_month requires a closed
    month), so reading a training window that reaches to yesterday through
    the client triggers a live API call AND a disk write -- see spec
    section 4, rule 4.

    Deliberately glob-based: it knows neither the file prefix nor any month
    arithmetic, so the cache layout stays owned solely by
    data/_entsoe_cache.py.

    **Files are grouped by shared columns before concatenating** (spec
    6.7.2, found live during the wiring probe, 2026-09-12, in two stages).

    Stage 1 finding: scheduled_exchanges/cross_border_flows share one
    cache_dir across all six neighbours (ops/store_sources.py::
    _fetch_border_flows), each neighbour's own month chunks carrying only
    *that* neighbour's single column (e.g. ``DE_LU_AT_2020-01.parquet`` has
    only ``scheduled_net_de_to_at``). A blind ``pd.concat(frames)``
    (row-wise) stacked all six neighbours' same-timestamp rows on top of
    each other instead of joining them side by side -- a duplicated index
    with >75% NaN per column on the real store, which then made a
    downstream ``pd.concat(..., axis=1)``
    (arena/live_inputs.py::assemble_price_model_inputs) raise
    ``InvalidIndexError`` outright.

    Stage 2 finding, caught by the same wiring probe run: grouping by
    *exact* column-set equality is too strict. ``generation``'s real cache
    has two distinct column sets -- one with ``gen_nuclear``, one without
    -- because Germany's last nuclear plants shut down in 2023, so
    ENTSO-E's response genuinely omits the "Nuclear" generation type from
    that month onward (entsoe_client.py::fetch_generation_by_type only
    emits a column when the type is actually present in the response,
    spec-legitimate, not a bug there). Exact-equality grouping treated
    those as two disjoint "neighbours" and joined them column-wise,
    producing duplicate column names (every shared column appearing
    twice). The real distinguishing property isn't identical columns, it's
    *any shared column at all*: files are grouped into connected
    components (union-find) where two files join the same component if
    they share at least one column name, transitively. A source with
    genuinely disjoint per-file columns (the neighbour case) still forms
    one component per neighbour; a source whose per-file columns merely
    drift over time while mostly overlapping (the generation case) forms
    one single component, exactly like a source with fully identical
    columns everywhere (day_ahead_price/load/wind_solar) already did.
    Frames within a component concatenate row-wise (time); components
    concatenate column-wise (an outer join on the row-merged, sorted time
    index of each).

    ``start``/``end`` slice AFTER reading rather than filtering month files.
    The whole ENTSO-E holding is roughly 60 MB of parquet; premature
    filtering would buy nothing and add month-boundary arithmetic, which is
    the exact class of bug this project has already had once.
    """
    if not cache_dir.exists():
        return pd.DataFrame()
    paths = sorted(cache_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in paths]
    groups = _group_frames_by_shared_columns(frames)

    per_group = [pd.concat(group).sort_index() for group in groups]
    combined = per_group[0] if len(per_group) == 1 else pd.concat(per_group, axis=1, join="outer")

    if isinstance(combined.index, pd.DatetimeIndex) and combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    combined = combined.sort_index()
    if start is not None or end is not None:
        combined = combined.loc[start:end]
    return combined


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
    # Non-blocking findings (6.7.1a, spec section 3.4/5.3): a fine per-column
    # NaN reading below LIVE_GROSS_NAN_FRACTION, a missing/extra carried
    # column. Deliberately a SEPARATE field from `reasons`, not appended to
    # it (spec section 4.1 point 3) -- a blocking and a non-blocking finding
    # must never land in the same list, or some future `if result.reasons`
    # check would start failing on a hint that was never meant to block.
    hints: tuple[str, ...] = ()
    # Absolute NaN cell count per CHECKED column over the live-mode fixed
    # window (spec section 5.3) -- only populated for mode="live". The
    # caller (scripts/sync_store.py) carries this into
    # SourceManifestEntry.live_nan_cell_counts.
    nan_cell_counts: dict[str, int] = field(default_factory=dict)


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

    ``carried_columns`` (6.7.1a, spec section 3.6) names columns that are
    fetched and stored but read by NO consumer under features/, models/, or
    evaluation/ -- derived by a real code search, not assumed, and pinned by
    tests/test_store.py's own source-scan regression test so the split can't
    silently go stale. ``expected_columns`` above is therefore, as of
    6.7.1a, the CHECKED set: present-required and NaN-gated in both modes.
    A carried column is never a blocking reason in either mode -- a missing
    one is a hint, and its NaN fraction (measured only in "rebuild" mode,
    for the owner's own future-development tracking; never measured at all
    in "live" mode, since nobody reads the number there) is reported, never
    gated. Spec section 2.4's A9 finding is the concrete trigger: a gap in
    gen_hard_coal -- read by nothing -- froze the whole "generation" fetch
    group, including gen_wind_onshore/_offshore/_solar, which the price
    model actually needs (features/lags.py's forecast-error lags).
    """

    expected_columns: frozenset[str]
    max_nan_fraction: dict[str, float] = field(default_factory=dict)
    grid_based: bool = True
    known_low_resolution_windows: dict[str, tuple[str, str]] = field(default_factory=dict)
    live_settling_columns: frozenset[str] = field(default_factory=frozenset)
    carried_columns: frozenset[str] = field(default_factory=frozenset)
    # relevant_only_eligible (2026-10-01 maintenance-timeout fix): True only
    # for a source scripts/sync_store.py's --mode relevant-only may skip
    # entirely during the gate-closure-adjacent window. Deliberately a
    # SEPARATE field from expected_columns/carried_columns above, not a
    # third value folded into that split: carried_columns answers "does any
    # feature/model/evaluation code read this column" (pinned by the
    # source-scan test below); this field answers a different question,
    # "is it operationally safe to skip fetching this source for a few
    # hours" -- and the two answers do NOT always agree. day_ahead_price_ec/
    # load_forecast_day_ahead_ec are fully carried but never eligible here
    # (actively read via arena/live_inputs.py::coalesce_price as the
    # ENTSO-E-outage fallback). eua_co2 is fully carried and has no
    # feature/model dependency either, but is also NOT eligible: real-code
    # search found arena/preflight.py::check_commodity_staleness reads it
    # by default, called unconditionally from run_daily_submission.py --
    # skipping its fetch here would make that staleness signal partly
    # self-inflicted instead of reflecting genuine upstream lag. Default
    # False is the safe choice -- a source must be explicitly proven
    # skip-eligible, not assumed so from carried_columns alone.
    relevant_only_eligible: bool = False


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
        # CHECKED (6.9 spec section 2.5, Owner 2026-09-24): only
        # load_forecast_day_ahead -- row 1 reads it, and row 2's Similar-Day-
        # Patch (arena/load_patch.py) needs it for the reference-day
        # completeness check. load_actual moved to carried below: no live
        # price-model row reads it directly (the actual-load lags spec
        # section 2.1 already dropped from every row's own requirement).
        expected_columns=frozenset({"load_forecast_day_ahead"}),
        # A9's real full-history run (2026-09-10): load_forecast_day_ahead had
        # two full missing calendar days (2022-02-xx, 2022-03-xx, 96
        # quarter-hours each) plus known DST fall-back hours -- same small/
        # isolated/permanent-gap category as generation's finding.
        max_nan_fraction={"load_forecast_day_ahead": 0.001},
        carried_columns=frozenset({"load_actual"}),
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
        # Entirely CARRIED (6.9 spec section 2.5, Owner 2026-09-24): the three
        # gen_wind_onshore/_offshore/_solar columns 6.7.1a checked (they fed
        # features/lags.py's forecast-error lags) join the eight already-
        # carried gen_* types below, since spec 6.9 section 2.1 drops the
        # forecast-error-lag track from every fallback-ladder row entirely
        # ("Keine Zeile verlangt diese Spalten mehr"). grid_based=False for
        # the same reason the two EC sources need it (see their own comment
        # below): with grid_based=True, the row-count coverage-gap check
        # would still block a sync on this now-fully-carried source
        # regardless of which columns are checked, reintroducing exactly the
        # C4 gen_hard_coal bug class this table already fixed once.
        expected_columns=frozenset(),
        carried_columns=frozenset(
            {
                "gen_wind_onshore",
                "gen_wind_offshore",
                "gen_solar",
                "gen_nuclear",
                "gen_lignite",
                "gen_hard_coal",
                "gen_gas",
                "gen_oil",
                "gen_biomass",
                "gen_hydro",
                "gen_other",
            }
        ),
        grid_based=False,
        # relevant_only_eligible (2026-10-01): confirmed by a real code
        # search of src/ and scripts/ that no gen_* column is read outside
        # entsoe_client.py (the fetch itself), features/availability.py
        # (bookkeeping, already excluded from the consumer scan below),
        # ops/availability_audit.py (a separate, non-live DQ audit), and the
        # already-accepted "computed but unrequired" features/lags.py /
        # evaluation/regimes.py hits for the three wind/solar columns (see
        # _unrequired_feature_columns below) -- none of that is the live
        # submission path (run_daily_submission.py / arena/).
        relevant_only_eligible=True,
    ),
    "scheduled_exchanges": SourceExpectation(
        # Entirely CARRIED (6.9 spec section 2.5, Owner 2026-09-24): all 12
        # neighbor columns, previously CHECKED because features/lags.py's
        # _total_flow_lag summed every one of them (6.7.1a finding) -- spec
        # 6.9 section 2.1 drops the cross-border-flow-lag track from every
        # fallback-ladder row entirely ("Keine Zeile verlangt diese Spalten
        # mehr"), so nothing reads this source any more. grid_based=False so
        # the row-count coverage-gap check (column-agnostic) does not still
        # block a sync on a now-fully-carried source -- same reasoning as
        # "generation" above.
        expected_columns=frozenset(),
        carried_columns=frozenset(f"scheduled_net_de_to_{n.lower()}" for n in NEIGHBORS),
        # grid_based=False disables the whole coverage/NaN block (see
        # validate_source), including the carried-column NaN-fraction hint
        # reporting -- the same trade-off already accepted for the two EC
        # sources below (own comment: "duplicating that here would be
        # redundant, not a gap"), not a new one. known_low_resolution_windows
        # is therefore dropped here too: with grid_based=False it would never
        # be consulted (dead configuration), unlike when this source was
        # still partly CHECKED.
        grid_based=False,
        # relevant_only_eligible (2026-10-01): one of the two sources behind
        # the near-miss maintain_store.yml timeout documented in
        # docs/bugs_in_live_system.md entry 6 -- a real code search found no
        # scheduled_net_de_to_* column read anywhere outside
        # entsoe_client.py/availability.py/availability_audit.py (same
        # reasoning as "generation" above).
        relevant_only_eligible=True,
    ),
    "cross_border_flows": SourceExpectation(
        # Entirely CARRIED -- see scheduled_exchanges' own comment above for
        # the reasoning (same fetch group, same dropped cross-border-flow-lag
        # track, spec 6.9 section 2.1/2.5) and for why grid_based=False also
        # means known_low_resolution_windows is dropped, not just moved.
        expected_columns=frozenset(),
        carried_columns=frozenset(f"physical_net_de_to_{n.lower()}" for n in NEIGHBORS),
        grid_based=False,
        # relevant_only_eligible (2026-10-01): the OTHER, bigger source
        # behind the near-miss documented in docs/bugs_in_live_system.md
        # entry 6 -- the live API cache-miss cascade for the current
        # month's physical_net_de_to_* columns (6 neighbours) dominated the
        # 1085s worst-observed "Sync the data store" step. Same "read by
        # nothing outside fetch/bookkeeping/audit code" finding as
        # scheduled_exchanges above.
        relevant_only_eligible=True,
    ),
    "ttf_gas": SourceExpectation(
        expected_columns=frozenset({"ttf_gas_eur_per_mwh"}), grid_based=False
    ),
    "eua_co2": SourceExpectation(
        # CARRIED (6.9 spec section 2.5, Owner 2026-09-24): no fallback-ladder
        # row reads EUA any more (spec section 2.1 -- rows 1/2 keep only TTF
        # gas, row 3 is gas-free entirely). grid_based is already False here
        # (commodities are non-grid, daily, trading-days-only), so no change
        # needed there.
        expected_columns=frozenset(),
        carried_columns=frozenset({"eua_co2_eur_per_t"}),
        grid_based=False,
        # NOT relevant_only_eligible, deliberately (2026-10-01): fully
        # carried for the feature/model split above, but a real code search
        # found arena/preflight.py::check_commodity_staleness reads
        # eua_co2_eur_per_t by default, called unconditionally from
        # run_daily_submission.py -- skipping this source's fetch in
        # relevant-only mode would make that staleness signal partly
        # self-inflicted (stale because we chose not to fetch, not because
        # of genuine upstream lag) rather than a clean measurement, which is
        # exactly the thing 6.7.2's deferred acceptance criterion is
        # waiting to observe for real. In practice this costs nothing:
        # _sync_commodity_source's own once-per-UTC-day cadence gate (spec
        # section 2.8) already limits this source to one real fetch a day
        # regardless, so an explicit relevant-only skip would rarely even
        # have been a no-op API call it saved.
    ),
    # Energy-Charts, a permanent second source (6.9 spec section 2.4/2.5):
    # "Ein EC-Fehler im Pflege-Job ist nur eine Warnung, der Lauf bleibt
    # grün. Die EC-Quellen gelten als carried." -- entirely carried (no
    # expected_columns at all), and grid_based=False deliberately, the same
    # choice already made for ttf_gas/eua_co2 above: with grid_based=True,
    # the coverage-gap check a few lines above runs unconditionally on the
    # whole frame regardless of whether any column is CHECKED, which would
    # reintroduce exactly the bug class 6.7.1a fixed for gen_hard_coal (a
    # fully-carried source still blocking a sync). EC's own grid is
    # measured and reported by its own scripts (build_grid_report,
    # unchanged from 6.8) -- duplicating that here would be redundant, not
    # a gap.
    "day_ahead_price_ec": SourceExpectation(
        expected_columns=frozenset(),
        carried_columns=frozenset({"day_ahead_price_ec"}),
        grid_based=False,
    ),
    "load_forecast_day_ahead_ec": SourceExpectation(
        expected_columns=frozenset(),
        carried_columns=frozenset({"load_forecast_day_ahead_ec"}),
        grid_based=False,
    ),
}

# A month is only judged for gaplessness once it is definitely over --
# mirrors the one-day publication buffer data/_entsoe_cache.py's own
# _is_complete_month() uses, re-derived independently rather than imported
# (spec section 2.4: the input control must not depend on the code it is
# there to double-check).
_COMPLETENESS_BUFFER: Final[pd.Timedelta] = pd.Timedelta(days=1)

# --- Two validation contexts, deliberately not one shared mechanism ---
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
#    under a day). A9/A10's original fix (LIVE_MAX_NAN_FRACTION, a ~5%
#    floor) turned out to still be the wrong shape of answer: 6.7.1a
#    (spec section 2.3, real 2026-09-11 finding) found the checked NaN
#    window itself was unstably sized -- anywhere from a few hours to
#    hundreds of days wide, depending on how large the current sync gap
#    happened to be -- so a *fraction* threshold over it could never mean
#    the same thing twice. LIVE_NAN_WINDOW_DAYS fixes the denominator
#    instead of tuning the threshold: the NaN fraction is now always
#    measured over the same [as_of - 7d, as_of) span regardless of the
#    sync gap, live_settling_columns' trailing exclusion shrinks that fixed
#    window by only a known fraction (24h of 7d) rather than shrinking an
#    already-variable one further, and 6.7.1a's second, layered change
#    (spec section 3.4) makes the live gate itself coarser: only gross
#    corruption (LIVE_GROSS_NAN_FRACTION) or a structural finding blocks; a
#    real but partial gap -- like the gen_hard_coal one that produced two
#    real red maintenance runs on 2026-09-11 -- is recorded (manifest +
#    log) and left for 6.7.2's own per-column freshness check to judge
#    against the actual submission's needs, not frozen out of the store by
#    a gate that cannot see which columns matter.
LIVE_NAN_WINDOW_DAYS: Final[int] = 7

# Only a NaN fraction at or above this, over the fixed LIVE_NAN_WINDOW_DAYS
# window, blocks a live sync (spec section 3.4) -- gross corruption (e.g.
# A11's 100%-NaN negative-probe month), not an ordinary partial gap.
# Chosen, not fitted: even a multi-day full outage of one CHECKED column
# stays well under 0.9 against a 7-day denominator, so this threshold is
# reached only by something close to "this column is not really there this
# run", never by the kind of gap 6.7.1a exists to stop from blocking.
LIVE_GROSS_NAN_FRACTION: Final[float] = 0.9

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
    validated (A9/A10, 2026-09-10; reworked again 6.7.1a, 2026-09-11 -- see
    LIVE_NAN_WINDOW_DAYS' own comment for the full reasoning):
    ``"rebuild"`` (the default) uses ``expectation.max_nan_fraction`` as-is
    over the full ``[period_start, period_end)`` window for CHECKED columns
    only, sized for scripts/rebuild_store.py's multi-year denominator;
    CARRIED columns (``expectation.carried_columns``) are measured and
    reported as hints, never gated. ``"live"`` is for scripts/
    sync_store.py's small incremental windows: CHECKED columns are measured
    over a FIXED ``[as_of - LIVE_NAN_WINDOW_DAYS, as_of)`` window regardless
    of how wide the actual sync gap is, with columns in
    ``expectation.live_settling_columns`` additionally getting the trailing
    ``LIVE_SETTLING_BUFFER`` excluded (TSO actuals only -- forecasts/
    schedules have no live-tail dynamic to exclude); only a fraction at or
    above ``LIVE_GROSS_NAN_FRACTION`` blocks, everything below is recorded
    as a hint plus an absolute cell count in ``nan_cell_counts`` and never
    gates. CARRIED columns are not measured at all in live mode -- nobody
    reads the number there. ``expectation.known_low_resolution_windows``
    are excluded from the NaN judgment in both modes and for both column
    classes, since those are real, dated historical facts, not a
    live-vs-rebuild or checked-vs-carried distinction. The coverage-gap
    check (gaplessness at the inferred resolution) is unaffected by any of
    this -- it always uses ``[period_start, period_end)``, a different
    question ("did this sync close the gap it meant to") than the NaN
    checks answer.

    Returns a result object rather than raising, so one bad source does
    not stop another source of the same run from being written (spec
    section 2.7) -- the caller (write_if_valid, scripts/sync_store.py)
    decides what a failing result means for the overall run.
    """
    reasons: list[str] = []
    hints: list[str] = []
    nan_cell_counts: dict[str, int] = {}

    checked = expectation.expected_columns
    carried = expectation.carried_columns
    actual_columns = set(frame.columns)
    missing_checked = checked - actual_columns
    if missing_checked:
        reasons.append(f"column mismatch -- missing: {sorted(missing_checked)}")
    missing_carried = carried - actual_columns
    if missing_carried:
        hints.append(
            f"missing carried column(s), not blocking (spec 6.7.1a section 3.6): "
            f"{sorted(missing_carried)}"
        )
    extra = actual_columns - checked - carried
    if extra:
        hints.append(
            f"unexpected extra column(s), not blocking (spec 6.7.1a section 3.6): {sorted(extra)}"
        )
    schema_ok = not missing_checked

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

    # Period-bound checks need a tz-aware UTC index and every CHECKED column
    # present to compare against period_start/period_end without raising --
    # skipped, not silently coerced, when either already failed above. A
    # missing/extra CARRIED or unknown column no longer blocks this (spec
    # 6.7.1a section 3.6) -- only schema_ok (checked columns only) gates it.
    if index_usable and index_is_utc and schema_ok:
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

            # Always computed (cheap, a single timestamp subtraction), not
            # only inside `if mode == "live"` -- unused in rebuild mode, but
            # keeping it unconditional avoids a possibly-unbound read below,
            # since pyright cannot correlate this mode check with the
            # separate one inside the per-column loop further down.
            nan_window_start = as_of - pd.Timedelta(days=LIVE_NAN_WINDOW_DAYS)
            if mode == "live":
                nan_check_period = frame[(dt_index >= nan_window_start) & (dt_index < as_of)]
            else:
                nan_check_period = frame[(dt_index >= period_start) & (dt_index < period_end)]

            if len(nan_check_period):
                for column in sorted(checked):
                    column_frame = nan_check_period
                    column_period_end = as_of if mode == "live" else period_end

                    if mode == "live" and column in expectation.live_settling_columns:
                        column_period_end = as_of - LIVE_SETTLING_BUFFER
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
                    nan_mask = column_frame[column].isna()
                    nan_fraction = float(nan_mask.mean())

                    if mode == "live":
                        nan_cell_counts[column] = int(nan_mask.sum())
                        if nan_fraction >= LIVE_GROSS_NAN_FRACTION:
                            reasons.append(
                                f"column {column!r}: NaN fraction {nan_fraction:.3f} over the "
                                f"live window {nan_window_start}..{column_period_end} at or "
                                f"above the gross-corruption threshold "
                                f"{LIVE_GROSS_NAN_FRACTION:.2f} (spec 6.7.1a section 3.4)"
                            )
                        elif nan_mask.any():
                            hints.append(
                                f"column {column!r}: {int(nan_mask.sum())} NaN cell(s) "
                                f"({nan_fraction:.3f}) over the live window "
                                f"{nan_window_start}..{column_period_end} -- not blocking "
                                "(spec 6.7.1a section 3.4)"
                            )
                    else:
                        allowed = expectation.max_nan_fraction.get(column, 0.0)
                        if nan_fraction > allowed:
                            reasons.append(
                                f"column {column!r}: NaN fraction {nan_fraction:.3f} "
                                f"exceeds allowed {allowed:.3f} in {period_start}..{column_period_end} "
                                f"(mode={mode!r}, outside any known_low_resolution_windows exclusion)"
                            )

                if mode == "rebuild":
                    # CARRIED columns are measured and reported here (spec
                    # 6.7.1a section 5.3: "im Rebuild berichtet") -- for the
                    # owner's own future-development tracking -- but never
                    # gated, in either mode.
                    for column in sorted(carried):
                        if column not in nan_check_period.columns:
                            continue
                        column_frame = nan_check_period
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
                        if nan_fraction > 0:
                            hints.append(
                                f"carried column {column!r}: NaN fraction {nan_fraction:.3f} "
                                f"in {period_start}..{period_end} -- never blocking, reported "
                                "for reference only (spec 6.7.1a section 3.6)"
                            )

    if previous is not None and index_usable and index_is_utc and len(frame.index):
        if previous.covered_end_utc is not None:
            prev_end = pd.Timestamp(previous.covered_end_utc)
            new_max = pd.Timestamp(frame.index.max())
            if new_max < prev_end:
                reasons.append(f"covered range shrank: new max {new_max} < previous {prev_end}")
        if len(frame) < previous.count:
            reasons.append(f"row count shrank: {len(frame)} < previous {previous.count}")

    return ValidationResult(
        source=name,
        ok=not reasons,
        reasons=tuple(reasons),
        hints=tuple(hints),
        nan_cell_counts=nan_cell_counts,
    )


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
    validate_source/write_if_valid's own split of concerns). A known-bad
    file that shows up again is never waved through here -- KNOWN_WEATHER_
    DEFECTS (data/weather_grid.py) only changes how a finding is reported
    below, never whether it is checked (docs/sprint6_auftrag_known_data_
    defects.md §3: "Eine als korrupt bekannte Datei ... muss weiterhin an
    validate_weather_run() scheitern").

    Also logs an inventory report distinguishing calendar days in range,
    present/usable run files, known-defective days (by category), and any
    NEWLY missing or corrupt day -- named individually, never folded into a
    count, so a real new provider gap is never silently absorbed into the
    known list's numbers (docs/sprint6_auftrag_known_data_defects.md §2).
    """
    paths = _matched_paths(root, ("weather_single_runs",))
    if not paths:
        return None, ("no cached weather run files found",)

    run_inits: list[pd.Timestamp] = []
    present_inits: set[pd.Timestamp] = set()
    for rel_path in paths:
        df = pd.read_parquet(root / rel_path)
        result = validate_weather_run(df)
        # Filename is "{YYYY-MM-DD}THHZ.parquet" (data/_weather_cache.py::
        # cache_path) -- the run's own init date, not read from file content
        # since HOURLY_VARIABLES columns hold forecast valid-times, not the
        # run init itself. This IS the KNOWN_WEATHER_DEFECTS key space
        # already (both are run-init timestamps) -- no delivery-day shift
        # needed here, unlike a caller that reports by delivery day.
        date_part, hour_part = rel_path.stem.split("T")
        run_init = pd.Timestamp(f"{date_part}T{hour_part.rstrip('Z')}:00:00", tz="UTC")
        if not result.ok:
            defect = KNOWN_WEATHER_DEFECTS.get(run_init)
            if defect is None or defect.category != KnownWeatherDefectCategory.PROVIDER_CORRUPT:
                logger.warning(
                    "New weather defect, not in KNOWN_WEATHER_DEFECTS: %s (%s)",
                    run_init.date(),
                    "; ".join(result.reasons),
                )
            return None, (f"{rel_path}: {'; '.join(result.reasons)}",)
        run_inits.append(run_init)
        present_inits.add(run_init.normalize())

    calendar_days = pd.date_range(min(run_inits).normalize(), max(run_inits).normalize(), freq="D")
    missing_days = [d for d in calendar_days if d not in present_inits]
    # A missing day here means "no file present" -- true both for a
    # PROVIDER_UNAVAILABLE day (never had one) and a PROVIDER_CORRUPT day
    # (had one, physically removed in 6.7.1 A9 once found bad); this scan
    # cannot and need not distinguish the two, so either category explains
    # an absence (confirmed: without this, the two known-corrupt days'
    # already-deleted files showed up here as unexplained "new" misses).
    known_missing = [d for d in missing_days if d in KNOWN_WEATHER_DEFECTS]
    new_missing = [d for d in missing_days if d not in known_missing]
    if new_missing:
        logger.warning(
            "New missing weather run(s), not in KNOWN_WEATHER_DEFECTS: %s",
            [d.date().isoformat() for d in new_missing],
        )
    logger.info(
        "Weather run inventory: %d calendar day(s) in range, %d present, %d usable, "
        "%d known-defective (missing, either category), %d newly missing",
        len(calendar_days),
        len(present_inits),
        len(run_inits),
        len(known_missing),
        len(new_missing),
    )

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
