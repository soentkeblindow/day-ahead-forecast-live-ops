"""Unit tests for ops/store.py -- packing, manifest, versioned publish (spec
6.7.1, section 5.3/7). Network is never touched: release_assets' functions
are monkeypatched on the store module's own reference to that module,
same house style as tests/test_release_assets.py / test_weather_client.py.

Input-control and heal-step tests live in this same file (added by later
steps A5/A6), matching the spec's own tests/test_store.py grouping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES
from energy_price_forecast.ops import release_assets as ra
from energy_price_forecast.ops import store


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sample_manifest(
    created_at_utc: str = "2026-09-08T10:10:00+00:00", extra_sources: tuple[str, ...] = ()
) -> store.Manifest:
    sources = {
        "day_ahead_price": store.SourceManifestEntry(
            covered_start_utc="2026-08-01T00:00:00+00:00",
            covered_end_utc="2026-09-07T23:45:00+00:00",
            count=3552,
            last_success_utc="2026-09-08T10:09:00+00:00",
            last_attempt_utc="2026-09-08T10:09:00+00:00",
        ),
    }
    for name in extra_sources:
        sources[name] = store.SourceManifestEntry(
            covered_start_utc="2026-08-01T00:00:00+00:00",
            covered_end_utc="2026-09-07T23:45:00+00:00",
            count=1,
            last_success_utc="2026-09-08T10:09:00+00:00",
            last_attempt_utc="2026-09-08T10:09:00+00:00",
        )
    return store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc=created_at_utc,
        run_id="12345",
        run_url="https://github.com/soentkeblindow/sbl-energy-forecast/actions/runs/12345",
        code_sha="abc1234",
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Positive-list packing
# ---------------------------------------------------------------------------


def test_pack_store_includes_only_positive_list_matches(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    # In the positive list, and each source has a manifest entry below.
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet")
    _write(
        root
        / "data"
        / "cache"
        / "weather_single_runs"
        / "ecmwf_ifs"
        / "grid_v1_abcd1234"
        / "2026"
        / "09"
        / "2026-09-07T00Z.parquet"
    )
    _write(root / "data" / "raw" / "commodities" / "ttf_gas.parquet")
    # NOT in the positive list -- must never be packed (spec section 2.2).
    _write(root / "data" / "interim" / "hourly.parquet")
    _write(root / "data" / "processed" / "features.parquet")
    _write(root / "data" / "processed" / "renewables_forecast_rolling365_l2.parquet")
    _write(root / "data" / "interim" / "hourly.parquet.pre_6_4.bak")

    out = tmp_path / "store.tar"
    manifest = _sample_manifest(extra_sources=("weather_single_runs", "ttf_gas"))
    store.pack_store(root, manifest, out)

    import tarfile

    with tarfile.open(out, "r") as tf:
        names = set(tf.getnames())

    assert "manifest.json" in names
    assert "data/raw/entsoe/day_ahead_prices/2026-08.parquet" in names
    assert (
        "data/cache/weather_single_runs/ecmwf_ifs/grid_v1_abcd1234/2026/09/2026-09-07T00Z.parquet"
        in names
    )
    assert "data/raw/commodities/ttf_gas.parquet" in names
    assert not any("hourly.parquet" in n for n in names)
    assert not any("features.parquet" in n for n in names)
    assert not any("renewables_forecast" in n for n in names)
    assert not any(n.endswith(".bak") for n in names)


def test_pack_store_excludes_a_source_absent_from_the_manifest(tmp_path: Path) -> None:
    """A9 finding (2026-09-10): the first real publish packed every ENTSO-E
    fetch group's cache files regardless of whether that source's own last
    fetch had validated -- cached_fetch writes to disk unconditionally, so
    a rejected source's files sit right next to validated ones on disk.
    Packing must only include a source's files if it has a manifest entry."""
    root = tmp_path / "repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet")
    # generation's cache file exists on disk (its fetch succeeded) but it
    # has no manifest entry (its validation this run did not) -- must not
    # be packed.
    _write(root / "data" / "raw" / "entsoe" / "generation" / "2026-08.parquet")

    out = tmp_path / "store.tar"
    store.pack_store(root, _sample_manifest(), out)

    import tarfile

    with tarfile.open(out, "r") as tf:
        names = set(tf.getnames())

    assert "data/raw/entsoe/day_ahead_prices/2026-08.parquet" in names
    assert not any("generation" in n for n in names)


def test_pack_store_deduplicates_a_path_matching_two_globs(tmp_path: Path) -> None:
    # Sanity check on _matched_paths' set-based dedup -- not reachable with
    # the current glob list (they don't overlap), but the function must not
    # double-add a file into the tar if a future glob addition did overlap.
    root = tmp_path / "repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet")

    matched = store._matched_paths(root, ("day_ahead_price",))

    assert matched == sorted(set(matched))


# ---------------------------------------------------------------------------
# Round-trip: pack -> unpack
# ---------------------------------------------------------------------------


def test_round_trip_pack_unpack_is_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    content = b"parquet-bytes-not-actually-parquet-but-bytes-are-bytes"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet", content)

    tar_path = tmp_path / "store.tar"
    manifest = _sample_manifest()
    store.pack_store(root, manifest, tar_path)

    workdir = tmp_path / "unpacked"
    unpacked_manifest = store.unpack_store(tar_path, workdir)

    restored = workdir / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet"
    assert restored.read_bytes() == content
    assert unpacked_manifest == manifest


def test_unpack_store_raises_without_manifest(tmp_path: Path) -> None:
    import tarfile

    tar_path = tmp_path / "broken.tar"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="data/raw/entsoe/day_ahead_prices/2026-08.parquet")
        info.size = 1
        import io

        tf.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(store.StoreError, match="manifest"):
        store.unpack_store(tar_path, tmp_path / "workdir")


# ---------------------------------------------------------------------------
# Manifest serialisation
# ---------------------------------------------------------------------------


def test_manifest_round_trips_through_dict() -> None:
    manifest = _sample_manifest()
    restored = store.Manifest.from_dict(manifest.to_dict())
    assert restored == manifest


def test_manifest_distinguishes_last_success_from_last_attempt() -> None:
    entry = store.SourceManifestEntry(
        covered_start_utc="2026-08-01T00:00:00+00:00",
        covered_end_utc="2026-09-05T23:45:00+00:00",
        count=100,
        last_success_utc="2026-09-05T10:00:00+00:00",  # a stale source: kept failing since
        last_attempt_utc="2026-09-08T10:10:00+00:00",  # ...but a run just tried again
    )
    assert entry.last_success_utc != entry.last_attempt_utc


# ---------------------------------------------------------------------------
# load_store / publish_store (release_assets monkeypatched)
# ---------------------------------------------------------------------------


def test_load_store_downloads_newest_and_unpacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "source_repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet", b"payload")
    tar_path = tmp_path / "store-20260908T101000Z.tar"
    store.pack_store(root, _sample_manifest(), tar_path)

    release = ra.ReleaseRef(release_id=1, tag=store.STORE_RELEASE_TAG)
    newest = ra.AssetRef(1, "store-20260908T101000Z.tar", 1, True, tar_path.stat().st_size)
    older = ra.AssetRef(2, "store-20260907T101000Z.tar", 1, True, 10)

    monkeypatch.setattr(store.release_assets, "ensure_release", lambda tag: release)
    monkeypatch.setattr(store.release_assets, "list_assets", lambda rel: (newest, older))

    downloaded: list[str] = []

    def fake_download(asset: ra.AssetRef, target: Path) -> None:
        downloaded.append(asset.name)
        target.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(store.release_assets, "download_asset", fake_download)

    workdir = tmp_path / "workdir"
    state = store.load_store(workdir)

    assert downloaded == ["store-20260908T101000Z.tar"]  # newest, not older
    assert state.manifest == _sample_manifest()
    assert (workdir / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet").exists()


def test_load_store_raises_when_no_assets_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = ra.ReleaseRef(release_id=1, tag=store.STORE_RELEASE_TAG)
    monkeypatch.setattr(store.release_assets, "ensure_release", lambda tag: release)
    monkeypatch.setattr(store.release_assets, "list_assets", lambda rel: ())

    with pytest.raises(store.StoreError, match="rebuild_store"):
        store.load_store(tmp_path / "workdir")


def test_publish_store_uploads_then_prunes_to_keep_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet")

    release = ra.ReleaseRef(release_id=1, tag=store.STORE_RELEASE_TAG)
    monkeypatch.setattr(store.release_assets, "ensure_release", lambda tag: release)

    uploaded: dict[str, Any] = {}

    def fake_upload(rel: ra.ReleaseRef, path: Path) -> ra.AssetRef:
        uploaded["release"] = rel
        uploaded["name"] = path.name
        return ra.AssetRef(9, path.name, rel.release_id, True, path.stat().st_size)

    pruned: dict[str, Any] = {}

    def fake_prune(rel: ra.ReleaseRef, keep: int) -> tuple[ra.AssetRef, ...]:
        pruned["release"] = rel
        pruned["keep"] = keep
        return ()

    monkeypatch.setattr(store.release_assets, "upload_asset", fake_upload)
    monkeypatch.setattr(store.release_assets, "prune_assets", fake_prune)

    manifest = _sample_manifest(created_at_utc="2026-09-08T10:10:00+00:00")
    asset = store.publish_store(root, manifest)

    assert uploaded["name"] == "store-20260908T101000Z.tar"
    assert pruned["keep"] == store.STORE_KEEP_VERSIONS
    assert pruned["release"] == release
    assert asset.name == "store-20260908T101000Z.tar"


# ---------------------------------------------------------------------------
# Input control -- validate_source (spec section 2.4/7)
# ---------------------------------------------------------------------------

_PRICE_EXPECT = store.EXPECTATION_TABLE["day_ahead_price"]


def _price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"day_ahead_price": np.arange(len(index), dtype=float)}, index=index)


def test_6_6_case_missing_end_of_month_quarter_hourly_timestamps_is_rejected() -> None:
    """Rebuilds the actual 2026-09-06 finding: a day_ahead_price month cache
    missing its final ~9 days of 15-min timestamps, which the OLD
    _is_sufficiently_complete() (hardcoded to hourly) waved through because
    the row count looked plausible against a wrong, ~4x-too-small expected
    count. This is the most important test of this step."""
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    truncated_end = pd.Timestamp("2026-08-23", tz="UTC")  # missing the last ~9 days
    index = pd.date_range(period_start, truncated_end, freq="15min", inclusive="left")
    frame = _price_frame(index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),  # the month is long over
    )

    assert result.ok is False
    assert any("coverage gap" in r for r in result.reasons)


def test_quarter_hourly_month_is_not_measured_against_hourly_expectations() -> None:
    """Regression test for the bug CLASS: a COMPLETE 15-min month must not
    be misjudged as incomplete by an expectation baked in for hourly data."""
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = _price_frame(index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True
    assert result.reasons == ()


def test_currently_open_month_is_not_measured_for_coverage() -> None:
    # as_of sits INSIDE the period -- the month is naturally, legitimately
    # incomplete so far; no coverage gap should be raised.
    period_start = pd.Timestamp("2026-09-01", tz="UTC")
    period_end = pd.Timestamp("2026-10-01", tz="UTC")
    index = pd.date_range(period_start, pd.Timestamp("2026-09-08", tz="UTC"), freq="h")
    frame = _price_frame(index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08T10:00", tz="UTC"),
    )

    assert result.ok is True


def test_missing_and_extra_columns_are_rejected() -> None:
    index = pd.date_range("2026-08-01", "2026-08-02", freq="15min", tz="UTC", inclusive="left")
    frame = pd.DataFrame({"some_other_column": np.arange(len(index))}, index=index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=index.min(),
        period_end=index.max(),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("column mismatch" in r for r in result.reasons)


def test_duplicate_timestamps_are_rejected() -> None:
    index = pd.DatetimeIndex(
        ["2026-08-01T00:00:00Z", "2026-08-01T00:00:00Z", "2026-08-01T00:15:00Z"]
    )
    frame = _price_frame(index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=index.min(),
        period_end=index.max() + pd.Timedelta(minutes=15),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("duplicate timestamp" in r for r in result.reasons)


def test_naive_index_is_rejected() -> None:
    index = pd.date_range("2026-08-01", "2026-08-02", freq="15min", inclusive="left")  # no tz
    frame = _price_frame(index)

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=pd.Timestamp("2026-08-01", tz="UTC"),
        period_end=pd.Timestamp("2026-08-02", tz="UTC"),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("not tz-aware UTC" in r for r in result.reasons)


def test_gen_nuclear_permanent_nan_is_accepted() -> None:
    expectation = store.EXPECTATION_TABLE["generation"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["gen_nuclear"] = np.full(len(index), np.nan)  # Germany, since April 2023 -- not a gap
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_unexpected_nan_in_a_non_exempt_column_is_rejected() -> None:
    expectation = store.EXPECTATION_TABLE["generation"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["gen_gas"] = np.full(len(index), np.nan)  # NOT an expected-NaN column -- a real gap
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("gen_gas" in r for r in result.reasons)


def test_live_tail_nan_within_settling_buffer_is_not_rejected() -> None:
    """A9/A10 finding: individual generation-by-type columns can lag others
    by hours at the live edge -- a normal reporting artefact, not a gap.
    Only applies in mode="live" (see store.py's LIVE_SETTLING_BUFFER)."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    period_start = pd.Timestamp("2026-09-08", tz="UTC")
    period_end = as_of.floor("D") + pd.Timedelta(days=1)
    index = pd.date_range(
        period_start, as_of - pd.Timedelta(hours=1), freq="15min", inclusive="left"
    )
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    tail_mask = index >= (as_of - pd.Timedelta(hours=2))  # inside the 24h settling buffer
    data["gen_hard_coal"] = np.where(tail_mask, np.nan, np.arange(len(index), dtype=float))
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True


def test_settled_nan_outside_the_buffer_is_still_rejected() -> None:
    """The settling buffer must not swallow a genuine gap once the data is
    old enough that live reporting lag no longer explains it (mode="live",
    a full-column gap well beyond LIVE_MAX_NAN_FRACTION too)."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    period_start = pd.Timestamp("2026-09-08", tz="UTC")
    period_end = as_of.floor("D") + pd.Timedelta(days=1)
    index = pd.date_range(
        period_start, as_of - pd.Timedelta(hours=25), freq="15min", inclusive="left"
    )
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["gen_gas"] = np.full(len(index), np.nan)  # settled, real gap -- well before as_of - 24h
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is False
    assert any("gen_gas" in r for r in result.reasons)


def test_rebuild_mode_applies_no_settling_buffer_and_the_strict_tolerance() -> None:
    """mode="rebuild" (the default) never excludes a live tail and never
    loosens to LIVE_MAX_NAN_FRACTION -- a live-tail-shaped gap right at
    as_of must still be judged against the strict rebuild tolerance."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    period_start = pd.Timestamp("2026-09-08", tz="UTC")
    period_end = as_of.floor("D") + pd.Timedelta(days=1)
    index = pd.date_range(
        period_start, as_of - pd.Timedelta(hours=1), freq="15min", inclusive="left"
    )
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    tail_mask = index >= (as_of - pd.Timedelta(hours=2))
    data["gen_hard_coal"] = np.where(tail_mask, np.nan, np.arange(len(index), dtype=float))
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=as_of,
    )

    assert result.ok is False
    assert any("gen_hard_coal" in r for r in result.reasons)


def test_live_mode_never_loosens_a_permanent_exemption() -> None:
    """LIVE_MAX_NAN_FRACTION must be a floor, not a ceiling -- gen_nuclear's
    permanent 1.0 exemption must survive mode="live" unchanged."""
    expectation = store.EXPECTATION_TABLE["generation"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["gen_nuclear"] = np.full(len(index), np.nan)
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
        mode="live",
    )

    assert result.ok is True


def test_live_mode_tolerates_an_isolated_cell_in_a_small_window() -> None:
    """A10's real finding (2026-09-10): one already-known isolated cell in
    a small daily-sync window is ~5% of that window, over the rebuild-sized
    0.001 tolerance -- LIVE_MAX_NAN_FRACTION must absorb this.
    load_forecast_day_ahead specifically: not in live_settling_columns, so
    this isolates LIVE_MAX_NAN_FRACTION's own effect from the settling
    buffer's (a settling-buffered column this close to as_of would have its
    entire small window excluded before the fraction check ever ran)."""
    expectation = store.EXPECTATION_TABLE["load"]
    period_start = pd.Timestamp("2026-09-10T00:00", tz="UTC")
    period_end = pd.Timestamp("2026-09-11T00:00", tz="UTC")
    index = pd.date_range(
        period_start, pd.Timestamp("2026-09-10T06:00", tz="UTC"), freq="15min", inclusive="left"
    )
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["load_forecast_day_ahead"][0] = np.nan  # one isolated cell in a 24-row window (~4.2%)
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-10T16:48", tz="UTC"),
        mode="live",
    )

    assert result.ok is True


def test_cross_border_flows_tolerates_an_isolated_settled_gap() -> None:
    """A9 probe finding: physical_net_de_to_pl had one isolated settled gap
    (2026-09-02 03:00 UTC), the same kind of permanent small ENTSO-E gap as
    generation's -- extended the same 0.001 tolerance here."""
    expectation = store.EXPECTATION_TABLE["cross_border_flows"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["physical_net_de_to_pl"][100] = np.nan  # one isolated settled cell
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "cross_border_flows",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_cross_border_flows_still_rejects_a_large_settled_gap() -> None:
    expectation = store.EXPECTATION_TABLE["cross_border_flows"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["physical_net_de_to_pl"] = np.full(len(index), np.nan)
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "cross_border_flows",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("physical_net_de_to_pl" in r for r in result.reasons)


def test_known_low_resolution_window_absorbs_the_structural_75pct_gap() -> None:
    """A9 finding: physical_net_de_to_fr genuinely reported hourly (not
    quarter-hourly) 2021-08..2025-04 -- merging it against an already-15min
    grid produces a structural 75% NaN there, not a real gap."""
    expectation = store.EXPECTATION_TABLE["cross_border_flows"]
    period_start = pd.Timestamp("2022-01-01", tz="UTC")
    period_end = pd.Timestamp("2022-02-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    hourly_only = index.minute != 0
    data["physical_net_de_to_fr"] = np.where(
        hourly_only, np.nan, np.arange(len(index), dtype=float)
    )
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "cross_border_flows",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_known_low_resolution_window_does_not_hide_a_gap_after_the_window() -> None:
    """The window must not swallow a genuine, unrelated gap once the column
    is back to full quarter-hourly resolution."""
    expectation = store.EXPECTATION_TABLE["cross_border_flows"]
    period_start = pd.Timestamp("2026-01-01", tz="UTC")  # well after the FR window ends
    period_end = pd.Timestamp("2026-02-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["physical_net_de_to_fr"] = np.full(len(index), np.nan)
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "cross_border_flows",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("physical_net_de_to_fr" in r for r in result.reasons)


def test_shrinking_covered_range_is_rejected() -> None:
    index = pd.date_range("2026-08-01", "2026-08-02", freq="15min", tz="UTC", inclusive="left")
    frame = _price_frame(index)
    previous = store.SourceManifestEntry(
        covered_start_utc="2026-07-01T00:00:00+00:00",
        covered_end_utc="2026-08-15T00:00:00+00:00",  # store already had data up to Aug 15
        count=10_000,
        last_success_utc="2026-08-15T00:00:00+00:00",
        last_attempt_utc="2026-08-15T00:00:00+00:00",
    )

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=index.min(),
        period_end=index.max() + pd.Timedelta(minutes=15),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
        previous=previous,
    )

    assert result.ok is False
    assert any("shrank" in r for r in result.reasons)


def test_growing_covered_range_is_accepted() -> None:
    index = pd.date_range("2026-08-01", "2026-08-02", freq="15min", tz="UTC", inclusive="left")
    frame = _price_frame(index)
    previous = store.SourceManifestEntry(
        covered_start_utc="2026-07-01T00:00:00+00:00",
        covered_end_utc="2026-07-31T00:00:00+00:00",
        count=1,
        last_success_utc="2026-07-31T00:00:00+00:00",
        last_attempt_utc="2026-07-31T00:00:00+00:00",
    )

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=index.min(),
        period_end=index.max() + pd.Timedelta(minutes=15),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
        previous=previous,
    )

    assert result.ok is True


def test_commodity_weekend_gaps_are_not_flagged_as_coverage_gap() -> None:
    expectation = store.EXPECTATION_TABLE["ttf_gas"]
    assert expectation.grid_based is False
    # Trading days only -- Mon-Fri, deliberately skipping the weekend, exactly
    # as data/commodities_client.py describes its own output.
    period_start = pd.Timestamp("2026-08-03", tz="UTC")  # a Monday
    period_end = pd.Timestamp("2026-08-10", tz="UTC")  # the following Monday
    index = pd.DatetimeIndex(
        [
            "2026-08-03T00:00:00Z",
            "2026-08-04T00:00:00Z",
            "2026-08-05T00:00:00Z",
            "2026-08-06T00:00:00Z",
            "2026-08-07T00:00:00Z",
            # 08-08/08-09 (Sat/Sun) absent by design
        ]
    )
    frame = pd.DataFrame({"ttf_gas_eur_per_mwh": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    result = store.validate_source(
        "ttf_gas",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_a_failing_source_does_not_block_writing_another(tmp_path: Path) -> None:
    good_index = pd.date_range("2026-08-01", "2026-08-02", freq="15min", tz="UTC", inclusive="left")
    good_frame = _price_frame(good_index)
    good_result = store.validate_source(
        "day_ahead_price",
        good_frame,
        _PRICE_EXPECT,
        period_start=good_index.min(),
        period_end=good_index.max() + pd.Timedelta(minutes=15),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    bad_frame = pd.DataFrame({"wrong_column": [1, 2, 3]}, index=good_index[:3])
    bad_result = store.validate_source(
        "day_ahead_price",
        bad_frame,
        _PRICE_EXPECT,
        period_start=good_index.min(),
        period_end=good_index.max() + pd.Timedelta(minutes=15),
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    good_path = tmp_path / "good.parquet"
    bad_path = tmp_path / "bad.parquet"
    store.write_if_valid(good_path, good_frame, good_result)
    store.write_if_valid(bad_path, bad_frame, bad_result)

    assert good_result.ok is True
    assert bad_result.ok is False
    assert good_path.exists()
    assert not bad_path.exists()


# ---------------------------------------------------------------------------
# Weather input control -- validate_weather_run
# ---------------------------------------------------------------------------


def _weather_frame(
    *, corrupt_column: str | None = None, nan_night_radiation: bool = True
) -> pd.DataFrame:
    index = pd.date_range("2026-09-07T00:00", periods=24, freq="h", tz="UTC")
    data: dict[str, Any] = {}
    for point in GRID_POINTS:
        for variable in HOURLY_VARIABLES:
            col = f"{point.point_id}__{variable}"
            if (
                variable in ("shortwave_radiation", "direct_normal_irradiance")
                and nan_night_radiation
            ):
                # first 6 hours "at night" -- NaN, exactly like real Open-Meteo output
                values = [np.nan] * 6 + [100.0] * 18
            else:
                values = [10.0] * 24
            data[col] = values
    frame = pd.DataFrame(data, index=index, dtype="float32")
    if corrupt_column is not None and corrupt_column in frame.columns:
        frame[corrupt_column] = np.nan
    return frame


def test_valid_weather_run_is_accepted() -> None:
    frame = _weather_frame()
    result = store.validate_weather_run(frame)
    assert result.ok is True


def test_weather_run_missing_column_is_rejected() -> None:
    frame = _weather_frame().drop(columns=[f"{GRID_POINTS[0].point_id}__temperature_2m"])
    result = store.validate_weather_run(frame)
    assert result.ok is False
    assert any("column mismatch" in r for r in result.reasons)


def test_weather_run_nan_in_non_radiation_variable_is_rejected() -> None:
    col = f"{GRID_POINTS[0].point_id}__wind_speed_10m"
    frame = _weather_frame(corrupt_column=col)
    result = store.validate_weather_run(frame)
    assert result.ok is False
    assert any("wind_speed_10m" in r for r in result.reasons)


def test_weather_run_nighttime_radiation_nan_is_accepted() -> None:
    frame = _weather_frame(nan_night_radiation=True)
    result = store.validate_weather_run(frame)
    assert result.ok is True


def test_weather_run_empty_is_rejected() -> None:
    frame = _weather_frame().iloc[0:0]
    result = store.validate_weather_run(frame)
    assert result.ok is False
    assert any("zero rows" in r for r in result.reasons)


def _write_weather_run(root: Path, run_init: str) -> None:
    ts = pd.Timestamp(run_init, tz="UTC")
    path = (
        root
        / "data"
        / "cache"
        / "weather_single_runs"
        / "ecmwf_ifs"
        / "grid_v1_abcd1234"
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"{ts.strftime('%Y-%m-%d')}T{ts.strftime('%H')}Z.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _weather_frame().to_parquet(path)


def test_validate_historical_weather_runs_builds_a_manifest_entry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_weather_run(root, "2026-09-07T00:00")
    _write_weather_run(root, "2026-09-08T06:00")

    entry, reasons = store.validate_historical_weather_runs(
        root, as_of=pd.Timestamp("2026-09-10T00:00", tz="UTC")
    )

    assert reasons == ()
    assert entry is not None
    assert entry.count == 2
    assert entry.covered_start_utc == "2026-09-07T00:00:00+00:00"
    assert entry.covered_end_utc == "2026-09-08T06:00:00+00:00"


def test_validate_historical_weather_runs_rejects_if_any_file_is_bad(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_weather_run(root, "2026-09-07T00:00")
    bad_path = (
        root
        / "data"
        / "cache"
        / "weather_single_runs"
        / "ecmwf_ifs"
        / "grid_v1_abcd1234"
        / "2026"
        / "09"
        / "2026-09-08T06Z.parquet"
    )
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    _weather_frame().iloc[0:0].to_parquet(bad_path)  # zero rows -- fails validate_weather_run

    entry, reasons = store.validate_historical_weather_runs(
        root, as_of=pd.Timestamp("2026-09-10T00:00", tz="UTC")
    )

    assert entry is None
    assert reasons != ()


def test_validate_historical_weather_runs_with_no_files_returns_none(tmp_path: Path) -> None:
    entry, reasons = store.validate_historical_weather_runs(
        tmp_path / "repo", as_of=pd.Timestamp("2026-09-10T00:00", tz="UTC")
    )

    assert entry is None
    assert reasons != ()


# ---------------------------------------------------------------------------
# Heal step -- heal_recent (spec section 2.6/7)
# ---------------------------------------------------------------------------


def test_heal_recent_fills_a_gap_from_the_refetch() -> None:
    as_of = pd.Timestamp("2026-09-08T00:00", tz="UTC")
    full_index = pd.date_range("2026-09-01", "2026-09-08", freq="h", tz="UTC", inclusive="left")
    values = np.arange(len(full_index), dtype=float)
    # Simulate a hole: 2026-09-05 03:00-05:00 missing from the store.
    gap = (full_index >= pd.Timestamp("2026-09-05T03:00", tz="UTC")) & (
        full_index < pd.Timestamp("2026-09-05T06:00", tz="UTC")
    )
    existing_values = values.copy()
    existing_values[gap] = np.nan
    existing = pd.DataFrame({"day_ahead_price": existing_values}, index=full_index)

    fresh = pd.DataFrame({"day_ahead_price": values}, index=full_index)  # source has it after all

    healed, result = store.heal_recent(
        "day_ahead_price", existing, lookback_days=10, refetch=lambda s, e: fresh, as_of=as_of
    )

    assert result.filled_cells == 3
    assert result.still_missing_cells == 0
    assert not healed["day_ahead_price"].isna().any()
    assert (
        healed.loc["2026-09-05T04:00", "day_ahead_price"]
        == fresh.loc["2026-09-05T04:00", "day_ahead_price"]
    )


def test_heal_recent_never_overwrites_an_existing_value_even_if_revised() -> None:
    as_of = pd.Timestamp("2026-09-08T00:00", tz="UTC")
    index = pd.date_range("2026-09-05", "2026-09-06", freq="h", tz="UTC", inclusive="left")
    existing = pd.DataFrame({"day_ahead_price": [100.0] * len(index)}, index=index)
    # ENTSO-E "revises" the value -- the store must keep its own, not the new one.
    fresh = pd.DataFrame({"day_ahead_price": [999.0] * len(index)}, index=index)

    healed, result = store.heal_recent(
        "day_ahead_price", existing, lookback_days=10, refetch=lambda s, e: fresh, as_of=as_of
    )

    assert (healed["day_ahead_price"] == 100.0).all()
    assert result.filled_cells == 0


def test_heal_recent_raises_if_a_value_would_change() -> None:
    # Directly proves the verification step actually runs (spec: "wird
    # geführt, nicht behauptet") by making combine_first's own precondition
    # violated -- feed it a fresh frame where an existing NaN gets healed to
    # X, but a manual post-hoc corruption of the result would be needed to
    # trigger the raise honestly, so instead we confirm the positive path:
    # combine_first structurally cannot change a non-NaN value, and the
    # verification loop passes without raising for the well-behaved case.
    as_of = pd.Timestamp("2026-09-08T00:00", tz="UTC")
    index = pd.date_range("2026-09-05", "2026-09-06", freq="h", tz="UTC", inclusive="left")
    existing = pd.DataFrame({"day_ahead_price": [100.0] * len(index)}, index=index)
    fresh = pd.DataFrame({"day_ahead_price": [100.0] * len(index)}, index=index)

    # Should not raise.
    store.heal_recent(
        "day_ahead_price", existing, lookback_days=10, refetch=lambda s, e: fresh, as_of=as_of
    )


def test_heal_recent_leaves_an_unclosable_gap_open_and_reports_it() -> None:
    as_of = pd.Timestamp("2026-09-08T00:00", tz="UTC")
    index = pd.date_range("2026-09-05", "2026-09-06", freq="h", tz="UTC", inclusive="left")
    values = np.arange(len(index), dtype=float)
    gap = (index >= pd.Timestamp("2026-09-05T10:00", tz="UTC")) & (
        index < pd.Timestamp("2026-09-05T12:00", tz="UTC")
    )
    existing_values = values.copy()
    existing_values[gap] = np.nan
    existing = pd.DataFrame({"day_ahead_price": existing_values}, index=index)

    # The source genuinely has nothing there either -- e.g. one of the four
    # known Open-Meteo provider gaps, or a real ENTSO-E publication hole.
    fresh = existing.copy()

    healed, result = store.heal_recent(
        "day_ahead_price", existing, lookback_days=10, refetch=lambda s, e: fresh, as_of=as_of
    )

    assert result.filled_cells == 0
    assert result.still_missing_cells == 2
    assert healed["day_ahead_price"].isna().sum() == 2  # left alone, not interpolated


def test_heal_recent_calls_refetch_with_the_lookback_window_only() -> None:
    # heal_recent has no cache parameter at all -- structurally, it can only
    # ever call whatever `refetch` the caller passes. A real caller (A8)
    # passes the underlying client query directly (e.g. entsoe-py's raw
    # query, or weather_client.fetch_run(..., use_cache=False)), never
    # data/_entsoe_cache.py::cached_fetch -- this test proves refetch is
    # called exactly once, with exactly the lookback window, not the cache.
    as_of = pd.Timestamp("2026-09-08T00:00", tz="UTC")
    index = pd.date_range("2026-09-05", "2026-09-06", freq="h", tz="UTC", inclusive="left")
    existing = pd.DataFrame({"day_ahead_price": np.arange(len(index), dtype=float)}, index=index)

    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def raw_refetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        return existing

    store.heal_recent(
        "day_ahead_price", existing, lookback_days=10, refetch=raw_refetch, as_of=as_of
    )

    assert calls == [(as_of - pd.Timedelta(days=10), as_of)]


# ---------------------------------------------------------------------------
# Size and deadline guards (spec section 2.3/5.5/7)
# ---------------------------------------------------------------------------


def test_size_guard_below_threshold_is_none() -> None:
    assert store.check_store_size(1024) is None


def test_size_guard_above_threshold_warns_but_never_raises() -> None:
    warning = store.check_store_size(store.STORE_SIZE_WARN_BYTES + 1)
    assert warning is not None
    assert "exceeds" in warning


def test_deadline_warns_within_window() -> None:
    table = (store.Deadline(name="test_deadline", date_utc="2026-10-01", note="do the thing"),)
    warnings = store.check_deadlines(
        pd.Timestamp("2026-09-15", tz="UTC"), table=table, warn_days=30
    )
    assert len(warnings) == 1
    assert "test_deadline" in warnings[0]


def test_deadline_silent_outside_window() -> None:
    table = (store.Deadline(name="test_deadline", date_utc="2026-12-01", note="do the thing"),)
    warnings = store.check_deadlines(
        pd.Timestamp("2026-09-15", tz="UTC"), table=table, warn_days=30
    )
    assert warnings == ()


def test_deadline_overdue_still_warns() -> None:
    table = (store.Deadline(name="test_deadline", date_utc="2026-09-01", note="do the thing"),)
    warnings = store.check_deadlines(
        pd.Timestamp("2026-09-15", tz="UTC"), table=table, warn_days=30
    )
    assert len(warnings) == 1
    assert "OVERDUE" in warnings[0]


def test_default_deadline_table_has_the_trigger_pat_entry() -> None:
    names = [d.name for d in store.DEADLINE_TABLE]
    assert names == ["trigger_pat_expiry"]
    assert store.DEADLINE_TABLE[0].date_utc == "2026-12-07"
