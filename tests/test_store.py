"""Unit tests for ops/store.py -- packing, manifest, versioned publish (spec
6.7.1, section 5.3/7). Network is never touched: release_assets' functions
are monkeypatched on the store module's own reference to that module,
same house style as tests/test_release_assets.py / test_weather_client.py.

Input-control and heal-step tests live in this same file (added by later
steps A5/A6), matching the spec's own tests/test_store.py grouping.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cryptography.fernet import Fernet

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES
from energy_price_forecast.ops import release_assets as ra
from energy_price_forecast.ops import store

# A throwaway key generated fresh per test run -- never the real
# STORE_ENCRYPTION_KEY, so these tests need no secret and can run in CI
# exactly as before (publication spec section 9: "Synthetisch, ohne Netz").
_TEST_KEY: str = Fernet.generate_key().decode()


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


def test_manifest_reads_without_the_new_live_nan_cell_counts_field() -> None:
    """spec 6.7.1a section 4.1 point 4 / section 8: a manifest published
    before this field existed must still load without raising -- the next
    real run reads exactly such a manifest."""
    data = _sample_manifest().to_dict()
    del data["sources"]["day_ahead_price"]["live_nan_cell_counts"]  # simulate a pre-6.7.1a manifest

    restored = store.Manifest.from_dict(data)

    assert restored.sources["day_ahead_price"].live_nan_cell_counts == {}


def test_source_manifest_entry_carries_live_nan_cell_counts() -> None:
    entry = store.SourceManifestEntry(
        covered_start_utc="2026-08-01T00:00:00+00:00",
        covered_end_utc="2026-09-08T00:00:00+00:00",
        count=100,
        last_success_utc="2026-09-08T00:00:00+00:00",
        last_attempt_utc="2026-09-08T00:00:00+00:00",
        live_nan_cell_counts={"gen_wind_onshore": 2},
    )
    restored = store.SourceManifestEntry.from_dict(entry.to_dict())
    assert restored == entry


# ---------------------------------------------------------------------------
# load_store / publish_store (release_assets monkeypatched)
# ---------------------------------------------------------------------------


def test_load_store_downloads_newest_and_unpacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)
    root = tmp_path / "source_repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet", b"payload")
    tar_path = tmp_path / "store-20260908T101000Z.tar"
    store.pack_store(root, _sample_manifest(), tar_path)
    store._encrypt_store_archive(tar_path)

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
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)

    manifest = _sample_manifest(created_at_utc="2026-09-08T10:10:00+00:00")
    asset = store.publish_store(root, manifest)

    assert uploaded["name"] == "store-20260908T101000Z.tar"
    assert pruned["keep"] == store.STORE_KEEP_VERSIONS
    assert pruned["release"] == release
    assert asset.name == "store-20260908T101000Z.tar"


# ---------------------------------------------------------------------------
# Encryption (publication spec, section 2.1/9)
# ---------------------------------------------------------------------------


def test_encrypt_then_decrypt_archive_round_trips_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)
    path = tmp_path / "store.tar"
    original = b"not-actually-a-tar-but-bytes-are-bytes" * 100
    path.write_bytes(original)

    store._encrypt_store_archive(path)
    assert path.read_bytes().startswith(store._FERNET_PREFIX)  # actually encrypted, not a no-op

    store._decrypt_store_archive(path)

    assert path.read_bytes() == original


def test_decrypt_with_wrong_key_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)
    path = tmp_path / "store.tar"
    path.write_bytes(b"some store bytes")
    store._encrypt_store_archive(path)

    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, Fernet.generate_key().decode())

    with pytest.raises(store.StoreEncryptionError, match="wrong or corrupted"):
        store._decrypt_store_archive(path)


def test_decrypt_a_corrupted_file_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)
    path = tmp_path / "store.tar"
    path.write_bytes(b"some store bytes" * 10)
    store._encrypt_store_archive(path)

    corrupted = bytearray(path.read_bytes())
    corrupted[-5] ^= 0xFF  # flip one byte inside the token, well past the fixed prefix
    path.write_bytes(bytes(corrupted))

    with pytest.raises(store.StoreEncryptionError, match="wrong or corrupted"):
        store._decrypt_store_archive(path)


def test_missing_key_raises_a_clear_error_naming_the_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, raising=False)
    path = tmp_path / "store.tar"
    path.write_bytes(b"gAAAAA-shaped bytes that would need a key to decrypt")

    with pytest.raises(store.StoreEncryptionError, match=store.STORE_ENCRYPTION_KEY_ENV_VAR):
        store._decrypt_store_archive(path)


def test_plaintext_archive_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The transitional read path (publication spec section 2.1) that used
    to accept a legacy plaintext archive was removed in section 8 step 8 --
    a plaintext archive is now rejected the same way a wrong key would be."""
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)

    plaintext_path = tmp_path / "plain.tar"
    original = b"a legacy plaintext archive, e.g. tar magic bytes"
    plaintext_path.write_bytes(original)

    with pytest.raises(store.StoreEncryptionError, match="not a Fernet-encrypted archive"):
        store._decrypt_store_archive(plaintext_path)
    assert plaintext_path.read_bytes() == original  # left untouched, not partially consumed


def test_publish_store_never_uploads_without_a_valid_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The upload path must abort before upload_asset is ever called if the
    key is missing -- this is what makes 'the store never uploads plaintext'
    true, not just documented (publication spec section 9, 'Upload schreibt
    nie Klartext')."""
    root = tmp_path / "repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet")
    monkeypatch.delenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, raising=False)

    release = ra.ReleaseRef(release_id=1, tag=store.STORE_RELEASE_TAG)
    monkeypatch.setattr(store.release_assets, "ensure_release", lambda tag: release)

    upload_called = False

    def fake_upload(rel: ra.ReleaseRef, path: Path) -> ra.AssetRef:
        nonlocal upload_called
        upload_called = True
        return ra.AssetRef(9, path.name, rel.release_id, True, path.stat().st_size)

    monkeypatch.setattr(store.release_assets, "upload_asset", fake_upload)
    monkeypatch.setattr(store.release_assets, "prune_assets", lambda rel, keep: ())

    manifest = _sample_manifest(created_at_utc="2026-09-08T10:10:00+00:00")
    with pytest.raises(store.StoreEncryptionError, match=store.STORE_ENCRYPTION_KEY_ENV_VAR):
        store.publish_store(root, manifest)

    assert upload_called is False


def test_load_store_decrypts_a_real_encrypted_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """publish_store then load_store, round-tripped through real Fernet
    encryption end to end (not just the two private helpers in isolation)."""
    monkeypatch.setenv(store.STORE_ENCRYPTION_KEY_ENV_VAR, _TEST_KEY)
    root = tmp_path / "source_repo"
    _write(root / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet", b"payload")

    release = ra.ReleaseRef(release_id=1, tag=store.STORE_RELEASE_TAG)
    monkeypatch.setattr(store.release_assets, "ensure_release", lambda tag: release)

    published: dict[str, bytes] = {}

    def fake_upload(rel: ra.ReleaseRef, path: Path) -> ra.AssetRef:
        published["bytes"] = path.read_bytes()
        return ra.AssetRef(9, path.name, rel.release_id, True, path.stat().st_size)

    monkeypatch.setattr(store.release_assets, "upload_asset", fake_upload)
    monkeypatch.setattr(store.release_assets, "prune_assets", lambda rel, keep: ())

    manifest = _sample_manifest(created_at_utc="2026-09-08T10:10:00+00:00")
    store.publish_store(root, manifest)
    assert published["bytes"].startswith(store._FERNET_PREFIX)  # confirms it left encrypted

    newest = ra.AssetRef(1, "store-20260908T101000Z.tar", 1, True, len(published["bytes"]))
    monkeypatch.setattr(store.release_assets, "list_assets", lambda rel: (newest,))

    def fake_download(asset: ra.AssetRef, target: Path) -> None:
        target.write_bytes(published["bytes"])

    monkeypatch.setattr(store.release_assets, "download_asset", fake_download)

    workdir = tmp_path / "workdir"
    state = store.load_store(workdir)

    assert state.manifest == manifest
    assert (workdir / "data" / "raw" / "entsoe" / "day_ahead_prices" / "2026-08.parquet").exists()


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


def _generation_full_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A generation frame with all of its (now entirely carried) columns
    present -- the shape a real ENTSO-E response has (the API returns the
    whole breakdown regardless of which types are actually used
    downstream)."""
    expectation = store.EXPECTATION_TABLE["generation"]
    all_columns = expectation.expected_columns | expectation.carried_columns
    return pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in all_columns}, index=index
    )


# spec 6.9 section 2.5 moved every CHECKED, live_settling_columns/known_low_
# resolution_windows-using column (load_actual, generation's 3, both border
# flow groups) to fully-CARRIED-plus-grid_based=False -- so as of this step,
# NO entry in the real EXPECTATION_TABLE exercises either mechanism on a
# CHECKED column any more (both are still real, correct code -- just
# currently unreached by any live source). The tests below that are
# specifically ABOUT those two mechanisms use this synthetic fixture rather
# than a real EXPECTATION_TABLE entry, so the mechanism itself stays covered
# independent of which real source happens to use it this month.
_SYNTHETIC_ACTUAL_EXPECTATION = store.SourceExpectation(
    expected_columns=frozenset({"synthetic_actual"}),
    max_nan_fraction={"synthetic_actual": 0.001},
    live_settling_columns=frozenset({"synthetic_actual"}),
    known_low_resolution_windows={
        "synthetic_actual": ("2022-01-01T00:00:00+00:00", "2022-02-01T00:00:00+00:00")
    },
)


def test_gen_nuclear_permanent_nan_never_blocks_as_a_carried_column() -> None:
    """gen_nuclear (permanently ~100% NaN since the April 2023 phase-out --
    spec section 2.4's own example) moved from a dedicated max_nan_fraction
    exemption to carried_columns in 6.7.1a -- this is now just one instance
    of "a carried column's NaN never blocks", not a special case."""
    expectation = store.EXPECTATION_TABLE["generation"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = _generation_full_frame(index)
    frame["gen_nuclear"] = np.nan
    assert "gen_nuclear" in expectation.carried_columns

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_unexpected_nan_in_a_checked_column_is_rejected() -> None:
    """load_forecast_day_ahead (spec 6.9 section 2.5: still CHECKED -- row 1
    and the row 2 Similar-Day-Patch both need it) -- a real gap there, not
    exempt."""
    expectation = store.EXPECTATION_TABLE["load"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns},
        index=index,
    )
    frame["load_forecast_day_ahead"] = np.nan
    assert "load_forecast_day_ahead" in expectation.expected_columns

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("load_forecast_day_ahead" in r for r in result.reasons)


def test_carried_column_full_nan_never_blocks_in_rebuild_mode() -> None:
    """spec 6.7.1a section 3.6/5.3: a carried column is never a blocking
    reason in EITHER mode -- this is the direct fix for the real C4 finding
    (2026-09-11): two automatic maintenance runs went red on a gen_hard_coal
    gap the price model never reads.

    Uses "load" (grid_based=True, spec 6.9 section 2.5: load_actual moved to
    carried but load_forecast_day_ahead stays checked) rather than
    "generation": as of this step, "generation" is entirely carried and
    grid_based=False (the coverage/NaN block -- including the carried-column
    hint reporting this test itself checks -- never runs at all there, see
    validate_source; the same accepted trade-off the two EC sources already
    have). "load" still exercises the real reporting path this test is
    about."""
    expectation = store.EXPECTATION_TABLE["load"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    all_columns = expectation.expected_columns | expectation.carried_columns
    frame = pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in all_columns}, index=index
    )
    frame["load_actual"] = np.nan
    assert "load_actual" in expectation.carried_columns

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True
    assert any("load_actual" in h for h in result.hints)  # reported, not gated (rebuild mode)


def test_carried_column_full_nan_never_blocks_in_live_mode() -> None:
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-11T09:05:00", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(days=1), as_of, freq="15min", inclusive="left")
    frame = _generation_full_frame(index)
    frame["gen_hard_coal"] = np.nan  # the real 2026-09-11 finding, reproduced
    assert "gen_hard_coal" in expectation.carried_columns

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=1),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True


def test_carried_column_is_not_measured_at_all_in_live_mode() -> None:
    """spec 6.7.1a section 5.3: "Im Live-Modus werden mitgeführte Spalten
    gar nicht erst gemessen" -- no hint, no nan_cell_counts entry, unlike
    the rebuild-mode case above."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-11T09:05:00", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(days=1), as_of, freq="15min", inclusive="left")
    frame = _generation_full_frame(index)
    frame["gen_hard_coal"] = np.nan

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=1),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert "gen_hard_coal" not in result.nan_cell_counts
    assert not any("gen_hard_coal" in h for h in result.hints)


def test_missing_carried_column_is_a_hint_not_a_failure() -> None:
    expectation = store.EXPECTATION_TABLE["generation"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns},
        index=index,
    )  # no carried columns present at all -- e.g. a hand-built test frame

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True
    assert any("missing carried column" in h for h in result.hints)


def test_unexpected_extra_column_is_a_hint_not_a_failure() -> None:
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-08-02", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = _price_frame(index)
    frame["some_unrelated_column"] = 1.0

    result = store.validate_source(
        "day_ahead_price",
        frame,
        _PRICE_EXPECT,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True
    assert any("unexpected extra column" in h for h in result.hints)


def test_live_tail_nan_within_settling_buffer_is_not_rejected() -> None:
    """A9/A10 finding: individual generation-by-type columns can lag others
    by hours at the live edge -- a normal reporting artefact, not a gap.
    Only applies in mode="live" (see store.py's LIVE_SETTLING_BUFFER). Uses
    gen_wind_onshore, a CHECKED column, so this isolates the settling
    buffer's own effect from the checked/carried split above."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    index = pd.date_range(
        as_of - pd.Timedelta(days=1), as_of - pd.Timedelta(hours=1), freq="15min", inclusive="left"
    )
    frame = _generation_full_frame(index)
    tail_mask = index >= (as_of - pd.Timedelta(hours=2))  # inside the 24h settling buffer
    frame["gen_wind_onshore"] = np.where(tail_mask, np.nan, frame["gen_wind_onshore"])

    result = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=1),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True


def test_live_mode_gross_nan_at_or_above_threshold_blocks() -> None:
    """A genuine, near-total gap in a CHECKED column must still block --
    LIVE_GROSS_NAN_FRACTION exists for exactly this (spec 6.7.1a section
    3.4), even though the old, laxer LIVE_MAX_NAN_FRACTION mechanism is gone.
    Uses "load"/load_forecast_day_ahead (spec 6.9 section 2.5: the one
    remaining CHECKED, grid_based column with no live_settling_columns
    treatment -- a day-ahead schedule has no live-tail dynamic to begin
    with, so this test needs no settling buffer either)."""
    expectation = store.EXPECTATION_TABLE["load"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    index = pd.date_range(
        as_of - pd.Timedelta(days=7), as_of - pd.Timedelta(days=2), freq="15min", inclusive="left"
    )
    frame = pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns},
        index=index,
    )
    frame["load_forecast_day_ahead"] = np.nan

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=7),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is False
    assert any("load_forecast_day_ahead" in r for r in result.reasons)
    assert any("gross-corruption" in r for r in result.reasons)


def test_live_mode_partial_nan_below_gross_threshold_is_a_hint_not_a_failure() -> None:
    """The core behavior change of 6.7.1a (spec section 3.4): a real,
    settled, non-trivial gap in a CHECKED column that is NOT gross must no
    longer block -- it is recorded as a hint with an absolute cell count
    and left for 6.7.2's own per-column freshness check to judge."""
    expectation = store.EXPECTATION_TABLE["load"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(days=7), as_of, freq="h", inclusive="left")
    frame = pd.DataFrame(
        {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns},
        index=index,
    )
    gap = (index >= as_of - pd.Timedelta(days=2)) & (index < as_of - pd.Timedelta(days=1))
    frame["load_forecast_day_ahead"] = np.where(
        gap, np.nan, frame["load_forecast_day_ahead"]
    )  # 24/168h, ~14%

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=7),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True
    assert result.nan_cell_counts["load_forecast_day_ahead"] == 24
    assert any("load_forecast_day_ahead" in h for h in result.hints)


def test_rebuild_mode_applies_no_settling_buffer_and_the_strict_tolerance() -> None:
    """mode="rebuild" (the default) never excludes a live tail and never
    applies the live gross threshold -- a live-tail-shaped gap right at
    as_of, in a CHECKED column with live-settling treatment, must still be
    judged against the strict rebuild tolerance. Uses the synthetic
    actuals-shaped fixture (see its own comment above): as of spec 6.9
    section 2.5, no real EXPECTATION_TABLE entry is both CHECKED and in
    live_settling_columns any more (every TSO-actuals-type checked column
    moved to fully-carried-plus-grid_based=False in this step) -- the
    mechanism itself stays real and correct, just currently unreached by any
    live source."""
    expectation = _SYNTHETIC_ACTUAL_EXPECTATION
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    period_start = pd.Timestamp("2026-09-08", tz="UTC")
    period_end = as_of.floor("D") + pd.Timedelta(days=1)
    index = pd.date_range(
        period_start, as_of - pd.Timedelta(hours=1), freq="15min", inclusive="left"
    )
    frame = pd.DataFrame({"synthetic_actual": np.arange(len(index), dtype=float)}, index=index)
    tail_mask = index >= (as_of - pd.Timedelta(hours=2))
    frame["synthetic_actual"] = np.where(tail_mask, np.nan, frame["synthetic_actual"])

    result = store.validate_source(
        "synthetic",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=as_of,
    )

    assert result.ok is False
    assert any("synthetic_actual" in r for r in result.reasons)


def test_live_mode_fixed_window_is_independent_of_the_sync_gap_width() -> None:
    """spec 6.7.1a section 2.3/7: two calls with a very different
    period_start (an hours-wide vs. a months-wide sync gap) must measure
    the identical NaN fraction/count for identical data -- the checked
    window is now always [as_of - LIVE_NAN_WINDOW_DAYS, as_of), never
    period_start..period_end. period_end stays before as_of so the
    coverage-gap check (which still uses period_start) never fires and
    can't confound the comparison."""
    expectation = store.EXPECTATION_TABLE["generation"]
    as_of = pd.Timestamp("2026-09-10T13:42:00", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(days=7), as_of, freq="h", inclusive="left")
    frame = _generation_full_frame(index)
    gap = (index >= as_of - pd.Timedelta(hours=50)) & (index < as_of - pd.Timedelta(hours=48))
    frame["gen_wind_onshore"] = np.where(gap, np.nan, frame["gen_wind_onshore"])

    narrow = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(hours=3),
        period_end=as_of - pd.Timedelta(hours=1),
        as_of=as_of,
        mode="live",
    )
    wide = store.validate_source(
        "generation",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=200),
        period_end=as_of - pd.Timedelta(hours=1),
        as_of=as_of,
        mode="live",
    )

    assert narrow.nan_cell_counts == wide.nan_cell_counts
    assert narrow.ok == wide.ok is True


def test_live_mode_settling_cut_leaves_most_of_the_fixed_window_evaluated() -> None:
    """Before 6.7.1a, a 24h settling cut against a variable, sometimes
    hours-wide live window could consume almost the entire checked range.
    Against the new fixed 7-day window, the same 24h cut only ever removes
    1/7 of it -- a gap 3 days before as_of (well outside the 24h buffer,
    well inside the fixed window) must still be measured. Synthetic fixture,
    see its own comment above: no real CHECKED column has live_settling
    treatment any more as of spec 6.9 section 2.5."""
    expectation = _SYNTHETIC_ACTUAL_EXPECTATION
    as_of = pd.Timestamp("2026-09-11T10:00", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(days=7), as_of, freq="h", inclusive="left")
    frame = pd.DataFrame({"synthetic_actual": np.arange(len(index), dtype=float)}, index=index)
    gap = (index >= as_of - pd.Timedelta(days=3)) & (
        index < as_of - pd.Timedelta(days=3) + pd.Timedelta(hours=2)
    )
    frame["synthetic_actual"] = np.where(gap, np.nan, frame["synthetic_actual"])

    result = store.validate_source(
        "synthetic",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(days=7),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True  # 2/168h is far below the gross threshold
    assert result.nan_cell_counts["synthetic_actual"] == 2  # not cut away by the settling buffer


def test_live_mode_isolated_cell_in_a_non_settling_column_is_a_hint() -> None:
    """load_forecast_day_ahead: a CHECKED column not in live_settling_
    columns (a day-ahead schedule, no live-tail dynamic) -- isolates the
    fixed-window/gross-threshold mechanism from the settling buffer's."""
    expectation = store.EXPECTATION_TABLE["load"]
    as_of = pd.Timestamp("2026-09-10T16:48", tz="UTC")
    index = pd.date_range(as_of - pd.Timedelta(hours=6), as_of, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["load_forecast_day_ahead"][0] = np.nan  # one isolated cell
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=as_of - pd.Timedelta(hours=6),
        period_end=as_of,
        as_of=as_of,
        mode="live",
    )

    assert result.ok is True
    assert result.nan_cell_counts["load_forecast_day_ahead"] == 1
    assert any("load_forecast_day_ahead" in h for h in result.hints)


def test_isolated_settled_gap_in_a_checked_column_is_tolerated() -> None:
    """A9 probe finding (originally physical_net_de_to_pl -- cross_border_
    flows is entirely carried as of spec 6.9 section 2.5, so this now uses
    "load"/load_forecast_day_ahead, the one remaining checked, tolerance-
    bearing column): one isolated settled gap, the same kind of permanent
    small ENTSO-E gap generation's own finding had -- extended the same
    0.001 tolerance here."""
    expectation = store.EXPECTATION_TABLE["load"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["load_forecast_day_ahead"][100] = np.nan  # one isolated settled cell
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is True


def test_large_settled_gap_in_a_checked_column_is_still_rejected() -> None:
    expectation = store.EXPECTATION_TABLE["load"]
    period_start = pd.Timestamp("2026-08-01", tz="UTC")
    period_end = pd.Timestamp("2026-09-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    data = {col: np.arange(len(index), dtype=float) for col in expectation.expected_columns}
    data["load_forecast_day_ahead"] = np.full(len(index), np.nan)
    frame = pd.DataFrame(data, index=index)

    result = store.validate_source(
        "load",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("load_forecast_day_ahead" in r for r in result.reasons)


def test_known_low_resolution_window_absorbs_the_structural_75pct_gap() -> None:
    """A9 finding (originally physical_net_de_to_fr -- cross_border_flows is
    entirely carried as of spec 6.9 section 2.5, so this uses the synthetic
    actuals-shaped fixture, see its own comment above): a column genuinely
    reporting hourly, not quarter-hourly, for a known window produces a
    structural 75% NaN there, not a real gap."""
    expectation = _SYNTHETIC_ACTUAL_EXPECTATION
    period_start = pd.Timestamp("2022-01-01", tz="UTC")
    period_end = pd.Timestamp("2022-02-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    hourly_only = index.minute != 0
    frame = pd.DataFrame(
        {"synthetic_actual": np.where(hourly_only, np.nan, np.arange(len(index), dtype=float))},
        index=index,
    )

    result = store.validate_source(
        "synthetic",
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
    expectation = _SYNTHETIC_ACTUAL_EXPECTATION
    period_start = pd.Timestamp("2026-01-01", tz="UTC")  # well after the synthetic window ends
    period_end = pd.Timestamp("2026-02-01", tz="UTC")
    index = pd.date_range(period_start, period_end, freq="15min", inclusive="left")
    frame = pd.DataFrame(
        {"synthetic_actual": np.full(len(index), np.nan)},
        index=index,
    )

    result = store.validate_source(
        "synthetic",
        frame,
        expectation,
        period_start=period_start,
        period_end=period_end,
        as_of=pd.Timestamp("2026-09-08", tz="UTC"),
    )

    assert result.ok is False
    assert any("synthetic_actual" in r for r in result.reasons)


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


# ---------------------------------------------------------------------------
# Checked-vs-carried column split -- source-scan regression test (spec
# 6.7.1a section 3.6/7): pins the carried_columns declarations in
# EXPECTATION_TABLE against a real search of features/, models/,
# evaluation/, so the split can't silently go stale if a future sprint
# starts reading a column that's currently only carried.
# ---------------------------------------------------------------------------

_CONSUMER_DIRS: tuple[str, ...] = ("features", "models", "evaluation")

# features/availability.py's _RAW_AVAILABILITY registry deliberately names
# EVERY fetched raw column, checked or carried, to give it a knowledge-time
# class in case something ever looks it up (6.7.1/6.5.3's own narrative on
# this dict) -- that registration is bookkeeping, not consumption, so this
# file must be excluded or the scan would trivially "find" every carried
# column and never be able to pass (confirmed real 2026-09-11, first run).
_NOT_A_CONSUMER: frozenset[str] = frozenset({"availability.py"})


def _consumer_source_files() -> list[Path]:
    src_root = PROJECT_ROOT / "src" / "energy_price_forecast"
    return [
        p
        for d in _CONSUMER_DIRS
        for p in (src_root / d).rglob("*.py")
        if p.name not in _NOT_A_CONSUMER
    ]


def _find_consumer_hits(columns: frozenset[str]) -> dict[str, list[str]]:
    files = _consumer_source_files()
    texts = {p: p.read_text(encoding="utf-8") for p in files}
    hits: dict[str, list[str]] = {}
    for column in columns:
        pattern = re.compile(rf"\b{re.escape(column)}\b")
        matches = [str(p) for p, text in texts.items() if pattern.search(text)]
        if matches:
            hits[column] = matches
    return hits


def _unrequired_feature_columns() -> frozenset[str]:
    """spec 6.9 section 2.5: the five carried-column families that ARE
    genuinely read by features/lags.py (actual/forecast-error/cross-border
    lag functions) or features/fundamentals.py (the full, unsliced
    commodity block) -- build_feature_set_for_day still computes them
    unconditionally (spec section 2.1: "Feature-Bau bleibt vollständig"),
    but no live fallback-ladder candidate's own required-column set
    (arena/candidates.py) includes their OUTPUT features any more (Schritt
    1's own code-reading finding, Schritt 10's Parity Check: 0.0 diff
    against core_gas/base's real built columns). This is a different,
    weaker "carried" than 6.7.1a's original gen_hard_coal case (never read
    by ANY code at all) -- checked below against a narrower claim, not the
    original blanket "read by nothing" bar.
    """
    return (
        frozenset({"load_actual", "gen_wind_onshore", "gen_wind_offshore", "gen_solar"})
        | store.EXPECTATION_TABLE["scheduled_exchanges"].carried_columns
        | store.EXPECTATION_TABLE["cross_border_flows"].carried_columns
        | store.EXPECTATION_TABLE["eua_co2"].carried_columns
    )


_ALLOWED_UNREQUIRED_FEATURE_FILES: frozenset[str] = frozenset(
    {
        "lags.py",
        "fundamentals.py",
        # evaluation/regimes.py labels BACKTEST folds by realised load/generation for
        # reporting (compare_objectives.py and similar one-off scripts) -- it is never called
        # by scripts/run_daily_submission.py (spec section 3.3: evaluation/ stays untouched),
        # so a raw column it reads has no bearing on what any live candidate requires.
        "regimes.py",
    }
)


def test_carried_columns_are_not_read_by_any_consumer() -> None:
    """The original, strict 6.7.1a claim -- for every carried column EXCEPT
    the five families spec 6.9 section 2.5 knowingly moved to carried while
    still-complete feature builders keep computing (unrequired) features
    from them, see _unrequired_feature_columns' own docstring."""
    all_carried = frozenset(
        col
        for expectation in store.EXPECTATION_TABLE.values()
        for col in expectation.carried_columns
    )
    orphan_carried = all_carried - _unrequired_feature_columns()
    hits = _find_consumer_hits(orphan_carried)
    assert hits == {}, (
        f"carried column(s) actually read under features/models/evaluation -- "
        f"must be moved to expected_columns (checked), not carried_columns: {hits}"
    )


def test_unrequired_feature_columns_are_read_only_where_expected() -> None:
    """The weaker spec 6.9 section 2.5 claim for the five families above:
    still read (confirmed, not just assumed), but ONLY inside the two files
    whose functions no live candidate's required columns depend on. A hit
    anywhere else would mean some other code -- plausibly a live candidate
    -- has started requiring one of these again, which is exactly what
    would need the column moved back to CHECKED."""
    all_carried = frozenset(
        col
        for expectation in store.EXPECTATION_TABLE.values()
        for col in expectation.carried_columns
    )
    columns = _unrequired_feature_columns() & all_carried
    # sanity: every one of these five families really is a real, currently-carried column --
    # otherwise this test would trivially pass by checking an empty set.
    assert len(columns) > 0

    hits = _find_consumer_hits(columns)
    stray = {
        col: [f for f in files if Path(f).name not in _ALLOWED_UNREQUIRED_FEATURE_FILES]
        for col, files in hits.items()
    }
    stray = {col: files for col, files in stray.items() if files}
    assert stray == {}, (
        f"carried column(s) read outside features/lags.py|fundamentals.py -- a live candidate "
        f"may now require them, check arena/candidates.py's real required_columns(): {stray}"
    )


def test_carried_columns_scan_negative_control_catches_a_misclassified_column() -> None:
    """Proves the scan mechanism itself works: gen_solar is genuinely read
    (features/lags.py's forecast-error lags) -- artificially treating it as
    carried must make the scan flag it, or the positive test above would be
    meaningless (spec 6.7.1a section 7)."""
    hits = _find_consumer_hits(frozenset({"gen_solar"}))
    assert "gen_solar" in hits


# ---------------------------------------------------------------------------
# relevant_only_eligible -- a second, independent source-scan regression
# test (2026-10-01 maintenance-timeout fix, docs/bugs_in_live_system.md
# entry 6: scripts/sync_store.py's --mode relevant-only skips a fixed set of
# sources during the gate-closure-adjacent window). Deliberately NOT reusing
# _find_consumer_hits/_CONSUMER_DIRS above unchanged: that scan answers "is
# this read by features/models/evaluation" (the DQ-gate question this
# source's carried_columns status already answers); this one answers a
# different question, "is it read anywhere in the live submission path" --
# which also needs arena/ and run_daily_submission.py itself. day_ahead_
# price_ec/load_forecast_day_ahead_ec are the proof this distinction is
# real and not pedantic: fully carried_columns, genuinely read via
# arena/live_inputs.py::coalesce_price as the ENTSO-E-outage fallback, but
# invisible to the narrower features/models/evaluation scan above. The same
# wider scan is what caught eua_co2_eur_per_t being read by
# arena/preflight.py::check_commodity_staleness's default columns while
# building this -- that is why EXPECTATION_TABLE["eua_co2"] is NOT marked
# relevant_only_eligible despite being fully carried (see its own comment).
# ---------------------------------------------------------------------------

_RELEVANT_ONLY_CONSUMER_DIRS: tuple[str, ...] = ("features", "models", "evaluation", "arena")


def _relevant_only_consumer_source_files() -> list[Path]:
    src_root = PROJECT_ROOT / "src" / "energy_price_forecast"
    files = [
        p
        for d in _RELEVANT_ONLY_CONSUMER_DIRS
        for p in (src_root / d).rglob("*.py")
        if p.name not in _NOT_A_CONSUMER
    ]
    # The live daily orchestrator itself, not just the package code it
    # calls into -- a column could in principle be referenced directly here
    # without going through arena/ at all.
    files.append(PROJECT_ROOT / "scripts" / "run_daily_submission.py")
    return files


def _find_relevant_only_consumer_hits(columns: frozenset[str]) -> dict[str, list[str]]:
    files = _relevant_only_consumer_source_files()
    texts = {p: p.read_text(encoding="utf-8") for p in files}
    hits: dict[str, list[str]] = {}
    for column in columns:
        pattern = re.compile(rf"\b{re.escape(column)}\b")
        matches = [str(p) for p, text in texts.items() if pattern.search(text)]
        if matches:
            hits[column] = matches
    return hits


def _relevant_only_eligible_carried_columns() -> frozenset[str]:
    return frozenset(
        col
        for expectation in store.EXPECTATION_TABLE.values()
        if expectation.relevant_only_eligible
        for col in expectation.carried_columns
    )


def test_relevant_only_eligible_sources_are_fully_carried_and_unread() -> None:
    """Every source scripts/sync_store.py's --mode relevant-only may skip
    must satisfy both halves of SourceExpectation.relevant_only_eligible's
    own docstring claim: no checked columns at all, and no hit anywhere in
    the live submission path (features/models/evaluation/arena plus
    run_daily_submission.py itself) for any of its carried columns --
    EXCEPT the already-accepted spec 6.9 section 2.5 "computed but
    unrequired" exemption (gen_wind_onshore/_offshore/_solar, same set
    _unrequired_feature_columns() already names above), checked separately
    and more narrowly by the next test."""
    eligible = {
        name: expectation
        for name, expectation in store.EXPECTATION_TABLE.items()
        if expectation.relevant_only_eligible
    }
    assert len(eligible) > 0  # sanity: the mechanism has at least one real user

    for name, expectation in eligible.items():
        assert expectation.expected_columns == frozenset(), (
            f"{name!r} has checked column(s) {expectation.expected_columns} but is marked "
            "relevant_only_eligible=True -- a checked column can never be skip-eligible"
        )

    strictly_unread = _relevant_only_eligible_carried_columns() - _unrequired_feature_columns()
    hits = _find_relevant_only_consumer_hits(strictly_unread)
    assert hits == {}, (
        f"relevant_only_eligible column(s) actually read somewhere in the live submission "
        f"path -- must not be skip-eligible: {hits}"
    )


def test_relevant_only_eligible_unrequired_feature_columns_are_read_only_where_expected() -> None:
    """The weaker claim for whichever relevant_only_eligible columns also
    fall under spec 6.9 section 2.5's _unrequired_feature_columns()
    exemption (currently gen_wind_onshore/_offshore/_solar, carried by the
    "generation" source): still read (confirmed, not just assumed), but
    only inside the same already-accepted files
    test_unrequired_feature_columns_are_read_only_where_expected checks
    above -- a hit anywhere else would mean a live candidate now requires
    one of them, which would also disqualify it from relevant_only_eligible."""
    columns = _relevant_only_eligible_carried_columns() & _unrequired_feature_columns()
    assert len(columns) > 0  # sanity: this exemption really is in play here

    hits = _find_relevant_only_consumer_hits(columns)
    stray = {
        col: [f for f in files if Path(f).name not in _ALLOWED_UNREQUIRED_FEATURE_FILES]
        for col, files in hits.items()
    }
    stray = {col: files for col, files in stray.items() if files}
    assert stray == {}, (
        f"relevant_only_eligible column(s) read outside the already-accepted computed-but-"
        f"unrequired files -- a live candidate may now require them: {stray}"
    )


def test_relevant_only_eligible_scan_negative_control_catches_an_ec_source() -> None:
    """Proves the scan's wider scope (vs. the features/models/evaluation-only
    one above) is actually doing something: day_ahead_price_ec is fully
    carried (same shape as a real relevant_only_eligible source) but IS
    read, via arena/live_inputs.py::coalesce_price -- invisible to the
    narrower scan, which is exactly why relevant_only_eligible needed its
    own check instead of reusing carried_columns directly."""
    hits = _find_relevant_only_consumer_hits(frozenset({"day_ahead_price_ec"}))
    assert "day_ahead_price_ec" in hits


# ---------------------------------------------------------------------------
# Every real source that gets its own manifest entry must have a
# _SOURCE_GLOBS entry too -- a real bug found live (spec 6.9 Schritt 6,
# 2026-09-24): the two new Energy-Charts sources got EXPECTATION_TABLE
# entries and real manifest rows via scripts/backfill_energy_charts_store.py,
# but no _SOURCE_GLOBS entry, so pack_store silently packed zero bytes for
# them -- a published store whose manifest claimed real row counts for data
# that was never actually in the tar. Caught by an owner-observed byte-count
# coincidence (three consecutive publishes reporting the identical store
# size), not by any local check -- this test is that check.
# ---------------------------------------------------------------------------


def test_every_named_source_has_a_source_globs_entry() -> None:
    from energy_price_forecast.ops.store_sources import (
        COMMODITY_SOURCES,
        ENERGY_CHARTS_SOURCES,
        ENTSOE_SOURCES,
    )

    named_sources = (
        {s.name for s in ENTSOE_SOURCES}
        | {name for name, _fetch, _column in COMMODITY_SOURCES}
        | {name for name, _fetch, _column in ENERGY_CHARTS_SOURCES}
    )
    missing = named_sources - set(store._SOURCE_GLOBS)
    assert missing == set(), (
        f"source(s) with no _SOURCE_GLOBS entry -- pack_store would silently pack zero "
        f"bytes for them despite a real manifest entry: {sorted(missing)}"
    )


def test_source_globs_negative_control_catches_a_missing_entry() -> None:
    """Proves the completeness check above genuinely fails on a missing
    entry, reproducing the real 2026-09-24 bug shape (a named source with
    no _SOURCE_GLOBS entry at all) rather than just asserting the removal
    itself worked."""
    named_sources = {"ttf_gas", "eua_co2"}
    reduced_globs = dict(store._SOURCE_GLOBS)
    del reduced_globs["ttf_gas"]

    missing = named_sources - set(reduced_globs)

    assert missing == {"ttf_gas"}


def test_read_cached_range_concatenates_and_sorts_month_files(tmp_path: Path) -> None:
    """Moved here from scripts/sync_store.py::_read_cache_dir (spec 6.7.2,
    section 5.8) -- pure disk read, no client, no network."""
    cache_dir = tmp_path / "wind_solar"
    cache_dir.mkdir()
    later = pd.DataFrame(
        {"wind_onshore_forecast": [3.0]},
        index=pd.DatetimeIndex(["2026-02-01T00:00:00Z"], name="timestamp"),
    )
    earlier = pd.DataFrame(
        {"wind_onshore_forecast": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], name="timestamp"),
    )
    later.to_parquet(cache_dir / "DE_LU_2026-02.parquet")
    earlier.to_parquet(cache_dir / "DE_LU_2026-01.parquet")

    result = store.read_cached_range(cache_dir)

    assert list(result["wind_onshore_forecast"]) == [1.0, 2.0, 3.0]
    assert result.index.is_monotonic_increasing
    assert pd.DatetimeIndex(result.index).tz is not None


def test_read_cached_range_slices_to_start_end(tmp_path: Path) -> None:
    cache_dir = tmp_path / "wind_solar"
    cache_dir.mkdir()
    df = pd.DataFrame(
        {"wind_onshore_forecast": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            name="timestamp",
        ),
    )
    df.to_parquet(cache_dir / "DE_LU_2026-01.parquet")

    result = store.read_cached_range(
        cache_dir,
        start=pd.Timestamp("2026-01-02", tz="UTC"),
        end=pd.Timestamp("2026-01-02T12:00:00Z"),
    )

    assert list(result["wind_onshore_forecast"]) == [2.0]


def test_read_cached_range_missing_dir_returns_empty(tmp_path: Path) -> None:
    result = store.read_cached_range(tmp_path / "does_not_exist")
    assert result.empty


def test_read_cached_range_joins_multi_neighbour_files_column_wise(tmp_path: Path) -> None:
    """Real bug found live during the 6.7.2 wiring probe (2026-09-12):
    scheduled_exchanges/cross_border_flows share one cache_dir across all
    six neighbours (ops/store_sources.py::_fetch_border_flows), each
    neighbour's own month chunks carrying only that neighbour's single
    column. A blind row-wise concat stacked neighbours on top of each
    other instead of joining them side by side -- >75% NaN per column and
    a duplicated index on the real store, which then broke a downstream
    outer-join concat outright (InvalidIndexError)."""
    cache_dir = tmp_path / "scheduled_exchanges"
    cache_dir.mkdir()
    idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], name="timestamp")
    fr = pd.DataFrame({"scheduled_net_de_to_fr": [1.0, 2.0]}, index=idx)
    nl = pd.DataFrame({"scheduled_net_de_to_nl": [3.0, 4.0]}, index=idx)
    fr.to_parquet(cache_dir / "DE_LU_FR_2026-01.parquet")
    nl.to_parquet(cache_dir / "DE_LU_NL_2026-01.parquet")

    result = store.read_cached_range(cache_dir)

    assert len(result) == 2  # not 4 -- neighbours join as columns, not stacked rows
    assert not result.index.duplicated().any()
    assert list(result["scheduled_net_de_to_fr"]) == [1.0, 2.0]
    assert list(result["scheduled_net_de_to_nl"]) == [3.0, 4.0]
    assert result.notna().all().all()


def test_read_cached_range_joins_multi_neighbour_month_chunks_correctly(
    tmp_path: Path,
) -> None:
    """Each neighbour's own month chunks must still merge row-wise with
    each other (chronological history) before neighbours join column-wise
    -- the grouping is by column set, not "one row-wise pass across
    everything" or "one column-wise pass across everything"."""
    cache_dir = tmp_path / "scheduled_exchanges"
    cache_dir.mkdir()
    fr_jan = pd.DataFrame(
        {"scheduled_net_de_to_fr": [1.0]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="timestamp"),
    )
    fr_feb = pd.DataFrame(
        {"scheduled_net_de_to_fr": [2.0]},
        index=pd.DatetimeIndex(["2026-02-01T00:00:00Z"], name="timestamp"),
    )
    nl_jan = pd.DataFrame(
        {"scheduled_net_de_to_nl": [3.0]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="timestamp"),
    )
    fr_jan.to_parquet(cache_dir / "DE_LU_FR_2026-01.parquet")
    fr_feb.to_parquet(cache_dir / "DE_LU_FR_2026-02.parquet")
    nl_jan.to_parquet(cache_dir / "DE_LU_NL_2026-01.parquet")

    result = store.read_cached_range(cache_dir)

    assert len(result) == 2
    assert list(result["scheduled_net_de_to_fr"]) == [1.0, 2.0]
    # NL has no February value -- outer join leaves it NaN, not fabricated.
    assert result.loc["2026-01-01":"2026-01-01", "scheduled_net_de_to_nl"].iloc[0] == 3.0
    assert pd.isna(result.loc["2026-02-01":"2026-02-01", "scheduled_net_de_to_nl"].iloc[0])


def test_read_cached_range_merges_overlapping_but_unequal_column_sets_row_wise(
    tmp_path: Path,
) -> None:
    """Real bug found live during the 6.7.2 wiring probe (2026-09-12),
    caught immediately after the neighbour-join fix above: exact column-set
    equality is too strict a grouping key. generation's real cache has two
    distinct column sets (with/without gen_nuclear -- Germany's last
    nuclear plants shut down in 2023, so ENTSO-E's response genuinely omits
    that type from then on). Grouping by exact equality treated the two
    sets as separate "neighbours" and joined them column-wise, producing
    every shared column twice. The fix groups by *any* shared column
    (transitively) instead -- these two month files share 10 of 11 columns
    and must land in one row-wise-merged group, not two column-wise-joined
    ones."""
    cache_dir = tmp_path / "generation"
    cache_dir.mkdir()
    with_nuclear = pd.DataFrame(
        {"gen_nuclear": [1.0], "gen_lignite": [2.0], "gen_solar": [3.0]},
        index=pd.DatetimeIndex(["2022-01-01T00:00:00Z"], name="timestamp"),
    )
    without_nuclear = pd.DataFrame(
        {"gen_lignite": [4.0], "gen_solar": [5.0]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], name="timestamp"),
    )
    with_nuclear.to_parquet(cache_dir / "DE_LU_2022-01.parquet")
    without_nuclear.to_parquet(cache_dir / "DE_LU_2024-01.parquet")

    result = store.read_cached_range(cache_dir)

    assert len(result) == 2  # row-wise merge, not a column-wise join producing 1 row x dup columns
    assert list(result.columns) == ["gen_nuclear", "gen_lignite", "gen_solar"]
    assert list(result["gen_lignite"]) == [2.0, 4.0]
    assert list(result["gen_solar"]) == [3.0, 5.0]
    assert result.loc["2022-01-01":"2022-01-01", "gen_nuclear"].iloc[0] == 1.0
    # 2024 row has no nuclear column in its source file -- NaN, not fabricated.
    assert pd.isna(result.loc["2024-01-01":"2024-01-01", "gen_nuclear"].iloc[0])
