"""Test for scripts/sync_store.py's weather sync step.

Glue scripts in this project are otherwise untested by convention (see
rebuild_store.py's own module docstring) -- this one function is the
exception, guarding a real, confirmed bug (A9, 2026-09-10): a weather run
that fetches successfully (HTTP 200) but fails validate_weather_run left
its bad cache file sitting on disk forever, ready to be swept into the
next publish_store call regardless of what the manifest actually claims is
covered. Same "found a real bug, add a regression test for exactly that"
precedent as tests/test_compare_arena_gate.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.data.weather_client import WeatherRunUnavailable, run_init_for_target_day
from energy_price_forecast.ops import store
from energy_price_forecast.ops.windows import next_delivery_day
from scripts import sync_store

_EMPTY_MANIFEST = store.Manifest(
    store_format_version=store.STORE_FORMAT_VERSION,
    created_at_utc="2026-09-12T09:00:00+00:00",
    run_id="1",
    run_url="",
    code_sha="abc",
    sources={},
)


def test_sync_weather_deletes_the_cache_file_on_failed_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache file must not already exist before the call: with the
    WEATHER_BACKFILL_DAYS path-check skip (docs/sprint6_fix_weather_run_offset.md
    section 3.1), an already-present file is trusted and never re-validated
    -- exactly the point of the skip. So this test simulates fetch_run's own
    real behavior (write to the cache path, then return the frame for
    _sync_weather to validate) rather than pre-seeding the file, to still
    genuinely exercise the invalid-after-download-delete path A9 guards."""
    run_init = pd.Timestamp("2026-09-07T00:00", tz="UTC")
    cache_file = tmp_path / "2026-09-07T00Z.parquet"

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        cache_file.write_bytes(b"stand-in bytes for a bad fetched run")
        return pd.DataFrame({"c": [1.0]})

    monkeypatch.setattr(sync_store, "run_init_for_target_day", lambda day: run_init)
    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", lambda ts, **kwargs: cache_file)
    monkeypatch.setattr(
        store,
        "validate_weather_run",
        lambda df: store.ValidationResult(
            source="weather_single_runs", ok=False, reasons=("synthetic failure",)
        ),
    )

    manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2026-09-07T10:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc",
        sources={},
    )
    log = sync_store.RunLog()

    entry = sync_store._sync_weather(manifest, pd.Timestamp("2026-09-07T10:00", tz="UTC"), log)

    assert not cache_file.exists()
    assert entry is None
    assert log.any_failure is True


# ---------------------------------------------------------------------------
# docs/sprint6_fix_weather_run_offset.md -- the weather-run-offset fix
# ---------------------------------------------------------------------------


def _never_existing_cache_path(tmp_path: Path):  # noqa: ANN201 -- returns a monkeypatch target
    def _path(run_init_utc: pd.Timestamp, **kwargs: object) -> Path:
        return tmp_path / f"{run_init_utc.isoformat().replace(':', '_')}.parquet"

    return _path


def test_sync_weather_fetches_the_run_tomorrows_submission_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 -- regression test for the real 2026-09-12 finding, fixed dates
    deliberate (reads as documentation of the bug): as_of=2026-09-12T09:00
    Europe/Berlin -> next delivery day 2026-09-13 -> the run tomorrow's
    submission needs is TODAY's own 00Z run (2026-09-12T00:00 UTC), not
    yesterday's (2026-09-11T00:00 UTC, all the pre-fix code ever requested).
    Confirmed red against the pre-fix single-run _sync_weather before this
    fix landed (docs/sprint6_fix_weather_run_offset.md section 6)."""
    as_of = pd.Timestamp("2026-09-12T09:00", tz="Europe/Berlin")
    expected_run_init = pd.Timestamp("2026-09-12T00:00", tz="UTC")

    requested_runs: list[pd.Timestamp] = []

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        requested_runs.append(run_init_utc)
        raise WeatherRunUnavailable(run_init_utc, "ecmwf_ifs", 400, "not published yet")

    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", _never_existing_cache_path(tmp_path))

    sync_store._sync_weather(_EMPTY_MANIFEST, as_of, sync_store.RunLog())

    assert expected_run_init in requested_runs


def test_sync_weather_day_zero_run_matches_next_delivery_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1 -- the most important test of the fix: the run _sync_weather asks
    for first (k=0) is bound to the SAME next_delivery_day()/
    run_init_for_target_day() call the live submission path makes, not a
    hardcoded date -- stays correct even if either side's own arithmetic
    changes later."""
    as_of = pd.Timestamp("2026-09-12T09:00", tz="Europe/Berlin")
    expected_run_init = run_init_for_target_day(next_delivery_day(as_of))

    requested_runs: list[pd.Timestamp] = []

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        requested_runs.append(run_init_utc)
        raise WeatherRunUnavailable(run_init_utc, "ecmwf_ifs", 400, "not published yet")

    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", _never_existing_cache_path(tmp_path))

    sync_store._sync_weather(_EMPTY_MANIFEST, as_of, sync_store.RunLog())

    assert requested_runs[0] == expected_run_init


def test_sync_weather_backfill_skips_cached_and_fetches_only_whats_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3 -- the trailing backfill window heals a hole (a day whose own run
    was never fetched) without wasting a call on a day already cached."""
    as_of = pd.Timestamp("2026-09-12T09:00", tz="Europe/Berlin")
    delivery_day = next_delivery_day(as_of)
    already_cached_run_init = run_init_for_target_day(delivery_day - dt.timedelta(days=1))
    cache_path = tmp_path / f"{already_cached_run_init.isoformat().replace(':', '_')}.parquet"
    cache_path.write_bytes(b"already there")

    call_count = 0

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        raise WeatherRunUnavailable(run_init_utc, "ecmwf_ifs", 400, "not published yet")

    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", _never_existing_cache_path(tmp_path))

    sync_store._sync_weather(_EMPTY_MANIFEST, as_of, sync_store.RunLog())

    # WEATHER_BACKFILL_DAYS + 1 candidate days total, one already cached --
    # fetch_run must be called for exactly the rest, never for the cached one.
    assert call_count == sync_store.WEATHER_BACKFILL_DAYS


def test_sync_weather_not_yet_available_newest_run_stays_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T4 -- the newest run not being published yet is normal operation
    (spec section 3.1's table), not a failure: green, logged, no manifest
    entry change."""
    as_of = pd.Timestamp("2026-09-12T09:00", tz="Europe/Berlin")

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        raise WeatherRunUnavailable(run_init_utc, "ecmwf_ifs", 400, "not published yet")

    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", _never_existing_cache_path(tmp_path))

    log = sync_store.RunLog()
    entry = sync_store._sync_weather(_EMPTY_MANIFEST, as_of, log)

    assert log.any_failure is False
    assert entry is None
    assert "not_yet_available" in log.get("weather_single_runs").validation


def test_sync_weather_extends_an_existing_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path with a real prior manifest entry: a validated new run
    extends covered_end_utc/count, and covered_start_utc (the store's own
    history start) is untouched."""
    as_of = pd.Timestamp("2026-09-12T09:00", tz="Europe/Berlin")
    previous_run_init = pd.Timestamp("2026-09-11T00:00", tz="UTC")
    manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2026-09-11T09:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc",
        sources={
            "weather_single_runs": store.SourceManifestEntry(
                covered_start_utc="2024-03-14T00:00:00+00:00",
                covered_end_utc=previous_run_init.isoformat(),
                count=500,
                last_success_utc="2026-09-11T09:00:00+00:00",
                last_attempt_utc="2026-09-11T09:00:00+00:00",
            )
        },
    )
    cache_path = tmp_path / f"{previous_run_init.isoformat().replace(':', '_')}.parquet"
    cache_path.write_bytes(b"already there")  # the backfill day already covered

    def _fake_fetch_run(run_init_utc: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        return pd.DataFrame({"c": [1.0]})

    monkeypatch.setattr(sync_store, "fetch_run", _fake_fetch_run)
    monkeypatch.setattr(sync_store, "weather_cache_path", _never_existing_cache_path(tmp_path))
    monkeypatch.setattr(
        store,
        "validate_weather_run",
        lambda df: store.ValidationResult(source="weather_single_runs", ok=True, reasons=()),
    )

    entry = sync_store._sync_weather(manifest, as_of, sync_store.RunLog())

    assert entry is not None
    assert entry.covered_start_utc == "2024-03-14T00:00:00+00:00"
    assert entry.covered_end_utc == "2026-09-12T00:00:00+00:00"
    # Two new runs written (k=0 and k=2; k=1 was already cached), each fetch
    # in this test succeeding validation.
    assert entry.count == 502
