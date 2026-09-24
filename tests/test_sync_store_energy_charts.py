"""Tests for scripts/sync_store.py's Energy-Charts sync step and its
ops/store_sources.py date-contract adapters (spec 6.9 section 2.4/5.8,
Schritt 6).

Energy-Charts sources are the one deliberate exception in this script: an
EC failure is always a warning, never a run failure (spec section 2.4,
"Ein EC-Fehler im Pflege-Job ist nur eine Warnung, der Lauf bleibt grün")
-- every test below that hits a failure path asserts `log.any_failure`
stays False, the opposite of what the equivalent ENTSO-E/commodity tests
would assert.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from energy_price_forecast.ops import store
from energy_price_forecast.ops.store_sources import _fetch_ec_load_forecast, _fetch_ec_price
from scripts import sync_store

# Patched where store_sources.py looks the name up (it imports these
# directly, so patching energy_price_forecast.data.energy_charts's own
# attribute would not affect store_sources' already-bound reference).
STORE_SOURCES_MODULE = "energy_price_forecast.ops.store_sources"


def _manifest(sources: dict[str, store.SourceManifestEntry]) -> store.Manifest:
    return store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2026-09-24T00:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc",
        sources=sources,
    )


# ---------------------------------------------------------------------------
# ops/store_sources.py adapters: dt.date/local-calendar-day contract ->
# RowFetchFn's UTC pd.Timestamp/half-open-[start, end) contract
# ---------------------------------------------------------------------------


def test_fetch_ec_price_converts_utc_timestamps_to_local_calendar_dates() -> None:
    captured: dict[str, dt.date] = {}

    def fake_fetch_price_range(start: dt.date, end: dt.date) -> pd.Series:
        captured["start"] = start
        captured["end"] = end
        return pd.Series([1.0], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), name="x")

    with patch(f"{STORE_SOURCES_MODULE}.fetch_price_range", fake_fetch_price_range):
        _fetch_ec_price(
            pd.Timestamp("2026-09-01T00:00:00+02:00"), pd.Timestamp("2026-09-03T00:00:00+02:00")
        )

    assert captured["start"] == dt.date(2026, 9, 1)
    # end is exclusive (half-open) -- the last INCLUSIVE local calendar day
    # is one day before the exclusive UTC boundary, not the boundary itself.
    assert captured["end"] == dt.date(2026, 9, 2)


def test_fetch_ec_price_returns_empty_frame_for_a_reversed_range_without_calling_the_api() -> None:
    """Defensive only -- scripts/sync_store.py's own _gap_start never
    produces start > end, but a malformed range must not silently reach
    the API with backwards bounds either."""
    with patch(f"{STORE_SOURCES_MODULE}.fetch_price_range") as fake:
        result = _fetch_ec_price(
            pd.Timestamp("2026-09-05T00:00:00Z"), pd.Timestamp("2026-09-01T00:00:00Z")
        )

    fake.assert_not_called()
    assert result.empty


def test_fetch_ec_load_forecast_renames_the_series_to_the_store_column() -> None:
    def fake_fetch_series_range(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        assert production_type == "load"
        return pd.Series([1.0], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), name="load")

    with patch(f"{STORE_SOURCES_MODULE}.fetch_series_range", fake_fetch_series_range):
        result = _fetch_ec_load_forecast(
            pd.Timestamp("2026-09-01T00:00:00Z"), pd.Timestamp("2026-09-02T00:00:00Z")
        )

    assert list(result.columns) == ["load_forecast_day_ahead_ec"]


# ---------------------------------------------------------------------------
# scripts/sync_store.py::_sync_energy_charts_source
# ---------------------------------------------------------------------------


def test_sync_energy_charts_source_writes_merged_file_and_returns_entry(tmp_path: Path) -> None:
    index = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=index)

    with patch.object(sync_store, "ENERGY_CHARTS_DIR", tmp_path):
        entry = sync_store._sync_energy_charts_source(
            "day_ahead_price_ec",
            fake_fetch,
            "x",
            _manifest({}),
            pd.Timestamp.now(tz="UTC"),
            sync_store.RunLog(),
        )

    assert entry is not None
    assert entry.count == 4
    on_disk = pd.read_parquet(tmp_path / "day_ahead_price_ec.parquet")
    assert on_disk["x"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_sync_energy_charts_source_fetched_every_run_no_once_a_day_gate(tmp_path: Path) -> None:
    """The one deliberate difference from _sync_commodity_source: EC has no
    "already attempted today" cadence gate (spec section 2.4 vs. the
    Yahoo-Finance-specific section 2.8 cadence rule)."""
    calls = 0

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"))

    as_of = pd.Timestamp("2026-09-24T09:00:00Z")
    previous = store.SourceManifestEntry(
        covered_start_utc="2026-09-01T00:00:00+00:00",
        covered_end_utc="2026-09-01T00:00:00+00:00",
        count=1,
        last_success_utc="2026-09-24T08:00:00+00:00",
        last_attempt_utc="2026-09-24T08:00:00+00:00",
    )

    with patch.object(sync_store, "ENERGY_CHARTS_DIR", tmp_path):
        sync_store._sync_energy_charts_source(
            "day_ahead_price_ec",
            fake_fetch,
            "x",
            _manifest({"day_ahead_price_ec": previous}),
            as_of,
            sync_store.RunLog(),
        )

    assert calls == 1


def test_sync_energy_charts_source_unreachable_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise ConnectionError("boom")

    log = sync_store.RunLog()
    with patch.object(sync_store, "ENERGY_CHARTS_DIR", tmp_path):
        entry = sync_store._sync_energy_charts_source(
            "day_ahead_price_ec", fake_fetch, "x", _manifest({}), pd.Timestamp.now(tz="UTC"), log
        )

    assert entry is None
    assert log.any_failure is False
    assert any("day_ahead_price_ec" in w for w in log.warnings)


def test_sync_energy_charts_source_merge_conflict_is_a_warning_not_a_raise(
    tmp_path: Path,
) -> None:
    """merge_existing_with_fresh raises ValueError on a genuine disagreement
    -- that must not propagate out of the EC sync step and fail the whole
    run (spec section 2.4), unlike every other source in this script."""
    index = pd.date_range("2026-09-01", periods=2, freq="D", tz="UTC")
    existing_df = pd.DataFrame({"x": [1.0, 2.0]}, index=index)
    existing_df.to_parquet(tmp_path / "day_ahead_price_ec.parquet")

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({"x": [999.0, 2.0]}, index=index)  # first value disagrees

    log = sync_store.RunLog()
    with patch.object(sync_store, "ENERGY_CHARTS_DIR", tmp_path):
        entry = sync_store._sync_energy_charts_source(
            "day_ahead_price_ec", fake_fetch, "x", _manifest({}), pd.Timestamp.now(tz="UTC"), log
        )

    assert entry is None
    assert log.any_failure is False
    assert any("day_ahead_price_ec" in w for w in log.warnings)
    # The on-disk file must be untouched -- the conflict was never written.
    on_disk = pd.read_parquet(tmp_path / "day_ahead_price_ec.parquet")
    assert on_disk["x"].tolist() == [1.0, 2.0]
