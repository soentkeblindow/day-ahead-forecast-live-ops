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

from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.ops import store
from scripts import sync_store


def test_sync_weather_deletes_the_cache_file_on_failed_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_init = pd.Timestamp("2026-09-07T00:00", tz="UTC")
    cache_file = tmp_path / "2026-09-07T00Z.parquet"
    cache_file.write_bytes(b"stand-in bytes for a bad cached run")

    monkeypatch.setattr(sync_store, "run_init_for_target_day", lambda day: run_init)
    monkeypatch.setattr(sync_store, "fetch_run", lambda *a, **k: pd.DataFrame({"c": [1.0]}))
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
