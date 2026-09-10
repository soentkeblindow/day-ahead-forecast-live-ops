"""Test for scripts/sync_store.py's ENTSO-E sync step.

Glue scripts in this project are otherwise untested by convention (see
rebuild_store.py's own module docstring) -- this one function is an
exception, guarding a real, confirmed bug found by A10's first live
wiring probe (2026-09-10): fetching only the gap window and then
validating it against the full historical previous.count/covered_start
made every incremental sync look like the store was shrinking ("row count
shrank: 67 < previous 83495"-style errors on every single ENTSO-E source),
and separately explained generation's "column mismatch" failure
(gen_nuclear has no real values at all in a gap-only window starting
after April 2023). Same "found a real bug, add a regression test for
exactly that" precedent as tests/test_compare_arena_gate.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_price_forecast.ops import store
from scripts import sync_store


def test_sync_entsoe_source_fetches_full_history_not_just_the_gap(tmp_path: Path) -> None:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    full_index = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        return pd.DataFrame({"day_ahead_price": range(len(full_index))}, index=full_index)

    source = sync_store.EntsoeSource(name="day_ahead_price", fetch=fake_fetch, cache_dir=tmp_path)
    previous = store.SourceManifestEntry(
        covered_start_utc="2020-01-01T00:00:00+00:00",
        covered_end_utc="2020-01-05T00:00:00+00:00",
        count=5,
        last_success_utc="2020-01-05T00:00:00+00:00",
        last_attempt_utc="2020-01-05T00:00:00+00:00",
    )
    manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2020-01-10T00:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc",
        sources={"day_ahead_price": previous},
    )
    log = sync_store.RunLog()
    as_of = pd.Timestamp("2020-01-10T00:00:00+00:00", tz="UTC")

    entry = sync_store._sync_entsoe_source(source, manifest, as_of, log)

    assert calls[0][0] == pd.Timestamp("2020-01-01T00:00:00+00:00", tz="UTC")
    assert entry is not None
    assert entry.count == 10
    assert not log.any_failure
