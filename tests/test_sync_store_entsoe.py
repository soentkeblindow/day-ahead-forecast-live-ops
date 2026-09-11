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
from unittest.mock import patch

import pandas as pd

from energy_price_forecast.data.entsoe_client import fetch_day_ahead_prices
from energy_price_forecast.ops import store
from scripts import sync_store

CLIENT_MODULE = "energy_price_forecast.data.entsoe_client"


def test_sync_entsoe_source_fetches_full_history_not_just_the_gap(tmp_path: Path) -> None:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    full_index = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")

    def fake_fetch(
        start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True
    ) -> pd.DataFrame:
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


def test_heal_refetch_with_use_cache_false_actually_persists_to_disk(tmp_path: Path) -> None:
    """Regression for 6.7.1's own section 2.1 finding, closed by 6.7.1a
    (spec section 5.4): before this fix, a heal refetch had no way to ask
    for use_cache=False at all -- data/entsoe_client.py's fetch_* functions
    took no such parameter -- so the only real caller (scripts/
    sync_store.py) compared a cached frame against itself, seconds apart,
    and filled nothing; separately, `healed` was computed but never
    written anywhere. Provably red on the unfixed code: calling
    fetch_day_ahead_prices(..., use_cache=False) below would raise
    TypeError on any version of entsoe_client.py before this spec, since
    the keyword did not exist.

    Drives the real components end to end -- entsoe_client.
    fetch_day_ahead_prices, _entsoe_cache.py's on-disk merge, and
    store.heal_recent -- with genuinely separate mock return values for the
    pre-heal cache write and the heal's own refetch, not the same frame
    compared against itself.
    """
    # 2026-09-20 + as_of 2026-09-25 (not 09-01): heal_recent's own window_
    # start (as_of - lookback_days) must stay inside the same calendar
    # month as the gap itself -- this mock, like the real ENTSO-E client,
    # returns the same series regardless of the (start, end) it's called
    # with, so a window spanning two months would silently mix August- and
    # September-labeled cache files with September-dated rows and produce
    # duplicate timestamps once cached_fetch concatenates both months'
    # chunks. Caught for real on the first run of this test (2026-09-11) --
    # a test-fixture bug, not a bug in the fix under test.
    with patch(f"{CLIENT_MODULE}.DATA_RAW", tmp_path), patch(f"{CLIENT_MODULE}._get_client") as gc:
        client = gc.return_value

        # Pre-heal on-disk state: one hour missing from the current month.
        idx_gappy = pd.date_range("2026-09-20", periods=24, freq="h", tz="UTC").delete(5)
        client.query_day_ahead_prices.return_value = pd.Series(50.0, index=idx_gappy)
        existing = fetch_day_ahead_prices(
            pd.Timestamp("2026-09-20", tz="UTC"), pd.Timestamp("2026-09-20T23:00", tz="UTC")
        )
        assert len(existing) == 23

        # ENTSO-E now genuinely has the missing hour too -- a distinct
        # return value from a real, cache-bypassing call, not the same
        # cached frame heal_recent would otherwise compare against itself.
        idx_full = pd.date_range("2026-09-20", periods=24, freq="h", tz="UTC")
        client.query_day_ahead_prices.return_value = pd.Series(50.0, index=idx_full)

        def refetch(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
            return fetch_day_ahead_prices(s, e, use_cache=False)

        healed, result = store.heal_recent(
            "day_ahead_price",
            existing,
            lookback_days=10,
            refetch=refetch,
            as_of=pd.Timestamp("2026-09-25T10:00", tz="UTC"),
        )

        on_disk = pd.read_parquet(
            tmp_path / "entsoe" / "day_ahead_prices" / "DE_LU_2026-09.parquet"
        )

    assert result.filled_cells > 0
    assert len(on_disk) == 24
    assert not on_disk["day_ahead_price"].isna().any()
    assert healed["day_ahead_price"].notna().all()
