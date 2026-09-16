"""Tests for scripts/sync_store.py's soft time budget (spec 6.7.3, section
2.1/5.2/7) -- reproduces the 2026-09-16 incident in miniature: a slow/dead
ENTSO-E must no longer be able to make the whole run miss
store.publish_store() entirely, discarding already-successful work (weather
included) along with it.

Clock is always injected (never real time) -- a fake clock's call count is
fully determined by the fixed loop structure under test (one call for
started_at, then one call per source per loop before that source's own
work), so each test can precisely place the budget trip at a known point
without waiting in real time.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from energy_price_forecast.ops import store
from scripts import sync_store


def _last_log_row(tmp_path: Path) -> dict[str, str]:
    """The one row _write_log_row wrote to the (monkeypatched) STORE_SYNC_LOG
    path, as a plain dict -- reads the real CSV back off disk rather than
    reaching into RunLog internals run_sync() never exposes to its caller."""
    frame = pd.read_csv(tmp_path / "store_sync.csv")
    row = frame.iloc[-1].to_dict()
    return {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}


def _manifest_with_entries(names: list[str]) -> store.Manifest:
    """A manifest where every named source already has a prior entry (so
    the heal loop's own "no covered_end_utc yet" skip never fires,
    independent of whatever the main loop did this run) -- count=1 is
    below any real fake fetch's row count below, so the real shrink-guard
    in _sync_entsoe_source never trips and masks what this test is
    actually about."""
    return store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2026-09-15T09:00:00+00:00",
        run_id="0",
        run_url="",
        code_sha="prev",
        sources={
            name: store.SourceManifestEntry(
                covered_start_utc="2026-01-01T00:00:00+00:00",
                covered_end_utc="2026-09-15T00:00:00+00:00",
                count=1,
                last_success_utc="2026-09-15T09:00:00+00:00",
                last_attempt_utc="2026-09-15T09:00:00+00:00",
            )
            for name in names
        },
    )


def _fake_source(name: str, tmp_path: Path, calls: list[str]) -> sync_store.EntsoeSource:
    """A fake source whose fetch fills exactly the real
    EXPECTATION_TABLE columns for ``name`` -- so the real (unmocked)
    validate_source() this drives genuinely passes, the same as it would
    against real data, rather than being bypassed."""
    columns = sorted(store.EXPECTATION_TABLE[name].expected_columns)

    def fetch(start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        calls.append(f"main:{name}")
        idx = pd.date_range("2026-09-15", periods=24, freq="h", tz="UTC")
        return pd.DataFrame({col: [1.0] * 24 for col in columns}, index=idx)

    return sync_store.EntsoeSource(name=name, fetch=fetch, cache_dir=tmp_path / name)


class _StepClock:
    """Returns 0.0 for the first ``free_calls`` calls (started_at plus
    however many budget checks should still pass), then a value past
    SYNC_SOFT_BUDGET_SECONDS forever after -- models "the budget runs out
    at exactly this point in the fixed loop sequence" without any real
    waiting."""

    def __init__(self, free_calls: int) -> None:
        self.free_calls = free_calls
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls <= self.free_calls:
            return 0.0
        return float(sync_store.SYNC_SOFT_BUDGET_SECONDS) + 1.0


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: store.Manifest,
    sources: list[sync_store.EntsoeSource],
    *,
    weather_calls: list[str],
) -> list[store.Manifest]:
    """Shared monkeypatching for all three tests: a fixed manifest to load,
    the fake ENTSO-E sources under test, no commodities (irrelevant to the
    budget guard, real yfinance calls would be pointless here), a
    call-counted weather stand-in, and a publish_store stand-in that
    records every manifest it was called with (so a test can assert on
    exactly what got published)."""
    monkeypatch.setattr(sync_store, "STORE_SYNC_LOG", tmp_path / "store_sync.csv")
    monkeypatch.setattr(sync_store, "ENTSOE_SOURCES", tuple(sources))
    monkeypatch.setattr(sync_store, "COMMODITY_SOURCES", ())
    monkeypatch.setattr(store, "load_store", lambda workdir: store.StoreState(workdir, manifest))

    def fake_sync_weather(
        manifest_: store.Manifest, as_of: pd.Timestamp, log: sync_store.RunLog
    ) -> store.SourceManifestEntry:
        weather_calls.append("weather")
        return store.SourceManifestEntry(
            covered_start_utc="2024-03-14T00:00:00+00:00",
            covered_end_utc=as_of.isoformat(),
            count=999,
            last_success_utc=as_of.isoformat(),
            last_attempt_utc=as_of.isoformat(),
        )

    monkeypatch.setattr(sync_store, "_sync_weather", fake_sync_weather)

    published: list[store.Manifest] = []

    def fake_publish_store(workdir: Path, manifest_: store.Manifest) -> SimpleNamespace:
        published.append(manifest_)
        return SimpleNamespace(size_bytes=1000)

    monkeypatch.setattr(store, "publish_store", fake_publish_store)
    return published


def test_budget_exhausted_is_a_pure_elapsed_time_comparison() -> None:
    """Direct test of the small pure helper itself, independent of any
    loop -- documents the exact boundary (>=, not >) and that it only
    depends on the two arguments, never real time."""
    assert (
        sync_store._budget_exhausted(0.0, lambda: sync_store.SYNC_SOFT_BUDGET_SECONDS - 1) is False
    )
    assert sync_store._budget_exhausted(0.0, lambda: sync_store.SYNC_SOFT_BUDGET_SECONDS) is True
    assert (
        sync_store._budget_exhausted(0.0, lambda: sync_store.SYNC_SOFT_BUDGET_SECONDS + 1) is True
    )
    # Only the elapsed delta matters, not the absolute clock value.
    started = 10_000.0
    assert sync_store._budget_exhausted(started, lambda: started + 1) is False


def test_sync_budget_normal_run_is_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a clock that never crosses the budget runs every source
    in both loops exactly as before this feature -- no skip, no warning,
    exit_status stays ok."""
    names = ["day_ahead_price", "load"]
    manifest = _manifest_with_entries(names)
    calls: list[str] = []
    weather_calls: list[str] = []
    sources = [_fake_source(n, tmp_path, calls) for n in names]
    published = _patch_common(monkeypatch, tmp_path, manifest, sources, weather_calls=weather_calls)

    clock: Callable[[], float] = lambda: 0.0  # noqa: E731 -- trivial, budget never exhausted
    exit_code = sync_store.run_sync(only=None, dry_run=False, clock=clock)

    assert exit_code == 0  # any_failure stayed False -- a budget skip always sets it
    # source.fetch is called from both the main loop and (existing_frame,
    # then a possible _refetch) the heal loop -- not asserting an exact
    # count, just that neither source was ever skipped outright.
    assert "main:day_ahead_price" in calls
    assert "main:load" in calls
    assert weather_calls == ["weather"]
    assert len(published) == 1
    assert "budget" not in _last_log_row(tmp_path)["warnings"]


def test_sync_budget_exhausted_in_heal_loop_still_publishes_weather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 (spec 6.7.3 section 7) -- reproduces the 2026-09-16 incident in
    miniature: the budget runs out only once the heal loop starts (both
    main-loop sources and weather already succeeded). Remaining heal steps
    must be skipped, a warning logged naming them, exit_status
    partial_failure -- but publish_store must still run, with weather's
    entry present in what gets published."""
    names = ["day_ahead_price", "load"]
    manifest = _manifest_with_entries(names)
    calls: list[str] = []
    weather_calls: list[str] = []
    sources = [_fake_source(n, tmp_path, calls) for n in names]
    published = _patch_common(monkeypatch, tmp_path, manifest, sources, weather_calls=weather_calls)

    # started_at (1) + one _budget_exhausted check per main-loop source (2)
    # all still free -- the heal loop's own first check is the one that
    # finally trips.
    clock = _StepClock(free_calls=1 + len(names))

    exit_code = sync_store.run_sync(only=None, dry_run=False, clock=clock)

    assert exit_code == 1  # partial_failure must redden the run
    assert calls == ["main:day_ahead_price", "main:load"]  # both main-loop fetches ran
    assert weather_calls == ["weather"]  # weather ran and succeeded, unaffected by the heal skip

    assert len(published) == 1, "publish_store must still run despite the budget skip"
    manifest_published = published[0]
    assert manifest_published.sources["weather_single_runs"].count == 999
    # The main loop ran fully for both (not budget-affected), so their
    # entries reflect the fresh 24-row fetch, not the prior manifest's
    # count=1 -- only the heal loop (healed_cells, not count) was skipped.
    for name in names:
        assert manifest_published.sources[name].count == 24

    row = _last_log_row(tmp_path)
    assert row["exit_status"] == "partial_failure"
    assert "heal loop" in row["warnings"]
    for name in names:
        assert name in row["warnings"]


def test_sync_budget_exhausted_in_main_loop_keeps_previous_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3 (spec 6.7.3 section 7) -- the budget is already gone before the
    main loop's very first source. Every source (main loop AND, as a
    consequence, the heal loop too) is skipped and keeps its previous
    manifest entry -- not treated as a validation failure, still
    published, still reddened for visibility."""
    names = ["day_ahead_price", "load"]
    manifest = _manifest_with_entries(names)
    calls: list[str] = []
    weather_calls: list[str] = []
    sources = [_fake_source(n, tmp_path, calls) for n in names]
    published = _patch_common(monkeypatch, tmp_path, manifest, sources, weather_calls=weather_calls)

    # started_at (1) is the only free call -- every _budget_exhausted check
    # from the very first main-loop source onward already trips.
    clock = _StepClock(free_calls=1)

    exit_code = sync_store.run_sync(only=None, dry_run=False, clock=clock)

    assert exit_code == 1
    assert calls == []  # no fetch was ever attempted
    assert weather_calls == ["weather"]  # weather sits outside the budget-guarded loops

    assert len(published) == 1, "publish_store must still run despite the budget skip"
    manifest_published = published[0]
    for name in names:
        assert manifest_published.sources[name].count == 1  # exactly the prior manifest value

    row = _last_log_row(tmp_path)
    assert row["exit_status"] == "partial_failure"
    assert "main fetch loop" in row["warnings"]
    for name in names:
        assert name in row["warnings"]  # untouched, not reverted
