"""Tests for scripts/sync_store.py's --mode all|relevant-only (2026-10-01,
docs/bugs_in_live_system.md entry 6): a near-miss on maintain_store.yml's
hard timeout, root-caused to a live API cache-miss cascade for
cross_border_flows -- a source fully carried and read by no live code (see
SourceExpectation.relevant_only_eligible). relevant-only mode skips it (and
the other two eligible sources) entirely during the gate-closure-adjacent
window; mode=all (the default) must remain byte-for-byte the prior
behavior.

Reuses test_sync_store_budget.py's own helper shapes rather than importing
them -- each sync_store test file is self-contained (no shared conftest
fixtures for this module), same pattern already established there.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from energy_price_forecast.ops import store
from scripts import sync_store


def _last_log_row(tmp_path: Path) -> dict[str, str]:
    frame = pd.read_csv(tmp_path / "store_sync.csv")
    row = frame.iloc[-1].to_dict()
    return {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}


def _manifest_with_entries(names: list[str]) -> store.Manifest:
    return store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc="2026-10-01T09:00:00+00:00",
        run_id="0",
        run_url="",
        code_sha="prev",
        sources={
            name: store.SourceManifestEntry(
                covered_start_utc="2026-01-01T00:00:00+00:00",
                covered_end_utc="2026-09-30T00:00:00+00:00",
                count=1,
                last_success_utc="2026-09-30T09:00:00+00:00",
                last_attempt_utc="2026-09-30T09:00:00+00:00",
            )
            for name in names
        },
    )


def _working_fake_source(name: str, tmp_path: Path, calls: list[str]) -> sync_store.EntsoeSource:
    """A normal, succeeding fake source -- same shape as
    test_sync_store_budget.py's own ``_fake_source``."""
    columns = sorted(store.EXPECTATION_TABLE[name].expected_columns)

    def fetch(start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        calls.append(f"main:{name}")
        idx = pd.date_range("2026-09-30", periods=24, freq="h", tz="UTC")
        return pd.DataFrame({col: [1.0] * 24 for col in columns}, index=idx)

    return sync_store.EntsoeSource(name=name, fetch=fetch, cache_dir=tmp_path / name)


def _must_not_be_called_source(name: str, tmp_path: Path) -> sync_store.EntsoeSource:
    """A fake source whose fetch raises if it is ever invoked -- the
    strongest possible proof that relevant-only mode's skip happens before
    any fetch attempt, not just that the eventual result is discarded."""

    def fetch(start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        raise AssertionError(f"{name!r} must not be fetched in --mode relevant-only")

    return sync_store.EntsoeSource(name=name, fetch=fetch, cache_dir=tmp_path / name)


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: store.Manifest,
    sources: list[sync_store.EntsoeSource],
) -> list[store.Manifest]:
    monkeypatch.setattr(sync_store, "STORE_SYNC_LOG", tmp_path / "store_sync.csv")
    monkeypatch.setattr(sync_store, "ENTSOE_SOURCES", tuple(sources))
    monkeypatch.setattr(sync_store, "COMMODITY_SOURCES", ())
    monkeypatch.setattr(store, "load_store", lambda workdir: store.StoreState(workdir, manifest))

    def fake_sync_weather(
        manifest_: store.Manifest, as_of: pd.Timestamp, log: sync_store.RunLog
    ) -> store.SourceManifestEntry:
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


# cross_border_flows is a real relevant_only_eligible source (see
# EXPECTATION_TABLE); day_ahead_price is a real non-eligible (checked) one
# -- using the real table rather than a monkeypatched one means this test
# would break, correctly, if a future change ever flipped either source's
# eligibility without updating this test's own assumption.
_ELIGIBLE = "cross_border_flows"
_NOT_ELIGIBLE = "day_ahead_price"


def test_relevant_only_mode_skips_eligible_source_in_both_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert store.EXPECTATION_TABLE[_ELIGIBLE].relevant_only_eligible is True
    assert store.EXPECTATION_TABLE[_NOT_ELIGIBLE].relevant_only_eligible is False

    manifest = _manifest_with_entries([_ELIGIBLE, _NOT_ELIGIBLE])
    calls: list[str] = []
    sources = [
        _must_not_be_called_source(_ELIGIBLE, tmp_path),
        _working_fake_source(_NOT_ELIGIBLE, tmp_path, calls),
    ]
    published = _patch_common(monkeypatch, tmp_path, manifest, sources)

    exit_code = sync_store.run_sync(only=None, dry_run=False, mode="relevant-only")

    # Not fetched (the fake would have raised) and not failed -- the
    # eligible source's previous manifest entry survives unchanged, exactly
    # like a budget-exhaustion skip's "previous state survives" semantics,
    # but without ever touching the API.
    assert exit_code == 0
    assert "main:day_ahead_price" in calls

    assert len(published) == 1
    published_entry = published[0].sources[_ELIGIBLE]
    assert published_entry.count == 1  # untouched prior manifest value

    row = _last_log_row(tmp_path)
    assert row["sync_mode"] == "relevant-only"
    assert row["exit_status"] == "ok"
    # The point of entry 6's fix: a deliberate skip must never look like
    # degradation -- no budget warning, no heal warning, nothing naming the
    # skipped source in the warnings column at all.
    assert row["warnings"] == ""
    assert f"{_ELIGIBLE}_validation" in row
    assert row[f"{_ELIGIBLE}_validation"] == sync_store._RELEVANT_ONLY_SKIP_VALIDATION


def test_mode_all_still_fetches_relevant_only_eligible_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: --mode all (the default, and the only behavior that
    existed before this feature) must fetch every source exactly as
    before, including ones relevant-only would skip."""
    manifest = _manifest_with_entries([_ELIGIBLE, _NOT_ELIGIBLE])
    calls: list[str] = []
    sources = [
        _working_fake_source(_ELIGIBLE, tmp_path, calls),
        _working_fake_source(_NOT_ELIGIBLE, tmp_path, calls),
    ]
    published = _patch_common(monkeypatch, tmp_path, manifest, sources)

    # mode omitted entirely -- proves the default itself, not just an
    # explicit mode="all" call site.
    exit_code = sync_store.run_sync(only=None, dry_run=False)

    assert exit_code == 0
    assert f"main:{_ELIGIBLE}" in calls
    assert f"main:{_NOT_ELIGIBLE}" in calls

    published_entry = published[0].sources[_ELIGIBLE]
    assert published_entry.count == 24  # the fresh fetch's row count, not the stale count=1

    row = _last_log_row(tmp_path)
    assert row["sync_mode"] == "all"
    assert row[f"{_ELIGIBLE}_validation"] == "ok"


def test_relevant_only_skip_is_a_pure_function_of_mode_and_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct test of the small helper, independent of any loop --
    monkeypatches EXPECTATION_TABLE itself so the commodity loop's own skip
    branch (currently dead in practice, since no real commodity source is
    eligible -- see EXPECTATION_TABLE["eua_co2"]'s own comment) is provably
    driven by the same generic field, not special-cased per loop."""
    fake_table = dict(store.EXPECTATION_TABLE)
    fake_table["ttf_gas"] = store.SourceExpectation(
        expected_columns=frozenset(), relevant_only_eligible=True
    )
    monkeypatch.setattr(store, "EXPECTATION_TABLE", fake_table)

    assert sync_store._relevant_only_skip("ttf_gas", "relevant-only") is True
    assert sync_store._relevant_only_skip("ttf_gas", "all") is False
    assert sync_store._relevant_only_skip(_NOT_ELIGIBLE, "relevant-only") is False
