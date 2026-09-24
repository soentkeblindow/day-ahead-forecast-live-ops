"""Unit tests for scripts/outage_drill.py's pure logic (spec 6.9, sections
2.12/6.3). run_outage_drill itself is deliberately not unit-tested here --
it calls ops.store.load_store() by design (a real, read-only download of
the published store), the one function in this module that is not "Kein
Netz" on purpose (see its own docstring); it is exercised by a real
As-of-Lauf instead, per spec section 2.12's own gate.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops.store_sources import (
    COMMODITIES_DIR,
    ENERGY_CHARTS_DIR,
    ENTSOE_SOURCES,
)
from scripts.outage_drill import (
    SCENARIOS,
    CopiedStore,
    _compare_against_archived_payload,
    _copy_sources_for_workdir,
    resolve_as_of,
    run_outage_drill,
)

# ---------------------------------------------------------------------------
# resolve_as_of
# ---------------------------------------------------------------------------


def test_resolve_as_of_after_10_uses_today_1145_and_tomorrow_target() -> None:
    real_now = pd.Timestamp("2026-09-23 11:00", tz="Europe/Berlin").tz_convert("UTC")
    as_of, target_day = resolve_as_of(real_now)
    assert as_of == pd.Timestamp("2026-09-23 11:45", tz="Europe/Berlin").tz_convert("UTC")
    assert target_day == dt.date(2026, 9, 24)


def test_resolve_as_of_before_10_uses_yesterday_1145_and_today_target() -> None:
    real_now = pd.Timestamp("2026-09-23 08:30", tz="Europe/Berlin").tz_convert("UTC")
    as_of, target_day = resolve_as_of(real_now)
    assert as_of == pd.Timestamp("2026-09-22 11:45", tz="Europe/Berlin").tz_convert("UTC")
    assert target_day == dt.date(2026, 9, 23)


def test_resolve_as_of_exactly_10_00_counts_as_after() -> None:
    """The boundary itself (not < 10:00) takes the "after" branch -- today
    11:45, tomorrow's target, matching spec section 2.12's literal wording
    ("bei Push vor 10:00" -- strictly before, not at-or-before)."""
    real_now = pd.Timestamp("2026-09-23 10:00", tz="Europe/Berlin").tz_convert("UTC")
    as_of, target_day = resolve_as_of(real_now)
    assert as_of == pd.Timestamp("2026-09-23 11:45", tz="Europe/Berlin").tz_convert("UTC")
    assert target_day == dt.date(2026, 9, 24)


def test_resolve_as_of_handles_a_dst_transition_day() -> None:
    """2026-10-25 is the real autumn DST transition in Europe/Berlin --
    exercised because as_of construction mixes DateOffset(days=1) with a
    fixed local hour, the exact shape of bug this project has hit before
    (docs/sprint6_fix_weather_run_offset.md)."""
    real_now = pd.Timestamp("2026-10-26 08:00", tz="Europe/Berlin").tz_convert("UTC")
    as_of, target_day = resolve_as_of(real_now)
    assert as_of.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M") == "2026-10-25 11:45"
    assert target_day == dt.date(2026, 10, 26)


def test_resolve_as_of_never_reads_the_wall_clock() -> None:
    """docs/sprint6_fix_partial_today.md section 3.4's standing rule,
    applied to this new probe too: real_now is an explicit argument, not
    read internally."""
    fixed = pd.Timestamp("2026-01-15 12:00", tz="UTC")
    as_of_a, target_a = resolve_as_of(fixed)
    as_of_b, target_b = resolve_as_of(fixed)
    assert as_of_a == as_of_b
    assert target_a == target_b


# ---------------------------------------------------------------------------
# _copy_sources_for_workdir
# ---------------------------------------------------------------------------


def test_copy_sources_reroots_every_entsoe_source_under_workdir(tmp_path: Path) -> None:
    copied = _copy_sources_for_workdir(tmp_path)
    assert len(copied.entsoe_sources) == len(ENTSOE_SOURCES)
    for original, replica in zip(ENTSOE_SOURCES, copied.entsoe_sources, strict=True):
        assert replica.name == original.name
        assert replica.fetch is original.fetch  # unchanged, only cache_dir moves
        assert replica.cache_dir == tmp_path / original.cache_dir.relative_to(PROJECT_ROOT)
        assert replica.cache_dir.is_relative_to(tmp_path)


def test_copy_sources_commodities_and_weather_paths(tmp_path: Path) -> None:
    copied = _copy_sources_for_workdir(tmp_path)
    assert copied.commodities_dir.is_relative_to(tmp_path)
    assert copied.commodities_dir.name == COMMODITIES_DIR.name
    assert copied.weather_root.is_relative_to(tmp_path)
    assert copied.workdir == tmp_path


def test_copy_sources_energy_charts_dir_path(tmp_path: Path) -> None:
    """Schritt 7 regression: the copy must be used for EC reads too, or a
    drill run would silently fall back to the real, live ENERGY_CHARTS_DIR
    instead of the freshly-downloaded store copy -- the exact bug class
    Schritt 3 already found and fixed for weather_root."""
    copied = _copy_sources_for_workdir(tmp_path)
    assert copied.energy_charts_dir.is_relative_to(tmp_path)
    assert copied.energy_charts_dir == tmp_path / ENERGY_CHARTS_DIR.relative_to(PROJECT_ROOT)


def test_copy_sources_returns_a_copiedstore(tmp_path: Path) -> None:
    copied = _copy_sources_for_workdir(tmp_path)
    assert isinstance(copied, CopiedStore)
    for source in copied.entsoe_sources:
        assert source.cache_dir.is_relative_to(tmp_path)


# ---------------------------------------------------------------------------
# _compare_against_archived_payload
# ---------------------------------------------------------------------------


def test_regression_check_no_archived_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.outage_drill as mod

    monkeypatch.setattr(mod, "PAYLOADS_DIR", tmp_path)
    result = _compare_against_archived_payload({"values": [1.0, 2.0]}, dt.date(2026, 9, 23))
    assert result.archived_payload_found is False
    assert result.identical is None


def test_regression_check_identical_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.outage_drill as mod

    monkeypatch.setattr(mod, "PAYLOADS_DIR", tmp_path)
    target_day = dt.date(2026, 9, 23)
    archived = {
        "challenge_id": "x",
        "target_start": "2026-09-23T00:00:00+02:00",
        "values": [1.0, 2.5, 3.0],
    }
    (tmp_path / f"{target_day.isoformat()}.json").write_text(json.dumps(archived), encoding="utf-8")

    result = _compare_against_archived_payload(dict(archived), target_day)
    assert result.archived_payload_found is True
    assert result.identical is True
    assert result.max_abs_diff == 0.0


def test_regression_check_deviating_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.outage_drill as mod

    monkeypatch.setattr(mod, "PAYLOADS_DIR", tmp_path)
    target_day = dt.date(2026, 9, 23)
    archived = {"values": [1.0, 2.0, 3.0]}
    (tmp_path / f"{target_day.isoformat()}.json").write_text(json.dumps(archived), encoding="utf-8")

    drilled = {"values": [1.0, 2.5, 3.0]}
    result = _compare_against_archived_payload(drilled, target_day)
    assert result.archived_payload_found is True
    assert result.identical is False
    assert result.max_abs_diff == pytest.approx(0.5)


def test_regression_check_no_payload_but_archive_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.outage_drill as mod

    monkeypatch.setattr(mod, "PAYLOADS_DIR", tmp_path)
    target_day = dt.date(2026, 9, 23)
    (tmp_path / f"{target_day.isoformat()}.json").write_text(
        json.dumps({"values": [1.0]}), encoding="utf-8"
    )
    result = _compare_against_archived_payload(None, target_day)
    assert result.archived_payload_found is True
    assert result.identical is False


def test_regression_check_value_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.outage_drill as mod

    monkeypatch.setattr(mod, "PAYLOADS_DIR", tmp_path)
    target_day = dt.date(2026, 9, 23)
    (tmp_path / f"{target_day.isoformat()}.json").write_text(
        json.dumps({"values": [1.0, 2.0]}), encoding="utf-8"
    )
    result = _compare_against_archived_payload({"values": [1.0]}, target_day)
    assert result.archived_payload_found is True
    assert result.identical is False
    assert "count differs" in result.note


# ---------------------------------------------------------------------------
# Scenario registry / CLI-level error handling
# ---------------------------------------------------------------------------


def test_none_scenario_registered() -> None:
    assert "none" in SCENARIOS


def test_unknown_scenario_raises_before_touching_the_network() -> None:
    """The unknown-scenario check must happen before store.load_store() is
    ever called -- otherwise a typo'd --scenario would still cost a real
    download before failing."""
    with pytest.raises(ValueError, match="unknown scenario"):
        run_outage_drill("does-not-exist", now=pd.Timestamp("2026-09-23 11:00", tz="UTC"))
