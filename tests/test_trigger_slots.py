"""Spec 8.0a section 7: a test over SLOTS proving the new weather-run
arrival-time probe entries collide with no maintenance/submission minute,
and cover the UTC measurement window both before and after the 2026-10-25
DST change.

No TS test runner exists anywhere in this repo (ops/trigger/package.json
has no test script, no jest/vitest devDependency, and no prior SLOTS
change ever shipped with one -- see docs/sprint6_step6_9_log.md's step 13
narrative). Introducing one for this single check would be a new-tooling
decision bigger than the check itself, so this follows the same
regex-parsing-from-Python pattern tests/test_workflow_script_entrypoints.py
already established for *.yml workflow files, applied here to index.ts's
plain-literal SLOTS array instead. `npx tsc --noEmit` (run locally, not in
CI -- see ci.yml, which never touches ops/trigger/ at all) is what actually
proves the TypeScript itself is well-typed; this test proves the SCHEDULE
DATA is collision-free, which tsc cannot check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from energy_price_forecast.config import PROJECT_ROOT

INDEX_TS = PROJECT_ROOT / "ops" / "trigger" / "src" / "index.ts"

# One slot object per match, allowing exactly one level of nested braces
# (the `inputs: { ... }` sub-object) -- matches this file's own one-line-
# per-slot formatting.
_SLOT_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}")
_LOCAL_TIME = re.compile(r'localTime:\s*"(\d{2}):(\d{2})"')
_WORKFLOW = re.compile(r'workflow:\s*"([^"]+)"')
_TIME_ZONE = re.compile(r'timeZone:\s*"(\w+)"')


class _ParsedSlot(TypedDict):
    minute_of_day: int
    workflow: str
    time_zone: str | None


def _parse_slots() -> list[_ParsedSlot]:
    text = INDEX_TS.read_text(encoding="utf-8")
    start = text.index("const SLOTS: Slot[] = [")
    end = text.index("\n];", start)
    body = text[start:end]

    slots: list[_ParsedSlot] = []
    for match in _SLOT_OBJECT.finditer(body):
        chunk = match.group(0)
        local_match = _LOCAL_TIME.search(chunk)
        workflow_match = _WORKFLOW.search(chunk)
        if local_match is None or workflow_match is None:
            continue  # a stray brace pair that isn't a Slot literal
        hour, minute = int(local_match.group(1)), int(local_match.group(2))
        tz_match = _TIME_ZONE.search(chunk)
        slots.append(
            {
                "minute_of_day": hour * 60 + minute,
                "workflow": workflow_match.group(1),
                "time_zone": tz_match.group(1) if tz_match else None,
            }
        )
    return slots


def test_at_least_one_slot_is_discovered() -> None:
    """Canary against a regex that silently matches nothing."""
    slots = _parse_slots()
    assert len(slots) >= 10


def test_no_utc_probe_slot_collides_with_a_local_slot_in_either_dst_regime() -> None:
    slots = _parse_slots()
    berlin_slots = [s for s in slots if s["time_zone"] != "utc"]
    utc_slots = [s for s in slots if s["time_zone"] == "utc"]

    assert utc_slots, "expected at least one UTC-anchored slot (the 8.0a probe)"
    assert berlin_slots, "expected at least one Europe/Berlin local slot (maintenance/submission)"

    # CEST = UTC+2 (before 2026-10-25), CET = UTC+1 (after). Berlin local
    # minute-of-day minus the offset gives the slot's effective UTC
    # minute-of-day in each regime.
    forbidden_cest = {(int(s["minute_of_day"]) - 120) % 1440 for s in berlin_slots}
    forbidden_cet = {(int(s["minute_of_day"]) - 60) % 1440 for s in berlin_slots}
    forbidden = forbidden_cest | forbidden_cet

    utc_minutes = {int(s["minute_of_day"]) for s in utc_slots}
    collisions = utc_minutes & forbidden
    assert not collisions, (
        f"UTC probe slot(s) collide with a maintenance/submission minute: {collisions}"
    )


def test_utc_probe_slots_form_a_15_minute_grid_covering_04_to_13_utc() -> None:
    slots = _parse_slots()
    probe_minutes = sorted(
        int(s["minute_of_day"])
        for s in slots
        if s["time_zone"] == "utc" and s["workflow"] == "weather_run_arrival_probe.yml"
    )

    assert probe_minutes, "no UTC-anchored weather_run_arrival_probe.yml slots found"
    # Covers the union of both named pairs' 1-7h windows (spec section 4):
    # ICON-D2 03 UTC ~04:00-08:00, the four 06 UTC pairs ~07:00-13:00.
    assert probe_minutes[0] <= 4 * 60 + 20  # starts by ~04:20
    assert probe_minutes[-1] >= 12 * 60 + 35  # ends no earlier than ~12:35

    gaps = [b - a for a, b in zip(probe_minutes, probe_minutes[1:], strict=False)]
    assert all(gap == 15 for gap in gaps), f"expected a uniform 15-minute grid, got gaps {gaps}"


def test_utc_probe_slots_are_identical_before_and_after_the_dst_change() -> None:
    """A UTC-anchored slot's wall-clock meaning must not shift with DST --
    the whole reason for the "utc" flag over plain "localTime" here. Since
    these slots carry no date, "before and after 2026-10-25" coverage is
    structural (there is nothing time-zone-dependent left to evaluate), so
    this test instead pins down the mechanism that makes that true: the
    scheduler must compare these slots against a raw UTC clock read, never
    against the Europe/Berlin-converted one the other slots use.
    """
    text = INDEX_TS.read_text(encoding="utf-8")
    assert 'slot.timeZone === "utc"' in text
    assert "actualUtc" in text
    # The UTC comparison must not route through localHHMM (the Intl/
    # Europe/Berlin conversion) -- it has its own, timezone-free helper.
    assert "function utcHHMM" in text
    utc_fn = text[text.index("function utcHHMM") : text.index("function minutesSinceMidnight")]
    assert "Intl.DateTimeFormat" not in utc_fn


def test_weather_run_arrival_probe_workflow_has_its_own_concurrency_group() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "weather_run_arrival_probe.yml"
    maintain_path = PROJECT_ROOT / ".github" / "workflows" / "maintain_store.yml"

    def _group(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^\s*group:\s*(\S+)", text, re.MULTILINE)
        assert match is not None, f"no concurrency group found in {path}"
        return match.group(1)

    probe_group = _group(workflow_path)
    maintain_group = _group(maintain_path)
    assert probe_group != maintain_group
    assert probe_group != "maintain-store"
