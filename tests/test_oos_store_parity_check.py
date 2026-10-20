"""Unit tests for scripts/oos_store_parity_check.py (spec 7.1, section
2.4) -- synthetic submission records/payloads/OOS-store rows only, no
real store or network access.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from energy_price_forecast.ops.oos_store import OosStoreRow, rows_to_frame
from scripts.oos_store_parity_check import (
    HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH,
    HALTEPUNKT_MIN_OFFENDING_DAYS,
    PARITY_CHECK_START_DAY,
    compare_day,
    real_submitted_rows,
    run_parity_check,
)

_START = PARITY_CHECK_START_DAY


def _record(target_day: dt.date, candidate: str, *, submitted: bool = True) -> dict[str, object]:
    return {
        "target_day": target_day.isoformat(),
        "candidate_selected": candidate,
        "submitted": submitted,
        "submission_mode": "live",
    }


def _qh_index(target_day: dt.date) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_day, tz="Europe/Berlin")
    end = pd.Timestamp(target_day + dt.timedelta(days=1), tz="Europe/Berlin")
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


def _oos_rows(target_day: dt.date, ladder_row: str, values: list[float]) -> list[OosStoreRow]:
    index = _qh_index(target_day)
    now = pd.Timestamp("2026-10-01T00:00:00", tz="UTC")
    return [
        OosStoreRow(
            target_day=target_day,
            timestamp_utc=ts,
            ladder_row=ladder_row,
            forecast=value,
            model_version="price_v1",
            weather_run_init_utc=None,
            source="backfill",
            created_at_utc=now,
        )
        for ts, value in zip(index, values, strict=True)
    ]


def _payload(target_day: dt.date, values: list[float]) -> dict[str, object]:
    return {
        "challenge_id": "2",
        "target_start": pd.Timestamp(target_day, tz="Europe/Berlin").isoformat(),
        "values": values,
    }


def test_real_submitted_rows_keeps_only_submitted_live_rows_on_or_after_start_day() -> None:
    before = _START - dt.timedelta(days=1)
    records = [
        _record(before, "core_gas"),  # too early -- ladder wasn't live yet
        _record(_START, "core_gas"),
        _record(_START + dt.timedelta(days=1), "base", submitted=False),  # never submitted
        {
            "target_day": (_START + dt.timedelta(days=2)).isoformat(),
            "candidate_selected": "base",
            "submitted": True,
            "submission_mode": "smoke",
        },  # smoke, not live
    ]

    result = real_submitted_rows(records, start_day=_START)

    assert result == {_START: "core_gas"}


def test_real_submitted_rows_keeps_the_last_accepted_record_for_a_day() -> None:
    records = [
        _record(_START, "core_gas"),
        _record(_START, "core_gas_loadpatch"),  # a later, same-day slot accepted a different row
    ]

    result = real_submitted_rows(records, start_day=_START)

    assert result == {_START: "core_gas_loadpatch"}


def test_compare_day_reports_zero_mean_abs_diff_on_an_exact_match() -> None:
    day = _START
    values = [50.0 + i * 0.1 for i in range(96)]
    oos_store = rows_to_frame(_oos_rows(day, "core_gas", values))
    payload = _payload(day, values)

    result = compare_day(oos_store, payload, day, "core_gas")

    assert result.n_matched_timestamps == 96
    assert result.mean_abs_diff == 0.0
    assert result.max_abs_diff == 0.0
    assert result.note == ""


def test_compare_day_reports_a_real_discrepancy() -> None:
    day = _START
    store_values = [50.0] * 96
    payload_values = [52.0] * 96  # every quarter-hour off by 2.0
    oos_store = rows_to_frame(_oos_rows(day, "core_gas", store_values))
    payload = _payload(day, payload_values)

    result = compare_day(oos_store, payload, day, "core_gas")

    assert result.mean_abs_diff == 2.0
    assert result.max_abs_diff == 2.0


def test_compare_day_reports_a_note_when_no_oos_store_row_exists() -> None:
    day = _START
    oos_store = rows_to_frame([])
    payload = _payload(day, [50.0] * 96)

    result = compare_day(oos_store, payload, day, "core_gas")

    assert result.mean_abs_diff is None
    assert "no OOS-store row" in result.note


def test_compare_day_reports_a_note_on_a_wrong_value_count() -> None:
    day = _START
    oos_store = rows_to_frame(_oos_rows(day, "core_gas", [50.0] * 96))
    payload = _payload(day, [50.0] * 90)  # wrong count, e.g. a corrupted payload

    result = compare_day(oos_store, payload, day, "core_gas")

    assert result.mean_abs_diff is None
    assert "expected 96" in result.note


def test_run_parity_check_passes_on_exact_matches(tmp_path: Path) -> None:
    day = _START
    values = [50.0] * 96
    oos_store = rows_to_frame(_oos_rows(day, "core_gas", values))

    submissions_log = tmp_path / "submissions.jsonl"
    submissions_log.write_text(json.dumps(_record(day, "core_gas")) + "\n", encoding="utf-8")
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir()
    (payloads_dir / f"{day.isoformat()}.json").write_text(
        json.dumps(_payload(day, values)), encoding="utf-8"
    )

    results, haltepunkt_triggered = run_parity_check(
        submissions_log=submissions_log,
        payloads_dir=payloads_dir,
        oos_store=oos_store,
        start_day=_START,
    )

    assert len(results) == 1
    assert haltepunkt_triggered is False


def test_run_parity_check_triggers_the_haltepunkt_on_more_than_a_handful_of_bad_days(
    tmp_path: Path,
) -> None:
    n_bad_days = HALTEPUNKT_MIN_OFFENDING_DAYS + 1
    submissions_log = tmp_path / "submissions.jsonl"
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir()

    frames = []
    lines = []
    for i in range(n_bad_days):
        day = _START + dt.timedelta(days=i)
        store_values = [50.0] * 96
        payload_values = [50.0 + HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH + 1.0] * 96
        frames.extend(_oos_rows(day, "core_gas", store_values))
        lines.append(json.dumps(_record(day, "core_gas")))
        (payloads_dir / f"{day.isoformat()}.json").write_text(
            json.dumps(_payload(day, payload_values)), encoding="utf-8"
        )
    submissions_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    oos_store = rows_to_frame(frames)

    results, haltepunkt_triggered = run_parity_check(
        submissions_log=submissions_log,
        payloads_dir=payloads_dir,
        oos_store=oos_store,
        start_day=_START,
    )

    assert len(results) == n_bad_days
    assert haltepunkt_triggered is True


def test_run_parity_check_does_not_trigger_at_exactly_the_boundary_count(tmp_path: Path) -> None:
    n_bad_days = HALTEPUNKT_MIN_OFFENDING_DAYS  # exactly at the boundary, not over it
    submissions_log = tmp_path / "submissions.jsonl"
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir()

    frames = []
    lines = []
    for i in range(n_bad_days):
        day = _START + dt.timedelta(days=i)
        store_values = [50.0] * 96
        payload_values = [50.0 + HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH + 1.0] * 96
        frames.extend(_oos_rows(day, "core_gas", store_values))
        lines.append(json.dumps(_record(day, "core_gas")))
        (payloads_dir / f"{day.isoformat()}.json").write_text(
            json.dumps(_payload(day, payload_values)), encoding="utf-8"
        )
    submissions_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    oos_store = rows_to_frame(frames)

    _, haltepunkt_triggered = run_parity_check(
        submissions_log=submissions_log,
        payloads_dir=payloads_dir,
        oos_store=oos_store,
        start_day=_START,
    )

    assert haltepunkt_triggered is False


def test_run_parity_check_excludes_a_missing_payload_day_from_the_haltepunkt_count(
    tmp_path: Path,
) -> None:
    day = _START
    submissions_log = tmp_path / "submissions.jsonl"
    submissions_log.write_text(json.dumps(_record(day, "core_gas")) + "\n", encoding="utf-8")
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir()  # no payload file written at all for this day
    oos_store = rows_to_frame(_oos_rows(day, "core_gas", [50.0] * 96))

    results, haltepunkt_triggered = run_parity_check(
        submissions_log=submissions_log,
        payloads_dir=payloads_dir,
        oos_store=oos_store,
        start_day=_START,
    )

    assert len(results) == 1
    assert results[0].mean_abs_diff is None
    assert results[0].note == "no real payload file"
    assert haltepunkt_triggered is False
