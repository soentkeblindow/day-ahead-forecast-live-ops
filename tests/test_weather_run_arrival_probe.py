"""Unit tests for data/weather_run_arrival_probe.py and its thin wiring
script scripts/probe_weather_run_arrival.py (spec 8.0a).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from energy_price_forecast.data.weather_grid import HOURLY_VARIABLES
from energy_price_forecast.data.weather_run_arrival_probe import (
    _classify_rate_limit,
    already_fully_covered,
    probe_pair,
    run_probe,
)
from energy_price_forecast.ops.windows import next_delivery_day

_RUN_ID = "test-run"


class _FakeResponse:
    def __init__(self, status_code: int, body: Any = None, *, not_json: bool = False) -> None:
        self.status_code = status_code
        self._body = body
        self._not_json = not_json

    def json(self) -> Any:
        if self._not_json:
            raise ValueError("mock: not JSON")
        return self._body


def _install_fake_get_sequence(
    monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    remaining = list(responses)

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        calls.append(params)
        return remaining.pop(0)

    import energy_price_forecast.data.weather_run_arrival_probe as module

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


def _full_payload(
    run_init_utc: pd.Timestamp,
    *,
    forecast_days: int = 2,
    null_variable_hours: dict[str, set[int]] | None = None,
    omit_variables: list[str] | None = None,
) -> dict[str, Any]:
    """A response covering ``forecast_days`` full days from run_init_utc,
    all HOURLY_VARIABLES present and non-null unless overridden by
    ``null_variable_hours`` (variable -> set of hour-offsets to null) or
    entirely dropped via ``omit_variables`` (simulates a model that does
    not deliver that variable at all)."""
    n_hours = forecast_days * 24
    times = [
        (run_init_utc + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n_hours)
    ]
    hourly: dict[str, Any] = {"time": times}
    null_variable_hours = null_variable_hours or {}
    omit_variables = omit_variables or []
    for variable in HOURLY_VARIABLES:
        if variable in omit_variables:
            continue
        nulled = null_variable_hours.get(variable, set())
        hourly[variable] = [None if i in nulled else 1.0 + i for i in range(n_hours)]
    return {"hourly": hourly}


# ---------------------------------------------------------------------------
# _classify_rate_limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("Daily API request limit exceeded", "rate_limited_daily"),
        ("Hourly API request limit exceeded", "rate_limited_hourly"),
        ("something else entirely", "rate_limited_unknown"),
        (None, "rate_limited_unknown"),
    ],
)
def test_classify_rate_limit(reason: str | None, expected: str) -> None:
    assert _classify_rate_limit(reason) == expected


# ---------------------------------------------------------------------------
# probe_pair: coverage, DST, missing variables, night-null radiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_day", "expected_hours"),
    [
        (dt.date(2026, 3, 29), 23),  # spring-forward DST day
        (dt.date(2026, 6, 15), 24),  # ordinary day
        (dt.date(2025, 10, 26), 25),  # fall-back DST day
    ],
)
def test_n_hours_expected_for_dst_days(
    monkeypatch: pytest.MonkeyPatch, target_day: dt.date, expected_hours: int
) -> None:
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, _full_payload(run_init_utc))])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["n_hours_expected"] == expected_hours
    assert row["n_hours_covered"] == expected_hours
    assert row["coverage_ratio"] == 1.0
    assert row["error_kind"] == ""


def test_partial_response_is_logged_as_partial_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    # Null one ordinary (non-radiation) variable at hour-offset 30, which
    # falls inside target_day's own 24 hours (run+24 .. run+48).
    payload = _full_payload(run_init_utc, null_variable_hours={"temperature_2m": {30}})
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, payload)])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    n_covered = cast(int, row["n_hours_covered"])
    n_expected = cast(int, row["n_hours_expected"])
    coverage_ratio = cast(float, row["coverage_ratio"])
    assert n_covered == n_expected - 1
    assert coverage_ratio < 1.0
    assert row["error_kind"] == "partial_coverage"


def test_missing_variable_is_recorded_and_does_not_reduce_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    payload = _full_payload(run_init_utc, omit_variables=["cloud_cover_low"])
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, payload)])

    row = probe_pair(
        "icon_eu", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["variables_missing"] == "cloud_cover_low"
    assert row["n_hours_covered"] == row["n_hours_expected"]
    assert row["coverage_ratio"] == 1.0
    assert row["error_kind"] == ""


def test_night_null_radiation_does_not_reduce_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    # Null both radiation variables for every hour of target_day --
    # exactly the "whole night" shape this rule exists for.
    payload = _full_payload(
        run_init_utc,
        null_variable_hours={
            "shortwave_radiation": set(range(24, 48)),
            "direct_normal_irradiance": set(range(24, 48)),
        },
    )
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, payload)])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["n_hours_covered"] == row["n_hours_expected"]
    assert row["coverage_ratio"] == 1.0
    assert row["error_kind"] == ""
    assert row["variables_missing"] == ""


@pytest.mark.parametrize(
    ("status", "expected_error_kind"),
    [
        (400, "not_available"),
        (503, "server_error"),
    ],
)
def test_non_200_statuses_are_green_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, status: int, expected_error_kind: str
) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(status, {})])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["error_kind"] == expected_error_kind
    assert row["http_status"] == str(status)


def test_malformed_200_body_is_green_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, None, not_json=True)])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["error_kind"] == "malformed_200_body"


def test_rate_limited_429_records_classified_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    _install_fake_get_sequence(
        monkeypatch, [_FakeResponse(429, {"reason": "Daily API request limit exceeded"})]
    )

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["error_kind"] == "rate_limited_daily"
    assert row["http_status"] == "429"


# ---------------------------------------------------------------------------
# already_fully_covered
# ---------------------------------------------------------------------------


def test_already_fully_covered_false_when_log_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "does_not_exist.csv"
    run_init_utc = pd.Timestamp("2026-10-10T06:00", tz="UTC")
    assert (
        already_fully_covered("ecmwf_ifs", run_init_utc, dt.date(2026, 10, 11), log_path) is False
    )


def test_already_fully_covered_true_after_a_full_coverage_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 6, 15)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    log_path = tmp_path / "weather_run_arrival_probe.csv"
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, _full_payload(run_init_utc))])

    row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )
    pd.DataFrame([row]).to_csv(log_path, index=False)

    assert already_fully_covered("ecmwf_ifs", run_init_utc, target_day, log_path) is True
    # A different model at the same run/target_day must not match.
    assert already_fully_covered("icon_eu", run_init_utc, target_day, log_path) is False


# ---------------------------------------------------------------------------
# run_probe: target day from next_delivery_day, windowing, early exit, 429 abort
# ---------------------------------------------------------------------------


def test_run_probe_uses_next_delivery_day_not_today(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spec section 7: a test must fail if "today" were measured instead
    of next_delivery_day(as_of)."""
    as_of = pd.Timestamp("2026-10-10T08:30", tz="UTC")
    wrong_target_day = as_of.tz_convert("Europe/Berlin").date()
    right_target_day = next_delivery_day(as_of)
    assert right_target_day != wrong_target_day  # sanity: the two must actually differ here

    run_init_utc = pd.Timestamp(right_target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(
        hours=6
    )
    log_path = tmp_path / "weather_run_arrival_probe.csv"
    _install_fake_get_sequence(monkeypatch, [_FakeResponse(200, _full_payload(run_init_utc))])

    rows = run_probe(as_of, run_id=_RUN_ID, log_path=log_path, pairs=(("ecmwf_ifs", 6),))

    assert len(rows) == 1
    assert rows[0]["target_day"] == right_target_day.isoformat()
    assert rows[0]["target_day"] != wrong_target_day.isoformat()


def test_run_probe_skips_a_pair_outside_its_own_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 6, 16)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    # as_of only 10 minutes after run_init -- below the 60-minute floor.
    as_of = run_init_utc + pd.Timedelta(minutes=10)
    log_path = tmp_path / "weather_run_arrival_probe.csv"
    calls = _install_fake_get_sequence(monkeypatch, [])

    rows = run_probe(as_of, run_id=_RUN_ID, log_path=log_path, pairs=(("ecmwf_ifs", 6),))

    assert rows == []
    assert calls == []
    assert not log_path.exists()


def test_run_probe_early_exit_skips_already_covered_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 6, 16)
    run_init_utc = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=6)
    as_of = run_init_utc + pd.Timedelta(hours=2)
    log_path = tmp_path / "weather_run_arrival_probe.csv"

    prior_row = probe_pair(
        "ecmwf_ifs", run_init_utc, target_day, run_id=_RUN_ID, as_of=as_of - pd.Timedelta(hours=1)
    )
    # Build the "prior full coverage" row directly, bypassing the HTTP
    # layer, then seed the log with it.
    prior_row = dict(prior_row)
    prior_row["http_status"] = "200"
    prior_row["coverage_ratio"] = 1.0
    pd.DataFrame([prior_row]).to_csv(log_path, index=False)

    calls = _install_fake_get_sequence(monkeypatch, [])

    rows = run_probe(as_of, run_id=_RUN_ID, log_path=log_path, pairs=(("ecmwf_ifs", 6),))

    assert rows == []
    assert calls == []


def test_run_probe_all_pairs_covered_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 6, 16)
    run_day = target_day - dt.timedelta(days=1)
    log_path = tmp_path / "weather_run_arrival_probe.csv"

    pairs = (("ecmwf_ifs", 6), ("icon_eu", 6))
    as_of = pd.Timestamp(run_day, tz="UTC") + pd.Timedelta(hours=8)  # 2h after both 06 UTC runs

    seeded_rows = []
    for model, run_hour in pairs:
        run_init_utc = pd.Timestamp(run_day, tz="UTC") + pd.Timedelta(hours=run_hour)
        row = {
            "probe_timestamp_utc": as_of.isoformat(),
            "run_id": _RUN_ID,
            "target_day": target_day.isoformat(),
            "model": model,
            "run_init_utc": run_init_utc.isoformat(),
            "minutes_since_run_init": 120.0,
            "minutes_to_gate_closure": 999.0,
            "http_status": "200",
            "error_kind": "",
            "n_hours_expected": 24,
            "n_hours_covered": 24,
            "coverage_ratio": 1.0,
            "variables_missing": "",
            "first_ts": "",
            "last_ts": "",
        }
        seeded_rows.append(row)
    pd.DataFrame(seeded_rows).to_csv(log_path, index=False)
    before = log_path.read_text(encoding="utf-8")

    calls = _install_fake_get_sequence(monkeypatch, [])

    rows = run_probe(as_of, run_id=_RUN_ID, log_path=log_path, pairs=pairs)

    assert rows == []
    assert calls == []
    assert log_path.read_text(encoding="utf-8") == before


def test_run_probe_aborts_remaining_pairs_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 6, 16)
    run_day = target_day - dt.timedelta(days=1)
    log_path = tmp_path / "weather_run_arrival_probe.csv"
    pairs = (("ecmwf_ifs", 6), ("icon_eu", 6))
    as_of = pd.Timestamp(run_day, tz="UTC") + pd.Timedelta(hours=8)

    calls = _install_fake_get_sequence(
        monkeypatch, [_FakeResponse(429, {"reason": "Daily API request limit exceeded"})]
    )

    rows = run_probe(as_of, run_id=_RUN_ID, log_path=log_path, pairs=pairs)

    assert len(calls) == 1  # the second pair was never attempted
    assert len(rows) == 1
    assert rows[0]["error_kind"] == "rate_limited_daily"
