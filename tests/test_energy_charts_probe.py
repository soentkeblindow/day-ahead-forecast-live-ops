"""Unit tests for data/energy_charts_probe.py and its thin wiring script
scripts/probe_energy_charts_forecast.py (docs/sprint6_auftrag_energy_charts_backup.md).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from energy_price_forecast.data.energy_charts_probe import (
    SERIES,
    EnergyChartsProbeAnomalyError,
    append_probe_row,
    probe_series,
)
from energy_price_forecast.ops.windows import local_day_bounds, next_delivery_day
from scripts.probe_energy_charts_forecast import run_probe

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


def _full_payload(
    target_day: dt.date, production_type: str, *, forecast_type: str = "day-ahead"
) -> dict[str, Any]:
    start, end = local_day_bounds(target_day)
    index = pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left"
    )
    return {
        "unix_seconds": [int(ts.timestamp()) for ts in index],
        "forecast_values": [1.0 + i for i in range(len(index))],
        "production_type": production_type,
        "forecast_type": forecast_type,
        "deprecated": False,
    }


def _install_fake_get(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        calls.append(params)
        return response

    import energy_price_forecast.data.energy_charts_probe as module

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


def _install_fake_get_sequence(
    monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    remaining = list(responses)

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        calls.append(params)
        return remaining.pop(0)

    import energy_price_forecast.data.energy_charts_probe as module

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# probe_series: request shape, coverage, DST
# ---------------------------------------------------------------------------


def test_full_coverage_response_gives_coverage_ratio_one(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    _install_fake_get(monkeypatch, _FakeResponse(200, _full_payload(target_day, "load")))

    row = probe_series("load", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["n_values"] == 96
    assert row["coverage_ratio"] == 1.0
    assert row["error_kind"] == ""
    assert row["http_status"] == "200"
    assert row["echoed_production_type"] == "load"


@pytest.mark.parametrize(
    ("target_day", "expected_slots"),
    [
        (dt.date(2026, 3, 29), 92),  # spring forward, 23h day
        (dt.date(2026, 10, 25), 100),  # fall back, 25h day
    ],
)
def test_coverage_ratio_uses_dst_aware_expected_slot_count(
    monkeypatch: pytest.MonkeyPatch, target_day: dt.date, expected_slots: int
) -> None:
    _install_fake_get(monkeypatch, _FakeResponse(200, _full_payload(target_day, "solar")))

    row = probe_series("solar", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["n_values"] == expected_slots
    assert row["coverage_ratio"] == 1.0


def test_partial_response_is_logged_as_partial_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response covering 40 of 96 slots must not be reported as a full
    success (spec section 3: 'Eine Antwort mit 40 von 96 Slots ist kein
    Erfolg')."""
    target_day = dt.date(2026, 9, 17)
    payload = _full_payload(target_day, "wind_onshore")
    payload["unix_seconds"] = payload["unix_seconds"][:40]
    payload["forecast_values"] = payload["forecast_values"][:40]
    _install_fake_get(monkeypatch, _FakeResponse(200, payload))

    row = probe_series("wind_onshore", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["n_values"] == 40
    # coverage_ratio is rounded to 4 decimals on write (clean CSV output),
    # so compare against that same precision rather than the raw ratio.
    assert row["coverage_ratio"] == pytest.approx(40 / 96, abs=1e-4)
    assert row["error_kind"] == "partial_coverage"


def test_null_values_within_response_are_excluded_from_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_day = dt.date(2026, 9, 17)
    payload = _full_payload(target_day, "wind_offshore")
    payload["forecast_values"][10] = None
    payload["forecast_values"][11] = None
    _install_fake_get(monkeypatch, _FakeResponse(200, payload))

    row = probe_series(
        "wind_offshore", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC")
    )

    assert row["n_values"] == 94
    assert row["error_kind"] == "partial_coverage"


# ---------------------------------------------------------------------------
# Red vs. green split (spec section 5)
# ---------------------------------------------------------------------------


def test_mismatched_echoed_production_type_raises_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    payload = _full_payload(target_day, "load")
    payload["production_type"] = "solar"  # echoes back something else than requested
    _install_fake_get(monkeypatch, _FakeResponse(200, payload))

    with pytest.raises(EnergyChartsProbeAnomalyError):
        probe_series("load", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))


def test_mismatched_echoed_forecast_type_raises_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    payload = _full_payload(target_day, "load", forecast_type="intraday")
    _install_fake_get(monkeypatch, _FakeResponse(200, payload))

    with pytest.raises(EnergyChartsProbeAnomalyError):
        probe_series("load", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))


def test_deprecated_true_raises_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    payload = _full_payload(target_day, "load")
    payload["deprecated"] = True
    _install_fake_get(monkeypatch, _FakeResponse(200, payload))

    with pytest.raises(EnergyChartsProbeAnomalyError):
        probe_series("load", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))


def test_not_yet_published_404_is_green_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    _install_fake_get(monkeypatch, _FakeResponse(404, {"detail": "no content available"}))

    row = probe_series("solar", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["error_kind"] == "not_available"
    assert row["http_status"] == "404"
    assert row["n_values"] == 0


def test_rate_limited_429_is_green_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    _install_fake_get(monkeypatch, _FakeResponse(429, {}))

    row = probe_series("solar", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["error_kind"] == "rate_limited"
    assert row["http_status"] == "429"


def test_server_error_is_green_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    target_day = dt.date(2026, 9, 17)
    _install_fake_get(monkeypatch, _FakeResponse(503, {}))

    row = probe_series("solar", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    assert row["error_kind"] == "server_error"
    assert row["http_status"] == "503"


# ---------------------------------------------------------------------------
# CSV header migration (the 2026-09-11 store_sync.csv incident, not repeated)
# ---------------------------------------------------------------------------


def test_append_probe_row_writes_header_on_first_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_day = dt.date(2026, 9, 17)
    _install_fake_get(monkeypatch, _FakeResponse(200, _full_payload(target_day, "load")))
    row = probe_series("load", target_day, run_id=_RUN_ID, as_of=pd.Timestamp.now(tz="UTC"))

    log_path = tmp_path / "probe.csv"
    append_probe_row(row, log_path)

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("probe_timestamp_utc,run_id,target_day")


def test_append_probe_row_migrates_header_on_new_column(tmp_path: Path) -> None:
    log_path = tmp_path / "probe.csv"
    append_probe_row({"production_type": "load", "n_values": 96}, log_path)
    append_probe_row(
        {"production_type": "solar", "n_values": 0, "a_future_column": "new"}, log_path
    )

    df = pd.read_csv(log_path)
    assert "a_future_column" in df.columns
    assert len(df) == 2
    assert df.loc[0, "production_type"] == "load"
    # pd.isna(df.loc[0, "a_future_column"]) triggers a spurious mypy
    # "Statement is unreachable" on the NEXT line, reproduced in isolation --
    # a pandas-stubs 3.0.5 vs. pandas 2.3.3 version-mismatch quirk, not a
    # real code issue. df["col"].isna().iloc[0] checks the identical value
    # without it.
    assert df["a_future_column"].isna().iloc[0]
    assert df.loc[1, "a_future_column"] == "new"


# ---------------------------------------------------------------------------
# run_probe: target day comes from next_delivery_day, never "today"
# ---------------------------------------------------------------------------


def test_run_probe_targets_next_delivery_day_not_today(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression-shaped negative control, same discipline as the weather-
    run-offset fix (docs/sprint6_fix_weather_run_offset.md): the probe must
    measure the day tomorrow's submission needs, not today's own delivery
    day. Fails if run_probe ever regresses to a locally-derived 'today'."""
    as_of = pd.Timestamp("2026-09-15T09:00", tz="Europe/Berlin")
    expected_target_day = next_delivery_day(as_of)
    assert expected_target_day != as_of.tz_convert("UTC").date()

    responses = [_FakeResponse(200, _full_payload(expected_target_day, s)) for s in SERIES]
    calls = _install_fake_get_sequence(monkeypatch, responses)

    sleeps: list[float] = []
    rows = run_probe(as_of, log_path=tmp_path / "probe.csv", sleep=sleeps.append)

    assert len(calls) == len(SERIES)
    for call in calls:
        assert call["start"] == expected_target_day.isoformat()
        assert call["end"] == expected_target_day.isoformat()
    for row in rows:
        assert row["target_day"] == expected_target_day.isoformat()
    assert sleeps == [30.0] * (len(SERIES) - 1)


def test_run_probe_writes_one_row_per_series_before_a_later_anomaly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later series raising must not erase rows already written for
    earlier series (spec section 5, row 4: 'Sonde wirft' is red, but the
    log up to that point is real data, not to be lost)."""
    as_of = pd.Timestamp("2026-09-15T09:00", tz="Europe/Berlin")
    target_day = next_delivery_day(as_of)

    bad_payload = _full_payload(target_day, SERIES[1])
    bad_payload["deprecated"] = True
    responses = [
        _FakeResponse(200, _full_payload(target_day, SERIES[0])),
        _FakeResponse(200, bad_payload),
    ]
    _install_fake_get_sequence(monkeypatch, responses)

    log_path = tmp_path / "probe.csv"
    with pytest.raises(EnergyChartsProbeAnomalyError):
        run_probe(as_of, log_path=log_path, sleep=lambda _s: None)

    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert df.loc[0, "production_type"] == SERIES[0]
