"""Unit tests for data/weather_client.py -- the Open-Meteo Single Runs client.

Fixture-based tests use a real recorded response
(tests/fixtures/open_meteo_single_run.json, recorded 2026-08-31: 18 points,
9 variables, run=2024-06-01T00:00, 30 consecutive hourly timestamps) and
monkeypatch ``requests.get`` so nothing in this file except the tests marked
``integration`` touches the network.

Determinism and lead-hour-coverage tests (spec 6.5.1, §7, both explicitly
"netzwerkabhängig, lokal") are deferred to the run-derivation test step
(§10, step 11) -- this file's scope matches its own work-order commit
message: request shape, response parsing, every fail-fast branch, single
attempt.

Every fetch_run() call here passes ``use_cache=False``: most tests share the
same ``_RUN`` timestamp with different (some deliberately corrupted) mock
responses, and the real per-run cache is keyed on nothing but the run --
without this, the first passing test would write a cache entry that later
tests would then silently hit instead of exercising their own mock response.
The cache itself is tested in tests/test_weather_cache.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.data.weather_client import WeatherRunUnavailable, fetch_run
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES, expected_columns

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "open_meteo_single_run.json"
_RUN = pd.Timestamp("2024-06-01T00:00", tz="UTC")
_N_HOURS = 30  # length baked into the fixture
_NOT_JSON = object()


def _load_fixture() -> list[dict[str, Any]]:
    with _FIXTURE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


class _FakeResponse:
    """Just enough of requests.Response for fetch_run's own code paths."""

    def __init__(self, status_code: int, body: Any, text: str | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self) -> Any:
        if self._body is _NOT_JSON:
            raise json.JSONDecodeError("mock non-JSON body", self.text, 0)
        return self._body


def _install_fake_get(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> list[dict[str, Any]]:
    """Monkeypatch requests.get; returns the list of captured params dicts,
    one per call, so tests can assert on request shape and call count."""
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        calls.append(params)
        return response

    import energy_price_forecast.data.weather_client as client_module

    monkeypatch.setattr(client_module.requests, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_uses_all_grid_points_models_run_no_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_get(monkeypatch, _FakeResponse(200, _load_fixture()))

    fetch_run(_RUN, forecast_days=1, use_cache=False)

    assert len(calls) == 1
    params = calls[0]
    assert params["latitude"] == ",".join(str(p.latitude) for p in GRID_POINTS)
    assert params["longitude"] == ",".join(str(p.longitude) for p in GRID_POINTS)
    assert params["hourly"] == ",".join(HOURLY_VARIABLES)
    assert params["models"] == "ecmwf_ifs"
    assert params["run"] == "2024-06-01T00:00"
    assert "timezone" not in params


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_response_parsing_against_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    df = fetch_run(_RUN, forecast_days=1, use_cache=False)

    assert df.shape == (_N_HOURS, 162)
    assert list(df.columns) == list(expected_columns())
    assert all(dtype == np.dtype("float32") for dtype in df.dtypes)
    assert df.index.names == ["run_init_utc", "valid_time_utc"]

    run_values = df.index.get_level_values("run_init_utc")
    assert (run_values == _RUN).all()

    valid_times = df.index.get_level_values("valid_time_utc")
    assert isinstance(valid_times, pd.DatetimeIndex)
    assert valid_times.tz is not None
    assert str(valid_times.tz) == "UTC"

    # Point -> column mapping via coordinate match (spike 6.5.0, F5: response
    # objects are in request order, so index i of the fixture is GRID_POINTS[i]).
    point = GRID_POINTS[3]
    obj = fixture[3]
    assert abs(obj["latitude"] - point.latitude) < 0.15
    assert abs(obj["longitude"] - point.longitude) < 0.15
    column = f"{point.point_id}__temperature_2m"
    expected_values = np.array(obj["hourly"]["temperature_2m"], dtype="float32")
    np.testing.assert_array_equal(df[column].to_numpy(), expected_values)


def test_night_hours_stay_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    df = fetch_run(_RUN, forecast_days=1, use_cache=False)

    column = f"{GRID_POINTS[0].point_id}__shortwave_radiation"
    assert pd.isna(df[column].iloc[0])  # 2024-06-01T00:00 UTC, night


# ---------------------------------------------------------------------------
# Fail-fast checks, each against a deliberately corrupted fixture (spec §5.3)
# ---------------------------------------------------------------------------


def test_wrong_object_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    fixture.pop()  # 17 objects instead of 18
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    with pytest.raises(ValueError, match="18"):
        fetch_run(_RUN, forecast_days=1, use_cache=False)


def test_shifted_coordinate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    fixture[0]["latitude"] += 1.0  # far beyond the 0.15° tolerance
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    with pytest.raises(ValueError, match=GRID_POINTS[0].point_id):
        fetch_run(_RUN, forecast_days=1, use_cache=False)


def test_missing_variable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    del fixture[2]["hourly"]["cloud_cover"]
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    with pytest.raises(ValueError, match="cloud_cover"):
        fetch_run(_RUN, forecast_days=1, use_cache=False)


def test_too_short_time_series_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    for obj in fixture:
        obj["hourly"] = {key: values[:10] for key, values in obj["hourly"].items()}
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    with pytest.raises(ValueError, match="24"):
        fetch_run(_RUN, forecast_days=1, use_cache=False)


def test_gap_in_timestamps_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    # Drop hour index 5 from every array in every object -- 29 hours remain,
    # still >= the 24-hour minimum for forecast_days=1, so this exercises the
    # gap check specifically rather than tripping the length check first.
    for obj in fixture:
        for key in list(obj["hourly"]):
            del obj["hourly"][key][5]
    _install_fake_get(monkeypatch, _FakeResponse(200, fixture))

    with pytest.raises(ValueError, match="gap"):
        fetch_run(_RUN, forecast_days=1, use_cache=False)


def test_200_with_non_json_body_raises_weather_run_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live-measured multi-point failure mode, not covered by spike 6.5.0's
    F6 (which only tested single-point requests): some invalid runs return
    HTTP 200 with a non-JSON streaming error body instead of a clean 400.
    See docs/sprint6_step6_5_1_log.md for the live evidence and the
    owner-confirmed decision to treat this the same as an explicit 400.
    """
    response = _FakeResponse(
        200,
        _NOT_JSON,
        text="Unexpected error while streaming data: modelRunUnavailable(...)",
    )
    calls = _install_fake_get(monkeypatch, response)

    with pytest.raises(WeatherRunUnavailable) as exc_info:
        fetch_run(_RUN, forecast_days=1, use_cache=False)

    assert len(calls) == 1
    assert exc_info.value.http_status == 200


# ---------------------------------------------------------------------------
# HTTP 400 -- WeatherRunUnavailable, one attempt, no substitute run (§2.3, §2.4)
# ---------------------------------------------------------------------------


def test_http_400_raises_weather_run_unavailable_with_run_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        400,
        {
            "error": True,
            "reason": "The requested model run is not available. "
            "Model: ecmwf_ifs, run: 2024-06-01T00:00Z",
        },
    )
    _install_fake_get(monkeypatch, response)

    with pytest.raises(WeatherRunUnavailable) as exc_info:
        fetch_run(_RUN, forecast_days=1, use_cache=False)

    assert "2024-06-01" in str(exc_info.value)


def test_single_attempt_no_retry_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(400, {"error": True, "reason": "not available"})
    calls = _install_fake_get(monkeypatch, response)

    with pytest.raises(WeatherRunUnavailable):
        fetch_run(_RUN, forecast_days=1, use_cache=False)

    assert len(calls) == 1


def test_no_substitute_run_requested_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(400, {"error": True, "reason": "not available"})
    calls = _install_fake_get(monkeypatch, response)

    with pytest.raises(WeatherRunUnavailable):
        fetch_run(_RUN, forecast_days=1, use_cache=False)

    requested_runs = {c["run"] for c in calls}
    assert requested_runs == {"2024-06-01T00:00"}


# ---------------------------------------------------------------------------
# run_init_utc input contract (not one of the six response-validation checks;
# a pre-flight guard on the caller-supplied argument, §5.3 docstring)
# ---------------------------------------------------------------------------


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        fetch_run(pd.Timestamp("2024-06-01T00:00"))


def test_invalid_run_hour_rejected() -> None:
    with pytest.raises(ValueError, match="0, 6, 12, 18"):
        fetch_run(pd.Timestamp("2024-06-01T03:00", tz="UTC"))
