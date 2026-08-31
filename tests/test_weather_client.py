"""Unit tests for data/weather_client.py -- the Open-Meteo Single Runs client.

Fixture-based tests use a real recorded response
(tests/fixtures/open_meteo_single_run.json, recorded 2026-08-31: 18 points,
9 variables, run=2024-06-01T00:00, 30 consecutive hourly timestamps) and
monkeypatch ``requests.get`` so nothing in this file except the tests marked
``integration`` touches the network.

Determinism and lead-hour-coverage tests (spec 6.5.1, §7, both explicitly
"netzwerkabhängig, lokal") were deferred from step 5's scope (request
shape, response parsing, every fail-fast branch, single attempt) and land
here in step 11 instead, alongside run_init_for_target_day and the
availability log -- both of the deferred tests are ``@pytest.mark.integration``
and therefore excluded from the default (and CI) run.

Every fetch_run() call here passes ``use_cache=False``: most tests share the
same ``_RUN`` timestamp with different (some deliberately corrupted) mock
responses, and the real per-run cache is keyed on nothing but the run --
without this, the first passing test would write a cache entry that later
tests would then silently hit instead of exercising their own mock response.
The cache itself is tested in tests/test_weather_cache.py.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests

from energy_price_forecast.data.weather_client import (
    WeatherRunUnavailable,
    fetch_run,
    log_availability_attempt,
    run_init_for_target_day,
)
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES, expected_columns
from energy_price_forecast.ops.windows import local_day_bounds

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
    assert params["wind_speed_unit"] == "ms"


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


# ---------------------------------------------------------------------------
# F11 measurement 2 as a standing test (spec 6.5.1, §5.2 / §7, work order
# step 9): live, local-only. See docs/sprint6_step6_5_1_log.md, "Schritt 8"
# for the full F10/F11 measurement writeup this codifies.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_lead_zero_of_12z_run_radiation_none_others_valued() -> None:
    """At lead 0 of a 12 UTC run there is no "preceding hour" within the run
    yet. Both radiation variables (mean-preceding-hour) must therefore be
    NaN there -- not a night effect, since 12:00 UTC is bright afternoon --
    while every instantaneous variable carries an ordinary value. If this
    ever flips, the API's aggregation convention for radiation has changed
    and 6.5.2 would be reading it wrong from then on.
    """
    run = pd.Timestamp("2025-06-21T12:00", tz="UTC")
    df = fetch_run(run, forecast_days=1, use_cache=False)
    lead_zero = df.iloc[0]

    for point in GRID_POINTS:
        assert pd.isna(lead_zero[f"{point.point_id}__shortwave_radiation"])
        assert pd.isna(lead_zero[f"{point.point_id}__direct_normal_irradiance"])
        assert not pd.isna(lead_zero[f"{point.point_id}__temperature_2m"])
        assert not pd.isna(lead_zero[f"{point.point_id}__surface_pressure"])


# ---------------------------------------------------------------------------
# run_init_for_target_day (spec §5.4a)
# ---------------------------------------------------------------------------


def test_run_init_for_target_day_normal_day() -> None:
    result = run_init_for_target_day(dt.date(2026, 8, 31))
    assert result == pd.Timestamp("2026-08-30T00:00", tz="UTC")


@pytest.mark.parametrize(
    "target_day",
    [
        dt.date(2026, 3, 29),  # spring-forward DST day
        dt.date(2026, 3, 30),  # day after
        dt.date(2025, 10, 26),  # fall-back DST day
        dt.date(2025, 10, 27),  # day after
    ],
)
def test_run_init_for_target_day_no_dst_shift(target_day: dt.date) -> None:
    result = run_init_for_target_day(target_day)
    expected = pd.Timestamp(target_day - dt.timedelta(days=1), tz="UTC")
    assert result == expected
    assert result.hour == 0


def test_run_init_for_target_day_gate_closure_assertion_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control (spec §7): an artificially-too-early gate closure
    must make the assertion fail, proving it is a live check rather than a
    comment that can silently drift."""
    import energy_price_forecast.data.weather_client as client_module

    def fake_gate_closure(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([pd.Timestamp("1900-01-01", tz="UTC")])

    monkeypatch.setattr(client_module, "gate_closure_for_index", fake_gate_closure)

    with pytest.raises(AssertionError):
        run_init_for_target_day(dt.date(2026, 8, 31))


def test_dst_conversion_produces_correct_local_hour_counts() -> None:
    """§7 "DST im Artefakt-Index": converting a run's UTC valid times to
    Europe/Berlin across both changeover days yields 23 (spring) / 25
    (autumn) local hours for the covered day, with no duplicate or missing
    timestamps. Purely a pandas/tz-database check -- no network needed.
    """
    cases = [
        (pd.Timestamp("2026-03-28T00:00", tz="UTC"), dt.date(2026, 3, 29), 23),
        (pd.Timestamp("2025-10-25T00:00", tz="UTC"), dt.date(2025, 10, 26), 25),
    ]
    for run, target_day, expected_hours in cases:
        utc_index = pd.date_range(run, periods=72, freq="h", tz="UTC")
        local_index = utc_index.tz_convert("Europe/Berlin")
        start, end = local_day_bounds(target_day)
        covered = local_index[(local_index >= start) & (local_index < end)]
        assert len(covered) == expected_hours
        assert covered.is_unique


# ---------------------------------------------------------------------------
# log_availability_attempt (spec §5.4b). Goes through the ordinary cache
# by design (spec §5.5) -- every test here redirects cache_path into
# tmp_path so it doesn't touch the real data/cache/.
# ---------------------------------------------------------------------------


def _redirect_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import energy_price_forecast.data.weather_client as client_module
    from energy_price_forecast.data._weather_cache import cache_path as real_cache_path

    monkeypatch.setattr(
        client_module,
        "cache_path",
        lambda run_init_utc, model: real_cache_path(run_init_utc, model, root=tmp_path / "cache"),
    )


def test_log_availability_attempt_success_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_cache(monkeypatch, tmp_path)
    log_path = tmp_path / "weather_availability_probe.csv"
    _install_fake_get(monkeypatch, _FakeResponse(200, _load_fixture()))

    row = log_availability_attempt(_RUN, forecast_days=1, log_path=log_path)

    assert row["available"] == "true"
    assert row["http_status"] == "200"
    assert row["n_hours"] == str(_N_HOURS)
    assert row["error"] == ""

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header + one row
    assert lines[0].split(",") == [
        "probe_time_utc",
        "run_init_utc",
        "model",
        "available",
        "http_status",
        "n_hours",
        "error",
    ]


def test_log_availability_attempt_failure_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_cache(monkeypatch, tmp_path)
    log_path = tmp_path / "weather_availability_probe.csv"
    _install_fake_get(monkeypatch, _FakeResponse(400, {"error": True, "reason": "not available"}))

    row = log_availability_attempt(_RUN, log_path=log_path)

    assert row["available"] == "false"
    assert row["http_status"] == "400"
    assert row["n_hours"] == ""
    assert row["error"] != ""


def test_log_availability_attempt_appends_not_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_cache(monkeypatch, tmp_path)
    log_path = tmp_path / "weather_availability_probe.csv"
    _install_fake_get(monkeypatch, _FakeResponse(200, _load_fixture()))

    log_availability_attempt(_RUN, forecast_days=1, log_path=log_path)
    log_availability_attempt(_RUN, forecast_days=1, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # header + two rows


def test_log_availability_attempt_writes_to_cache_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spec §5.5: a successful probe leaves the run cached, so 6.7's later
    same-day submission computation finds it already there. A second
    attempt for the same run must therefore make zero further HTTP calls."""
    import energy_price_forecast.data.weather_client as client_module

    _redirect_cache(monkeypatch, tmp_path)
    log_path = tmp_path / "weather_availability_probe.csv"
    calls = _install_fake_get(monkeypatch, _FakeResponse(200, _load_fixture()))

    log_availability_attempt(_RUN, forecast_days=1, log_path=log_path)
    assert len(calls) == 1

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("requests.get must not be called on a cache hit")

    monkeypatch.setattr(client_module.requests, "get", fail_if_called)

    row = log_availability_attempt(_RUN, forecast_days=1, log_path=log_path)
    assert row["available"] == "true"


# ---------------------------------------------------------------------------
# Determinism and lead-hour coverage (F8), deferred from step 5 -- both live,
# local-only (spec §7)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_determinism_same_run_fetched_twice() -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    first = fetch_run(run, forecast_days=1, use_cache=False)
    second = fetch_run(run, forecast_days=1, use_cache=False)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.integration
@pytest.mark.parametrize("target_day", [dt.date(2025, 6, 16), dt.date(2025, 1, 16)])
def test_lead_hour_coverage_full_local_day(target_day: dt.date) -> None:
    """F8 as a standing test: the 00 UTC run of D-1 must cover the complete
    local calendar day D, in both seasons. Cross-checked against the API's
    own timezone=Europe/Berlin parameter -- the one place in this repo that
    parameter is used (spec §2.1) -- as an independent verification of our
    own UTC -> local conversion.
    """
    run = run_init_for_target_day(target_day)
    df = fetch_run(run, forecast_days=3, use_cache=False)
    valid_times = pd.DatetimeIndex(df.index.get_level_values("valid_time_utc").unique())
    local_times = valid_times.tz_convert("Europe/Berlin")

    start, end = local_day_bounds(target_day)
    covered = [t for t in local_times if start <= t < end]
    assert len(covered) == 24

    response = requests.get(
        "https://single-runs-api.open-meteo.com/v1/forecast",
        params={
            "latitude": GRID_POINTS[0].latitude,
            "longitude": GRID_POINTS[0].longitude,
            "hourly": "temperature_2m",
            "models": "ecmwf_ifs",
            "run": run.strftime("%Y-%m-%dT%H:%M"),
            "forecast_days": 3,
            "timezone": "Europe/Berlin",
        },
        timeout=30,
    )
    response.raise_for_status()
    api_local_times = pd.to_datetime(response.json()["hourly"]["time"])
    target_day_str = target_day.isoformat()
    api_covered = [t for t in api_local_times if t.date().isoformat() == target_day_str]
    assert len(api_covered) == 24
