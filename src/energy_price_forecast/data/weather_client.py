"""Client for the Open-Meteo Single Runs API.

This is the only weather archive used anywhere in this repo (spec 6.5.1, §4
rule 4; plan decision 9). Neither the Historical Forecast API nor the
Previous Runs API appear here, in tests or in production code: both are
convenient and both leak. Historical Forecast splices together the first
hours of consecutive runs, so an afternoon-of-D feature would draw on a run
started on D itself; Previous Runs uses a fixed lag to the valid time, so an
evening-of-D feature would draw on a run published after our submission
deadline. Only the Single Runs API with an explicit ``run=`` parameter is
leakage-free by construction.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any, Final

import pandas as pd
import requests

from energy_price_forecast.data._weather_cache import cache_path, read_cached_run, write_cached_run
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES, expected_columns

logger = logging.getLogger(__name__)

_BASE_URL: Final[str] = "https://single-runs-api.open-meteo.com/v1/forecast"
_TIMEOUT_S: Final[float] = 30.0
_VALID_RUN_HOURS: Final[frozenset[int]] = frozenset({0, 6, 12, 18})
_COORDINATE_TOLERANCE_DEG: Final[float] = 0.15

# Transport-level retry only: a dropped connection or DNS blip, not an
# unpublished run. Deliberately small and separate from both existing
# retry helpers in this repo (arena/catalog.py, data/_entsoe_retry.py) --
# neither fits: this module must NOT retry a 429 or wait for a run to
# appear (spec §2.3, §2.6); that decision belongs to the caller (the bulk
# script's hard-stop rule, or the scheduler re-running the whole job).
_MAX_TRANSPORT_RETRIES: Final[int] = 2


class WeatherRunUnavailable(RuntimeError):  # noqa: N818 -- name fixed by spec 6.5.1, §5.3
    """Raised when a named model run cannot be retrieved.

    Carries the requested run, the model id and the HTTP status. The API
    returns the same HTTP 400 for a run that does not exist and for one that
    has not been published yet (spike 6.5.0, F6), so this exception does not
    distinguish them either -- the caller decides by clock time whether it is
    worth trying again later. Under GitHub Actions that decision belongs to
    the schedule, not to this module (spec 2.3).

    There is deliberately NO fallback to another run: the model is trained
    only on 00 UTC runs of D-1, and feeding it an older run would be a silent
    train/serve skew (spec 2.4, decisions 5b and 10). The submission policy in
    6.7 catches this and does not submit.
    """

    def __init__(
        self,
        run_init_utc: pd.Timestamp,
        model: str,
        http_status: int | None,
        reason: str | None = None,
    ) -> None:
        self.run_init_utc = run_init_utc
        self.model = model
        self.http_status = http_status
        self.reason = reason
        message = f"weather run unavailable: model={model!r} run={run_init_utc.isoformat()!r} status={http_status}"
        if reason:
            message += f" reason={reason!r}"
        super().__init__(message)


def _call_with_transport_retry[T](
    fn: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep
) -> T:
    """Retry fn() on a dropped connection or timeout only.

    Any HTTP response that actually arrives -- 200, 400, 429, 5xx -- is
    returned as-is and is not retried here: interpreting it (run missing vs.
    rate-limited vs. server error) is the caller's job (spec §2.3, §2.6).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= _MAX_TRANSPORT_RETRIES:
                raise
            wait = 2.0**attempt
            logger.warning(
                "Weather API transport error -- retrying in %.0fs (retry %d/%d): %s",
                wait,
                attempt + 1,
                _MAX_TRANSPORT_RETRIES,
                exc,
            )
            attempt += 1
            sleep(wait)


def _check_object_count(payload: list[dict[str, Any]]) -> None:
    if len(payload) != len(GRID_POINTS):
        raise ValueError(f"expected {len(GRID_POINTS)} objects in the response, got {len(payload)}")


def _check_coordinates(payload: list[dict[str, Any]]) -> None:
    for point, obj in zip(GRID_POINTS, payload, strict=True):
        lat_diff = abs(obj["latitude"] - point.latitude)
        lon_diff = abs(obj["longitude"] - point.longitude)
        if lat_diff > _COORDINATE_TOLERANCE_DEG or lon_diff > _COORDINATE_TOLERANCE_DEG:
            raise ValueError(
                f"point {point.point_id!r}: requested ({point.latitude}, {point.longitude}), "
                f"delivered ({obj['latitude']}, {obj['longitude']}) -- "
                f"exceeds {_COORDINATE_TOLERANCE_DEG}° tolerance"
            )


def _check_variables_present(payload: list[dict[str, Any]]) -> None:
    for point, obj in zip(GRID_POINTS, payload, strict=True):
        missing = [v for v in HOURLY_VARIABLES if v not in obj["hourly"]]
        if missing:
            raise ValueError(f"point {point.point_id!r} is missing variables: {missing}")


def _check_time_length(payload: list[dict[str, Any]], forecast_days: int) -> None:
    expected_min = forecast_days * 24
    n_hours = len(payload[0]["hourly"]["time"])
    if n_hours < expected_min:
        raise ValueError(f"expected at least {expected_min} hourly timestamps, got {n_hours}")


def _check_time_gapless(valid_times: pd.DatetimeIndex) -> None:
    diffs = valid_times[1:] - valid_times[:-1]
    bad = diffs != pd.Timedelta(hours=1)
    if bad.any():
        first_gap = int(bad.argmax())
        raise ValueError(
            f"non-hourly gap in returned timestamps between "
            f"{valid_times[first_gap]!r} and {valid_times[first_gap + 1]!r}"
        )


def _parse_response(payload: list[dict[str, Any]], run_init_utc: pd.Timestamp) -> pd.DataFrame:
    valid_times = pd.DatetimeIndex(pd.to_datetime(payload[0]["hourly"]["time"], utc=True))
    _check_time_gapless(valid_times)

    data: dict[str, list[float | None]] = {}
    for point, obj in zip(GRID_POINTS, payload, strict=True):
        for variable in HOURLY_VARIABLES:
            data[f"{point.point_id}__{variable}"] = obj["hourly"][variable]

    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([run_init_utc] * len(valid_times)), valid_times],
        names=["run_init_utc", "valid_time_utc"],
    )
    df = pd.DataFrame(data, index=index, columns=expected_columns()).astype("float32")
    return df


def fetch_run(
    run_init_utc: pd.Timestamp,
    *,
    model: str = "ecmwf_ifs",
    forecast_days: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch one named ECMWF IFS HRES run for the full fixed grid. One attempt.

    ``run_init_utc`` must be tz-aware UTC and an hour the model actually runs
    (00/06/12/18). There is no "latest run" mode: an implicitly chosen run is
    exactly what leaks in a backtest (abstract, decision 9). Derive the run
    from the target day with ``run_init_for_target_day`` instead.

    All 18 grid points travel in a single request as comma-separated
    coordinate lists; the response is a list of 18 objects in request order
    (measured in spike 6.5.0, F5). No ``timezone`` parameter is sent, so
    timestamps come back in UTC (spec 2.1).

    Returns a frame indexed by (run_init_utc, valid_time_utc) with 162 columns
    named ``{point_id}__{variable}``, dtype float32.

    With ``use_cache=True`` (default), a cache hit (data/_weather_cache.py)
    short-circuits the whole request -- no HTTP call is made at all. A miss
    fetches and then writes the cache entry, so a live probe run in the
    morning (spec §5.4c) leaves the run ready for the same day's later
    consumers. The cache is pure storage (no interval arithmetic, spec
    §2.5): it is keyed on the run and the grid/variable schema only, so it
    assumes callers use a consistent ``forecast_days`` across a given
    ``(model, run)`` -- true for every caller in this repo.

    ONE ATTEMPT, NO WAITING. If the run is not there, this raises. It does not
    sleep, poll or retry on a timer -- repetition is the scheduler's job
    (spec 2.3). Transport-level retries from the shared HTTP helper still
    apply; those cover a dropped connection, not an unpublished run.
    """
    if run_init_utc.tzinfo is None:
        raise ValueError(f"run_init_utc must be tz-aware, got a naive timestamp: {run_init_utc!r}")
    run_init_utc = run_init_utc.tz_convert("UTC")
    if (
        run_init_utc.hour not in _VALID_RUN_HOURS
        or run_init_utc.minute != 0
        or run_init_utc.second != 0
    ):
        raise ValueError(
            f"run_init_utc must be exactly one of {sorted(_VALID_RUN_HOURS)} UTC, "
            f"got {run_init_utc!r}"
        )

    path = cache_path(run_init_utc, model) if use_cache else None
    if path is not None:
        cached = read_cached_run(path)
        if cached is not None:
            logger.info("Weather cache hit: %s", path)
            return cached

    params = {
        "latitude": ",".join(str(p.latitude) for p in GRID_POINTS),
        "longitude": ",".join(str(p.longitude) for p in GRID_POINTS),
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": model,
        "run": run_init_utc.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": forecast_days,
    }

    def do_request() -> requests.Response:
        return requests.get(_BASE_URL, params=params, timeout=_TIMEOUT_S)

    response = _call_with_transport_retry(do_request)

    if response.status_code != 200:
        reason: str | None = None
        with contextlib.suppress(ValueError):
            reason = response.json().get("reason")
        raise WeatherRunUnavailable(run_init_utc, model, response.status_code, reason)

    try:
        payload: list[dict[str, Any]] = response.json()
    except ValueError as exc:
        # Spike 6.5.0's F6 measured a clean HTTP 400 for an invalid run, but
        # only for single-point requests. Live-tested against the actual
        # 18-point request shape this client uses: some invalid runs (before
        # archive start, an invalid run hour) instead return HTTP 200 with a
        # non-JSON streaming error body ("Unexpected error while streaming
        # data: modelRunUnavailable(...)") -- an undocumented multi-point-only
        # failure mode F6 never covered (see docs/sprint6_step6_5_1_log.md).
        # Any 200 response whose body isn't valid JSON is treated the same as
        # an explicit 400: the API has nothing usable for this run either way.
        raise WeatherRunUnavailable(
            run_init_utc, model, response.status_code, response.text[:200]
        ) from exc

    # NOTE: no check that the delivered run matches the requested run. Spike
    # 6.5.0, F6 measured that an invalid/unpublished run gets HTTP 400 with
    # no data at all -- there is no silent substitute run to guard against.
    _check_object_count(payload)
    _check_coordinates(payload)
    _check_variables_present(payload)
    _check_time_length(payload, forecast_days)

    # None values are expected and kept as NaN: spike 6.5.0, F9 measured that
    # shortwave_radiation / direct_normal_irradiance are None at night --
    # geometrically correct, not a data gap. A missing *key* is the error
    # case, and _check_variables_present already covers that.
    df = _parse_response(payload, run_init_utc)

    if path is not None:
        write_cached_run(df, path)

    return df
