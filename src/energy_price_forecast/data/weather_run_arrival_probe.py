"""Weather-run arrival-time probe (spec 8.0a).

Collects, for a fixed list of (model, run hour) pairs, WHEN a run becomes
fully retrievable over the Open-Meteo Single Runs API relative to gate
closure -- a number that exists nowhere in the provider's own
documentation (only rough ranges). This module decides nothing and feeds
no store/feature/submission path (spec section 6); it only measures and
appends to ``logs/weather_run_arrival_probe.csv``. The evaluation happens
in 8.0b.

Deliberately separate from ``weather_client.py``'s ``fetch_run``: that
function requests the full 18-point grid in one call and fails fast on
anything short of exactly that shape (spec 6.5.1 decision 10 -- the live
path must never silently request less than it needs). This probe
requests exactly ONE grid point (arrival time depends on the run, not the
location, spec section 2.4) and must keep running even when a point, a
variable or the whole response is partial -- the opposite failure
posture, which is why it is not a variant of ``fetch_run``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from energy_price_forecast.data.energy_charts import append_probe_row
from energy_price_forecast.data.weather_client import run_init_at_hour_for_target_day
from energy_price_forecast.data.weather_grid import GRID_POINTS, HOURLY_VARIABLES
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.ops.windows import (
    expected_timestamp_count,
    local_day_bounds,
    next_delivery_day,
)

logger = logging.getLogger(__name__)

_BASE_URL: Final[str] = "https://single-runs-api.open-meteo.com/v1/forecast"
_TIMEOUT_S: Final[float] = 30.0
_FORECAST_DAYS: Final[int] = 2  # run day + target day, spec 2.2

# Fixed, named list (spec 2.1) -- not guessed, confirmed live 2026-10-10
# against a known-old, certainly-archived run (2026-06-01, both 03 and 06
# UTC) before being added here: all four accept the short (no "dwd_")
# spelling, matching this project's existing "ecmwf_ifs" convention, and
# all four returned genuinely distinct forecast values at the same
# coordinate/run/time (not the same model under two aliases). A deliberately
# invalid model string was also tried in the same session and correctly
# got HTTP 400 -- the API does validate "models", so this one-time check is
# meaningful, not a coincidence of a lenient endpoint.
MODEL_RUN_PAIRS: Final[tuple[tuple[str, int], ...]] = (
    ("ecmwf_ifs", 6),
    ("icon_d2", 6),
    ("icon_d2", 3),
    ("icon_eu", 6),
    ("icon_global", 6),
)

# "von einer Stunde bis sieben Stunden nach Laufbeginn" (spec section 4) --
# a pair outside this window is neither attempted nor logged: too early,
# nothing has changed since the last probe; too late, the arrival-time
# question this probe exists to answer is already settled either way.
_WINDOW_MIN_MINUTES: Final[float] = 60.0
_WINDOW_MAX_MINUTES: Final[float] = 420.0

# One arbitrary, fixed land point from the existing versioned grid (spec
# section 2.4: "Die Ankunftszeit hängt am Lauf, nicht am Ort"). st_center
# (central Germany, no coastline) -- avoids any point-specific quirk a
# coastal/offshore point might have, though per the spec's own reasoning
# any point would do.
_PROBE_POINT = next(p for p in GRID_POINTS if p.point_id == "st_center")

# The two variables measured (not assumed, spec 6.5.1 §5.2/§2.2) to be null
# at night as a geometric fact, not a gap (weather_grid.py's own
# VARIABLE_TIME_CONVENTION / weather_client.py's "None values are expected
# ... geometrically correct" note). Treating a null here as "missing" would
# make 100% coverage permanently unreachable for the ~half of any day that
# is nighttime, defeating the early-exit this probe relies on (spec 2.4) --
# so a null in either of these two never reduces n_hours_covered, matching
# the established project convention rather than reinventing one.
_NIGHT_NULLABLE_VARIABLES: Final[frozenset[str]] = frozenset(
    {"shortwave_radiation", "direct_normal_irradiance"}
)

_LOG_PATH: Final[Path] = Path("logs/weather_run_arrival_probe.csv")
_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "probe_timestamp_utc",
    "run_id",
    "target_day",
    "model",
    "run_init_utc",
    "minutes_since_run_init",
    "minutes_to_gate_closure",
    "http_status",
    "error_kind",
    "n_hours_expected",
    "n_hours_covered",
    "coverage_ratio",
    "variables_missing",
    "first_ts",
    "last_ts",
)


def _classify_rate_limit(reason: str | None) -> str:
    """ "stündlich oder täglich, am reason-Text erkennbar" (spec 2.4 point 5).

    Measured text from the real bulk fetch (docs/sprint6_step6_5_1_log.md):
    a daily-limit 429 carries "Daily API request limit exceeded" in
    ``reason``; no hourly-limit 429 text has been observed in this repo's
    own history, so that branch is inferred from the daily one's shape,
    not measured -- "unknown" is the honest fallback when neither
    substring matches, rather than guessing which one it must be.
    """
    if reason is None:
        return "rate_limited_unknown"
    lowered = reason.lower()
    if "daily" in lowered:
        return "rate_limited_daily"
    if "hourly" in lowered:
        return "rate_limited_hourly"
    return "rate_limited_unknown"


def already_fully_covered(
    model: str, run_init_utc: pd.Timestamp, target_day: dt.date, log_path: Path = _LOG_PATH
) -> bool:
    """True if a prior row already recorded full coverage for this exact
    (model, run_init_utc, target_day) -- the early-exit check (spec 2.4
    point 3). A missing log file or no matching row is "not yet", not an
    error: the very first probe of the day always reaches this False.
    """
    if not log_path.exists():
        return False
    existing = pd.read_csv(log_path)
    if existing.empty:
        return False
    match = existing[
        (existing["model"] == model)
        & (existing["run_init_utc"] == run_init_utc.isoformat())
        & (existing["target_day"] == target_day.isoformat())
        & (existing["http_status"].astype(str) == "200")
        & (existing["coverage_ratio"] >= 1.0)
    ]
    return not match.empty


def probe_pair(
    model: str,
    run_init_utc: pd.Timestamp,
    target_day: dt.date,
    *,
    run_id: str,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    """One measurement of one (model, run) pair against ``target_day``.

    Every outcome in spec section 5's green rows (not yet there, missing
    variable, 429/timeout/5xx, partial coverage) is recorded in the
    returned row; this function never raises for any of them. A caller
    checking section 5's one red, config-error row does so separately
    (spec section 2.1's one-time pre-flight check), not here.
    """
    if as_of.tzinfo is None:
        raise ValueError(f"as_of must be tz-aware, got a naive timestamp: {as_of!r}")

    day_start, day_end = local_day_bounds(target_day)
    gate_closure_utc = gate_closure_for_index(pd.DatetimeIndex([day_start]))[0]
    n_hours_expected = expected_timestamp_count(
        day_start.tz_convert("UTC"), day_end.tz_convert("UTC"), 60
    )

    row: dict[str, object] = {
        "probe_timestamp_utc": as_of.isoformat(),
        "run_id": run_id,
        "target_day": target_day.isoformat(),
        "model": model,
        "run_init_utc": run_init_utc.isoformat(),
        "minutes_since_run_init": round((as_of - run_init_utc).total_seconds() / 60.0, 1),
        "minutes_to_gate_closure": round((gate_closure_utc - as_of).total_seconds() / 60.0, 1),
        "http_status": "",
        "error_kind": "",
        "n_hours_expected": n_hours_expected,
        "n_hours_covered": 0,
        "coverage_ratio": 0.0,
        "variables_missing": "",
        "first_ts": "",
        "last_ts": "",
    }
    assert set(row) == set(_LOG_COLUMNS), "row keys must exactly match _LOG_COLUMNS"

    params: dict[str, str | int | float] = {
        "latitude": _PROBE_POINT.latitude,
        "longitude": _PROBE_POINT.longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": model,
        "run": run_init_utc.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": _FORECAST_DAYS,
        "wind_speed_unit": "ms",
    }

    try:
        response = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT_S)
    except (requests.ConnectionError, requests.Timeout) as exc:
        row["error_kind"] = f"transport_error:{exc.__class__.__name__}"
        return row

    row["http_status"] = str(response.status_code)

    if response.status_code == 429:
        reason: str | None = None
        with contextlib.suppress(ValueError):
            reason = response.json().get("reason")
        row["error_kind"] = _classify_rate_limit(reason)
        return row
    if response.status_code >= 500:
        row["error_kind"] = "server_error"
        return row
    if response.status_code != 200:
        # Not-yet-published, an invalid run and (pre-go-live only, spec
        # 2.1) an invalid model id all return the identical HTTP 400 --
        # weather_client.py's fetch_run documents the same ambiguity for
        # the production grid request. The recurring probe logs this as
        # the expected "not there yet" measurement (spec 5, green row 1);
        # distinguishing a config error is the one-time pre-flight check's
        # job, not this function's.
        row["error_kind"] = "not_available"
        return row

    try:
        payload: dict[str, Any] = response.json()
    except ValueError:
        row["error_kind"] = "malformed_200_body"
        return row

    hourly: dict[str, Any] = payload.get("hourly", {})
    variables_missing = [v for v in HOURLY_VARIABLES if v not in hourly]
    row["variables_missing"] = ";".join(variables_missing)

    times_raw = hourly.get("time", [])
    timestamps = (
        pd.DatetimeIndex(pd.to_datetime(times_raw, utc=True))
        if times_raw
        else pd.DatetimeIndex([], tz="UTC")
    )
    if len(timestamps):
        row["first_ts"] = timestamps.min().isoformat()
        row["last_ts"] = timestamps.max().isoformat()

    checked_variables = [v for v in HOURLY_VARIABLES if v not in variables_missing]
    target_hours = pd.date_range(
        day_start.tz_convert("UTC"), day_end.tz_convert("UTC"), freq="h", inclusive="left"
    )

    n_hours_covered = 0
    for hour in target_hours:
        positions = timestamps.get_indexer_for(pd.DatetimeIndex([hour]))
        pos = positions[0] if len(positions) else -1
        if pos < 0:
            continue
        hour_ok = True
        for variable in checked_variables:
            if variable in _NIGHT_NULLABLE_VARIABLES:
                continue
            # Safe to index directly without a fallback: checked_variables
            # is HOURLY_VARIABLES minus variables_missing, and
            # variables_missing is exactly "not a key in hourly" -- so
            # every variable reached here is guaranteed present as a key.
            if hourly[variable][pos] is None:
                hour_ok = False
                break
        if hour_ok:
            n_hours_covered += 1

    row["n_hours_covered"] = n_hours_covered
    row["coverage_ratio"] = (
        round(n_hours_covered / n_hours_expected, 4) if n_hours_expected else 0.0
    )
    if n_hours_covered < n_hours_expected:
        row["error_kind"] = "partial_coverage"

    return row


def run_probe(
    as_of: pd.Timestamp,
    *,
    run_id: str,
    log_path: Path = _LOG_PATH,
    pairs: tuple[tuple[str, int], ...] = MODEL_RUN_PAIRS,
) -> list[dict[str, object]]:
    """Probe every pair in ``pairs`` against the delivery day ``as_of``'s
    next submission needs, appending one row per pair actually attempted.

    A pair is attempted only if it is inside its own 1-7h window (spec
    section 4) AND not already fully covered in the log (spec 2.4 point
    3) -- neither case writes a row, since nothing new was measured. On a
    429, the row for that pair IS written (the rate limit itself is the
    measurement, spec 2.4 point 5) and the run stops there, never
    attempting the remaining pairs.

    Every timing computation below is derived from the passed-in
    ``as_of``, never from a fresh ``pd.Timestamp.now()`` read inside the
    loop -- the standing rule from docs/sprint6_fix_partial_today.md §3.4
    ("jeder Probe pinnt as_of explizit, nie die Wanduhr"). In production
    ``as_of`` already IS "now" (the caller passes a fresh read once,
    before this loop starts), so this changes nothing live; it is what
    makes the window/early-exit logic deterministically testable at all.
    """
    target_day = next_delivery_day(as_of)

    rows: list[dict[str, object]] = []
    for model, run_hour in pairs:
        run_init_utc = run_init_at_hour_for_target_day(target_day, run_hour)
        minutes_since_run_init = (as_of - run_init_utc).total_seconds() / 60.0

        if not (_WINDOW_MIN_MINUTES <= minutes_since_run_init <= _WINDOW_MAX_MINUTES):
            continue
        if already_fully_covered(model, run_init_utc, target_day, log_path):
            continue

        row = probe_pair(model, run_init_utc, target_day, run_id=run_id, as_of=as_of)
        append_probe_row(row, log_path)
        rows.append(row)
        logger.info("probed %s run=%s for %s: %s", model, run_init_utc, target_day, row)

        error_kind = row["error_kind"]
        if isinstance(error_kind, str) and error_kind.startswith("rate_limited"):
            logger.warning("rate limited (%s) -- aborting remaining pairs this run", error_kind)
            break

    return rows
