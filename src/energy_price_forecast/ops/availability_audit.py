"""Raw-series availability audit: measures what the model would actually see.

Uses the same data clients as the model pipeline (data/entsoe_client.py,
data/commodities_client.py), not a separate HTTP probe -- an audit against a
differently-built client would measure availability *for that client*, not
for the pipeline this repo actually runs (spec §4.1).

The checklist is derived from two sources, never hardcoded here:
_RAW_AVAILABILITY (features/availability.py) for base columns and their
class, and NEIGHBORS (data/entsoe_client.py) for the concrete cross-border
column names the scheduled_*/physical_* prefixes expand to (spec §5.1,
§13). Errors from a single series are data, not a job failure (spec §4.4):
a fetch-group failure produces status="error" rows for every column that
group would have served, and run_audit() itself never raises for that
reason -- only building the checklist, computing windows, or the arena
catalog call can raise here, and all three are close to unrecoverable.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from energy_price_forecast.arena.catalog import ChallengeSpec, canonical_hash, get_challenge
from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.data.entsoe_client import (
    NEIGHBORS,
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)
from energy_price_forecast.features.availability import _RAW_AVAILABILITY, Availability
from energy_price_forecast.ops.windows import (
    expected_timestamp_count,
    local_day_bounds,
    local_window_bounds,
)

LOCAL_TZ = "Europe/Berlin"
ARENA_CHALLENGE_ID = "2"  # Day-Ahead Prices | Germany-Luxembourg | Point Forecast (spec §5.4)
TRAINING_WINDOW_DAYS = 90
COMMODITY_LOOKBACK_DAYS = 10  # long enough to cross any weekend/holiday gap
COMMODITY_STALE_THRESHOLD_DAYS = 4  # readability only, not a submit/no-submit threshold (spec §5.5)
GATE_CLOSURE_LOCAL_HOUR = (
    12  # duplicated from market_time.py's constant; see module docstring above
)

AVAILABILITY_COLUMNS = [
    "run_ts_utc",
    "run_ts_local",
    "target_date",
    "series",
    "availability_class",
    "window_kind",
    "window_start_local",
    "window_end_local",
    "native_resolution_min",
    "expected_count",
    "present_count",
    "coverage_ratio",
    "latest_target_local",
    "latest_target_offset_days",
    "staleness_days",
    "status",
    "error",
]
CHALLENGE_CATALOG_COLUMNS = [
    "run_ts_utc",
    "challenge_id",
    "name",
    "resolution",
    "timezone",
    "deadline",
    "target_start",
    "target_end",
    "expected_values",
    "allow_multiple",
    "selection_policy",
    "precision_decimals",
    "allow_negative",
    "max_forecast_points",
    "spec_sha256",
]
AUDIT_RUNS_COLUMNS = [
    "run_ts_utc",
    "run_ts_local",
    "local_date",
    "target_date",
    "windows_checked",
    "n_series",
    "n_client_calls",
    "duration_s",
    "n_ok",
    "n_partial",
    "n_missing",
    "n_error",
    "trigger",
]


@dataclass(frozen=True)
class SeriesSpec:
    """One raw column on the checklist and where to fetch it from."""

    column: str
    availability_class: Availability
    fetch_group: str


FetchFn = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]

# fetch_group -> the client function that produces it. Grouped, not one
# column at a time: fetch_load returns load_actual and
# load_forecast_day_ahead together, fetch_generation_by_type returns all
# gen_* columns together, and the two border-flow functions return all
# NEIGHBORS at once -- calling per-column would multiply client calls (and
# burst load, spec §8) for no benefit.
_FETCH_FUNCTIONS: dict[str, FetchFn] = {
    "day_ahead_price": fetch_day_ahead_prices,
    "load": fetch_load,
    "wind_solar_forecast": fetch_wind_solar_forecast,
    "generation": fetch_generation_by_type,
    "scheduled_exchanges": fetch_scheduled_exchanges,
    "cross_border_flows": fetch_cross_border_flows,
    "ttf_gas": fetch_ttf_gas,
    "eua_co2": fetch_eua_co2,
}

_BASE_FETCH_GROUP: dict[str, str] = {
    "day_ahead_price": "day_ahead_price",
    "load_forecast_day_ahead": "load",
    "load_actual": "load",
    "wind_onshore_forecast": "wind_solar_forecast",
    "wind_offshore_forecast": "wind_solar_forecast",
    "solar_forecast": "wind_solar_forecast",
    "gen_nuclear": "generation",
    "gen_lignite": "generation",
    "gen_hard_coal": "generation",
    "gen_gas": "generation",
    "gen_oil": "generation",
    "gen_biomass": "generation",
    "gen_hydro": "generation",
    "gen_wind_onshore": "generation",
    "gen_wind_offshore": "generation",
    "gen_solar": "generation",
    "gen_other": "generation",
    "ttf_gas_eur_per_mwh": "ttf_gas",
    "eua_co2_eur_per_t": "eua_co2",
}


def build_checklist() -> list[SeriesSpec]:
    """Derive the raw-series checklist from the registry + NEIGHBORS (spec §5.1)."""
    items = [
        SeriesSpec(column, cls, _BASE_FETCH_GROUP[column])
        for column, cls in _RAW_AVAILABILITY.items()
    ]
    for neighbor in NEIGHBORS:
        suffix = neighbor.lower()
        items.append(
            SeriesSpec(
                f"scheduled_net_de_to_{suffix}", Availability.DA_FIXED, "scheduled_exchanges"
            )
        )
        items.append(
            SeriesSpec(f"physical_net_de_to_{suffix}", Availability.RT_ACTUAL, "cross_border_flows")
        )
    return items


# ---------------------------------------------------------------------------
# Window helpers (spec §5.2)
# ---------------------------------------------------------------------------


def _critical_fetch_window(
    cls: Availability, target_date: dt.date
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The range actually fetched for cls's critical check.

    Equal to the evaluation window except for DA_FORECAST, which is fetched
    one calendar day earlier so latest_target_local/offset_days can tell
    "not yet published" apart from "genuinely missing" (spec §5.2).
    """
    if cls is Availability.DA_FORECAST:
        return local_window_bounds(target_date, days_back_start=1, days_back_end=-1)
    if cls is Availability.DA_FIXED:
        return local_window_bounds(target_date, days_back_start=1, days_back_end=0)
    if cls is Availability.RT_ACTUAL:
        return local_window_bounds(target_date, days_back_start=2, days_back_end=1)
    raise ValueError(f"no critical fetch window for {cls}")


def _critical_eval_window(
    cls: Availability, target_date: dt.date
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The window expected_count/present_count/coverage_ratio refer to.

    Always the nominal (unextended) window -- window_start_local/
    window_end_local never show the fetch range, only this (spec §5.5).
    """
    if cls is Availability.DA_FORECAST:
        return local_day_bounds(target_date)
    return _critical_fetch_window(cls, target_date)


def _training_window(target_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    return local_window_bounds(target_date, days_back_start=TRAINING_WINDOW_DAYS, days_back_end=0)


def _group_fetch_range(
    specs: list[SeriesSpec], target_date: dt.date, *, include_training: bool
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Union of every column's needed range in this fetch group, one call covers all of them."""
    starts = []
    ends = []
    for spec in specs:
        s, e = _critical_fetch_window(spec.availability_class, target_date)
        starts.append(s)
        ends.append(e)
    if include_training:
        s, e = _training_window(target_date)
        starts.append(s)
        ends.append(e)
    return min(starts), max(ends)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _detect_native_resolution(non_na_index: pd.DatetimeIndex) -> int | None:
    """Native resolution in minutes, from the typical spacing of non-NaN timestamps.

    Uses the full fetched series, not just the evaluation-window slice: a
    DA_FORECAST series can be empty for D itself (not yet published) while
    still having plenty of D-1 points to infer the resolution from.
    """
    if len(non_na_index) < 2:
        return None
    diffs = non_na_index.to_series().diff().dropna()
    if diffs.empty:
        return None
    minutes = diffs.mode().iloc[0].total_seconds() / 60
    return int(minutes) if minutes > 0 else None


def _as_datetime_series(series: pd.Series) -> pd.Series:
    """Normalize to a tz-aware (UTC) DatetimeIndex series.

    The data clients' empty-result path (e.g. fetch_day_ahead_prices on
    NoMatchingDataError) returns pd.DataFrame(columns=[...]) -- a real
    RangeIndex, not an empty DatetimeIndex -- so a genuinely-missing series
    needs this before any timestamp comparison touches its index.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def _slice_window(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    series = _as_datetime_series(series)
    start_utc = start.tz_convert("UTC")
    end_utc = end.tz_convert("UTC")
    return series[(series.index >= start_utc) & (series.index < end_utc)]


def _empty_row(
    run_ts_utc: pd.Timestamp,
    run_ts_local: pd.Timestamp,
    target_date: dt.date,
    spec: SeriesSpec,
    window_kind: str,
) -> dict[str, Any]:
    return {
        "run_ts_utc": run_ts_utc.isoformat(),
        "run_ts_local": run_ts_local.isoformat(),
        "target_date": target_date.isoformat(),
        "series": spec.column,
        "availability_class": spec.availability_class.name,
        "window_kind": window_kind,
        "window_start_local": None,
        "window_end_local": None,
        "native_resolution_min": None,
        "expected_count": None,
        "present_count": None,
        "coverage_ratio": None,
        "latest_target_local": None,
        "latest_target_offset_days": None,
        "staleness_days": None,
        "status": None,
        "error": None,
    }


def _availability_row(
    df: pd.DataFrame | None,
    error: Exception | None,
    spec: SeriesSpec,
    window_kind: str,
    target_date: dt.date,
    run_ts_utc: pd.Timestamp,
    run_ts_local: pd.Timestamp,
) -> dict[str, Any]:
    row = _empty_row(run_ts_utc, run_ts_local, target_date, spec, window_kind)

    eval_start, eval_end = (
        _critical_eval_window(spec.availability_class, target_date)
        if window_kind == "critical"
        else _training_window(target_date)
    )
    row["window_start_local"] = eval_start.isoformat()
    row["window_end_local"] = eval_end.isoformat()

    if error is not None:
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"[:200]
        return row

    assert df is not None
    raw_series = df[spec.column] if spec.column in df.columns else pd.Series(dtype=float)
    full_series = _as_datetime_series(raw_series)
    non_na_index = cast(pd.DatetimeIndex, full_series.dropna().index)
    native_resolution = _detect_native_resolution(non_na_index)
    present = int(_slice_window(full_series, eval_start, eval_end).notna().sum())
    row["present_count"] = present

    if native_resolution is None:
        row["status"] = "missing" if present == 0 else "partial"
    else:
        expected = expected_timestamp_count(eval_start, eval_end, native_resolution)
        coverage = round(present / expected, 4) if expected else 0.0
        row["native_resolution_min"] = native_resolution
        row["expected_count"] = expected
        row["coverage_ratio"] = coverage
        row["status"] = "ok" if coverage == 1.0 else "partial" if coverage > 0 else "missing"

    if spec.availability_class is Availability.DA_FORECAST and window_kind == "critical":
        fetch_start, fetch_end = _critical_fetch_window(spec.availability_class, target_date)
        latest_in_fetch = _slice_window(full_series, fetch_start, fetch_end).dropna()
        if len(latest_in_fetch):
            latest_local = latest_in_fetch.index.max().tz_convert(LOCAL_TZ)
            row["latest_target_local"] = latest_local.isoformat()
            row["latest_target_offset_days"] = (latest_local.date() - target_date).days

    return row


def _commodity_row(
    df: pd.DataFrame | None,
    error: Exception | None,
    spec: SeriesSpec,
    target_date: dt.date,
    run_ts_utc: pd.Timestamp,
    run_ts_local: pd.Timestamp,
) -> dict[str, Any]:
    """COMMODITY gets one row per run (window_kind="critical"), not two.

    There is no separate "training" fact for a single most-recent-settlement
    measurement -- computing it twice with identical content would just be
    redundant log lines. Checked every run (not gated by is_first_run_of_day)
    so staleness can be observed intraday if a backfill lands.
    """
    row = _empty_row(run_ts_utc, run_ts_local, target_date, spec, "critical")
    lookback_start, lookback_end = local_window_bounds(
        target_date, days_back_start=COMMODITY_LOOKBACK_DAYS, days_back_end=0
    )
    row["window_start_local"] = lookback_start.isoformat()
    row["window_end_local"] = lookback_end.isoformat()

    if error is not None:
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"[:200]
        return row

    assert df is not None
    series = df[spec.column] if spec.column in df.columns else pd.Series(dtype=float)
    non_na = series.dropna()
    if non_na.empty:
        row["status"] = "missing"
        return row

    last_local_date = non_na.index.max().tz_convert(LOCAL_TZ).date()
    staleness = (target_date - last_local_date).days
    row["staleness_days"] = staleness
    row["status"] = "ok" if staleness <= COMMODITY_STALE_THRESHOLD_DAYS else "partial"
    return row


def _challenge_catalog_row(
    challenge: ChallengeSpec, target_date: dt.date, run_ts_utc: pd.Timestamp
) -> dict[str, Any]:
    """Deadline/target_start are computed here, not read off the challenge.

    ChallengeSpec deliberately doesn't carry them (arena/catalog.py); this
    audit already knows D and the fixed 12:00-local-on-D-1 gate closure, so
    it derives them the same way any submission caller would, without
    importing market_time.py (Entscheidung 5).
    """
    target_start, target_end = local_day_bounds(target_date)
    deadline_day_start, _ = local_day_bounds(target_date - dt.timedelta(days=1))
    deadline = deadline_day_start + pd.Timedelta(hours=GATE_CLOSURE_LOCAL_HOUR)
    expected_values = expected_timestamp_count(
        target_start, target_end, challenge.resolution_minutes
    )
    submission_window = challenge.raw.get("submission_window", {})

    return {
        "run_ts_utc": run_ts_utc.isoformat(),
        "challenge_id": challenge.challenge_id,
        "name": challenge.name,
        "resolution": challenge.resolution_minutes,
        "timezone": challenge.timezone,
        "deadline": deadline.isoformat(),
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "expected_values": expected_values,
        "allow_multiple": submission_window.get("allow_multiple"),
        "selection_policy": submission_window.get("selection_policy"),
        "precision_decimals": challenge.precision_decimals,
        "allow_negative": challenge.allow_negative,
        "max_forecast_points": challenge.max_forecast_points,
        "spec_sha256": canonical_hash(challenge.raw),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditOutcome:
    availability_rows: list[dict[str, Any]]
    challenge_catalog_row: dict[str, Any]
    audit_run_row: dict[str, Any]


def run_audit(now_utc: pd.Timestamp, trigger: str, *, is_first_run_of_day: bool) -> AuditOutcome:
    """Run one audit pass. now_utc is the measurement, not a schedule (spec §5.3)."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be tz-aware")

    run_ts_local = now_utc.tz_convert(LOCAL_TZ)
    target_date = (run_ts_local + pd.DateOffset(days=1)).date()  # D = tomorrow, spec §5.4
    windows_checked = "critical+training" if is_first_run_of_day else "critical"

    checklist = build_checklist()
    by_group: dict[str, list[SeriesSpec]] = {}
    for spec in checklist:
        by_group.setdefault(spec.fetch_group, []).append(spec)

    availability_rows: list[dict[str, Any]] = []
    n_client_calls = 0
    start_time = time.monotonic()

    for group, specs in by_group.items():
        is_commodity_group = specs[0].availability_class is Availability.COMMODITY

        if is_commodity_group:
            fetch_start, fetch_end = local_window_bounds(
                target_date, days_back_start=COMMODITY_LOOKBACK_DAYS, days_back_end=0
            )
        else:
            fetch_start, fetch_end = _group_fetch_range(
                specs, target_date, include_training=is_first_run_of_day
            )

        try:
            df: pd.DataFrame | None = _FETCH_FUNCTIONS[group](fetch_start, fetch_end)
            error: Exception | None = None
        except Exception as exc:  # spec §4.4: a series' error is data, not a job failure
            df = None
            error = exc
        n_client_calls += 1

        for spec in specs:
            if is_commodity_group:
                availability_rows.append(
                    _commodity_row(df, error, spec, target_date, now_utc, run_ts_local)
                )
            else:
                window_kinds = ["critical", "training"] if is_first_run_of_day else ["critical"]
                for window_kind in window_kinds:
                    availability_rows.append(
                        _availability_row(
                            df, error, spec, window_kind, target_date, now_utc, run_ts_local
                        )
                    )

    duration_s = time.monotonic() - start_time

    challenge = get_challenge(ARENA_CHALLENGE_ID)
    challenge_catalog_row = _challenge_catalog_row(challenge, target_date, now_utc)

    status_counts = {"ok": 0, "partial": 0, "missing": 0, "error": 0}
    for row in availability_rows:
        status_counts[row["status"]] += 1

    audit_run_row = {
        "run_ts_utc": now_utc.isoformat(),
        "run_ts_local": run_ts_local.isoformat(),
        "local_date": run_ts_local.date().isoformat(),
        "target_date": target_date.isoformat(),
        "windows_checked": windows_checked,
        "n_series": len(checklist),
        "n_client_calls": n_client_calls,
        "duration_s": round(duration_s, 3),
        "n_ok": status_counts["ok"],
        "n_partial": status_counts["partial"],
        "n_missing": status_counts["missing"],
        "n_error": status_counts["error"],
        "trigger": trigger,
    }

    return AuditOutcome(availability_rows, challenge_catalog_row, audit_run_row)


# ---------------------------------------------------------------------------
# CSV persistence (spec §5.4 steps 5-8)
# ---------------------------------------------------------------------------


def is_first_run_of_local_date(audit_runs_path: Path, local_date: dt.date) -> bool:
    """No state outside the repo, no time-window guessing (spec §5.2)."""
    if not audit_runs_path.exists():
        return True
    existing = pd.read_csv(audit_runs_path, usecols=["local_date"], dtype={"local_date": str})
    return local_date.isoformat() not in set(existing["local_date"])


def append_csv_rows(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    """Append rows to path, creating it (with header) if it doesn't exist yet.

    Never rewrites existing bytes -- mode="a" only. Column order is always
    `columns`, regardless of dict key insertion order, so it stays stable
    even if a future row builder starts including keys in a different order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)
