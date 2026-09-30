"""Source bindings shared by scripts/sync_store.py and
scripts/rebuild_store.py: which real client function and on-disk cache
directory backs each source name in ops/store.py::EXPECTATION_TABLE.

Lives in the installed package (not under scripts/, which has no
__init__.py and is only importable as a package under pytest's own
pythonpath setup -- a direct ``python scripts/rebuild_store.py`` invocation
does not put the repo root on sys.path, so a `from scripts.sync_store
import ...` cross-script import breaks outside pytest). Both maintenance
scripts import from here instead, so the binding is defined exactly once.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import pandas as pd

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.data.energy_charts import (
    PRICE_COLUMN,
    fetch_price_range,
    fetch_series_range,
)
from energy_price_forecast.data.entsoe_client import (
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)
from energy_price_forecast.ops.windows import LOCAL_TZ

RowFetchFn = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]

# entsoe-py's underlying HTTP client embeds ENTSOE_API_KEY directly in the
# request URL as a securityToken query parameter -- a transient HTTPError's
# own str() therefore contains it verbatim (requests.HTTPError includes the
# full request URL). Found leaked into a committed audit log this way
# (publication spec, Sicherheitsprüfung section 3.1: a real token, truncated
# but reconstructable across several commits, in logs/availability.csv) --
# the truncation that produced the leak was incidental, not the cause; any
# fixed-length slice of an unredacted URL leaks a prefix. Applied wherever an
# exception from an ENTSO-E fetch is turned into a string that gets
# logged/persisted, not only at the point the leak was first found.
_SECURITY_TOKEN_RE: Final = re.compile(r"(securityToken=)[^&\s]+")


def redact_secrets(text: str) -> str:
    """Replace any ENTSO-E securityToken query-parameter value in text with
    a fixed placeholder. Safe to call unconditionally (a no-op if no token
    is present), so callers don't need to reason about which specific
    exception could contain one."""
    return _SECURITY_TOKEN_RE.sub(r"\1***REDACTED***", text)


class EntsoeFetchFn(Protocol):
    """All six data/entsoe_client.py fetch_* functions match this shape
    (6.7.1a): two positional args plus the keyword-only ``use_cache`` every
    one of them now carries. A plain RowFetchFn alias can't express the
    extra keyword, and scripts/sync_store.py's heal step needs to call
    through it with ``use_cache=False`` (spec section 5.4)."""

    def __call__(
        self, start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class EntsoeSource:
    name: str
    fetch: EntsoeFetchFn
    cache_dir: Path


ENTSOE_SOURCES: tuple[EntsoeSource, ...] = (
    EntsoeSource(
        "day_ahead_price", fetch_day_ahead_prices, DATA_RAW / "entsoe" / "day_ahead_prices"
    ),
    EntsoeSource("load", fetch_load, DATA_RAW / "entsoe" / "load"),
    EntsoeSource("wind_solar", fetch_wind_solar_forecast, DATA_RAW / "entsoe" / "wind_solar"),
    EntsoeSource("generation", fetch_generation_by_type, DATA_RAW / "entsoe" / "generation"),
    EntsoeSource(
        "scheduled_exchanges",
        fetch_scheduled_exchanges,
        DATA_RAW / "entsoe" / "scheduled_exchanges",
    ),
    EntsoeSource(
        "cross_border_flows", fetch_cross_border_flows, DATA_RAW / "entsoe" / "cross_border_flows"
    ),
)

# (source name, fetch function, raw column name)
COMMODITY_SOURCES: tuple[tuple[str, RowFetchFn, str], ...] = (
    ("ttf_gas", fetch_ttf_gas, "ttf_gas_eur_per_mwh"),
    ("eua_co2", fetch_eua_co2, "eua_co2_eur_per_t"),
)

# New cache location (spec 6.7.1 A1 finding): data/commodities_client.py had
# no on-disk cache before this spec -- the store needs one to have anything
# to pack.
COMMODITIES_DIR: Path = PROJECT_ROOT / "data" / "raw" / "commodities"

# Energy-Charts as a permanent second source (spec 6.9 section 2.4/3.2,
# Schritt 6). Deliberately a NEW location, not the Sprint 6.8 bulk-fetch
# artifact directories under data/raw/energy_charts/price|public_power_
# forecast/ -- those are monthly-file-per-series historical artifacts with
# their own layout (Teil 1/2/3, scripts/fetch_energy_charts_*_history.py),
# not the store's own single-file-per-source cache shape. scripts/
# backfill_energy_charts_store.py (one-time) reads FROM the 6.8 artifacts
# and writes INTO this directory; scripts/sync_store.py's ongoing
# maintenance runs only ever read/write here.
ENERGY_CHARTS_DIR: Path = PROJECT_ROOT / "data" / "raw" / "energy_charts" / "store"

_EC_LOAD_FORECAST_COLUMN: Final = "load_forecast_day_ahead_ec"


def _fetch_ec_price(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Adapts fetch_price_range's dt.date/local-calendar-day, both-ends-
    inclusive contract to the RowFetchFn shape sync_store.py's generic loop
    expects (UTC pd.Timestamp, half-open [start, end))."""
    start_date = start.tz_convert(LOCAL_TZ).date()
    end_date = (end - pd.Timedelta(seconds=1)).tz_convert(LOCAL_TZ).date()
    if start_date > end_date:
        return pd.DataFrame({PRICE_COLUMN: []}, index=pd.DatetimeIndex([], tz="UTC"))
    return fetch_price_range(start_date, end_date).to_frame()


def _fetch_ec_load_forecast(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Same date-contract adaptation as _fetch_ec_price, for the ``load``
    series of fetch_series_range -- renamed to the store's own
    load_forecast_day_ahead_ec column (fetch_series_range itself names the
    series after the requested production_type, "load", which would
    otherwise collide with ENTSO-E's own ``load`` fetch group name)."""
    start_date = start.tz_convert(LOCAL_TZ).date()
    end_date = (end - pd.Timedelta(seconds=1)).tz_convert(LOCAL_TZ).date()
    if start_date > end_date:
        return pd.DataFrame({_EC_LOAD_FORECAST_COLUMN: []}, index=pd.DatetimeIndex([], tz="UTC"))
    series = fetch_series_range("load", start_date, end_date)
    return series.rename(_EC_LOAD_FORECAST_COLUMN).to_frame()


# (source name, fetch function, raw column name) -- same shape as
# COMMODITY_SOURCES, but a separate tuple: EC updates far more often than
# once a day, so scripts/sync_store.py's EC sync must NOT reuse
# _sync_commodity_source's once-a-day cadence gate (spec section 2.8 was
# written for Yahoo Finance's own daily-close semantics, not this).
ENERGY_CHARTS_SOURCES: tuple[tuple[str, RowFetchFn, str], ...] = (
    ("day_ahead_price_ec", _fetch_ec_price, PRICE_COLUMN),
    ("load_forecast_day_ahead_ec", _fetch_ec_load_forecast, _EC_LOAD_FORECAST_COLUMN),
)


def code_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        return out.stdout.strip()[:12]
    except Exception:  # noqa: BLE001 -- best-effort only, never fatal for a maintenance run
        return "unknown"


def run_id_and_url() -> tuple[str, str]:
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    server = os.getenv("GITHUB_SERVER_URL")
    repo = os.getenv("GITHUB_REPOSITORY")
    if server and repo and run_id != "local":
        return run_id, f"{server}/{repo}/actions/runs/{run_id}"
    return run_id, ""
