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
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.data.entsoe_client import (
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)

RowFetchFn = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class EntsoeSource:
    name: str
    fetch: RowFetchFn
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
