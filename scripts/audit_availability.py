"""Thin CLI for the daily raw-series availability audit.

Logic lives in energy_price_forecast.ops.availability_audit so it stays
testable; this script only wires real time, real env, and real files
together (spec §9).

--part is required, not defaulted: "availability" needs ENTSOE_API_KEY and
never touches the Arena API, "catalog" needs ARENA_API_KEY and never touches
ENTSO-E/Yahoo. The workflow (.github/workflows/audit.yml) runs these as two
separate steps, each given only the one secret it needs, via env: (spec
§5.4, §4.5). "both" is a local-dev convenience -- it needs both keys, which
is fine on a developer's own machine where one .env already holds both, but
the workflow never uses it.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops.availability_audit import (
    AUDIT_RUNS_COLUMNS,
    AVAILABILITY_COLUMNS,
    CHALLENGE_CATALOG_COLUMNS,
    LOCAL_TZ,
    append_csv_rows,
    build_challenge_catalog_row,
    is_first_run_of_local_date,
    run_availability_audit,
)

LOGS_DIR = PROJECT_ROOT / "logs"


def _run_availability(now_utc: pd.Timestamp, trigger: str) -> None:
    audit_runs_path = LOGS_DIR / "audit_runs.csv"
    local_date = now_utc.tz_convert(LOCAL_TZ).date()
    is_first_run = is_first_run_of_local_date(audit_runs_path, local_date)

    outcome = run_availability_audit(now_utc, trigger, is_first_run_of_day=is_first_run)

    append_csv_rows(outcome.availability_rows, LOGS_DIR / "availability.csv", AVAILABILITY_COLUMNS)
    append_csv_rows([outcome.audit_run_row], audit_runs_path, AUDIT_RUNS_COLUMNS)

    print(f"Availability audit complete: {outcome.audit_run_row}")


def _run_catalog(now_utc: pd.Timestamp) -> None:
    row = build_challenge_catalog_row(now_utc)
    append_csv_rows([row], LOGS_DIR / "challenge_catalog.csv", CHALLENGE_CATALOG_COLUMNS)
    print(f"Challenge catalog snapshot complete: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=["availability", "catalog", "both"])
    args = parser.parse_args()

    now_utc = pd.Timestamp.now(tz="UTC")
    trigger = os.getenv("GITHUB_EVENT_NAME", "manual")

    if args.part in ("availability", "both"):
        _run_availability(now_utc, trigger)
    if args.part in ("catalog", "both"):
        _run_catalog(now_utc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
