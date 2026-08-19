"""Thin CLI for the daily raw-series availability audit.

Logic lives in energy_price_forecast.ops.availability_audit so it stays
testable; this script only wires real time, real env, and real files
together (spec §9).
"""

from __future__ import annotations

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
    is_first_run_of_local_date,
    run_audit,
)

LOGS_DIR = PROJECT_ROOT / "logs"


def main() -> int:
    now_utc = pd.Timestamp.now(tz="UTC")
    trigger = os.getenv("GITHUB_EVENT_NAME", "manual")

    audit_runs_path = LOGS_DIR / "audit_runs.csv"
    local_date = now_utc.tz_convert(LOCAL_TZ).date()
    is_first_run = is_first_run_of_local_date(audit_runs_path, local_date)

    outcome = run_audit(now_utc, trigger, is_first_run_of_day=is_first_run)

    append_csv_rows(outcome.availability_rows, LOGS_DIR / "availability.csv", AVAILABILITY_COLUMNS)
    append_csv_rows(
        [outcome.challenge_catalog_row],
        LOGS_DIR / "challenge_catalog.csv",
        CHALLENGE_CATALOG_COLUMNS,
    )
    append_csv_rows([outcome.audit_run_row], audit_runs_path, AUDIT_RUNS_COLUMNS)

    print(f"Audit run complete: {outcome.audit_run_row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
