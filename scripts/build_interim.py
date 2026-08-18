"""Build the hourly interim Parquet file from raw data.

This script is a thin wrapper around `build_interim_hourly()` from the data layer.
It ensures the interim file exists before feature building (Sprint 2.3.3).

Usage:
    uv run python scripts/build_interim.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--out PATH]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.loaders import build_interim_hourly

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly interim data from raw data.")
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date (UTC) in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end", type=str, default="2025-12-31", help="End date (UTC) in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/hourly.parquet"),
        help="Output path for the interim Parquet file.",
    )
    args = parser.parse_args()

    # Fail fast if raw data is missing
    raw_dir = Path("data/raw")
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"Raw data not found in {raw_dir}. "
            "Run the ingest step first (e.g., via notebooks or scripts)."
        )

    logger.info("Building interim hourly data from %s to %s", args.start, args.end)
    df = build_interim_hourly(
        start=pd.Timestamp(args.start, tz="UTC"),
        end=pd.Timestamp(args.end, tz="UTC"),
        path=args.out,
    )

    # Log key invariants
    logger.info("Interim data built successfully:")
    logger.info("  - Rows: %d", len(df))
    logger.info("  - Columns: %d", len(df.columns))
    logger.info("  - Index start: %s", df.index.min())
    logger.info("  - Index end: %s", df.index.max())
    logger.info("  - Timezone: %s", df.index.tz)

    # Check for regular hourly grid (2.1 invariant)
    expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
    if len(df) != expected_hours:
        logger.warning(
            "Irregular hourly grid detected: %d rows for %d expected hours", len(df), expected_hours
        )


if __name__ == "__main__":
    main()
