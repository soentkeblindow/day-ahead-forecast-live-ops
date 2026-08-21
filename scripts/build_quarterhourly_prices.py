"""Build the native quarter-hourly day-ahead price artefact.

Usage:
    uv run python scripts/build_quarterhourly_prices.py [--end YYYY-MM-DD]

Defaults --end to two days from now: day-ahead prices are auction-clearing
results published a day ahead of delivery, so "now" would silently truncate
already-published prices for (local) tomorrow. See
energy_price_forecast.data.quarterhourly.build_quarterhourly_prices.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.quarterhourly import QUARTERHOUR_START, build_quarterhourly_prices

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the native quarter-hourly day-ahead price artefact."
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (UTC) in YYYY-MM-DD format. Defaults to now + 2 days.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/quarterhourly_prices.parquet"),
        help="Output path for the quarter-hourly Parquet file.",
    )
    args = parser.parse_args()

    end = (
        pd.Timestamp(args.end, tz="UTC")
        if args.end
        else pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)
    )

    logger.info("Building native quarter-hourly prices from %s to %s", QUARTERHOUR_START, end)
    df = build_quarterhourly_prices(end=end, path=args.out)

    logger.info("Quarter-hourly prices built successfully:")
    logger.info("  - Rows: %d", len(df))
    logger.info("  - Index start: %s", df.index.min())
    logger.info("  - Index end: %s", df.index.max())


if __name__ == "__main__":
    main()
