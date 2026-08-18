"""Build and persist the feature matrix (Sprint 2.3.3).

Usage:
    uv run python scripts/build_features.py [--data-path PATH] [--out PATH]
"""

import argparse
import logging
from pathlib import Path

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.features.build import build_feature_matrix, trim_warmup

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist the feature matrix.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/interim/hourly.parquet"),
        help="Path to the interim hourly Parquet file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Output path for the feature matrix.",
    )
    args = parser.parse_args()

    # Fail fast if interim file is missing
    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Interim file not found at {args.data_path}. Run scripts/build_interim.py first."
        )

    logger.info("Loading interim data from %s", args.data_path)
    df = load_interim_hourly(args.data_path)

    logger.info("Building feature matrix")
    matrix = build_feature_matrix(df)

    logger.info("Trimming warm-up rows")
    matrix = trim_warmup(matrix)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.out)
    logger.info("Feature matrix written to %s", args.out)

    # Log summary for sanity checks
    logger.info("Feature matrix summary:")
    logger.info("  - Rows: %d", len(matrix))
    logger.info("  - Columns: %d", len(matrix.columns))
    logger.info("  - Index start: %s", matrix.index.min())
    logger.info("  - Index end: %s", matrix.index.max())

    # Log NaN counts per column (EUA NaN region should survive)
    nan_counts = matrix.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        logger.info("  - No NaN values in any column")
    else:
        for col, count in nan_cols.items():
            pct = 100 * count / len(matrix)
            logger.info("  - %s: %d NaN (%.1f%%)", col, count, pct)


if __name__ == "__main__":
    main()
