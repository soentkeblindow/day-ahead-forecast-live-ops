"""Extend the hourly interim and feature Parquet files forward in time.

Unlike scripts/build_interim.py and scripts/build_features.py (which write an
arbitrary [start, end] window and would silently overwrite the whole file),
this script only fetches timestamps after the last existing row, appends
them, and verifies the pre-existing rows are byte-for-byte unchanged before
writing anything. See docs/sprint6_step6_4_spec.md section 2.4 (regime 1):
hourly.parquet and features.parquet may be extended forward but never
rebuilt from scratch, because ENTSO-E revises historical series and a
rebuild would invalidate the sprint 6.1 provenance proof.

Usage:
    uv run python scripts/extend_interim.py [--end YYYY-MM-DD]
"""

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.entsoe_client import AREA_DE_LU
from energy_price_forecast.data.loaders import load_all_data, load_interim_hourly
from energy_price_forecast.data.normalize import to_hourly
from energy_price_forecast.features.build import build_feature_matrix, trim_warmup
from energy_price_forecast.features.config import FeatureConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_INTERIM_PATH = Path("data/interim/hourly.parquet")
_FEATURES_PATH = Path("data/processed/features.parquet")

# Longest lookback any feature reaches back (rolling-mean edge). Pre-existing
# feature rows within this many hours of the old boundary may legitimately
# change on rebuild -- they previously drew on an incomplete rolling window.
_MAX_LOOKBACK_HOURS = FeatureConfig().max_lookback_hours()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backup(path: Path) -> Path:
    backup_path = path.with_name(path.name + ".pre_6_4.bak")
    shutil.copy2(path, backup_path)
    logger.info("Backed up %s to %s (sha256=%s)", path, backup_path, _sha256(path))
    return backup_path


def extend_interim_hourly(
    path: Path,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    """Fetch and append only rows after the current last timestamp.

    Raises if the pre-existing rows are not exactly reproduced (equals,
    including NaN pattern) after the append -- never overwrites on failure.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- nothing to extend.")

    existing = load_interim_hourly(path)
    last_ts = pd.Timestamp(existing.index.max())
    new_start = last_ts + pd.Timedelta(hours=1)

    if new_start >= end:
        logger.info(
            "Nothing to extend: last existing row %s is already at or past requested end %s",
            last_ts,
            end,
        )
        return existing

    _backup(path)

    raw_new = load_all_data(new_start, end, area)
    new_hourly = to_hourly(raw_new)

    combined = pd.concat([existing, new_hourly]).sort_index()
    if combined.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps after concatenating existing and new rows -- "
            "the fetch window overlapped the existing data unexpectedly."
        )

    overlap = combined.loc[existing.index]
    if not overlap.equals(existing):
        diff_cols = [c for c in existing.columns if not overlap[c].equals(existing[c])]
        raise ValueError(
            f"Pre-existing hourly rows changed after extension in columns: {diff_cols}. "
            "Aborting without overwriting -- see spec 6.4 section 5.1 (a1)."
        )

    combined.to_parquet(path)
    logger.info(
        "Extended %s: %d -> %d rows (%s -> %s)",
        path,
        len(existing),
        len(combined),
        combined.index.min(),
        combined.index.max(),
    )
    return combined


def extend_features(
    interim_path: Path,
    features_path: Path,
) -> pd.DataFrame:
    """Rebuild the feature matrix from the (now-extended) interim data.

    The feature pipeline is a deterministic function of the interim data, so
    pre-existing feature rows must reproduce exactly -- except within
    _MAX_LOOKBACK_HOURS of the old boundary, where a legitimately incomplete
    rolling window at the time of the old build can now complete itself.
    """
    hourly = load_interim_hourly(interim_path)
    rebuilt = trim_warmup(build_feature_matrix(hourly))

    if not features_path.exists():
        features_path.parent.mkdir(parents=True, exist_ok=True)
        rebuilt.to_parquet(features_path)
        logger.info("No pre-existing features.parquet -- wrote %d rows fresh.", len(rebuilt))
        return rebuilt

    existing = pd.read_parquet(features_path)
    _backup(features_path)

    old_boundary = pd.Timestamp(existing.index.max())
    edge_cutoff = old_boundary - pd.Timedelta(hours=_MAX_LOOKBACK_HOURS)
    common_idx = existing.index.intersection(rebuilt.index)
    stable_idx = common_idx[common_idx <= edge_cutoff]
    edge_idx = common_idx[common_idx > edge_cutoff]

    stable_old = existing.loc[stable_idx]
    stable_new = rebuilt.loc[stable_idx]
    if not stable_new.equals(stable_old):
        diff_cols = [c for c in existing.columns if not stable_new[c].equals(stable_old[c])]
        raise ValueError(
            f"Pre-existing feature rows changed outside the {_MAX_LOOKBACK_HOURS}h edge "
            f"window in columns: {diff_cols}. Aborting without overwriting -- "
            "see spec 6.4 section 5.1 (a1)."
        )

    if not edge_idx.empty:
        n_changed = (~rebuilt.loc[edge_idx].eq(existing.loc[edge_idx]).all(axis=1)).sum()
        logger.info(
            "%d of %d rows in the last %dh before the old boundary (%s) changed on rebuild "
            "-- expected rolling-window edge effect, not a finding.",
            n_changed,
            len(edge_idx),
            _MAX_LOOKBACK_HOURS,
            old_boundary,
        )

    rebuilt.to_parquet(features_path)
    logger.info(
        "Extended %s: %d -> %d rows (%s -> %s)",
        features_path,
        len(existing),
        len(rebuilt),
        rebuilt.index.min(),
        rebuilt.index.max(),
    )
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend hourly interim and feature Parquet files forward, without rewriting history."
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (UTC) in YYYY-MM-DD format. Defaults to now.",
    )
    parser.add_argument("--interim-path", type=Path, default=_INTERIM_PATH)
    parser.add_argument("--features-path", type=Path, default=_FEATURES_PATH)
    args = parser.parse_args()

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")

    logger.info("Extending %s to %s", args.interim_path, end)
    extend_interim_hourly(args.interim_path, end)

    logger.info("Extending %s from the now-extended interim data", args.features_path)
    extend_features(args.interim_path, args.features_path)


if __name__ == "__main__":
    main()
