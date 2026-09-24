"""Sprint 6.9, Schritt 6, §6.2: one-time backfill of Energy-Charts price and
load-forecast history into the permanent store.

Reads the Sprint 6.8 bulk-fetch artifacts (``data/raw/energy_charts/price/``,
``data/raw/energy_charts/public_power_forecast/``, from scripts/
fetch_energy_charts_price_history.py and scripts/fetch_energy_charts_
forecast_history.py), extends each to ``as_of`` via a live fetch of just
the gap since those artifacts' own last-cached day (merged with the
existing-wins ``merge_existing_with_fresh``, spec section 5.8's "Muster
_merge_fresh"), writes into the store's own single-file Energy-Charts cache
(``ops/store_sources.py::ENERGY_CHARTS_DIR``), and publishes via the normal
path.

Never re-derives ``ENERGY_CHARTS_SOURCES``' own (name, fetch, column)
bindings -- reuses the exact same tuple scripts/sync_store.py's ongoing
maintenance runs use, so the one-time backfill and the daily top-up can
never silently diverge on which client function or column name backs a
source.

Also reproduces the price-identity re-check the whole 6.8 outage plan
already established once (docs/sprint6_step6_8_log.md's own "exact_fraction
= 1.0000" finding): now against the STORE's fresh EC copy specifically, not
the raw artifact -- an "erneuter Preis-Abgleich" (spec section 6.2), a
different measurement even though the expected answer is the same.

Single ``store.load_store()`` call for the whole operation (never twice
around a patch-then-publish sequence -- see docs/feedback_typechecking.md's
sibling lesson in this codebase's own dev-workflow notes: load_store()
overwrites on-disk files, not just the manifest).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import pandas as pd

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.energy_charts import merge_existing_with_fresh
from energy_price_forecast.ops import store
from energy_price_forecast.ops.store_sources import (
    ENERGY_CHARTS_DIR,
    ENERGY_CHARTS_SOURCES,
    ENTSOE_SOURCES,
    code_sha,
    run_id_and_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PRICE_RAW_DIR = DATA_RAW / "energy_charts" / "price"
_LOAD_RAW_DIR = DATA_RAW / "energy_charts" / "public_power_forecast"
_EC_LOAD_FORECAST_COLUMN = "load_forecast_day_ahead_ec"


def _read_price_history() -> pd.DataFrame:
    """Sprint 6.8 Teil 0's own monthly price artifact, concatenated. Column
    already named ``day_ahead_price_ec`` (fetch_energy_charts_price_
    history.py's own PRICE_COLUMN, unchanged since this repo's Schritt 4
    move)."""
    files = sorted(_PRICE_RAW_DIR.glob("DE_LU_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no price history under {_PRICE_RAW_DIR} -- run "
            "scripts/fetch_energy_charts_price_history.py first"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(f"price history: {dupe_count} duplicate timestamp(s) across raw files")
    return frame


def _read_load_forecast_history() -> pd.DataFrame:
    """Sprint 6.8 Teil 1's own monthly load-forecast artifact, concatenated
    and renamed from the raw ``load`` column (named after the Energy-Charts
    production_type, scripts/fetch_energy_charts_forecast_history.py) to
    the store's own ``load_forecast_day_ahead_ec``."""
    files = sorted(_LOAD_RAW_DIR.glob("load_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no load-forecast history under {_LOAD_RAW_DIR} -- run "
            "scripts/fetch_energy_charts_forecast_history.py first"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(
            f"load-forecast history: {dupe_count} duplicate timestamp(s) across raw files"
        )
    return frame.rename(columns={"load": _EC_LOAD_FORECAST_COLUMN})


_RAW_HISTORY_READERS: dict[str, Callable[[], pd.DataFrame]] = {
    "day_ahead_price_ec": _read_price_history,
    "load_forecast_day_ahead_ec": _read_load_forecast_history,
}


def _extend_and_write(
    name: str,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    column: str,
    *,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reads the matching 6.8 raw artifact, fetches just the gap through
    ``as_of``, merges (existing wins, verified), writes the store's own
    single-file cache, and returns (merged frame, a small report dict)."""
    history = _RAW_HISTORY_READERS[name]()
    gap_start = history.index.max()
    fresh = fetch(gap_start, as_of)
    merged = merge_existing_with_fresh(history, fresh[column], column=column)

    path = ENERGY_CHARTS_DIR / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, compression="snappy")

    report = {
        "name": name,
        "n_rows": len(merged),
        "covered_start_utc": merged.index.min().isoformat(),
        "covered_end_utc": merged.index.max().isoformat(),
        "raw_artifact_rows": len(history),
        "raw_artifact_max": history.index.max().isoformat(),
        "n_new_rows": len(merged) - len(history),
    }
    logger.info(
        "%s: %d row(s) total (%d new since the raw artifact's own %s), covering %s..%s",
        name,
        report["n_rows"],
        report["n_new_rows"],
        report["raw_artifact_max"],
        report["covered_start_utc"],
        report["covered_end_utc"],
    )
    return merged, report


def _price_identity_check(ec_price: pd.Series) -> dict[str, object]:
    """Re-checks the 6.8 outage plan's own foundational assumption ("EC
    price == ENTSO-E price") against the store's ENTSO-E day_ahead_price,
    now that the store has its own fresh EC copy (spec section 6.2: "ein
    erneuter Preis-Abgleich gegen ENTSO-E im Speicher"). Expected
    exact_fraction = 1.0000, per docs/sprint6_step6_8_log.md's own already-
    measured finding over the raw artifact -- this measures the STORE's
    copy specifically, a different (if expectedly identical) check."""
    price_source = next(s for s in ENTSOE_SOURCES if s.name == "day_ahead_price")
    entsoe_price = store.read_cached_range(price_source.cache_dir)["day_ahead_price"]

    common = ec_price.index.intersection(entsoe_price.index)
    pair = pd.DataFrame({"ec": ec_price.loc[common], "store": entsoe_price.loc[common]}).dropna()
    diff = pair["ec"] - pair["store"]
    result: dict[str, object] = {
        "n": int(len(pair)),
        "exact_fraction": float((pair["ec"] == pair["store"]).mean())
        if len(pair)
        else float("nan"),
        "diff_max_abs": float(diff.abs().max()) if len(diff) else float("nan"),
    }
    logger.info("price identity re-check (store EC vs. store ENTSO-E): %s", result)
    return result


def run_backfill(*, as_of: pd.Timestamp | None = None) -> dict[str, object]:
    resolved_as_of = as_of or pd.Timestamp.now(tz="UTC")

    state = store.load_store(PROJECT_ROOT)
    manifest = state.manifest

    reports: list[dict[str, object]] = []
    new_sources = dict(manifest.sources)
    merged_by_name: dict[str, pd.DataFrame] = {}

    for name, fetch, column in ENERGY_CHARTS_SOURCES:
        merged, report = _extend_and_write(name, fetch, column, as_of=resolved_as_of)
        reports.append(report)
        merged_by_name[name] = merged
        new_sources[name] = store.SourceManifestEntry(
            covered_start_utc=merged.index.min().isoformat(),
            covered_end_utc=merged.index.max().isoformat(),
            count=len(merged),
            last_success_utc=resolved_as_of.isoformat(),
            last_attempt_utc=resolved_as_of.isoformat(),
        )

    identity_check = _price_identity_check(
        merged_by_name["day_ahead_price_ec"]["day_ahead_price_ec"]
    )
    if identity_check["exact_fraction"] != 1.0:
        raise ValueError(
            f"price identity check failed: exact_fraction={identity_check['exact_fraction']!r}, "
            "expected 1.0 -- see docs/sprint6_step6_8_log.md for the original measurement this "
            "was supposed to reproduce; a Rückfrage is warranted before publishing"
        )

    run_id, run_url = run_id_and_url()
    new_manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc=resolved_as_of.isoformat(),
        run_id=f"ec-backfill-{run_id}",
        run_url=run_url,
        code_sha=code_sha(),
        sources=new_sources,
    )
    asset = store.publish_store(PROJECT_ROOT, new_manifest)

    summary: dict[str, object] = {
        "as_of": resolved_as_of.isoformat(),
        "sources": reports,
        "price_identity_check": identity_check,
        "published_bytes": asset.size_bytes,
    }
    logger.info("backfill summary: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        type=lambda s: pd.Timestamp(s, tz="UTC"),
        default=None,
        help="override as_of (UTC), default now -- for reproducible manual testing",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_backfill(as_of=args.as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
