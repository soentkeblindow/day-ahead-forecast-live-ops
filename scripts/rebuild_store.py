"""Rebuild the persistent data store from zero (spec 6.7.1, section 5.6) --
the store's emergency exit. Without this script a damaged or lost store is
a project standstill, which is the only reason it exists; it runs
manually, never on a schedule.

Fetches every source over its full range, validates, packs, uploads.
Resumable by construction: data/entsoe_client.py's cached_fetch() writes
each month's cache file as it succeeds and skips months already cached on
a re-run, so a re-invocation after a hard stop below picks up roughly
where it left off without re-fetching months already on disk.

Hard stop on a detected rate limit or five consecutive per-source
failures (spec section 5.6, mirrors the same rule scripts/
build_weather_artefact.py already applies per-run for the weather bulk
fetch, spec 6.5.1 section 2.6) -- reports how far it got and lets the
owner decide, no automatic retry past that.

    uv run python scripts/rebuild_store.py --from 2020-01-01

No test for this script (glue), consistent with project practice
(scripts/sync_store.py, scripts/build_weather_artefact.py).
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops import store
from energy_price_forecast.ops.store_sources import (
    COMMODITIES_DIR,
    COMMODITY_SOURCES,
    ENTSOE_SOURCES,
    code_sha,
    run_id_and_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 5
# ENTSO-E's earliest reliably available day-ahead history for DE_LU
# (CLAUDE.md, README "Provenance" section) -- not derived from code, since
# no constant for it exists anywhere in the untouched source-repo code.
_DEFAULT_FROM = "2020-01-01"


def _is_rate_limited(exc: Exception) -> bool:
    """Best-effort 429 detection across entsoe-py's own exception surface,
    which -- unlike WeatherRunUnavailable -- does not carry a structured
    HTTP status code. String-matching is the only signal available without
    changing data/entsoe_client.py (spec section 3.3 leaves it untouched)."""
    return "429" in str(exc) or "Too Many Requests" in str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_date", default=_DEFAULT_FROM)
    args = parser.parse_args()

    as_of = pd.Timestamp.now(tz="UTC")
    from_ts = pd.Timestamp(args.from_date, tz="UTC")

    new_sources: dict[str, store.SourceManifestEntry] = {}
    consecutive_failures = 0

    for source in ENTSOE_SOURCES:
        logger.info("Fetching %r from %s to %s ...", source.name, from_ts.date(), as_of.date())
        try:
            frame = source.fetch(from_ts, as_of)
        except Exception as exc:  # noqa: BLE001 -- exact type varies across entsoe-py call sites
            consecutive_failures += 1
            logger.error("Source %r failed: %s", source.name, exc)
            if _is_rate_limited(exc):
                raise RuntimeError(
                    f"Rate limited while fetching {source.name!r} -- hard stop per spec section "
                    "5.6. Re-run this script later: already-cached months are skipped."
                ) from exc
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{_MAX_CONSECUTIVE_FAILURES} consecutive source failures, most recently "
                    f"{source.name!r} -- hard stop per spec section 5.6."
                ) from exc
            continue

        consecutive_failures = 0
        expectation = store.EXPECTATION_TABLE[source.name]
        result = store.validate_source(
            source.name,
            frame,
            expectation,
            period_start=from_ts,
            period_end=as_of.floor("D") + pd.Timedelta(days=1),
            as_of=as_of,
            previous=None,
        )
        if not result.ok:
            logger.error("Source %r failed validation: %s", source.name, result.reasons)
            continue

        new_sources[source.name] = store.SourceManifestEntry(
            covered_start_utc=frame.index.min().isoformat(),
            covered_end_utc=frame.index.max().isoformat(),
            count=len(frame),
            last_success_utc=as_of.isoformat(),
            last_attempt_utc=as_of.isoformat(),
        )
        logger.info("Source %r: %d rows, validated OK.", source.name, len(frame))

    for name, fetch, _column in COMMODITY_SOURCES:
        logger.info("Fetching commodity %r from %s to %s ...", name, from_ts.date(), as_of.date())
        try:
            fresh = fetch(from_ts, as_of)
        except Exception as exc:  # noqa: BLE001
            logger.error("Commodity source %r failed: %s", name, exc)
            continue

        expectation = store.EXPECTATION_TABLE[name]
        result = store.validate_source(
            name,
            fresh,
            expectation,
            period_start=from_ts,
            period_end=as_of,
            as_of=as_of,
            previous=None,
        )
        if not result.ok:
            logger.error("Commodity source %r failed validation: %s", name, result.reasons)
            continue

        path = COMMODITIES_DIR / f"{name}.parquet"
        store.write_if_valid(path, fresh, result)
        new_sources[name] = store.SourceManifestEntry(
            covered_start_utc=fresh.index.min().isoformat() if len(fresh) else None,
            covered_end_utc=fresh.index.max().isoformat() if len(fresh) else None,
            count=len(fresh),
            last_success_utc=as_of.isoformat(),
            last_attempt_utc=as_of.isoformat(),
        )
        logger.info("Commodity %r: %d rows, validated OK.", name, len(fresh))

    logger.info(
        "Weather: not re-fetched here -- use scripts/build_weather_artefact.py "
        "for the historical bulk fetch (spec 6.5.1), which already implements the same "
        "hard-stop rule for the per-day weather run loop. Validating whatever is "
        "already cached in data/cache/weather_single_runs/ before packing it."
    )
    weather_entry, weather_reasons = store.validate_historical_weather_runs(PROJECT_ROOT, as_of)
    if weather_entry is None:
        logger.error("Weather run files failed validation, not packed: %s", weather_reasons)
    else:
        new_sources["weather_single_runs"] = weather_entry
        logger.info("Weather: %d cached run(s), validated OK.", weather_entry.count)

    run_id, run_url = run_id_and_url()
    manifest = store.Manifest(
        store_format_version=store.STORE_FORMAT_VERSION,
        created_at_utc=as_of.isoformat(),
        run_id=run_id,
        run_url=run_url,
        code_sha=code_sha(),
        sources=new_sources,
    )

    asset = store.publish_store(PROJECT_ROOT, manifest)
    logger.info("Rebuilt store published: %s (%d bytes)", asset.name, asset.size_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
