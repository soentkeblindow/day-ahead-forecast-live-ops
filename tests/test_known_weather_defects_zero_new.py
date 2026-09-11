"""Local reality check for KNOWN_WEATHER_DEFECTS (data/weather_grid.py):
validate_historical_weather_runs() must report zero NEW weather defects
against the real, currently-cached weather bundle -- the standing proof
that the list and reality agree (docs/sprint6_auftrag_known_data_defects.md
§7). Needs the full local weather cache (gitignored, ~900 files) built by
scripts/build_weather_artefact.py -- skipped, not a CI gate, same pattern
as test_feature_integration.py's own real-data smoke tests.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops import store

_WEATHER_CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "weather_single_runs"

_needs_real_weather_cache = pytest.mark.skipif(
    not _WEATHER_CACHE_ROOT.exists() or not any(_WEATHER_CACHE_ROOT.rglob("*.parquet")),
    reason="local weather cache not present -- local reality check, not a CI gate",
)


@_needs_real_weather_cache
def test_validate_historical_weather_runs_reports_zero_new_defects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="energy_price_forecast.ops.store"):
        entry, reasons = store.validate_historical_weather_runs(
            PROJECT_ROOT, as_of=pd.Timestamp.now(tz="UTC")
        )

    assert entry is not None, f"unexpected validation failure: {reasons}"
    new_defect_warnings = [
        r.message
        for r in caplog.records
        if r.message.startswith("New weather defect")
        or r.message.startswith("New missing weather run")
    ]
    assert new_defect_warnings == [], (
        "KNOWN_WEATHER_DEFECTS no longer matches reality -- new, unexplained defect(s) found: "
        f"{new_defect_warnings}"
    )
