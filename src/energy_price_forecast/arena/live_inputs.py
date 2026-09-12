"""Store-raw-data assembly for the daily submission job (spec 6.7.2,
section 5.8 -- owner-approved module placement, 2026-09-12, decided via
Rueckfrage since data/ must stay untouched and this logic has no other
named home in spec section 3.2).

Both functions here are the disk-only equivalents of data/loaders.py's two
merged artefact loaders, sourced from the store's raw per-source cache
files instead of a static pre-built artefact -- neither calls a fetch_*
function or touches the network (spec section 4, rule 4: "the job calls no
fetch client, it reads files"). Every real fetch_* call would also risk a
disk write, since data/_entsoe_cache.py::cached_fetch never treats the
still-open calendar month as a cache hit.

- assemble_price_model_inputs() mirrors data/loaders.py::load_interim_hourly()
  (itself load_all_data() + normalize.to_hourly()) -- reads every ENTSO-E
  source via ops.store.read_cached_range() and both commodities via a plain
  whole-file parquet read.
- read_weather_runs() mirrors data/loaders.py::load_interim_weather() --
  concatenates the individual per-run cache files data/_weather_cache.py
  already writes, via its own pure cache_path()/read_cached_run(), never
  weather_client.fetch_run() (which goes to the network on a cache miss).
- read_quarterhourly_prices() mirrors data/quarterhourly.py::build_quarterhourly_prices()
  (the native, unresampled quarter-hourly series the shape profile needs,
  spec section 5.6) -- same QUARTERHOUR_START filter and dedup, but reads
  the day_ahead_price cache dir via ops.store.read_cached_range() instead
  of calling fetch_day_ahead_prices().

target_hourly for evaluation/renewables_walkforward.py::run_renewables_backtest
is deliberately NOT a separate reader here: scripts/train_renewables.py
derives it as ``hourly[list(TARGET_COLUMNS.values())]`` from the same
merged, to_hourly()-normalised frame load_interim_hourly() returns -- a raw,
un-normalised read of the wind_solar cache dir would silently diverge from
that (to_hourly() time-averages every MW column onto the hourly grid).
Callers must slice assemble_price_model_inputs()'s own output the same way,
never read wind_solar separately, or the live path would train the
renewables model on differently-aggregated data than 6.5.2/6.6 measured.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from energy_price_forecast.data._weather_cache import CACHE_ROOT, cache_path, read_cached_run
from energy_price_forecast.data.loaders import COMMODITY_FFILL_LIMIT, time_limited_ffill
from energy_price_forecast.data.normalize import to_hourly
from energy_price_forecast.data.quarterhourly import QUARTERHOUR_START
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.ops import store
from energy_price_forecast.ops.store_sources import (
    COMMODITIES_DIR,
    COMMODITY_SOURCES,
    ENTSOE_SOURCES,
    EntsoeSource,
    RowFetchFn,
)

DEFAULT_WEATHER_MODEL = "ecmwf_ifs"


def assemble_price_model_inputs(
    *,
    entsoe_sources: tuple[EntsoeSource, ...] = ENTSOE_SOURCES,
    commodity_sources: tuple[tuple[str, RowFetchFn, str], ...] = COMMODITY_SOURCES,
    commodities_dir: Path = COMMODITIES_DIR,
) -> pd.DataFrame:
    """The full merged, hourly-normalised price-model input frame, read
    purely from the store's on-disk raw cache (spec section 4, rule 4).

    Replicates data/loaders.py::load_all_data's body (outer join of all
    six ENTSO-E sources plus both commodities, commodity forward-fill,
    drop rows without day_ahead_price) followed by
    data/normalize.py::to_hourly -- exactly the two steps
    build_interim_hourly chains, so this function's output has the same
    shape load_interim_hourly() returns. Neither data/loaders.py nor
    data/normalize.py is modified; both are imported and called unchanged
    (spec section 3.3).

    Every ENTSO-E source is read via ops.store.read_cached_range() (no
    date bounds: the full available history, matching how the historical
    interim artefact was itself built over the whole range at once) --
    never a fetch_* function. Both commodity sources are read as whole
    single-file parquet reads (ops/store_sources.py::COMMODITIES_DIR),
    since scripts/sync_store.py keeps one undivided file per commodity,
    unlike the ENTSO-E sources' month-chunked cache. A missing commodity
    file contributes an empty, correctly-named column, mirroring
    load_all_data's own fetch-failure fallback shape.
    """
    frames = [store.read_cached_range(source.cache_dir) for source in entsoe_sources]

    for name, _fetch, column in commodity_sources:
        path = commodities_dir / f"{name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            # A DatetimeIndex (not the default RangeIndex an empty
            # pd.DataFrame(columns=[...]) would carry) and an explicit float
            # dtype: pd.concat's outer join needs the former to keep the
            # merged index a DatetimeIndex (required by normalize.to_hourly
            # below), and the latter avoids a silent object-dtype column
            # that ffill would then warn about downcasting.
            frames.append(
                pd.DataFrame(
                    {column: pd.Series(dtype="float64")}, index=pd.DatetimeIndex([], tz="UTC")
                )
            )

    merged = pd.concat(frames, axis=1, join="outer")

    for _name, _fetch, column in commodity_sources:
        merged[column] = time_limited_ffill(merged[column], limit_hours=COMMODITY_FFILL_LIMIT)

    merged = merged[merged["day_ahead_price"].notna()]
    return to_hourly(merged)


def read_weather_runs(
    target_days: Iterable[dt.date],
    *,
    model: str = DEFAULT_WEATHER_MODEL,
    root: Path = CACHE_ROOT,
) -> pd.DataFrame:
    """The weather reconstruction input for every ``target_day`` given,
    read purely from the per-run cache (spec section 4, rule 4).

    Maps each target_day to its one allowed run
    (data/weather_client.py::run_init_for_target_day) and reads that run's
    cache file directly (data/_weather_cache.py::cache_path/read_cached_run
    -- never weather_client.fetch_run(), which goes to the network on a
    cache miss). A target_day whose run is not cached is silently omitted,
    not an error here: features/nwp_fundamentals.py's whole-day-out policy
    (spec 6.5.3) is what turns a missing day into a skipped one downstream,
    and arena/preflight.py::check_reconstruction_inputs is what turns a
    missing *target* day's run into a silent day at the top of a run --
    this function only assembles what is actually on disk.

    Returns an empty DataFrame if no requested run is cached at all.
    """
    frames = []
    for target_day in target_days:
        run_init = run_init_for_target_day(target_day)
        cached = read_cached_run(cache_path(run_init, model, root=root))
        if cached is not None:
            frames.append(cached)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def read_quarterhourly_prices(
    *,
    entsoe_sources: tuple[EntsoeSource, ...] = ENTSOE_SOURCES,
    start: pd.Timestamp = QUARTERHOUR_START,
) -> pd.DataFrame:
    """The native, unresampled quarter-hourly day-ahead price series the
    shape profile (models/bridge.py::fit_shape_profile) needs (spec section
    5.6), read purely from the store's on-disk raw cache.

    Mirrors data/quarterhourly.py::build_quarterhourly_prices -- same
    ``start`` default (QUARTERHOUR_START, the 2025-09-30 resolution
    cutover), same sort + duplicate-drop -- but reads the day_ahead_price
    cache dir via ops.store.read_cached_range() instead of calling
    fetch_day_ahead_prices() (spec section 4, rule 4). data/quarterhourly.py
    itself is not modified (spec section 3.3).
    """
    source = next(s for s in entsoe_sources if s.name == "day_ahead_price")
    df = store.read_cached_range(source.cache_dir, start=start)
    df = df.sort_index()
    return df[~df.index.duplicated(keep="first")]
