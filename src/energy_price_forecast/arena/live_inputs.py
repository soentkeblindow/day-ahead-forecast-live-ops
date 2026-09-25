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
    ENERGY_CHARTS_DIR,
    ENTSOE_SOURCES,
    EntsoeSource,
    RowFetchFn,
)

DEFAULT_WEATHER_MODEL = "ecmwf_ifs"

_PRICE_COLUMN = "day_ahead_price"
_EC_PRICE_COLUMN = "day_ahead_price_ec"
_EC_LOAD_FORECAST_COLUMN = "load_forecast_day_ahead_ec"


def coalesce_price(entsoe: pd.Series, ec: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """ENTSO-E where present, Energy-Charts where ENTSO-E is missing (spec
    6.9 section 5.3).

    Returns (merged series, provenance mask [True = value came from
    Energy-Charts], number of cells where both were present and differed).
    ENTSO-E always wins on a conflict, silently, never raising (owner
    decision 2026-09-23): a changed rounding convention on either side
    must never take the live system down. Operates on whatever native grid
    the two inputs already share (hourly before, quarter-hourly from
    data/quarterhourly.py::QUARTERHOUR_START) -- callers coalesce before
    any further resampling, never after, so every downstream consumer
    (price lags, the price-model label, the shape profile, the persistence
    value) sees the same merged series (spec section 2.4).
    """
    combined = pd.DataFrame({"entsoe": entsoe, "ec": ec})
    merged = combined["entsoe"].combine_first(combined["ec"])
    provenance = combined["entsoe"].isna() & combined["ec"].notna()

    both_present = combined["entsoe"].notna() & combined["ec"].notna()
    n_conflicts = int((both_present & (combined["entsoe"] != combined["ec"])).sum())

    name = str(entsoe.name) if entsoe.name is not None else None
    return merged.rename(name), provenance.rename(name), n_conflicts


def _read_ec_price(*, root: Path = ENERGY_CHARTS_DIR) -> pd.Series:
    """The store's own Energy-Charts price cache (scripts/sync_store.py's
    ``day_ahead_price_ec`` single-file cache, spec section 2.4) -- an empty,
    correctly-named series if the file does not exist yet (mirrors
    assemble_price_model_inputs' own missing-commodity-file fallback)."""
    path = root / "day_ahead_price_ec.parquet"
    if not path.exists():
        return pd.Series(
            dtype="float64", name=_EC_PRICE_COLUMN, index=pd.DatetimeIndex([], tz="UTC")
        )
    return pd.read_parquet(path)[_EC_PRICE_COLUMN]


def _read_ec_load_forecast(*, root: Path = ENERGY_CHARTS_DIR) -> pd.Series:
    """The store's own Energy-Charts load-forecast cache -- same
    empty-fallback discipline as _read_ec_price."""
    path = root / "load_forecast_day_ahead_ec.parquet"
    if not path.exists():
        return pd.Series(
            dtype="float64", name=_EC_LOAD_FORECAST_COLUMN, index=pd.DatetimeIndex([], tz="UTC")
        )
    return pd.read_parquet(path)[_EC_LOAD_FORECAST_COLUMN]


def assemble_price_model_inputs(
    *,
    entsoe_sources: tuple[EntsoeSource, ...] = ENTSOE_SOURCES,
    commodity_sources: tuple[tuple[str, RowFetchFn, str], ...] = COMMODITY_SOURCES,
    commodities_dir: Path = COMMODITIES_DIR,
    energy_charts_dir: Path = ENERGY_CHARTS_DIR,
) -> pd.DataFrame:
    """The full merged, hourly-normalised price-model input frame, read
    purely from the store's on-disk raw cache (spec section 4, rule 4).

    Replicates data/loaders.py::load_all_data's body (outer join of all
    six ENTSO-E sources plus both commodities, commodity forward-fill)
    followed by data/normalize.py::to_hourly -- exactly the two steps
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

    Unlike load_all_data, this function does **not** drop rows without a
    real day_ahead_price (docs/sprint6_fix_partial_today.md section 3.1,
    superseding the narrower, single-day ``keep_rows_for`` exemption from
    docs/sprint6_fix_future_target_day.md). That drop exists in
    load_all_data because the historical backtest always needs a real
    price label; this live assembly step feeds more than one consumer
    (the renewables reconstruction reads its own forecast columns off the
    very same frame, via scripts/run_daily_submission.py::run_renewables_step),
    and completeness is a per-consumer question, not a per-row one. A row
    missing only its price still carries perfectly real
    load_forecast_day_ahead/wind/solar-forecast/scheduled-flow values that
    a price-blind drop would needlessly take down with it -- observed live
    2026-09-13 for both a genuine future delivery day (no price yet, by
    design) and, on the same real day, an unrelated one-off gap in
    *today's own* published price (confirmed a one-off against
    logs/availability.csv: 63 prior daily audit runs all had it, this is
    the first miss). Each consumer now decides for itself what it needs:
    the renewables step reads its own forecast columns regardless of
    price; the price model's own training step
    (scripts/run_daily_submission.py::fit_predict_expand) drops any
    training-window row whose own price is still missing right before fit,
    the same defensive placement already used for renewables labels
    (evaluation/renewables_walkforward.py's own dropna-before-fit).

    For a PAST target day this is bit-identical to the old drop-then-keep
    behaviour: historically every row with any real column value also has
    a real price, so no row this function would have kept before is newly
    dropped, and no row it would have dropped before (all-NaN across every
    source) is newly kept, since pd.concat's outer join never invents an
    index entry with zero source data.

    Energy-Charts, since spec 6.9 section 5.3/2.4 (Schritt 7): the store's
    own ``day_ahead_price_ec`` is coalesced into ``day_ahead_price`` via
    coalesce_price() -- ENTSO-E wins where present, EC only fills a real
    gap -- BEFORE to_hourly() runs, so every hourly consumer of this
    column (price lags, the price-model label) sees the already-merged
    series, not a second, separate one. The raw ``day_ahead_price_ec``
    column itself is dropped afterwards -- it has fully done its job by
    feeding the coalesce, and leaving it in would make to_hourly() treat
    it as an ordinary MW column to time-average, which it structurally is
    not. ``load_forecast_day_ahead_ec`` is deliberately NOT coalesced with
    ENTSO-E's own ``load_forecast_day_ahead`` (spec section 2.4: "Zeile 1
    liest nur ENTSO-E, Zeile 2 nur EC") -- it survives as its own column
    for whichever fallback-ladder row (Schritt 10) reads it directly.
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

    frames.append(_read_ec_price(root=energy_charts_dir).to_frame())
    frames.append(_read_ec_load_forecast(root=energy_charts_dir).to_frame())

    merged = pd.concat(frames, axis=1, join="outer")

    for _name, _fetch, column in commodity_sources:
        merged[column] = time_limited_ffill(merged[column], limit_hours=COMMODITY_FFILL_LIMIT)

    coalesced_price, _provenance, _n_conflicts = coalesce_price(
        merged[_PRICE_COLUMN], merged[_EC_PRICE_COLUMN]
    )
    merged[_PRICE_COLUMN] = coalesced_price
    merged = merged.drop(columns=[_EC_PRICE_COLUMN])

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
    and arena/preflight.py::check_weather_run is what turns a
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
    energy_charts_dir: Path = ENERGY_CHARTS_DIR,
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

    Coalesced with Energy-Charts (spec 6.9 section 5.3/2.4, Schritt 7) via
    the same coalesce_price() assemble_price_model_inputs() uses -- the
    shape profile is one of the four named price consumers that must see
    the merged series, not the ENTSO-E-only one.
    """
    source = next(s for s in entsoe_sources if s.name == "day_ahead_price")
    df = store.read_cached_range(source.cache_dir, start=start)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    ec_price = _read_ec_price(root=energy_charts_dir)
    ec_price = ec_price[ec_price.index >= start]
    # Built directly from the coalesced series, not assigned back into df's
    # own index -- coalesce_price's result can have MORE rows than df alone
    # (a slot ENTSO-E is missing entirely, not just NaN within an existing
    # row), and a real gap-fill must add that row, not silently drop it.
    coalesced_price, _provenance, _n_conflicts = coalesce_price(df[_PRICE_COLUMN], ec_price)
    return coalesced_price.sort_index().to_frame()


# Matches the Shape Profile's own established default window
# (models/bridge.py::fit_shape_profile's own n_days, Sprint 6.4/6.8) --
# purely for this report's own "did EC feed the shape window" question,
# never passed into fit_shape_profile itself.
_SHAPE_WINDOW_DAYS = 28


def price_provenance_report(
    *,
    entsoe_sources: tuple[EntsoeSource, ...] = ENTSOE_SOURCES,
    energy_charts_dir: Path = ENERGY_CHARTS_DIR,
    as_of: pd.Timestamp,
) -> tuple[dict[str, str], int]:
    """Per-consumer price provenance and the overall conflict count (spec
    6.9 section 5.7's protocol_version 4 ``price_provenance``/
    ``price_source_conflicts`` fields).

    Reads the same two underlying series assemble_price_model_inputs()/
    read_quarterhourly_prices() each coalesce internally, independently --
    so this can never diverge from them on WHICH cells get merged, only on
    when it happens to run -- purely to report, for each of the four named
    price consumers (spec section 5.7: ``training_labels``, ``price_lags``,
    ``shape_window``, ``persistence``), whether at least one
    Energy-Charts-sourced cell fell inside that consumer's own relevant
    window.

    ``training_labels``/``price_lags``/``persistence`` all ultimately read
    from the same trained-on price history -- reported here over the FULL
    available window as a deliberately conservative, over-inclusive proxy
    for whichever exact training window Schritt 8/10 end up using (reporting
    "energy_charts" too eagerly is the safe direction; reporting it too
    rarely is not). ``shape_window`` uses the last _SHAPE_WINDOW_DAYS days
    before ``as_of``, matching the Shape Profile's own established default.
    """
    source = next(s for s in entsoe_sources if s.name == "day_ahead_price")
    entsoe_price = store.read_cached_range(source.cache_dir)[_PRICE_COLUMN]
    ec_price = _read_ec_price(root=energy_charts_dir)

    _merged, provenance, n_conflicts = coalesce_price(entsoe_price, ec_price)

    shape_window_start = as_of - pd.Timedelta(days=_SHAPE_WINDOW_DAYS)
    shape_provenance = provenance[provenance.index >= shape_window_start]

    def _label(mask: pd.Series) -> str:
        return "energy_charts" if bool(mask.any()) else "entsoe"

    report = {
        "training_labels": _label(provenance),
        "price_lags": _label(provenance),
        "shape_window": _label(shape_provenance),
        "persistence": _label(provenance),
    }
    return report, n_conflicts
