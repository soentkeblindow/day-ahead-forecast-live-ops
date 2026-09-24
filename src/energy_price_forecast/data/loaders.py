import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.data.entsoe_client import (
    AREA_DE_LU,
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)
from energy_price_forecast.data.quarterhourly import validate_delivery_day_slot_counts

logger = logging.getLogger(__name__)

_COMMODITY_COLUMNS = ["ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"]
# 14 days = 336 hours (spec 6.9, section 2.11). Raised from the original
# 4-day limit (spec 6.7.2, section 2.3) after measuring every real gap >4
# days in the full commodity history (scripts/run_daily_submission.py's
# caller, arena/live_inputs.py, applies the same limit): TTF gas has one
# 5-day gap (2024-03-28 to 2024-04-02), EUA CO2 has ten, all 5 days
# (clustered around Christmas/New Year and Easter each year). Confirming
# this from the data rather than the owner's stated expectation is
# deliberate -- it matched exactly, but it was measured, not assumed.
# Raised a second time, 7 to 14, per the owner's 6.9 finding that a TTF feed
# dead longer than the ffill limit fails all three fallback-ladder rows at
# once -- the owner's own re-check of the full history (2026-09-23) still
# found no real gap longer than the original 4 days, so this second raise is
# bounded headroom against a future outage, not new evidence of a longer
# one; the feature values themselves stay bit-identical, nothing to
# recompute. Public (not module-private) so the live path can reuse the
# exact same number -- a different limit between backtest and live would be
# the train/serve skew Entscheidung 10 forbids.
COMMODITY_FFILL_LIMIT = 14 * 24
# Kept at the OLD 4-day limit, non-blocking: a gap this deep would have been
# a silent day before this change. Flagged in the submission run's protocol
# and summary (spec 6.7.2, section 2.3) so it stays visible, without
# reintroducing a hard stop for a still-legitimate, if unusually stale, price.
COMMODITY_STALENESS_WARN_DAYS = 4


def time_limited_ffill(series: pd.Series, *, limit_hours: int) -> pd.Series:
    """Forward-fill honoring a real wall-clock time limit, not a row count.

    ``Series.ffill(limit=N)`` counts ROWS, which silently means a
    different real time span depending on the index's own resolution -- a
    real bug found live during the 6.7.2 wiring probe (2026-09-12):
    day_ahead_price (and therefore this merged frame's own index) has
    been quarter-hourly since 2025-09-30 (data/quarterhourly.py::
    QUARTERHOUR_START), so ``COMMODITY_FFILL_LIMIT=168`` covered only 42
    real hours there instead of the intended 7 days -- silently
    contradicting this file's own COMMODITY_FFILL_LIMIT docstring
    (Entscheidung 10: one number, meant identically by every caller,
    including the live path in arena/live_inputs.py, which reuses this
    function directly).

    Correct regardless of the index's own resolution: every NaN is
    filled from the most recent real (non-NaN) value, then reverted to
    NaN wherever that value's own age exceeds ``limit_hours``.
    """
    filled = series.ffill()
    last_valid_time = series.index.to_series().where(series.notna()).ffill()
    age = series.index.to_series() - last_valid_time
    return filled.where(age <= pd.Timedelta(hours=limit_hours))


def _log_fetch(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.info("%s: 0 rows (empty)", name)
    else:
        logger.info("%s: %d rows (%s to %s)", name, len(df), df.index.min(), df.index.max())
    return df


def load_all_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    """Load and merge all data sources into a single hourly DataFrame.

    Calls all six ENTSO-E fetchers and both commodities fetchers, then merges
    them on the hourly UTC index using an outer join. The day-ahead price
    column is required: rows where the price is missing are dropped.

    Commodity prices (TTF gas, EUA CO2) come at daily granularity and are
    forward-filled to hourly resolution with a 14-day limit (raised from 4
    to 7 days, spec 6.7.2 section 2.3, then to 14, spec 6.9 section 2.11 --
    see COMMODITY_FFILL_LIMIT). This bridges
    weekends and the real multi-day gaps measured in the full history but
    leaves genuine long outages visible as NaN for downstream data quality
    analysis.
    This is the only resampling done by the loader; all other forward-fill
    or interpolation decisions are deferred to feature engineering (Sprint 2).

    Parameters
    ----------
    start : pd.Timestamp
        Start of the requested time range. Converted to UTC at entry.
    end : pd.Timestamp
        End of the requested time range (inclusive). Converted to UTC.
    area : str, default "DE_LU"
        ENTSO-E area code. Currently only DE_LU is fully supported; other
        areas would require neighbor-list adjustments in entsoe_client.py.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with UTC DatetimeIndex. Columns include the
        day-ahead price (target), all load and renewable forecasts/actuals,
        all generation types, all six neighbor scheduled exchanges and
        physical flows, plus the two commodity prices forward-filled to
        hourly. Columns missing from individual fetchers are not synthesised.
    """
    start_utc = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    end_utc = end.tz_convert("UTC") if end.tzinfo is not None else end.tz_localize("UTC")

    if start_utc >= end_utc:
        raise ValueError(f"start must be before end, got start={start_utc!r}, end={end_utc!r}")

    frames = [
        _log_fetch(
            "fetch_day_ahead_prices",
            fetch_day_ahead_prices(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_load",
            fetch_load(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_wind_solar_forecast",
            fetch_wind_solar_forecast(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_generation_by_type",
            fetch_generation_by_type(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_scheduled_exchanges",
            fetch_scheduled_exchanges(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_cross_border_flows",
            fetch_cross_border_flows(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_ttf_gas",
            fetch_ttf_gas(start_utc, end_utc),
        ),
        _log_fetch(
            "fetch_eua_co2",
            fetch_eua_co2(start_utc, end_utc),
        ),
    ]

    df = pd.concat(frames, axis=1, join="outer")

    # Forward-fill commodity columns only (daily → hourly granularity).
    # All other gaps remain as NaN so EDA can see real data-quality issues.
    for col in _COMMODITY_COLUMNS:
        if col in df.columns:
            df[col] = time_limited_ffill(df[col], limit_hours=COMMODITY_FFILL_LIMIT)

    # Rows without a target price are unusable for modelling; drop them early
    # so they don't distort predictor gap-visualisation in EDA.
    df = df[df["day_ahead_price"].notna()]

    dt_index = pd.DatetimeIndex(df.index)
    assert dt_index.tz is not None, "merged index must be timezone-aware"
    assert str(dt_index.tz) == "UTC", f"merged index timezone must be UTC, got {dt_index.tz!r}"

    logger.info("merged result: %d rows × %d columns", len(df), len(df.columns))

    return df


# ---------------------------------------------------------------------------
# Interim layer — normalised hourly grid
# ---------------------------------------------------------------------------

_INTERIM_PATH = Path("data/interim/hourly.parquet")

# Timestamp of the resolution break in the raw data (hourly → 15-min).
_BREAK_TS = pd.Timestamp("2025-09-30 22:00", tz="UTC")


def build_interim_hourly(
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path = _INTERIM_PATH,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    """Fetch raw data, normalise to an hourly grid, and persist as Parquet.

    The resulting DataFrame is written to `path` and also returned so the
    caller can inspect it without a second read. If the parent directory does
    not exist it is created automatically.

    After writing, logs daily mean load for the two days on either side of
    the resolution break as a continuity sanity check.
    """
    from energy_price_forecast.data.normalize import to_hourly

    raw = load_all_data(start, end, area)
    hourly = to_hourly(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(path)
    logger.info("wrote %d hourly rows to %s", len(hourly), path)

    # Continuity sanity: daily mean load around the resolution break.
    window_start = _BREAK_TS - pd.Timedelta("2D")
    window_end = _BREAK_TS + pd.Timedelta("2D")
    if "load_actual" in hourly.columns:
        snippet = hourly.loc[window_start:window_end, "load_actual"]
        daily = snippet.resample("D").mean()
        logger.info(
            "load_actual daily mean around resolution break (MW):\n%s",
            daily.to_string(),
        )

    return hourly


def load_interim_hourly(path: Path = _INTERIM_PATH) -> pd.DataFrame:
    """Load the normalised hourly DataFrame from Parquet.

    Raises FileNotFoundError if the file does not exist (run
    build_interim_hourly first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Interim Parquet not found at {path!r}. Run build_interim_hourly() to generate it."
        )
    df = pd.read_parquet(path)
    logger.info("loaded %d hourly rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Processed layer — engineered feature matrix
# ---------------------------------------------------------------------------

_FEATURES_PATH = Path("data/processed/features.parquet")


def load_processed_features(
    path: str | Path = _FEATURES_PATH,
) -> pd.DataFrame:
    """Load the engineered feature matrix (X only, no target) from step 2.3.

    The matrix carries a UTC DatetimeIndex on a regular hourly grid, already
    warm-up-trimmed. NaN is expected only in the EUA-CO2 region (pre-Oct-2021,
    by design D4); model-side imputation happens in the LassoForecaster.

    Fails loudly with a pointer to scripts/build_features.py if the file is
    missing -- never silently return an empty frame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"processed features not found at {path} -- run scripts/build_features.py first"
        )
    df = pd.read_parquet(path)
    logger.info("loaded %d feature rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Interim layer — native quarter-hourly day-ahead prices (sprint 6.4)
# ---------------------------------------------------------------------------

_QUARTERHOURLY_PATH = Path("data/interim/quarterhourly_prices.parquet")


def load_interim_quarterhourly(path: Path = _QUARTERHOURLY_PATH) -> pd.DataFrame:
    """Load the native (unresampled) quarter-hourly day-ahead price series.

    Raises FileNotFoundError if the file does not exist (run
    data.quarterhourly.build_quarterhourly_prices() first). Validates every
    local delivery day's slot count before returning -- see
    quarterhourly.validate_delivery_day_slot_counts; a day with a gap raises
    rather than being silently skipped or interpolated (spec 6.4, §2.6).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Quarter-hourly Parquet not found at {path!r}. "
            "Run data.quarterhourly.build_quarterhourly_prices() to generate it."
        )
    df = pd.read_parquet(path)
    validate_delivery_day_slot_counts(pd.DatetimeIndex(df.index))
    logger.info("loaded %d quarter-hourly rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Interim layer — raw ECMWF IFS weather artefact (sprint 6.5.1)
# ---------------------------------------------------------------------------

_WEATHER_PATH = Path("data/interim/weather_ifs_run00.parquet")


def load_interim_weather(path: Path = _WEATHER_PATH) -> pd.DataFrame:
    """Load the raw weather artefact (scripts/build_weather_artefact.py).

    Raises FileNotFoundError if the file does not exist. Validates the
    artefact's shape -- MultiIndex level names, column count, dtype -- before
    returning, rather than passing a structurally wrong frame downstream
    (spec 6.5.1, §5.7). Does not re-check the exact column set against
    weather_grid.expected_columns(): that's already enforced file-by-file
    when build_weather_artefact.py reads each cached run
    (data/_weather_cache.py::read_cached_run), so a merged artefact with the
    wrong shape here means something outside that pipeline touched the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Weather Parquet not found at {path!r}. "
            "Run scripts/build_weather_artefact.py to generate it."
        )
    df = pd.read_parquet(path)

    index_names = list(df.index.names)
    if index_names != ["run_init_utc", "valid_time_utc"]:
        raise ValueError(
            f"weather artefact at {path!r} has index names {index_names!r}, "
            "expected ['run_init_utc', 'valid_time_utc']"
        )
    if len(df.columns) != 162:
        raise ValueError(
            f"weather artefact at {path!r} has {len(df.columns)} columns, expected 162"
        )
    bad_dtypes = {col: dtype for col, dtype in df.dtypes.items() if dtype != "float32"}
    if bad_dtypes:
        raise ValueError(f"weather artefact at {path!r} has non-float32 columns: {bad_dtypes}")

    logger.info("loaded %d weather rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Processed layer — renewables reconstruction artefact (sprint 6.5.2/6.5.3)
# ---------------------------------------------------------------------------

_RENEWABLES_PREDICTIONS_PATH = Path("data/processed/renewables_forecast_rolling365_l2.parquet")


def load_renewables_predictions(path: Path = _RENEWABLES_PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the headline-variant renewables reconstruction artefact
    (evaluation/renewables_walkforward.run_renewables_backtest's output,
    written by scripts/train_renewables.py --variant rolling365_l2).

    Raises FileNotFoundError if the file does not exist. Only the headline
    variant is a valid feature source (spec 6.5.3, section 4.2) -- the two
    comparison variants (expanding_l2, rolling365_quantile) are for 6.5.4's
    own DM-test comparison, not for feeding the price model.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Renewables predictions Parquet not found at {path!r}. "
            "Run scripts/train_renewables.py --variant rolling365_l2 to generate it."
        )
    df = pd.read_parquet(path)

    index_names = list(df.index.names)
    if index_names != ["run_init_utc", "valid_time_utc"]:
        raise ValueError(
            f"renewables predictions artefact at {path!r} has index names {index_names!r}, "
            "expected ['run_init_utc', 'valid_time_utc']"
        )

    logger.info("loaded %d renewables prediction rows from %s", len(df), path)
    return df
