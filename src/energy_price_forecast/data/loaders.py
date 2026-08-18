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

logger = logging.getLogger(__name__)

_COMMODITY_COLUMNS = ["ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"]
# 4 days = 96 hours: bridges weekends (2 days) and the longest common holiday gap
# (Good Friday to Easter Monday = 4 days). Genuine outages ≥ 5 days stay visible as NaN.
_COMMODITY_FFILL_LIMIT = 4 * 24


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
    forward-filled to hourly resolution with a 4-day limit. This bridges
    weekends and typical holiday gaps (Easter, Christmas) but leaves
    multi-day outages visible as NaN for downstream data quality analysis.
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
            df[col] = df[col].ffill(limit=_COMMODITY_FFILL_LIMIT)

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
