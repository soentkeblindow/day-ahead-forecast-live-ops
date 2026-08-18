import functools
import logging
from collections.abc import Callable

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from energy_price_forecast.config import DATA_RAW, get_entsoe_token
from energy_price_forecast.data._entsoe_cache import cached_fetch
from energy_price_forecast.data._entsoe_retry import call_with_retry

logger = logging.getLogger(__name__)

AREA_DE_LU = "DE_LU"
NEIGHBORS = ["FR", "NL", "AT", "PL", "CH", "DK_1"]

# Maps entsoe-py column names to our gen_* column names.
_ENTSOE_TO_GEN_COL: dict[str, str] = {
    "Nuclear": "gen_nuclear",
    "Fossil Brown coal/Lignite": "gen_lignite",
    "Fossil Hard coal": "gen_hard_coal",
    "Fossil Gas": "gen_gas",
    "Fossil Oil": "gen_oil",
    "Biomass": "gen_biomass",
    "Wind Onshore": "gen_wind_onshore",
    "Wind Offshore": "gen_wind_offshore",
    "Solar": "gen_solar",
}

# Hydro sub-types that are summed into gen_hydro.
_HYDRO_TYPES = frozenset(
    {
        "Hydro Pumped Storage",
        "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir",
    }
)

_GEN_COLUMNS = [
    "gen_nuclear",
    "gen_lignite",
    "gen_hard_coal",
    "gen_gas",
    "gen_oil",
    "gen_biomass",
    "gen_hydro",
    "gen_wind_onshore",
    "gen_wind_offshore",
    "gen_solar",
    "gen_other",
]

_WIND_SOLAR_COLUMNS = [
    "wind_onshore_forecast",
    "wind_offshore_forecast",
    "solar_forecast",
]


@functools.cache
def _get_client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=get_entsoe_token())


def _get_gen_series(gen_df: pd.DataFrame, source_name: str) -> pd.Series | None:
    """Extract the Actual Aggregated series for a generation type.

    query_generation() can return either a flat or a MultiIndex DataFrame
    (second level: "Actual Aggregated" / "Actual Consumption"). We prefer
    "Actual Aggregated" when both sub-columns exist.
    """
    if isinstance(gen_df.columns, pd.MultiIndex):
        cols = [c for c in gen_df.columns if c[0] == source_name]
        if not cols:
            return None
        agg = [c for c in cols if "Aggregated" in c[1]]
        key = agg[0] if agg else cols[0]
        return gen_df[key]
    elif source_name in gen_df.columns:
        return gen_df[source_name]
    return None


def fetch_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "day_ahead_prices"

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()
        try:
            series = call_with_retry(lambda: client.query_day_ahead_prices(area, start=s, end=e))
        except NoMatchingDataError:
            logger.warning("No day-ahead prices from ENTSO-E for %s–%s", s, e)
            return pd.DataFrame(columns=["day_ahead_price"])
        return series.tz_convert("UTC").rename("day_ahead_price").to_frame()

    return cached_fetch(start, end, cache_dir, area, fetch_fn)


def fetch_load(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "load"
    _columns = ["load_actual", "load_forecast_day_ahead"]

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()

        try:
            actual = (
                call_with_retry(lambda: client.query_load(area, start=s, end=e))["Actual Load"]
                .tz_convert("UTC")
                .rename("load_actual")
            )
        except NoMatchingDataError:
            logger.warning("No load actual from ENTSO-E for %s–%s", s, e)
            actual = pd.Series(dtype=float, name="load_actual")

        try:
            forecast = (
                call_with_retry(lambda: client.query_load_forecast(area, start=s, end=e))[
                    "Forecasted Load"
                ]
                .tz_convert("UTC")
                .rename("load_forecast_day_ahead")
            )
        except NoMatchingDataError:
            logger.warning("No load forecast from ENTSO-E for %s–%s", s, e)
            forecast = pd.Series(dtype=float, name="load_forecast_day_ahead")

        if actual.empty and forecast.empty:
            return pd.DataFrame(columns=_columns)

        # Partial failure: reindex the empty Series to the other's index so that
        # pd.concat always produces a two-column DataFrame. The all-NaN column is
        # cached as-is — the 90%-row coverage check sees the row count as sufficient.
        # To force a re-fetch of later-backfilled ENTSO-E data, delete the cache
        # file manually.
        if actual.empty:
            actual = actual.reindex(forecast.index)
        elif forecast.empty:
            forecast = forecast.reindex(actual.index)

        return pd.concat([actual, forecast], axis=1)

    return cached_fetch(start, end, cache_dir, area, fetch_fn)


def fetch_wind_solar_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "wind_solar"

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()

        try:
            forecast_df = call_with_retry(
                lambda: client.query_wind_and_solar_forecast(area, start=s, end=e)
            ).tz_convert("UTC")
        except NoMatchingDataError:
            logger.warning("No wind/solar forecast from ENTSO-E for %s–%s", s, e)
            forecast_df = pd.DataFrame()

        parts: list[pd.Series] = []

        for src, dst in [
            ("Wind Onshore", "wind_onshore_forecast"),
            ("Wind Offshore", "wind_offshore_forecast"),
            ("Solar", "solar_forecast"),
        ]:
            if not forecast_df.empty and src in forecast_df.columns:
                parts.append(forecast_df[src].rename(dst))

        if not parts:
            return pd.DataFrame(columns=_WIND_SOLAR_COLUMNS)
        return pd.concat(parts, axis=1)

    return cached_fetch(start, end, cache_dir, area, fetch_fn)


def fetch_generation_by_type(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "generation"

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()

        try:
            gen_df = call_with_retry(
                lambda: client.query_generation(area, start=s, end=e)
            ).tz_convert("UTC")
        except NoMatchingDataError:
            logger.warning("No generation data from ENTSO-E for %s–%s", s, e)
            return pd.DataFrame(columns=_GEN_COLUMNS)

        if isinstance(gen_df.columns, pd.MultiIndex):
            all_types: set[str] = {c[0] for c in gen_df.columns}
        else:
            all_types = set(gen_df.columns)

        result: dict[str, pd.Series] = {}

        for src, dst in _ENTSOE_TO_GEN_COL.items():
            series = _get_gen_series(gen_df, src)
            if series is not None:
                result[dst] = series

        hydro_parts = [gs for t in _HYDRO_TYPES if (gs := _get_gen_series(gen_df, t)) is not None]
        if hydro_parts:
            result["gen_hydro"] = pd.concat(hydro_parts, axis=1).sum(axis=1, min_count=1)

        known = set(_ENTSOE_TO_GEN_COL.keys()) | _HYDRO_TYPES
        other_types = all_types - known
        other_parts = [gs for t in other_types if (gs := _get_gen_series(gen_df, t)) is not None]
        if other_parts:
            result["gen_other"] = pd.concat(other_parts, axis=1).sum(axis=1, min_count=1)

        if not result:
            return pd.DataFrame(columns=_GEN_COLUMNS)
        return pd.DataFrame(result)

    return cached_fetch(start, end, cache_dir, area, fetch_fn)


def _build_neighbor_fetch_fn(
    area: str,
    neighbor: str,
    col_name: str,
    cache_subdir: str,
    query_fn: Callable[[str, str, pd.Timestamp, pd.Timestamp], pd.Series],
) -> Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
    """Return the per-neighbor fetch function used by cached_fetch.

    Extracted from the loop in _fetch_border_flows to avoid the late-binding
    closure problem: all values are bound as function parameters here, not
    resolved lazily from the enclosing loop scope.
    """

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        try:
            export = call_with_retry(lambda: query_fn(area, neighbor, s, e)).tz_convert("UTC")
            import_ = call_with_retry(lambda: query_fn(neighbor, area, s, e)).tz_convert("UTC")
        except NoMatchingDataError:
            logger.warning("No %s data for %s↔%s for %s–%s", cache_subdir, area, neighbor, s, e)
            return pd.DataFrame(columns=[col_name])
        return (export - import_).rename(col_name).to_frame()

    return fetch_fn


def _fetch_border_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str,
    cache_subdir: str,
    col_prefix: str,
    query_fn: Callable[[str, str, pd.Timestamp, pd.Timestamp], pd.Series],
) -> pd.DataFrame:
    """Shared loop logic for scheduled_exchanges and cross_border_flows.

    For each neighbor, calls query_fn in both directions and caches the resulting
    net-flow Series (positive = export from area to neighbor) as a single-column
    Parquet. The six per-neighbor DataFrames are concatenated column-wise.
    """
    cache_dir = DATA_RAW / "entsoe" / cache_subdir
    neighbor_frames: list[pd.DataFrame] = []

    for neighbor in NEIGHBORS:
        col_name = f"{col_prefix}_de_to_{neighbor.lower()}"
        file_prefix = f"{area}_{neighbor}"
        fetch_fn = _build_neighbor_fetch_fn(area, neighbor, col_name, cache_subdir, query_fn)
        neighbor_frames.append(cached_fetch(start, end, cache_dir, file_prefix, fetch_fn))

    if not neighbor_frames:
        return pd.DataFrame()
    return pd.concat(neighbor_frames, axis=1)


def fetch_scheduled_exchanges(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    def query_pair(from_a: str, to_a: str, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
        return _get_client().query_scheduled_exchanges(from_a, to_a, start=s, end=e, dayahead=True)

    return _fetch_border_flows(start, end, area, "scheduled_exchanges", "scheduled_net", query_pair)


def fetch_cross_border_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    def query_pair(from_a: str, to_a: str, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
        return _get_client().query_crossborder_flows(from_a, to_a, start=s, end=e)

    return _fetch_border_flows(start, end, area, "cross_border_flows", "physical_net", query_pair)
