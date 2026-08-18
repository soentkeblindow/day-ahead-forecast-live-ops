"""Unit tests for the ENTSO-E client fetchers.

All tests mock _get_client() — the boundary between our code and the ENTSO-E
library — so only the API class itself is replaced. Cache and retry logic run
against real implementation code.
"""

import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from entsoe.exceptions import NoMatchingDataError
from freezegun import freeze_time

from energy_price_forecast.data._entsoe_retry import EntsoeFetchError
from energy_price_forecast.data.entsoe_client import (
    _GEN_COLUMNS,
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)

MODULE = "energy_price_forecast.data.entsoe_client"

# call_with_retry uses time.sleep as a default parameter; cached_fetch provides
# no way to inject a custom sleep into fetch_fn. We therefore patch sleep at
# the _entsoe_retry module level rather than using call_with_retry's injectable
# parameter.
SLEEP = "energy_price_forecast.data._entsoe_retry.time.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(start: str, periods: int) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.Series(50.0, index=idx)


def _write_month_cache(root: Path, filename: str, series: pd.Series) -> None:
    cache_dir = root / "entsoe" / "day_ahead_prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    series.to_frame("day_ahead_price").to_parquet(cache_dir / filename, compression="snappy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_mock() -> Generator[MagicMock, None, None]:
    with patch(f"{MODULE}._get_client") as p:
        yield p.return_value


@pytest.fixture
def cache_root(tmp_path: Path) -> Generator[Path, None, None]:
    with patch(f"{MODULE}.DATA_RAW", tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 1: complete month cached → no API call
# ---------------------------------------------------------------------------


def test_cache_hit(client_mock: MagicMock, cache_root: Path) -> None:
    _write_month_cache(cache_root, "DE_LU_2022-01.parquet", _make_series("2022-01-01", 744))

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_not_called()
    assert len(result) == 744
    assert list(result.columns) == ["day_ahead_price"]


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 2: no cached file → API called, file written
# ---------------------------------------------------------------------------


def test_cache_miss_calls_api_and_writes_file(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.return_value = _make_series("2022-01-01", 744)

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()
    assert (cache_root / "entsoe" / "day_ahead_prices" / "DE_LU_2022-01.parquet").exists()
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 3: current month bypasses cache, re-fetches
# ---------------------------------------------------------------------------


@freeze_time("2026-04-15 12:00:00")
def test_current_month_is_always_refetched(client_mock: MagicMock, cache_root: Path) -> None:
    _write_month_cache(cache_root, "DE_LU_2026-04.parquet", _make_series("2026-04-01", 240))
    client_mock.query_day_ahead_prices.return_value = _make_series("2026-04-01", 576)

    fetch_day_ahead_prices(
        pd.Timestamp("2026-04-01", tz="UTC"),
        pd.Timestamp("2026-04-24 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 4: NoMatchingDataError → empty DataFrame with columns
# ---------------------------------------------------------------------------


def test_empty_response_returns_empty_dataframe(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.side_effect = NoMatchingDataError

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert result.empty
    assert list(result.columns) == ["day_ahead_price"]


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 5: 3 transient failures then success → correct result
# ---------------------------------------------------------------------------


def test_retry_on_transient_error(client_mock: MagicMock, cache_root: Path) -> None:
    with patch(SLEEP):
        client_mock.query_day_ahead_prices.side_effect = [
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
            _make_series("2022-01-01", 744),
        ]

        result = fetch_day_ahead_prices(
            pd.Timestamp("2022-01-01", tz="UTC"),
            pd.Timestamp("2022-01-31 23:00", tz="UTC"),
        )

    assert client_mock.query_day_ahead_prices.call_count == 4
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 6: permanent failure → EntsoeFetchError raised
# ---------------------------------------------------------------------------


def test_retry_exhaustion_raises_fetch_error(client_mock: MagicMock, cache_root: Path) -> None:
    with patch(SLEEP):
        client_mock.query_day_ahead_prices.side_effect = requests.ConnectionError()

        with pytest.raises(EntsoeFetchError):
            fetch_day_ahead_prices(
                pd.Timestamp("2022-01-01", tz="UTC"),
                pd.Timestamp("2022-01-31 23:00", tz="UTC"),
            )


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 7: 3-month range → 3 API calls, concatenated result
# ---------------------------------------------------------------------------


def test_chunk_aggregation(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.side_effect = [
        _make_series("2022-01-01", 744),  # January:  31 days × 24 h
        _make_series("2022-02-01", 672),  # February: 28 days × 24 h
        _make_series("2022-03-01", 744),  # March:    31 days × 24 h
    ]

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-03-31 23:00", tz="UTC"),
    )

    assert client_mock.query_day_ahead_prices.call_count == 3
    assert len(result) == 744 + 672 + 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 8: <90% rows for complete month → re-fetch
# ---------------------------------------------------------------------------


def test_incomplete_cache_triggers_refetch(client_mock: MagicMock, cache_root: Path) -> None:
    # 240 rows (10 days) is well below 90% of the 744 hours expected for January
    _write_month_cache(cache_root, "DE_LU_2022-01.parquet", _make_series("2022-01-01", 240))
    client_mock.query_day_ahead_prices.return_value = _make_series("2022-01-01", 744)

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_load — Test 1: both APIs return data → two-column DataFrame
# ---------------------------------------------------------------------------


def test_fetch_load_returns_both_columns(client_mock: MagicMock, cache_root: Path) -> None:
    idx = pd.date_range("2022-01-01", periods=744, freq="h", tz="UTC")
    client_mock.query_load.return_value = pd.DataFrame({"Actual Load": 30_000.0}, index=idx)
    client_mock.query_load_forecast.return_value = pd.DataFrame(
        {"Forecasted Load": 31_000.0}, index=idx
    )

    result = fetch_load(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert list(result.columns) == ["load_actual", "load_forecast_day_ahead"]
    assert len(result) == 744
    assert result["load_actual"].notna().all()
    assert result["load_forecast_day_ahead"].notna().all()


# ---------------------------------------------------------------------------
# fetch_load — Test 2: one API fails → other column NaN, schema preserved
# ---------------------------------------------------------------------------


def test_fetch_load_preserves_schema_on_partial_failure(
    client_mock: MagicMock, cache_root: Path
) -> None:
    idx = pd.date_range("2022-01-01", periods=744, freq="h", tz="UTC")
    client_mock.query_load.return_value = pd.DataFrame({"Actual Load": 30_000.0}, index=idx)
    client_mock.query_load_forecast.side_effect = NoMatchingDataError

    result = fetch_load(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert list(result.columns) == ["load_actual", "load_forecast_day_ahead"]
    assert len(result) == 744
    assert result["load_actual"].notna().all()
    assert result["load_forecast_day_ahead"].isna().all()


# ---------------------------------------------------------------------------
# fetch_wind_solar_forecast — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_wind_solar_forecast_returns_expected_columns(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _forecast_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {"Wind Onshore": 5000.0, "Wind Offshore": 2000.0, "Solar": 100.0}, index=idx
        )

    client_mock.query_wind_and_solar_forecast.side_effect = _forecast_se

    result = fetch_wind_solar_forecast(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    expected = {
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
    }
    assert not result.empty
    assert set(result.columns) == expected
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert len(result) == 72


# ---------------------------------------------------------------------------
# fetch_wind_solar_forecast — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_wind_solar_forecast_handles_empty_response(
    client_mock: MagicMock, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client_mock.query_wind_and_solar_forecast.side_effect = NoMatchingDataError

    with caplog.at_level(logging.WARNING):
        result = fetch_wind_solar_forecast(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-03 23:00", tz="UTC"),
        )

    assert result.empty
    expected = {
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
    }
    assert set(result.columns) == expected
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_wind_solar_forecast — Test 3: cache hit on second call
# ---------------------------------------------------------------------------


def test_fetch_wind_solar_forecast_uses_cache_on_second_call(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _forecast_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {"Wind Onshore": 5000.0, "Wind Offshore": 2000.0, "Solar": 100.0}, index=idx
        )

    client_mock.query_wind_and_solar_forecast.side_effect = _forecast_se

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31 23:00", tz="UTC")

    result1 = fetch_wind_solar_forecast(start, end)
    forecast_calls = client_mock.query_wind_and_solar_forecast.call_count

    result2 = fetch_wind_solar_forecast(start, end)

    assert client_mock.query_wind_and_solar_forecast.call_count == forecast_calls
    # Parquet does not preserve DatetimeIndex frequency; check_freq=False ignores that.
    pd.testing.assert_frame_equal(result1, result2, check_freq=False)


# ---------------------------------------------------------------------------
# fetch_wind_solar_forecast — Test 4 (extra): partial sources
# ---------------------------------------------------------------------------


def test_wind_solar_handles_partial_sources(client_mock: MagicMock, cache_root: Path) -> None:
    def _forecast_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {"Wind Onshore": 5000.0, "Wind Offshore": 2000.0, "Solar": 100.0}, index=idx
        )

    client_mock.query_wind_and_solar_forecast.side_effect = _forecast_se

    result = fetch_wind_solar_forecast(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert not result.empty
    assert "wind_onshore_forecast" in result.columns
    assert "wind_offshore_forecast" in result.columns
    assert "solar_forecast" in result.columns


# ---------------------------------------------------------------------------
# fetch_generation_by_type — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_generation_by_type_returns_expected_columns(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _gen_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Nuclear": 8000.0,
                "Fossil Gas": 4000.0,
                "Wind Onshore": 5000.0,
                "Solar": 100.0,
                "Hydro Run-of-river and poundage": 2000.0,
            },
            index=idx,
        )

    client_mock.query_generation.side_effect = _gen_se

    result = fetch_generation_by_type(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert not result.empty
    for col in ("gen_nuclear", "gen_gas", "gen_wind_onshore", "gen_solar", "gen_hydro"):
        assert col in result.columns
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert len(result) == 72


# ---------------------------------------------------------------------------
# fetch_generation_by_type — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_generation_by_type_handles_empty_response(
    client_mock: MagicMock, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client_mock.query_generation.side_effect = NoMatchingDataError

    with caplog.at_level(logging.WARNING):
        result = fetch_generation_by_type(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-03 23:00", tz="UTC"),
        )

    assert result.empty
    assert set(result.columns) == set(_GEN_COLUMNS)
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_generation_by_type — Test 3: cache hit on second call
# ---------------------------------------------------------------------------


def test_fetch_generation_by_type_uses_cache_on_second_call(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _gen_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame({"Nuclear": 8000.0, "Wind Onshore": 5000.0}, index=idx)

    client_mock.query_generation.side_effect = _gen_se

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31 23:00", tz="UTC")

    result1 = fetch_generation_by_type(start, end)
    calls_after_first = client_mock.query_generation.call_count

    result2 = fetch_generation_by_type(start, end)

    assert client_mock.query_generation.call_count == calls_after_first
    pd.testing.assert_frame_equal(result1, result2, check_freq=False)


# ---------------------------------------------------------------------------
# fetch_generation_by_type — Test 4 (extra): hydro sub-types summed into gen_hydro
# ---------------------------------------------------------------------------


def test_generation_aggregates_hydro_subtypes(client_mock: MagicMock, cache_root: Path) -> None:
    def _gen_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Hydro Pumped Storage": 500.0,
                "Hydro Run-of-river and poundage": 1000.0,
                "Hydro Water Reservoir": 800.0,
            },
            index=idx,
        )

    client_mock.query_generation.side_effect = _gen_se

    result = fetch_generation_by_type(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert "gen_hydro" in result.columns
    assert (result["gen_hydro"] == 2300.0).all()


# ---------------------------------------------------------------------------
# fetch_generation_by_type — Test 5 (extra): unknown types collected into gen_other
# ---------------------------------------------------------------------------


def test_generation_collects_unknown_types_into_other(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _gen_se(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Nuclear": 8000.0,
                "Geothermal": 50.0,  # not in _ENTSOE_TO_GEN_COL or _HYDRO_TYPES
            },
            index=idx,
        )

    client_mock.query_generation.side_effect = _gen_se

    result = fetch_generation_by_type(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert "gen_other" in result.columns
    assert (result["gen_other"] == 50.0).all()


# ---------------------------------------------------------------------------
# fetch_scheduled_exchanges — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_scheduled_exchanges_returns_expected_columns(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _se(
        from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp, dayahead: bool = False
    ) -> pd.Series:
        return pd.Series(1000.0, index=pd.date_range(start, end, freq="h", tz="UTC"))

    client_mock.query_scheduled_exchanges.side_effect = _se

    result = fetch_scheduled_exchanges(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    expected = {
        "scheduled_net_de_to_fr",
        "scheduled_net_de_to_nl",
        "scheduled_net_de_to_at",
        "scheduled_net_de_to_pl",
        "scheduled_net_de_to_ch",
        "scheduled_net_de_to_dk_1",
    }
    assert not result.empty
    assert set(result.columns) == expected
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert len(result) == 72


# ---------------------------------------------------------------------------
# fetch_scheduled_exchanges — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_scheduled_exchanges_handles_empty_response(
    client_mock: MagicMock, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client_mock.query_scheduled_exchanges.side_effect = NoMatchingDataError

    with caplog.at_level(logging.WARNING):
        result = fetch_scheduled_exchanges(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-03 23:00", tz="UTC"),
        )

    expected = {
        "scheduled_net_de_to_fr",
        "scheduled_net_de_to_nl",
        "scheduled_net_de_to_at",
        "scheduled_net_de_to_pl",
        "scheduled_net_de_to_ch",
        "scheduled_net_de_to_dk_1",
    }
    assert result.empty
    assert set(result.columns) == expected
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_scheduled_exchanges — Test 3: cache hit on second call
# ---------------------------------------------------------------------------


def test_fetch_scheduled_exchanges_uses_cache_on_second_call(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _se(
        from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp, dayahead: bool = False
    ) -> pd.Series:
        return pd.Series(1000.0, index=pd.date_range(start, end, freq="h", tz="UTC"))

    client_mock.query_scheduled_exchanges.side_effect = _se

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31 23:00", tz="UTC")

    result1 = fetch_scheduled_exchanges(start, end)
    calls_after_first = (
        client_mock.query_scheduled_exchanges.call_count
    )  # 12: 6 neighbors × 2 directions

    result2 = fetch_scheduled_exchanges(start, end)

    assert client_mock.query_scheduled_exchanges.call_count == calls_after_first
    pd.testing.assert_frame_equal(result1, result2, check_freq=False)


# ---------------------------------------------------------------------------
# fetch_scheduled_exchanges — Test 4 (extra): net flow = export − import
# ---------------------------------------------------------------------------


def test_scheduled_exchanges_computes_net_flow(client_mock: MagicMock, cache_root: Path) -> None:
    def _se(
        from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp, dayahead: bool = False
    ) -> pd.Series:
        if "FR" not in (from_a, to_a):
            raise NoMatchingDataError()
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.Series(100.0 if from_a == "DE_LU" else 30.0, index=idx)

    client_mock.query_scheduled_exchanges.side_effect = _se

    result = fetch_scheduled_exchanges(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert "scheduled_net_de_to_fr" in result.columns
    assert (result["scheduled_net_de_to_fr"] == 70.0).all()


# ---------------------------------------------------------------------------
# fetch_cross_border_flows — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_cross_border_flows_returns_expected_columns(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _cbf(from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return pd.Series(1000.0, index=pd.date_range(start, end, freq="h", tz="UTC"))

    client_mock.query_crossborder_flows.side_effect = _cbf

    result = fetch_cross_border_flows(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    expected = {
        "physical_net_de_to_fr",
        "physical_net_de_to_nl",
        "physical_net_de_to_at",
        "physical_net_de_to_pl",
        "physical_net_de_to_ch",
        "physical_net_de_to_dk_1",
    }
    assert not result.empty
    assert set(result.columns) == expected
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert len(result) == 72


# ---------------------------------------------------------------------------
# fetch_cross_border_flows — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_cross_border_flows_handles_empty_response(
    client_mock: MagicMock, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client_mock.query_crossborder_flows.side_effect = NoMatchingDataError

    with caplog.at_level(logging.WARNING):
        result = fetch_cross_border_flows(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-03 23:00", tz="UTC"),
        )

    expected = {
        "physical_net_de_to_fr",
        "physical_net_de_to_nl",
        "physical_net_de_to_at",
        "physical_net_de_to_pl",
        "physical_net_de_to_ch",
        "physical_net_de_to_dk_1",
    }
    assert result.empty
    assert set(result.columns) == expected
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_cross_border_flows — Test 3: cache hit on second call
# ---------------------------------------------------------------------------


def test_fetch_cross_border_flows_uses_cache_on_second_call(
    client_mock: MagicMock, cache_root: Path
) -> None:
    def _cbf(from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return pd.Series(1000.0, index=pd.date_range(start, end, freq="h", tz="UTC"))

    client_mock.query_crossborder_flows.side_effect = _cbf

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31 23:00", tz="UTC")

    result1 = fetch_cross_border_flows(start, end)
    calls_after_first = (
        client_mock.query_crossborder_flows.call_count
    )  # 12: 6 neighbors × 2 directions

    result2 = fetch_cross_border_flows(start, end)

    assert client_mock.query_crossborder_flows.call_count == calls_after_first
    pd.testing.assert_frame_equal(result1, result2, check_freq=False)


# ---------------------------------------------------------------------------
# fetch_cross_border_flows — Test 4 (extra): net flow = export − import
# ---------------------------------------------------------------------------


def test_cross_border_flows_computes_net_flow(client_mock: MagicMock, cache_root: Path) -> None:
    def _cbf(from_a: str, to_a: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if "FR" not in (from_a, to_a):
            raise NoMatchingDataError()
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.Series(200.0 if from_a == "DE_LU" else 80.0, index=idx)

    client_mock.query_crossborder_flows.side_effect = _cbf

    result = fetch_cross_border_flows(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-03 23:00", tz="UTC"),
    )

    assert "physical_net_de_to_fr" in result.columns
    assert (result["physical_net_de_to_fr"] == 120.0).all()


# ---------------------------------------------------------------------------
# Integration test — real API call, excluded from CI by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_day_ahead_prices_live() -> None:
    """Single real ENTSO-E API call for local verification only."""
    result = fetch_day_ahead_prices(
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-01 23:00", tz="UTC"),
    )
    assert not result.empty
    assert "day_ahead_price" in result.columns
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
