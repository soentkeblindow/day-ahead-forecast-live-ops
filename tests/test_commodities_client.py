"""Unit tests for the commodities client fetchers.

All tests mock yf.Ticker — the boundary between our code and yfinance — so
no real HTTP calls are made. The tests cover happy path, empty response,
timezone normalization, and the suspicious-ticker heuristic.
"""

import logging
from unittest.mock import patch

import pandas as pd
import pytest

from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas

MODULE = "energy_price_forecast.data.commodities_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_df(start: str, periods: int, tz: str | None = None) -> pd.DataFrame:
    """Build a daily close-price DataFrame as yfinance would return it."""
    idx = pd.date_range(start, periods=periods, freq="D", tz=tz)
    return pd.DataFrame({"Close": 50.0}, index=idx)


# ---------------------------------------------------------------------------
# fetch_ttf_gas — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_ttf_gas_returns_expected_columns() -> None:
    mock_df = _make_daily_df("2024-01-01", 30)

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = mock_df

        result = fetch_ttf_gas(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-30", tz="UTC"),
        )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["ttf_gas_eur_per_mwh"]
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert str(index.tz) == "UTC"
    assert len(result) == 30


# ---------------------------------------------------------------------------
# fetch_ttf_gas — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_ttf_gas_handles_empty_response(caplog: pytest.LogCaptureFixture) -> None:
    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        with caplog.at_level(logging.WARNING):
            result = fetch_ttf_gas(
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-30", tz="UTC"),
            )

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["ttf_gas_eur_per_mwh"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_ttf_gas — Test 3: timezone normalization
# ---------------------------------------------------------------------------


def test_fetch_ttf_gas_normalizes_timezone_to_utc() -> None:
    # Variant A: yfinance returns naive index → must be tagged as UTC.
    naive_df = _make_daily_df("2024-01-01", 5, tz=None)

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = naive_df
        result_a = fetch_ttf_gas(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"),
        )

    index_a = result_a.index
    assert isinstance(index_a, pd.DatetimeIndex)
    assert index_a.tz is not None
    assert str(index_a.tz) == "UTC"

    # Variant B: yfinance returns Europe/London index (BST in summer = UTC+1).
    # 2024-07-01 00:00 Europe/London = 2024-06-30 23:00 UTC.
    london_df = _make_daily_df("2024-07-01", 5, tz="Europe/London")

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = london_df
        result_b = fetch_ttf_gas(
            pd.Timestamp("2024-07-01", tz="UTC"),
            pd.Timestamp("2024-07-05", tz="UTC"),
        )

    index_b = result_b.index
    assert isinstance(index_b, pd.DatetimeIndex)
    assert index_b.tz is not None
    assert str(index_b.tz) == "UTC"
    # First London midnight in summer shifts back one hour in UTC.
    assert index_b[0] == pd.Timestamp("2024-06-30 23:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# fetch_eua_co2 — Test 1: happy path
# ---------------------------------------------------------------------------


def test_fetch_eua_co2_returns_expected_columns() -> None:
    mock_df = _make_daily_df("2024-01-01", 30)

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = mock_df

        result = fetch_eua_co2(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-30", tz="UTC"),
        )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["eua_co2_eur_per_t"]
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert str(index.tz) == "UTC"
    assert len(result) == 30


# ---------------------------------------------------------------------------
# fetch_eua_co2 — Test 2: empty response
# ---------------------------------------------------------------------------


def test_fetch_eua_co2_handles_empty_response(caplog: pytest.LogCaptureFixture) -> None:
    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        with caplog.at_level(logging.WARNING):
            result = fetch_eua_co2(
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-30", tz="UTC"),
            )

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["eua_co2_eur_per_t"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# fetch_eua_co2 — Test 3: timezone normalization
# ---------------------------------------------------------------------------


def test_fetch_eua_co2_normalizes_timezone_to_utc() -> None:
    # Variant A: naive index → UTC.
    naive_df = _make_daily_df("2024-01-01", 5, tz=None)

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = naive_df
        result_a = fetch_eua_co2(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"),
        )

    index_a = result_a.index
    assert isinstance(index_a, pd.DatetimeIndex)
    assert index_a.tz is not None
    assert str(index_a.tz) == "UTC"

    # Variant B: Europe/London index → UTC (BST in summer = UTC+1).
    london_df = _make_daily_df("2024-07-01", 5, tz="Europe/London")

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = london_df
        result_b = fetch_eua_co2(
            pd.Timestamp("2024-07-01", tz="UTC"),
            pd.Timestamp("2024-07-05", tz="UTC"),
        )

    index_b = result_b.index
    assert isinstance(index_b, pd.DatetimeIndex)
    assert index_b.tz is not None
    assert str(index_b.tz) == "UTC"
    assert index_b[0] == pd.Timestamp("2024-06-30 23:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Suspicious-ticker heuristic — Test 7: warns when rows are far below expected
# ---------------------------------------------------------------------------


def test_fetch_ttf_gas_warns_on_suspiciously_few_rows(caplog: pytest.LogCaptureFixture) -> None:
    # Only 3 rows for a ~5-year range; expected ~1300 trading days.
    sparse_df = _make_daily_df("2023-01-01", 3)

    with patch(f"{MODULE}.yf.Ticker") as mock_ticker_cls:
        mock_ticker_cls.return_value.history.return_value = sparse_df

        with caplog.at_level(logging.WARNING):
            result = fetch_ttf_gas(
                pd.Timestamp("2018-01-01", tz="UTC"),
                pd.Timestamp("2023-01-01", tz="UTC"),
            )

    # Data is still returned — no filtering, no exception.
    assert not result.empty
    assert len(result) == 3

    warning_messages = " ".join(r.message for r in caplog.records)
    assert "ticker" in warning_messages.lower()
