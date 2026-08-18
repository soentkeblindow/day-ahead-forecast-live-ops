"""Unit tests for data loaders: load_all_data and load_processed_features."""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_price_forecast.data.loaders import load_all_data, load_processed_features

MODULE = "energy_price_forecast.data.loaders"

# ---------------------------------------------------------------------------
# Shared index constants
# ---------------------------------------------------------------------------

_HOURLY_3D = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
_DAILY_3D = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
_HOURLY_7D = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")

_START = pd.Timestamp("2024-01-01", tz="UTC")
_END = pd.Timestamp("2024-01-03 23:00", tz="UTC")
_START_7D = pd.Timestamp("2024-01-01", tz="UTC")
_END_7D = pd.Timestamp("2024-01-07 23:00", tz="UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_returns(
    hourly_idx: pd.DatetimeIndex | None = None,
    daily_idx: pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the default mock return values for all eight fetchers."""
    h = hourly_idx if hourly_idx is not None else _HOURLY_3D
    d = daily_idx if daily_idx is not None else _DAILY_3D
    return {
        "fetch_day_ahead_prices": pd.DataFrame({"day_ahead_price": 50.0}, index=h),
        "fetch_load": pd.DataFrame(
            {"load_actual": 30_000.0, "load_forecast_day_ahead": 31_000.0}, index=h
        ),
        "fetch_wind_solar_forecast": pd.DataFrame(
            {
                "wind_onshore_forecast": 5_000.0,
                "wind_offshore_forecast": 2_000.0,
                "solar_forecast": 100.0,
            },
            index=h,
        ),
        "fetch_generation_by_type": pd.DataFrame(
            {
                "gen_nuclear": 8_000.0,
                "gen_lignite": 5_000.0,
                "gen_hard_coal": 3_000.0,
                "gen_gas": 4_000.0,
                "gen_oil": 100.0,
                "gen_biomass": 500.0,
                "gen_hydro": 1_000.0,
                "gen_wind_onshore": 5_000.0,
                "gen_wind_offshore": 2_000.0,
                "gen_solar": 100.0,
                "gen_other": 50.0,
            },
            index=h,
        ),
        "fetch_scheduled_exchanges": pd.DataFrame(
            {
                "scheduled_net_de_to_fr": 1_000.0,
                "scheduled_net_de_to_nl": 500.0,
                "scheduled_net_de_to_at": 200.0,
                "scheduled_net_de_to_pl": 300.0,
                "scheduled_net_de_to_ch": 100.0,
                "scheduled_net_de_to_dk_1": 150.0,
            },
            index=h,
        ),
        "fetch_cross_border_flows": pd.DataFrame(
            {
                "physical_net_de_to_fr": 900.0,
                "physical_net_de_to_nl": 450.0,
                "physical_net_de_to_at": 180.0,
                "physical_net_de_to_pl": 280.0,
                "physical_net_de_to_ch": 90.0,
                "physical_net_de_to_dk_1": 130.0,
            },
            index=h,
        ),
        "fetch_ttf_gas": pd.DataFrame({"ttf_gas_eur_per_mwh": 50.0}, index=d),
        "fetch_eua_co2": pd.DataFrame({"eua_co2_eur_per_t": 30.0}, index=d),
    }


@contextmanager
def _patch_fetchers(
    hourly_idx: pd.DatetimeIndex | None = None,
    daily_idx: pd.DatetimeIndex | None = None,
    **overrides: pd.DataFrame,
) -> Generator[dict[str, MagicMock], None, None]:
    """Patch all eight loader fetchers; keyword overrides replace specific returns."""
    returns = _default_returns(hourly_idx, daily_idx)
    returns.update(overrides)
    with ExitStack() as stack:
        mocks: dict[str, MagicMock] = {
            name: stack.enter_context(patch(f"{MODULE}.{name}", return_value=df))
            for name, df in returns.items()
        }
        yield mocks


# ---------------------------------------------------------------------------
# Test 1: happy path — all expected columns present
# ---------------------------------------------------------------------------


def test_load_all_data_returns_expected_columns() -> None:
    with _patch_fetchers():
        result = load_all_data(_START, _END)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    index = result.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None
    assert str(index.tz) == "UTC"

    expected_columns = {
        "day_ahead_price",
        "load_actual",
        "load_forecast_day_ahead",
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
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
        "scheduled_net_de_to_fr",
        "scheduled_net_de_to_nl",
        "scheduled_net_de_to_at",
        "scheduled_net_de_to_pl",
        "scheduled_net_de_to_ch",
        "scheduled_net_de_to_dk_1",
        "physical_net_de_to_fr",
        "physical_net_de_to_nl",
        "physical_net_de_to_at",
        "physical_net_de_to_pl",
        "physical_net_de_to_ch",
        "physical_net_de_to_dk_1",
        "ttf_gas_eur_per_mwh",
        "eua_co2_eur_per_t",
    }
    assert expected_columns.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Test 2a: commodity forward-fill propagates across weekends
# ---------------------------------------------------------------------------


def test_load_all_data_forward_fills_commodities_over_weekends() -> None:
    # TTF has daily values Mon–Fri only (Jan 1–5); other mocks cover all 7 days.
    ttf_idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    ttf_df = pd.DataFrame({"ttf_gas_eur_per_mwh": 50.0}, index=ttf_idx)
    eua_df = pd.DataFrame({"eua_co2_eur_per_t": 30.0}, index=ttf_idx)

    with _patch_fetchers(
        hourly_idx=_HOURLY_7D,
        fetch_ttf_gas=ttf_df,
        fetch_eua_co2=eua_df,
    ):
        result = load_all_data(_START_7D, _END_7D)

    # Sat (Jan 6) and Sun (Jan 7) must be filled from Friday's value;
    # 71 NaN slots (Fri 01:00 → Sun 23:00) are all within the 96-hour limit.
    assert result["ttf_gas_eur_per_mwh"].isna().sum() == 0
    assert result["eua_co2_eur_per_t"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test 2b: commodity forward-fill stops after 4-day (96-hour) limit
# ---------------------------------------------------------------------------


def test_load_all_data_forward_fill_respects_4_day_limit() -> None:
    # TTF: value at Jan 1 00:00 UTC (50.0) and Jan 7 00:00 UTC (60.0).
    # 143 NaN slots between them; limit=96 stops after Jan 5 00:00 UTC.
    ttf_idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-07", tz="UTC")]
    )
    ttf_df = pd.DataFrame({"ttf_gas_eur_per_mwh": [50.0, 60.0]}, index=ttf_idx)

    with _patch_fetchers(hourly_idx=_HOURLY_7D, fetch_ttf_gas=ttf_df):
        result = load_all_data(_START_7D, _END_7D)

    # Jan 5 00:00 UTC = the 96th filled slot → still filled.
    ts_96th = pd.Timestamp("2024-01-05 00:00", tz="UTC")
    assert result.at[ts_96th, "ttf_gas_eur_per_mwh"] == 50.0

    # Jan 5 01:00 UTC = the 97th → limit exceeded, must be NaN.
    ts_97th = pd.Timestamp("2024-01-05 01:00", tz="UTC")
    assert pd.isna(result.at[ts_97th, "ttf_gas_eur_per_mwh"])


# ---------------------------------------------------------------------------
# Test 3: rows without a target price are dropped
# ---------------------------------------------------------------------------


def test_load_all_data_drops_rows_without_target() -> None:
    # Day-ahead returns only the first 48 of 72 hours.
    day_ahead_df = pd.DataFrame({"day_ahead_price": 50.0}, index=_HOURLY_3D[:48])

    with _patch_fetchers(fetch_day_ahead_prices=day_ahead_df):
        result = load_all_data(_START, _END)

    assert len(result) == 48
    assert result["day_ahead_price"].notna().all()


# ---------------------------------------------------------------------------
# Test 4: predictor NaN values are preserved (outer join, no fill)
# ---------------------------------------------------------------------------


def test_load_all_data_keeps_nans_in_predictors() -> None:
    # Wind-solar returns only the first 60 of 72 hours; last 12 are absent.
    wind_solar_df = pd.DataFrame(
        {
            "wind_onshore_forecast": 5_000.0,
            "wind_offshore_forecast": 2_000.0,
            "solar_forecast": 100.0,
        },
        index=_HOURLY_3D[:60],
    )

    with _patch_fetchers(fetch_wind_solar_forecast=wind_solar_df):
        result = load_all_data(_START, _END)

    assert len(result) == 72
    assert result["wind_onshore_forecast"].isna().sum() == 12


# ---------------------------------------------------------------------------
# Test 5: start >= end raises ValueError
# ---------------------------------------------------------------------------


def test_load_all_data_validates_time_range() -> None:
    with pytest.raises(ValueError, match="start must be before end"):
        load_all_data(
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# Test 6: naive timestamps are interpreted as UTC
# ---------------------------------------------------------------------------


def test_load_all_data_normalizes_naive_timestamps() -> None:
    with _patch_fetchers() as mocks:
        load_all_data(
            pd.Timestamp("2024-01-01"),  # naive
            pd.Timestamp("2024-01-03 23:00"),  # naive
        )

    called_start: Any = mocks["fetch_day_ahead_prices"].call_args[0][0]
    assert isinstance(called_start, pd.Timestamp)
    assert called_start.tzinfo is not None
    assert str(called_start.tz) == "UTC"


# ---------------------------------------------------------------------------
# load_processed_features
# ---------------------------------------------------------------------------


def test_load_processed_features_loads_parquet(tmp_path: pytest.TempPathFactory) -> None:
    idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"feat_a": 1.0, "feat_b": 2.0}, index=idx)
    path = tmp_path / "features.parquet"  # type: ignore[operator]
    df.to_parquet(path)

    result = load_processed_features(path)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feat_a", "feat_b"]
    assert len(result) == 48


def test_load_processed_features_missing_file_raises(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "does_not_exist.parquet"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError, match="build_features.py"):
        load_processed_features(missing)
