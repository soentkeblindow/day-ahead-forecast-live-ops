"""Unit tests for arena/live_inputs.py -- store-raw-data assembly for the
daily submission job (spec 6.7.2, section 5.8). Everything here is pure
disk I/O against tmp_path fixtures: no client, no network, matching the
module's own "the job calls no fetch client" contract.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from energy_price_forecast.arena.live_inputs import (
    DEFAULT_WEATHER_MODEL,
    assemble_price_model_inputs,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.data._weather_cache import cache_path, write_cached_run
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import expected_columns
from energy_price_forecast.ops.store_sources import EntsoeSource
from energy_price_forecast.ops.windows import local_day_bounds


def _entsoe_source(name: str, cache_dir: Path) -> EntsoeSource:
    def _never_call(
        start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True
    ) -> pd.DataFrame:
        raise AssertionError("assemble_price_model_inputs must never call a fetch function")

    return EntsoeSource(name=name, fetch=_never_call, cache_dir=cache_dir)


def _never_call_row_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raise AssertionError("assemble_price_model_inputs must never call a fetch function")


def _synthetic_weather_frame(run_init_utc: pd.Timestamp, n_hours: int = 2) -> pd.DataFrame:
    valid_times = pd.date_range(run_init_utc, periods=n_hours, freq="h", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([run_init_utc] * n_hours), valid_times],
        names=["run_init_utc", "valid_time_utc"],
    )
    columns = expected_columns()
    data = {col: [1.0 * i for i in range(n_hours)] for col in columns}
    return pd.DataFrame(data, index=index, columns=columns).astype("float32")


# ---------------------------------------------------------------------------
# assemble_price_model_inputs
# ---------------------------------------------------------------------------


def test_assemble_price_model_inputs_reads_disk_only_and_joins_sources(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [50.0, 51.0, 52.0, 53.0, 54.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame(
        {"load_actual": [1.0] * 5, "load_forecast_day_ahead": [1.0] * 5}, index=idx
    ).to_parquet(load_dir / "DE_LU_2024-01.parquet")

    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()
    ttf_idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
    pd.DataFrame({"ttf_gas_eur_per_mwh": [10.0]}, index=ttf_idx).to_parquet(
        commodities_dir / "ttf_gas.parquet"
    )
    # eua_co2.parquet deliberately absent -- exercises the missing-file fallback.

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        # Isolates from this machine's own real, populated store (spec 6.9
        # Schritt 6 backfill wrote real data under the real ENERGY_CHARTS_DIR)
        # -- an empty, never-written tmp_path dir reproduces the exact prior
        # behaviour (_read_ec_price's own missing-file fallback).
        energy_charts_dir=tmp_path / "energy_charts",
    )

    assert list(result.index) == list(idx)
    assert list(result["day_ahead_price"]) == [50.0, 51.0, 52.0, 53.0, 54.0]
    # ttf_gas known only at hour 0 -- forward-filled across all 5 hours (well within the 7-day limit).
    assert (result["ttf_gas_eur_per_mwh"] == 10.0).all()
    # eua_co2 has no cache file at all -- column present, entirely NaN, never a KeyError.
    assert result["eua_co2_eur_per_t"].isna().all()


def test_assemble_price_model_inputs_forward_fill_respects_7_real_days_at_quarter_hourly_resolution(
    tmp_path: Path,
) -> None:
    """Reproduces the real 2026-09-12 bug end to end, through the actual
    live path: day_ahead_price has been quarter-hourly since 2025-09-30,
    so a row-count ffill(limit=168) covered only 42 real hours instead of
    the intended 7 days -- found live during the 6.7.2 wiring probe."""
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    qh_idx = pd.date_range("2024-01-01", periods=4 * 24 * 10, freq="15min", tz="UTC")
    pd.DataFrame({"day_ahead_price": 50.0}, index=qh_idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame({"load_actual": 1000.0}, index=qh_idx).to_parquet(
        load_dir / "DE_LU_2024-01.parquet"
    )

    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()
    # One real value, never repeated -- everything after it is pure ffill.
    ttf_idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
    pd.DataFrame({"ttf_gas_eur_per_mwh": [50.0]}, index=ttf_idx).to_parquet(
        commodities_dir / "ttf_gas.parquet"
    )

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        # Isolates from this machine's own real, populated store (spec 6.9
        # Schritt 6 backfill wrote real data under the real ENERGY_CHARTS_DIR)
        # -- an empty, never-written tmp_path dir reproduces the exact prior
        # behaviour (_read_ec_price's own missing-file fallback).
        energy_charts_dir=tmp_path / "energy_charts",
    )

    # 96 real hours after the last known value -- well past the ~42-hour span
    # a row-count ffill(limit=168) would have covered on this quarter-hourly
    # index, but still within the intended 7 real days.
    still_within_7_days = pd.Timestamp("2024-01-05 00:00", tz="UTC")
    assert result.loc[still_within_7_days, "ttf_gas_eur_per_mwh"] == 50.0


def test_assemble_price_model_inputs_keeps_a_row_missing_only_price(tmp_path: Path) -> None:
    """docs/sprint6_fix_partial_today.md section 3.1: a row is no longer
    dropped from the merged frame just because day_ahead_price is missing --
    completeness is a per-consumer question (the renewables reconstruction
    reads its own forecast columns off this same frame, regardless of
    price), not a per-row one decided here. Real 2026-09-13 finding: this
    is what happens to every genuine future delivery day (no price yet, by
    design) and, that same day, to an unrelated one-off gap in *today's*
    own price."""
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    # Price only covers the first two hours.
    pd.DataFrame({"day_ahead_price": [50.0, 51.0]}, index=idx[:2]).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame(
        {"load_actual": [1.0] * 4, "load_forecast_day_ahead": [1.0] * 4}, index=idx
    ).to_parquet(load_dir / "DE_LU_2024-01.parquet")

    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        # Isolates from this machine's own real, populated store (spec 6.9
        # Schritt 6 backfill wrote real data under the real ENERGY_CHARTS_DIR)
        # -- an empty, never-written tmp_path dir reproduces the exact prior
        # behaviour (_read_ec_price's own missing-file fallback).
        energy_charts_dir=tmp_path / "energy_charts",
    )

    assert len(result) == 4
    assert result["day_ahead_price"].iloc[:2].notna().all()
    assert result["day_ahead_price"].iloc[2:].isna().all()  # not invented
    assert result["load_forecast_day_ahead"].notna().all()  # real column survives throughout


def test_assemble_price_model_inputs_priceless_days_keep_other_real_data(
    tmp_path: Path,
) -> None:
    """The multi-day shape: local 2024-01-01 has a real price all day;
    2024-01-02/03 have none at all, but real load data throughout -- all
    three now survive identically (no day needs naming, unlike the old
    keep_rows_for exemption this fix replaces). Boundaries come from
    local_day_bounds, not a naive UTC date_range slice, so they align with
    what the SUT itself means by "day" (Europe/Berlin, not UTC)."""
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    day1_start, _ = local_day_bounds(dt.date(2024, 1, 1))
    _, day3_end = local_day_bounds(dt.date(2024, 1, 3))
    full_idx = pd.date_range(
        day1_start.tz_convert("UTC"), day3_end.tz_convert("UTC"), freq="h", inclusive="left"
    )
    day2_start, _ = local_day_bounds(dt.date(2024, 1, 2))
    priced_idx = full_idx[full_idx < day2_start.tz_convert("UTC")]  # only local 2024-01-01
    pd.DataFrame({"day_ahead_price": 50.0}, index=priced_idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame({"load_actual": 1.0, "load_forecast_day_ahead": 1.0}, index=full_idx).to_parquet(
        load_dir / "DE_LU_2024-01.parquet"
    )
    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        # Isolates from this machine's own real, populated store (spec 6.9
        # Schritt 6 backfill wrote real data under the real ENERGY_CHARTS_DIR)
        # -- an empty, never-written tmp_path dir reproduces the exact prior
        # behaviour (_read_ec_price's own missing-file fallback).
        energy_charts_dir=tmp_path / "energy_charts",
    )

    # Local (Europe/Berlin) calendar date, not the raw UTC one: local
    # midnight on 2024-01-03 is 2024-01-02T23:00 UTC (CET, UTC+1), so a
    # UTC-only .date comparison would misplace an hour across the boundary.
    result_local_dates = pd.DatetimeIndex(result.index).tz_convert("Europe/Berlin").date
    day1 = result[result_local_dates == dt.date(2024, 1, 1)]
    day2 = result[result_local_dates == dt.date(2024, 1, 2)]
    day3 = result[result_local_dates == dt.date(2024, 1, 3)]
    assert len(day1) == len(day2) == len(day3) == 24
    assert day1["day_ahead_price"].notna().all()
    assert day2["day_ahead_price"].isna().all() and day2["load_forecast_day_ahead"].notna().all()
    assert day3["day_ahead_price"].isna().all() and day3["load_forecast_day_ahead"].notna().all()


# ---------------------------------------------------------------------------
# Energy-Charts coalescing (spec 6.9 section 5.3/2.4, Schritt 7)
# ---------------------------------------------------------------------------


def test_assemble_price_model_inputs_fills_a_real_price_gap_from_energy_charts(
    tmp_path: Path,
) -> None:
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [50.0, float("nan"), 52.0, 53.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame(
        {"load_actual": [1.0] * 4, "load_forecast_day_ahead": [1.0] * 4}, index=idx
    ).to_parquet(load_dir / "DE_LU_2024-01.parquet")

    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()
    energy_charts_dir = tmp_path / "energy_charts"
    energy_charts_dir.mkdir()
    pd.DataFrame({"day_ahead_price_ec": [50.0, 99.0, 52.0, 53.0]}, index=idx).to_parquet(
        energy_charts_dir / "day_ahead_price_ec.parquet"
    )

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        energy_charts_dir=energy_charts_dir,
    )

    assert result["day_ahead_price"].tolist() == [50.0, 99.0, 52.0, 53.0]
    # The raw EC column has done its job feeding the coalesce and does not
    # linger in the output (it would otherwise be misread by to_hourly() as
    # an ordinary MW column).
    assert "day_ahead_price_ec" not in result.columns


def test_assemble_price_model_inputs_keeps_ec_load_forecast_as_its_own_column(
    tmp_path: Path,
) -> None:
    """spec section 2.4: "Zeile 1 liest nur ENTSO-E, Zeile 2 nur EC" -- the
    two load forecasts must never be merged into one another."""
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()

    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [50.0, 51.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame(
        {"load_actual": [1.0, 1.0], "load_forecast_day_ahead": [10.0, 11.0]}, index=idx
    ).to_parquet(load_dir / "DE_LU_2024-01.parquet")

    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()
    energy_charts_dir = tmp_path / "energy_charts"
    energy_charts_dir.mkdir()
    pd.DataFrame({"load_forecast_day_ahead_ec": [20.0, 21.0]}, index=idx).to_parquet(
        energy_charts_dir / "load_forecast_day_ahead_ec.parquet"
    )

    result = assemble_price_model_inputs(
        entsoe_sources=(
            _entsoe_source("day_ahead_price", price_dir),
            _entsoe_source("load", load_dir),
        ),
        commodity_sources=(
            ("ttf_gas", _never_call_row_fetch, "ttf_gas_eur_per_mwh"),
            ("eua_co2", _never_call_row_fetch, "eua_co2_eur_per_t"),
        ),
        commodities_dir=commodities_dir,
        energy_charts_dir=energy_charts_dir,
    )

    assert result["load_forecast_day_ahead"].tolist() == [10.0, 11.0]
    assert result["load_forecast_day_ahead_ec"].tolist() == [20.0, 21.0]


def test_read_quarterhourly_prices_fills_a_real_gap_from_energy_charts(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()
    idx = pd.date_range("2025-09-30 22:00", periods=4, freq="15min", tz="UTC")
    pd.DataFrame({"day_ahead_price": [1.0, 2.0, 3.0]}, index=idx[[0, 1, 3]]).to_parquet(
        price_dir / "DE_LU_2025-09.parquet"
    )  # the third quarter-hour is missing entirely, not just NaN

    energy_charts_dir = tmp_path / "energy_charts"
    energy_charts_dir.mkdir()
    # Position 3 matches ENTSO-E's own real 3.0 -- only position 2 (missing
    # from ENTSO-E entirely) is a genuine gap-fill, not a conflict.
    pd.DataFrame({"day_ahead_price_ec": [1.0, 2.0, 99.0, 3.0]}, index=idx).to_parquet(
        energy_charts_dir / "day_ahead_price_ec.parquet"
    )

    result = read_quarterhourly_prices(
        entsoe_sources=(_entsoe_source("day_ahead_price", price_dir),),
        energy_charts_dir=energy_charts_dir,
    )

    assert list(result.index) == list(idx)
    assert result["day_ahead_price"].tolist() == [1.0, 2.0, 99.0, 3.0]


# ---------------------------------------------------------------------------
# read_weather_runs
# ---------------------------------------------------------------------------


def test_read_weather_runs_concatenates_available_runs_and_skips_missing(tmp_path: Path) -> None:
    root = tmp_path / "weather_single_runs"
    day1 = dt.date(2024, 6, 2)
    day2 = dt.date(2024, 6, 3)  # deliberately not written -- must be silently skipped
    day3 = dt.date(2024, 6, 4)

    run1 = run_init_for_target_day(day1)
    run3 = run_init_for_target_day(day3)
    write_cached_run(
        _synthetic_weather_frame(run1), cache_path(run1, DEFAULT_WEATHER_MODEL, root=root)
    )
    write_cached_run(
        _synthetic_weather_frame(run3), cache_path(run3, DEFAULT_WEATHER_MODEL, root=root)
    )

    result = read_weather_runs([day1, day2, day3], root=root)

    assert set(result.index.get_level_values("run_init_utc")) == {run1, run3}
    assert len(result) == 4


def test_read_weather_runs_returns_empty_frame_when_nothing_cached(tmp_path: Path) -> None:
    result = read_weather_runs([dt.date(2024, 6, 2)], root=tmp_path / "weather_single_runs")
    assert result.empty


def test_read_weather_runs_with_no_target_days_returns_empty_frame(tmp_path: Path) -> None:
    result = read_weather_runs([], root=tmp_path / "weather_single_runs")
    assert result.empty


# ---------------------------------------------------------------------------
# read_quarterhourly_prices
# ---------------------------------------------------------------------------


def test_read_quarterhourly_prices_filters_to_the_cutover_and_dedups(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()

    pre_cutover = pd.date_range("2025-09-30 20:00", periods=2, freq="h", tz="UTC")
    post_cutover = pd.date_range("2025-09-30 22:00", periods=4, freq="15min", tz="UTC")
    pd.DataFrame({"day_ahead_price": [1.0, 2.0]}, index=pre_cutover).to_parquet(
        price_dir / "DE_LU_2025-09_a.parquet"
    )
    pd.DataFrame({"day_ahead_price": [3.0, 4.0, 5.0, 6.0]}, index=post_cutover).to_parquet(
        price_dir / "DE_LU_2025-09_b.parquet"
    )
    # Duplicate of the first post-cutover row -- must be dropped, first wins.
    pd.DataFrame({"day_ahead_price": [999.0]}, index=post_cutover[:1]).to_parquet(
        price_dir / "DE_LU_2025-09_c.parquet"
    )

    result = read_quarterhourly_prices(
        entsoe_sources=(_entsoe_source("day_ahead_price", price_dir),),
        energy_charts_dir=tmp_path / "energy_charts",
    )

    assert list(result.index) == list(post_cutover)
    assert list(result["day_ahead_price"]) == [3.0, 4.0, 5.0, 6.0]
