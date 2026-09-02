"""Tests for Sprint 6.5.3: NWP-reconstruction feature integration and the
knowledge-time contract (leakage) it must satisfy.

Most tests use synthetic data (fast, portable, run everywhere). A handful of
real-data smoke tests are marked skipif when the local interim/processed
artefacts are absent (same pattern as test_quarterhourly.py's own local
integrity checks) -- they are not a CI gate, those artefacts are gitignored.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.loaders import load_interim_hourly, load_renewables_predictions
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.features.availability import LeakageError, assert_no_leakage
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.fundamentals import build_forecast_fundamentals
from energy_price_forecast.features.nwp_fundamentals import (
    IncompleteReconstructionError,
    build_nwp_forecast_fundamentals,
    build_residual_load_nwp,
)
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.ops.windows import local_day_bounds

_HOURLY_PATH = PROJECT_ROOT / "data" / "interim" / "hourly.parquet"
_RENEWABLES_PATH = PROJECT_ROOT / "data" / "processed" / "renewables_forecast_rolling365_l2.parquet"

_needs_real_data = pytest.mark.skipif(
    not _HOURLY_PATH.exists() or not _RENEWABLES_PATH.exists(),
    reason="local interim/processed data not present -- local integrity check, not a CI gate",
)


def _local_hourly_index(target_day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _synthetic_predictions(
    target_index: pd.DatetimeIndex,
    run_init: pd.Timestamp,
    *,
    won: float = 8000.0,
    woff: float = 2000.0,
    solar: float = 5000.0,
) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([run_init] * len(target_index), tz="UTC"), target_index],
        names=["run_init_utc", "valid_time_utc"],
    )
    return pd.DataFrame(
        {
            "wind_onshore_mw_pred": np.full(len(target_index), won),
            "wind_offshore_mw_pred": np.full(len(target_index), woff),
            "solar_mw_pred": np.full(len(target_index), solar),
        },
        index=index,
    )


def _synthetic_load_df(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"load_forecast_day_ahead": 40000.0}, index=target_index)


def _make_full_df_with_lags(
    periods: int = 10 * 24, start: str = "2024-01-01 00:00"
) -> pd.DataFrame:
    """Same shape as test_features.py's own fixture of the same purpose --
    every column build_feature_set_for_day's unchanged calendar/commodity/
    lag pieces need, filled with constant values. Insufficient lookback for
    a target day near the start only produces NaN in unrelated lag columns
    (not an error, not a leakage violation), so exact lookback depth doesn't
    matter for the properties these tests check."""
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "day_ahead_price": np.full(periods, 50.0),
            "load_actual": np.full(periods, 40000.0),
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "gen_wind_onshore": np.full(periods, 8000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "gen_wind_offshore": np.full(periods, 2000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "gen_solar": np.full(periods, 5000.0),
            "solar_forecast": np.full(periods, 5000.0),
            "scheduled_net_de_to_AT": np.full(periods, 1000.0),
            "scheduled_net_de_to_BE": np.full(periods, 500.0),
            "physical_net_de_to_AT": np.full(periods, 1200.0),
            "physical_net_de_to_BE": np.full(periods, 600.0),
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": np.full(periods, 70.0),
        },
        index=idx,
    )


_TARGET_DAY = dt.date(2024, 1, 8)


# ---------------------------------------------------------------------------
# 5.1 Leakage test with negative control -- the core of this step
# ---------------------------------------------------------------------------


def test_leakage_positive_nwp_features_known_before_gate_closure() -> None:
    target_index = _local_hourly_index(_TARGET_DAY)
    run_init = run_init_for_target_day(_TARGET_DAY)
    predictions = _synthetic_predictions(target_index, run_init)
    df = _synthetic_load_df(target_index)

    feats = build_nwp_forecast_fundamentals(df, predictions, target_index)
    assert_no_leakage(feats)  # must not raise

    gc = gate_closure_for_index(target_index)
    for f in feats:
        assert (f.knowledge_time <= gc).all(), f.name


def test_leakage_negative_control_run_from_target_day_itself_is_caught() -> None:
    """Deliberately break the contract: substitute a run initialised on D
    itself (instead of D-1) and confirm the leakage check demonstrably goes
    red (spec 6.5.3 section 5.1) -- a leakage test that has never actually
    been run red is a claim, not a proof."""
    target_index = _local_hourly_index(_TARGET_DAY)
    bad_run_init = pd.Timestamp(_TARGET_DAY, tz="UTC")  # D's own 00:00 UTC, not D-1's
    predictions = _synthetic_predictions(target_index, bad_run_init)
    df = _synthetic_load_df(target_index)

    feats = build_nwp_forecast_fundamentals(df, predictions, target_index)
    with pytest.raises(LeakageError):
        assert_no_leakage(feats)


def test_leakage_negative_control_run_from_d_plus_one_is_caught() -> None:
    target_index = _local_hourly_index(_TARGET_DAY)
    bad_run_init = pd.Timestamp(_TARGET_DAY + dt.timedelta(days=1), tz="UTC")
    predictions = _synthetic_predictions(target_index, bad_run_init)
    df = _synthetic_load_df(target_index)

    feats = build_nwp_forecast_fundamentals(df, predictions, target_index)
    with pytest.raises(LeakageError):
        assert_no_leakage(feats)


# ---------------------------------------------------------------------------
# 5.2 Identity probe for the residual load formula
# ---------------------------------------------------------------------------


def test_residual_load_formula_identity_against_tso_series() -> None:
    """Feeding build_residual_load_nwp the TSO-sourced Features instead of
    the NWP reconstruction must reproduce the original residual_load_forecast
    column bit-exact (spec 6.5.3 section 5.2) -- proves the *formula* was
    reproduced exactly, not just that a different source was wired in."""
    df = _make_full_df_with_lags()
    target_index = _local_hourly_index(_TARGET_DAY)
    old_feats = {f.name: f for f in build_forecast_fundamentals(df, target_index)}

    new_residual = build_residual_load_nwp(
        old_feats["load_forecast_day_ahead"],
        old_feats["wind_onshore_forecast"],
        old_feats["wind_offshore_forecast"],
        old_feats["solar_forecast"],
    )

    pd.testing.assert_series_equal(
        new_residual.values, old_feats["residual_load_forecast"].values, check_names=False
    )
    pd.testing.assert_series_equal(
        new_residual.knowledge_time, old_feats["residual_load_forecast"].knowledge_time
    )


# ---------------------------------------------------------------------------
# 5.3 No DA_FORECAST column in the live feature set
# ---------------------------------------------------------------------------

_FORBIDDEN_LIVE_COLUMNS = frozenset(
    {"wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"}
)


def _assert_no_da_forecast_renewables_column(columns: Iterable[str]) -> None:
    leaked = _FORBIDDEN_LIVE_COLUMNS & set(columns)
    if leaked:
        raise AssertionError(
            f"DA_FORECAST-class renewables column(s) leaked into the live feature set: "
            f"{sorted(leaked)}"
        )


def test_no_da_forecast_renewables_column_in_live_feature_set() -> None:
    df = _make_full_df_with_lags()
    target_index = _local_hourly_index(_TARGET_DAY)
    run_init = run_init_for_target_day(_TARGET_DAY)
    predictions = _synthetic_predictions(target_index, run_init)

    matrix = build_feature_set_for_day(_TARGET_DAY, df, predictions)
    _assert_no_da_forecast_renewables_column(matrix.columns)  # must not raise


def test_no_da_forecast_check_negative_control_catches_old_columns() -> None:
    """The check above is not vacuously true: it correctly flags the old,
    TSO-based fundamentals' own column set (spec 6.5.3 section 5.3)."""
    df = _make_full_df_with_lags()
    target_index = _local_hourly_index(_TARGET_DAY)
    old_columns = [f.name for f in build_forecast_fundamentals(df, target_index)]

    with pytest.raises(AssertionError, match="leaked"):
        _assert_no_da_forecast_renewables_column(old_columns)


# ---------------------------------------------------------------------------
# 5.4 Remaining: backtest==live identity, DST days, NaN policy, no clock
# ---------------------------------------------------------------------------


def test_backtest_and_live_calls_produce_identical_output_for_the_same_day() -> None:
    """E7 as a test, not just an assertion (spec 6.5.3 section 5.4): two
    calls with the same target_day and the same already-loaded data return
    bit-identical output -- there is only one code path to call twice."""
    df = _make_full_df_with_lags()
    target_index = _local_hourly_index(_TARGET_DAY)
    run_init = run_init_for_target_day(_TARGET_DAY)
    predictions = _synthetic_predictions(target_index, run_init)

    m1 = build_feature_set_for_day(_TARGET_DAY, df, predictions)
    m2 = build_feature_set_for_day(_TARGET_DAY, df, predictions)
    pd.testing.assert_frame_equal(m1, m2)


@pytest.mark.parametrize(
    ("target_day", "expected_rows"),
    [
        (dt.date(2025, 3, 30), 23),  # DE/LU DST start 2025 -- spring forward
        (dt.date(2025, 10, 26), 25),  # DE/LU DST end 2025 -- fall back
    ],
)
def test_dst_days_correct_row_count_and_single_run_init(
    target_day: dt.date, expected_rows: int
) -> None:
    target_index = _local_hourly_index(target_day)
    assert len(target_index) == expected_rows

    run_init = run_init_for_target_day(target_day)
    predictions = _synthetic_predictions(target_index, run_init)
    df = _synthetic_load_df(target_index)

    feats = build_nwp_forecast_fundamentals(df, predictions, target_index)
    by_name = {f.name: f for f in feats}
    for f in feats:
        assert len(f.values) == expected_rows, f.name

    # Only the three pure NWP_RECONSTRUCTION features carry run_init as their
    # knowledge time on the nose -- load stays DA_FORECAST (gate closure) and
    # residual_load_forecast_nwp is the elementwise max of all four, which
    # equals load's later gate closure, not run_init (spec 6.5.3 combine()
    # semantics, unchanged).
    for name in ("wind_onshore_forecast_nwp", "wind_offshore_forecast_nwp", "solar_forecast_nwp"):
        kt = by_name[name].knowledge_time
        assert kt.nunique() == 1, name
        assert kt.iloc[0] == run_init, name


def test_incomplete_reconstruction_raises_on_a_single_nan_value() -> None:
    """One NaN hour in a required reconstruction series is enough to reject
    the whole target day (spec 6.5.3 section 3.3) -- never filled."""
    target_index = _local_hourly_index(_TARGET_DAY)
    run_init = run_init_for_target_day(_TARGET_DAY)
    predictions = _synthetic_predictions(target_index, run_init)
    predictions.loc[predictions.index[3], "wind_onshore_mw_pred"] = float("nan")
    df = _synthetic_load_df(target_index)

    with pytest.raises(IncompleteReconstructionError, match="wind_onshore_mw_pred"):
        build_nwp_forecast_fundamentals(df, predictions, target_index)


def test_incomplete_reconstruction_raises_when_an_hour_is_missing_entirely() -> None:
    target_index = _local_hourly_index(_TARGET_DAY)
    run_init = run_init_for_target_day(_TARGET_DAY)
    predictions = _synthetic_predictions(target_index, run_init).iloc[1:]
    df = _synthetic_load_df(target_index)

    with pytest.raises(IncompleteReconstructionError, match="missing entirely"):
        build_nwp_forecast_fundamentals(df, predictions, target_index)


_CLOCK_ACCESS_PATTERNS = (
    "datetime.now(",
    "date.today(",
    "Timestamp.now(",
    "Timestamp.today(",
    ".now()",
    ".today()",
)


def test_builder_source_never_touches_the_system_clock() -> None:
    """spec 6.5.3 section 3.4 (E7): the day-builder and the NWP fundamentals
    it calls into may only ever derive knowledge from the target_day
    argument, never datetime.now()/date.today()/Timestamp.now() -- checked
    by source inspection, since datetime.date/datetime and pd.Timestamp are
    immutable C-extension types that cannot be monkeypatched to prove a
    negative dynamically (confirmed: assigning to date.today/datetime.now
    raises TypeError on this Python)."""
    modules = [
        PROJECT_ROOT / "src" / "energy_price_forecast" / "features" / "build.py",
        PROJECT_ROOT / "src" / "energy_price_forecast" / "features" / "nwp_fundamentals.py",
    ]
    for path in modules:
        text = path.read_text(encoding="utf-8")
        for pattern in _CLOCK_ACCESS_PATTERNS:
            assert pattern not in text, f"{path} appears to call the system clock via {pattern!r}"


# ---------------------------------------------------------------------------
# Real-data smoke tests (local integrity checks, not a CI gate)
# ---------------------------------------------------------------------------


@_needs_real_data
def test_real_data_smoke_normal_day() -> None:
    df = load_interim_hourly()
    predictions = load_renewables_predictions()
    matrix = build_feature_set_for_day(dt.date(2025, 6, 15), df, predictions)
    assert len(matrix) == 24
    _assert_no_da_forecast_renewables_column(matrix.columns)


@_needs_real_data
def test_real_data_smoke_dst_fall_back_day_is_excluded() -> None:
    """The one real DST fall-back day in the artefact's coverage window
    (2025-10-26) has a genuine single-hour ENTSO-E gap (wind_onshore_forecast
    and solar_forecast both NaN at 2025-10-25 22:00 UTC) that propagates to a
    whole-day skip in the upstream renewables walk-forward
    (docs/sprint6_step6_5_3_log.md has the full root-cause trace) -- the NaN
    policy here must exclude the day outright, not partially deliver it."""
    df = load_interim_hourly()
    predictions = load_renewables_predictions()
    with pytest.raises(IncompleteReconstructionError):
        build_feature_set_for_day(dt.date(2025, 10, 26), df, predictions)


@_needs_real_data
def test_real_data_backtest_and_live_calls_agree() -> None:
    df = load_interim_hourly()
    predictions = load_renewables_predictions()
    m1 = build_feature_set_for_day(dt.date(2025, 6, 15), df, predictions)
    m2 = build_feature_set_for_day(dt.date(2025, 6, 15), df, predictions)
    pd.testing.assert_frame_equal(m1, m2)
