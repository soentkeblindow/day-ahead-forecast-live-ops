"""Unit tests for the quarter-hourly Energy-Arena walk-forward harness."""

import datetime as dt

import numpy as np
import pandas as pd

from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.evaluation.arena_walkforward import (
    _hourly_persistence_forecast,
    run_arena_backtest,
    run_live_gate_backtest,
)
from energy_price_forecast.features.build import (
    build_feature_set_for_day,
    build_original_feature_set_for_day,
)

_TZ = "Europe/Berlin"
_SHAPE = np.array([2.0, -2.0, 1.0, -1.0])


class _RecordingMeanModel:
    """Minimal Forecaster: predicts the training mean, records every fit()
    call's training index so tests can assert on it directly."""

    def __init__(self) -> None:
        self.fit_calls: list[pd.DatetimeIndex] = []
        self._mean = 0.0

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        self.fit_calls.append(pd.DatetimeIndex(y_train.index))
        self._mean = float(y_train.mean())

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        return pd.Series(self._mean, index=test_index)


def _build_hourly(start: str, n_days: int, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n_days * 24, freq="h")
    rng = np.random.default_rng(seed)
    y = pd.Series(50 + rng.normal(0, 5, len(idx)), index=idx)
    x = pd.DataFrame({"f1": rng.normal(0, 1, len(idx))}, index=idx)
    return y, x


def _build_quarterhourly(
    y_hourly: pd.Series, start_date: dt.date, n_days: int, shape: np.ndarray = _SHAPE
) -> pd.Series:
    # Calendar-date arithmetic on a bare date, then a fresh per-day tz
    # localization -- NOT pd.date_range(freq="D") on an already tz-aware
    # start, which advances by a fixed 24h and produces overlapping /
    # duplicate timestamps across a DST boundary (exactly the pitfall spec
    # 6.4 section 11 warns about).
    idx_parts: list[pd.DatetimeIndex] = []
    val_parts: list[np.ndarray] = []
    for i in range(n_days):
        day = start_date + dt.timedelta(days=i)
        day_start = pd.Timestamp(day, tz=_TZ)
        day_end = pd.Timestamp(day + dt.timedelta(days=1), tz=_TZ)
        qh = pd.date_range(day_start, day_end, freq="15min", inclusive="left")
        for h in range(len(qh) // 4):
            hour_ts = qh[h * 4].floor("h")
            base = float(y_hourly.get(hour_ts, 50.0))
            idx_parts.append(qh[h * 4 : (h + 1) * 4])
            val_parts.append(base + shape)
    index = pd.DatetimeIndex(np.concatenate([list(p) for p in idx_parts])).tz_convert("UTC")
    return pd.Series(np.concatenate(val_parts), index=index)


def test_fold_row_counts_match_dst_expectation() -> None:
    """A window spanning the 2026 spring-forward day (2026-03-29) yields 92
    rows for that delivery day and 96 for its ordinary neighbours."""
    y_hourly, x_hourly = _build_hourly("2026-01-01", 120)
    prices_qh = _build_quarterhourly(y_hourly, dt.date(2026, 3, 15), 25)  # covers 03-15..04-08

    result = run_arena_backtest(
        y_hourly,
        x_hourly,
        prices_qh,
        _RecordingMeanModel(),
        shape_window_days=7,
        train_span_days=30,
    )

    spring_day = pd.Timestamp("2026-03-29", tz=_TZ)
    spring_rows = result[result["delivery_day"] == spring_day]
    assert len(spring_rows) == 92
    assert (spring_rows["n_slots_in_day"] == 92).all()

    ordinary_day = pd.Timestamp("2026-03-25", tz=_TZ)
    ordinary_rows = result[result["delivery_day"] == ordinary_day]
    assert len(ordinary_rows) == 96
    assert (ordinary_rows["n_slots_in_day"] == 96).all()


def test_no_fold_trains_on_data_at_or_after_its_delivery_day() -> None:
    y_hourly, x_hourly = _build_hourly("2026-01-01", 90)
    prices_qh = _build_quarterhourly(y_hourly, dt.date(2026, 2, 15), 20)

    model = _RecordingMeanModel()
    result = run_arena_backtest(
        y_hourly, x_hourly, prices_qh, model, shape_window_days=7, train_span_days=30
    )

    delivery_days = pd.DatetimeIndex(result["delivery_day"].drop_duplicates().sort_values())
    assert len(model.fit_calls) == len(delivery_days)
    for train_index, delivery_day in zip(model.fit_calls, delivery_days, strict=True):
        assert train_index.max() < delivery_day


def test_start_day_is_first_with_n_full_prior_days_not_hardcoded() -> None:
    y_hourly, x_hourly = _build_hourly("2026-01-01", 90)
    qh_start = dt.date(2026, 2, 1)
    prices_qh = _build_quarterhourly(y_hourly, qh_start, 15)

    result = run_arena_backtest(
        y_hourly,
        x_hourly,
        prices_qh,
        _RecordingMeanModel(),
        shape_window_days=7,
        train_span_days=30,
    )

    first_delivery_day = result["delivery_day"].min()
    expected_first_day = pd.Timestamp(qh_start, tz=_TZ) + pd.Timedelta(days=7)
    assert first_delivery_day == expected_first_day


def test_output_shape_and_index_integrity() -> None:
    y_hourly, x_hourly = _build_hourly("2026-01-01", 90)
    prices_qh = _build_quarterhourly(y_hourly, dt.date(2026, 2, 1), 15)

    result = run_arena_backtest(
        y_hourly,
        x_hourly,
        prices_qh,
        _RecordingMeanModel(),
        shape_window_days=7,
        train_span_days=30,
    )

    expected_columns = {
        "y_true",
        "pred_bridge_shape",
        "pred_bridge_flat",
        "pred_baseline",
        "delivery_day",
        "n_slots_in_day",
    }
    assert set(result.columns) == expected_columns
    assert not result.index.has_duplicates

    for _, group in result.groupby("delivery_day"):
        gaps = group.index.to_series().diff().dropna().unique()
        assert len(gaps) == 1
        assert gaps[0] == pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------------
# 6.6: run_live_gate_backtest -- two feature sets, both resolutions
# ---------------------------------------------------------------------------


class _RecordingMeanModelPerCandidate(_RecordingMeanModel):
    """Same recording behaviour as _RecordingMeanModel; kept as a distinct
    name so a test can hold one instance per candidate without confusion."""


def _build_full_hourly_df(periods: int, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """Every raw column build_feature_set_for_day/build_original_feature_set_for_day
    need, filled with constant values -- same shape as
    test_feature_integration.py's own fixture of the same purpose."""
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


def _build_renewables_predictions_for_range(days: list[dt.date]) -> pd.DataFrame:
    """One synthetic NWP-reconstruction block per day, correct run_init_utc
    (spec 6.5.3's knowledge-time contract), concatenated into one artefact
    covering every day in ``days``."""
    frames = []
    for day in days:
        start = pd.Timestamp(day, tz=_TZ)
        end = pd.Timestamp(day + dt.timedelta(days=1), tz=_TZ)
        target_index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
        run_init = run_init_for_target_day(day)
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(target_index), tz="UTC"), target_index],
            names=["run_init_utc", "valid_time_utc"],
        )
        frames.append(
            pd.DataFrame(
                {
                    "wind_onshore_mw_pred": np.full(len(target_index), 8000.0),
                    "wind_offshore_mw_pred": np.full(len(target_index), 2000.0),
                    "solar_mw_pred": np.full(len(target_index), 5000.0),
                },
                index=index,
            )
        )
    return pd.concat(frames)


_LIVE_GATE_DAYS = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]


def test_two_feature_sets_differ_as_expected() -> None:
    """spec 6.6 section 7: 'original' carries the three TSO forecast columns
    plus renewable_share_forecast; 'live' carries their _nwp counterparts
    and none of the originals. Checked as set membership, not a sample."""
    df = _build_full_hourly_df(30 * 24)
    predictions = _build_renewables_predictions_for_range(_LIVE_GATE_DAYS[:30])
    target_day = dt.date(2024, 1, 20)

    original = build_original_feature_set_for_day(target_day, df)
    live = build_feature_set_for_day(target_day, df, predictions)

    tso_only = {
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
        "renewable_share_forecast",
    }
    nwp_only = {
        "wind_onshore_forecast_nwp",
        "wind_offshore_forecast_nwp",
        "solar_forecast_nwp",
        "renewable_share_forecast_nwp",
    }

    assert tso_only <= set(original.columns)
    assert not (tso_only & set(live.columns))
    assert nwp_only <= set(live.columns)
    assert not (nwp_only & set(original.columns))


def test_run_live_gate_backtest_quarterhourly_produces_two_candidates() -> None:
    df = _build_full_hourly_df(60 * 24)
    predictions = _build_renewables_predictions_for_range(_LIVE_GATE_DAYS)
    y_hourly = df["day_ahead_price"]
    prices_qh = _build_quarterhourly(y_hourly, dt.date(2024, 1, 15), 35)

    result = run_live_gate_backtest(
        df,
        predictions,
        _RecordingMeanModelPerCandidate(),
        _RecordingMeanModelPerCandidate(),
        resolution="quarterhourly",
        prices_qh=prices_qh,
        shape_window_days=5,
        train_span_days=10,
    )

    assert set(result.columns) == {
        "y_true",
        "pred_original",
        "pred_live",
        "pred_baseline",
        "delivery_day",
        "n_slots_in_day",
    }
    assert not result.index.has_duplicates
    assert result["delivery_day"].nunique() > 0
    assert result["pred_original"].notna().all()
    assert result["pred_live"].notna().all()


def test_run_live_gate_backtest_hourly_resolution_no_shape_profile() -> None:
    """spec 6.6 section 7: --resolution hourly builds no shape profile and
    never produces quarter-hourly slot counts (92/96/100)."""
    df = _build_full_hourly_df(60 * 24)
    predictions = _build_renewables_predictions_for_range(_LIVE_GATE_DAYS)

    result = run_live_gate_backtest(
        df,
        predictions,
        _RecordingMeanModelPerCandidate(),
        _RecordingMeanModelPerCandidate(),
        resolution="hourly",
        train_span_days=10,
    )

    assert set(result.columns) == {
        "y_true",
        "pred_original",
        "pred_live",
        "pred_baseline",
        "delivery_day",
        "n_slots_in_day",
    }
    assert result["n_slots_in_day"].between(20, 26).all()
    assert not result["n_slots_in_day"].isin([92, 96, 100]).any()


def test_missing_nwp_day_excludes_the_fold_for_both_candidates() -> None:
    """spec 6.6 section 3.4: a day the live feature set cannot build
    (incomplete NWP reconstruction) is excluded for the original candidate
    too -- pred_original and pred_live are always defined on identical
    delivery days."""
    df = _build_full_hourly_df(60 * 24)
    excluded_day = dt.date(2024, 1, 25)
    predictions = _build_renewables_predictions_for_range(
        [d for d in _LIVE_GATE_DAYS if d != excluded_day]
    )

    result = run_live_gate_backtest(
        df,
        predictions,
        _RecordingMeanModelPerCandidate(),
        _RecordingMeanModelPerCandidate(),
        resolution="hourly",
        train_span_days=10,
    )

    delivery_days = pd.DatetimeIndex(result["delivery_day"].unique())
    assert excluded_day not in delivery_days.date


def test_both_models_train_only_on_data_before_delivery_day() -> None:
    """spec 6.6 section 7: the training window ends at D-1 and never
    includes D -- for both feature sets."""
    df = _build_full_hourly_df(60 * 24)
    predictions = _build_renewables_predictions_for_range(_LIVE_GATE_DAYS)
    y_hourly = df["day_ahead_price"]
    prices_qh = _build_quarterhourly(y_hourly, dt.date(2024, 1, 15), 35)

    model_live = _RecordingMeanModelPerCandidate()
    model_original = _RecordingMeanModelPerCandidate()
    result = run_live_gate_backtest(
        df,
        predictions,
        model_live,
        model_original,
        resolution="quarterhourly",
        prices_qh=prices_qh,
        shape_window_days=5,
        train_span_days=10,
    )

    delivery_days = pd.DatetimeIndex(result["delivery_day"].drop_duplicates().sort_values())
    assert len(model_live.fit_calls) == len(delivery_days)
    assert len(model_original.fit_calls) == len(delivery_days)
    for train_index, delivery_day in zip(model_live.fit_calls, delivery_days, strict=True):
        assert train_index.max() < delivery_day
    for train_index, delivery_day in zip(model_original.fit_calls, delivery_days, strict=True):
        assert train_index.max() < delivery_day


def test_hourly_persistence_baseline_matches_d_minus_1_wall_clock() -> None:
    """spec 6.6 section 3.3: the hourly baseline is the realised price of
    the same wall-clock hour on D-1 -- not the mean, not D itself."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)  # strictly increasing, no ties
    target_day = pd.Timestamp("2024-01-03", tz=_TZ)

    baseline = _hourly_persistence_forecast(y, target_day, tz=_TZ)

    prev_day_start = pd.Timestamp("2024-01-02", tz=_TZ)
    prev_day_end = prev_day_start + pd.Timedelta(days=1)
    expected = y.loc[(y.index >= prev_day_start) & (y.index < prev_day_end)].to_numpy()
    np.testing.assert_array_equal(baseline.to_numpy(), expected)
