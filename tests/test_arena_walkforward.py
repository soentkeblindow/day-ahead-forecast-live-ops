"""Unit tests for the quarter-hourly Energy-Arena walk-forward harness."""

import datetime as dt

import numpy as np
import pandas as pd

from energy_price_forecast.evaluation.arena_walkforward import run_arena_backtest

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
