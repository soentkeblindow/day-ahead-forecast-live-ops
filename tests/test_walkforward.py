import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.metrics import summarise
from energy_price_forecast.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.models.baseline import SimilarDayNaive

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC")


def _flat_series(index: pd.DatetimeIndex, value: float = 50.0) -> pd.Series:
    return pd.Series(value, index=index, name="day_ahead_price")


class _SpyModel:
    """Tracks fit/predict call counts without any real computation."""

    def __init__(self) -> None:
        self.fit_count = 0
        self.predict_count = 0

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        self.fit_count += 1

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        self.predict_count += 1
        return pd.Series(0.0, index=test_index, name="y_pred")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_non_utc_index_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=48, freq="h")  # tz-naive
    with pytest.raises(ValueError, match="UTC"):
        list(walk_forward_splits(idx, test_start="2024-01-02"))


def test_rolling_without_span_raises() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-10 22:00")
    with pytest.raises(ValueError, match="train_span_days"):
        list(walk_forward_splits(idx, test_start="2024-01-05", window="rolling"))


# ---------------------------------------------------------------------------
# Leakage — the core contract
# ---------------------------------------------------------------------------


def test_no_future_in_train_expanding() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-20 22:00")
    folds = list(walk_forward_splits(idx, test_start="2024-01-10"))
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_index.max() < fold.test_index.min(), (
            f"Leakage on {fold.delivery_day}: "
            f"train.max={fold.train_index.max()} >= test.min={fold.test_index.min()}"
        )


def test_no_future_in_train_rolling() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-20 22:00")
    folds = list(
        walk_forward_splits(idx, test_start="2024-01-10", window="rolling", train_span_days=5)
    )
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_index.max() < fold.test_index.min()


# ---------------------------------------------------------------------------
# Expanding window
# ---------------------------------------------------------------------------


def test_expanding_window_grows() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-15 22:00")
    folds = list(walk_forward_splits(idx, test_start="2024-01-05"))
    assert len(folds) >= 2
    for prev, curr in zip(folds, folds[1:], strict=False):
        assert len(curr.train_index) > len(prev.train_index)


def test_expanding_train_is_superset_of_previous() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-10 22:00")
    folds = list(walk_forward_splits(idx, test_start="2024-01-04"))
    for prev, curr in zip(folds, folds[1:], strict=False):
        # Every timestamp in the previous training set must appear in the next
        assert set(prev.train_index).issubset(set(curr.train_index))


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------


def test_rolling_window_has_constant_span() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-20 22:00")
    folds = list(
        walk_forward_splits(idx, test_start="2024-01-10", window="rolling", train_span_days=7)
    )
    assert len(folds) > 0
    for fold in folds:
        local_days = fold.train_index.tz_convert("Europe/Berlin").normalize().nunique()
        assert local_days == 7, f"Expected 7 training days, got {local_days}"


def test_rolling_skips_folds_without_full_window() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-10 22:00")
    folds = list(
        walk_forward_splits(idx, test_start="2024-01-03", window="rolling", train_span_days=7)
    )
    # Every yielded fold must have a full 7-day training window
    for fold in folds:
        local_days = fold.train_index.tz_convert("Europe/Berlin").normalize().nunique()
        assert local_days == 7


# ---------------------------------------------------------------------------
# Gate closure
# ---------------------------------------------------------------------------


def test_gate_closure_summer() -> None:
    # D = 2024-06-15 (CEST UTC+2): gate_closure on D-1 = 2024-06-14 12:00 CEST = 10:00 UTC
    idx = _hourly_utc("2024-06-12 22:00", "2024-06-15 21:00")
    folds = list(walk_forward_splits(idx, test_start="2024-06-15", test_end="2024-06-15"))
    assert len(folds) == 1
    expected = pd.Timestamp("2024-06-14 10:00", tz="UTC")
    assert folds[0].gate_closure == expected, f"Got {folds[0].gate_closure}"


def test_gate_closure_winter() -> None:
    # D = 2024-01-15 (CET UTC+1): gate_closure on D-1 = 2024-01-14 12:00 CET = 11:00 UTC
    idx = _hourly_utc("2024-01-12 23:00", "2024-01-15 22:00")
    folds = list(walk_forward_splits(idx, test_start="2024-01-15", test_end="2024-01-15"))
    assert len(folds) == 1
    expected = pd.Timestamp("2024-01-14 11:00", tz="UTC")
    assert folds[0].gate_closure == expected, f"Got {folds[0].gate_closure}"


# ---------------------------------------------------------------------------
# DST — delivery day length regression tests
# ---------------------------------------------------------------------------


def test_dst_spring_forward_23_hours() -> None:
    # 2024-03-31: clocks spring forward at 02:00 CET → 03:00 CEST; 23-hour day.
    # UTC range: 2024-03-30 23:00 to 2024-03-31 21:00 (23 timestamps).
    idx = _hourly_utc("2024-03-28 23:00", "2024-03-31 21:00")
    folds = list(walk_forward_splits(idx, test_start="2024-03-31", test_end="2024-03-31"))
    assert len(folds) == 1, "Expected one fold for the spring-forward day"
    assert len(folds[0].test_index) == 23, (
        f"Expected 23 hours for DST spring day, got {len(folds[0].test_index)}"
    )


def test_dst_fall_back_25_hours() -> None:
    # 2024-10-27: clocks fall back at 03:00 CEST → 02:00 CET; 25-hour day.
    # UTC range: 2024-10-26 22:00 to 2024-10-27 22:00 (25 timestamps).
    idx = _hourly_utc("2024-10-25 22:00", "2024-10-27 22:00")
    folds = list(walk_forward_splits(idx, test_start="2024-10-27", test_end="2024-10-27"))
    assert len(folds) == 1, "Expected one fold for the fall-back day"
    assert len(folds[0].test_index) == 25, (
        f"Expected 25 hours for DST autumn day, got {len(folds[0].test_index)}"
    )


# ---------------------------------------------------------------------------
# refit_every
# ---------------------------------------------------------------------------


def test_refit_every_controls_fit_cadence() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-20 22:00")
    y = _flat_series(idx)
    folds = list(walk_forward_splits(idx, test_start="2024-01-10", test_end="2024-01-19"))
    n = len(folds)
    assert n == 10

    spy = _SpyModel()
    run_backtest(y, spy, folds, refit_every=3)

    expected_fits = sum(1 for i in range(n) if i % 3 == 0)  # folds 0, 3, 6, 9
    assert spy.fit_count == expected_fits
    assert spy.predict_count == n


def test_refit_every_1_fits_every_fold() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-10 22:00")
    y = _flat_series(idx)
    folds = list(walk_forward_splits(idx, test_start="2024-01-05", test_end="2024-01-09"))
    spy = _SpyModel()
    run_backtest(y, spy, folds, refit_every=1)
    assert spy.fit_count == len(folds)


def test_fold_0_always_fits() -> None:
    idx = _hourly_utc("2024-01-01 23:00", "2024-01-10 22:00")
    y = _flat_series(idx)
    folds = list(walk_forward_splits(idx, test_start="2024-01-05", test_end="2024-01-05"))
    spy = _SpyModel()
    run_backtest(y, spy, folds, refit_every=99)
    assert spy.fit_count == 1


# ---------------------------------------------------------------------------
# SimilarDayNaive correctness
# ---------------------------------------------------------------------------


def test_naive_tuesday_24h_lag() -> None:
    # Berlin Mon 2024-01-08: UTC 2024-01-07 23:00 – 2024-01-08 22:00
    # Berlin Tue 2024-01-09: UTC 2024-01-08 23:00 – 2024-01-09 22:00
    idx = _hourly_utc("2024-01-07 23:00", "2024-01-09 22:00")
    # Unique values so each position is identifiable
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)

    train_idx = idx[idx < pd.Timestamp("2024-01-08 23:00", tz="UTC")]  # Mon
    test_idx = idx[idx >= pd.Timestamp("2024-01-08 23:00", tz="UTC")]  # Tue

    pred = SimilarDayNaive().predict(test_idx, history=prices.loc[train_idx])

    assert not pred.isna().any(), "No NaN expected for 24h lag with complete history"
    # Tue test_idx[i] - 24h == Mon train_idx[i] → pred should equal prices[train_idx]
    np.testing.assert_array_equal(pred.to_numpy(), prices.loc[train_idx].to_numpy())


def test_naive_monday_168h_lag() -> None:
    # Build 8 Berlin days starting Mon 2024-01-01.
    # Berlin Jan 1 (Mon): UTC 2023-12-31 23:00 – 2024-01-01 22:00
    # Berlin Jan 8 (Mon): UTC 2024-01-07 23:00 – 2024-01-08 22:00  ← test day
    idx = _hourly_utc("2023-12-31 23:00", "2024-01-08 22:00")  # 9 Berlin days
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)

    train_idx = idx[idx < pd.Timestamp("2024-01-07 23:00", tz="UTC")]
    test_idx = idx[idx >= pd.Timestamp("2024-01-07 23:00", tz="UTC")]

    pred = SimilarDayNaive().predict(test_idx, history=prices.loc[train_idx])

    assert not pred.isna().any(), "No NaN expected: Jan 8 - 168h falls on Jan 1 (in history)"
    # 168h lag: test_idx[i] - 168h should be train_idx position 7*24 before
    expected = prices.reindex(test_idx - pd.Timedelta(hours=168)).to_numpy()
    np.testing.assert_array_equal(pred.to_numpy(), expected)


def test_naive_missing_lag_yields_nan() -> None:
    # Only 1 hour of history; all lags (24h or 168h) point outside it → all NaN.
    idx = _hourly_utc("2024-01-07 23:00", "2024-01-08 22:00")  # Mon Jan 8
    history = _flat_series(idx[:1])  # single row; no 7-day lookback available
    test_idx = idx

    pred = SimilarDayNaive().predict(test_idx, history=history)
    assert pred.isna().all(), "Expected all NaN when lag history is absent"


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_e2e_smoke(tmp_path: "Path") -> None:  # type: ignore[name-defined]  # noqa: F821
    import mlflow

    # 14 training days + 7 test days; constant price → SimilarDay MAE = 0
    idx = _hourly_utc("2023-12-18 23:00", "2024-01-07 22:00")
    y = _flat_series(idx, value=60.0)

    folds = list(walk_forward_splits(idx, test_start="2024-01-02", test_end="2024-01-07"))
    assert len(folds) == 6

    predictions = run_backtest(y, SimilarDayNaive(), folds)
    summary = summarise(predictions)

    assert all(np.isfinite(v) for v in summary.values()), "All metrics must be finite"
    assert summary["mae"] == pytest.approx(0.0, abs=1e-9)

    # Verify MLflow run is recorded
    tracking_uri = f"file:{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("smoke_test")
    with mlflow.start_run(run_name="smoke"):
        mlflow.log_metrics(summary)

    runs = mlflow.search_runs(experiment_names=["smoke_test"])
    assert len(runs) == 1, "Expected exactly one MLflow run"
