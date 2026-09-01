import datetime as dt
import logging

import numpy as np
import pandas as pd
import pytest
import pytz

from energy_price_forecast.data import capacity as capacity_mod
from energy_price_forecast.data.capacity import CapacitySource
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import expected_columns
from energy_price_forecast.evaluation.renewables_walkforward import (
    persistence_baseline_cf,
    run_renewables_backtest,
)
from energy_price_forecast.features.renewables_forecast import _target_hours_for_day
from energy_price_forecast.models.renewables import RenewablesModel


def _patch_capacity_anchors(monkeypatch: pytest.MonkeyPatch, value: float = 1000.0) -> None:
    anchors = pd.Series(
        [value, value], index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], tz="UTC")
    )
    monkeypatch.setitem(
        capacity_mod._ANCHOR_LOADERS, CapacitySource.PUBLIC_REGISTRY, lambda pt: anchors
    )


def _weather_for_days(days: list[dt.date]) -> pd.DataFrame:
    cols = expected_columns()
    rng = np.random.default_rng(0)
    frames = []
    for day in days:
        run_init = run_init_for_target_day(day)
        hours = _target_hours_for_day(day)
        valid_times = pd.DatetimeIndex(hours.union(hours + pd.Timedelta(hours=1)))
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(valid_times), tz="UTC"), valid_times],
            names=["run_init_utc", "valid_time_utc"],
        )
        # Small positive noise around a stable baseline -- keeps wind speeds,
        # radiation etc. plausible (non-negative) without caring about exact
        # physical realism; this file tests fold mechanics, not forecast skill.
        data = {c: 5.0 + rng.normal(0, 0.1, len(valid_times)) for c in cols}
        frames.append(pd.DataFrame(data, index=idx, dtype="float64"))
    return pd.concat(frames)


def _target_hourly_for_days(days: list[dt.date], base: float = 300.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    all_hours = pd.DatetimeIndex(sorted({h for day in days for h in _target_hours_for_day(day)}))
    data = {
        "wind_onshore_forecast": base + rng.normal(0, 20, len(all_hours)),
        "wind_offshore_forecast": base * 1.5 + rng.normal(0, 20, len(all_hours)),
        "solar_forecast": np.clip(base * 0.3 + rng.normal(0, 20, len(all_hours)), 0, None),
    }
    return pd.DataFrame(data, index=all_hours)


def _consecutive_days(start: str, n: int) -> list[dt.date]:
    start_date = dt.date.fromisoformat(start)
    return [start_date + dt.timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Fold boundaries and refit_every
# ---------------------------------------------------------------------------


def test_training_window_never_reaches_into_the_test_day(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch)
    days = _consecutive_days("2025-06-01", 20)
    weather = _weather_for_days(days)
    target_hourly = _target_hourly_for_days(days)

    # Track the training window in force at the time of each prediction --
    # not just at fit time, since a model fit at fold i keeps predicting
    # with that same training data through fold i+refit_every-1.
    current_train_max: dict[str, pd.Timestamp] = {}
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original_fit = RenewablesModel.fit
    original_predict = RenewablesModel.predict_capacity_factor

    def spy_fit(self: RenewablesModel, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        current_train_max[self.production_type.value] = pd.DatetimeIndex(
            x_train.index.get_level_values("valid_time_utc")
        ).max()
        original_fit(self, x_train, y_train)

    def spy_predict(self: RenewablesModel, test_index: pd.Index, x_test: pd.DataFrame) -> pd.Series:
        test_min = pd.DatetimeIndex(x_test.index.get_level_values("valid_time_utc")).min()
        train_max = current_train_max.get(self.production_type.value)
        if train_max is not None:
            pairs.append((train_max, test_min))
        return original_predict(self, test_index, x_test)

    monkeypatch.setattr(RenewablesModel, "fit", spy_fit)
    monkeypatch.setattr(RenewablesModel, "predict_capacity_factor", spy_predict)

    run_renewables_backtest(
        target_hourly, weather, window="rolling", train_span_days=5, refit_every=1
    )

    assert pairs  # the spies actually recorded something
    assert all(train_max < test_min for train_max, test_min in pairs)


def test_refit_every_controls_the_number_of_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch)
    days = _consecutive_days("2025-06-01", 20)
    weather = _weather_for_days(days)
    target_hourly = _target_hourly_for_days(days)

    fit_calls = {"count": 0}
    original_fit = RenewablesModel.fit

    def counting_fit(self: RenewablesModel, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        fit_calls["count"] += 1
        original_fit(self, x_train, y_train)

    monkeypatch.setattr(RenewablesModel, "fit", counting_fit)

    refit_every = 3
    run_renewables_backtest(
        target_hourly, weather, window="rolling", train_span_days=5, refit_every=refit_every
    )

    # Exactly 3 targets refit on the same cadence -- fit count must be an
    # exact multiple of 3, and refitting less often than every fold means
    # strictly fewer fits than one per fold per target.
    n_evaluated_days = 20 - 5  # rough upper bound: days after the 5-day training floor
    assert fit_calls["count"] % 3 == 0
    assert fit_calls["count"] < n_evaluated_days * 3


# ---------------------------------------------------------------------------
# First evaluable day: derived, and shared between window variants
# ---------------------------------------------------------------------------


def test_first_evaluable_day_is_derived_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch)
    days = _consecutive_days("2025-06-01", 15)
    weather = _weather_for_days(days)
    target_hourly = _target_hourly_for_days(days)

    train_span_days = 6
    result = run_renewables_backtest(
        target_hourly, weather, window="rolling", train_span_days=train_span_days, refit_every=2
    )

    first_valid_time = pd.DatetimeIndex(result.index.get_level_values("valid_time_utc")).min()
    first_local_day = first_valid_time.tz_convert("Europe/Berlin").date()
    expected_first_day = days[0] + dt.timedelta(days=train_span_days)
    assert first_local_day == expected_first_day


def test_rolling_and_expanding_start_on_the_same_first_day(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch)
    days = _consecutive_days("2025-06-01", 15)
    weather = _weather_for_days(days)
    target_hourly = _target_hourly_for_days(days)

    rolling = run_renewables_backtest(
        target_hourly, weather, window="rolling", train_span_days=6, refit_every=2
    )
    expanding = run_renewables_backtest(
        target_hourly, weather, window="expanding", min_history_days=6, refit_every=2
    )

    rolling_first = pd.DatetimeIndex(rolling.index.get_level_values("valid_time_utc")).min()
    expanding_first = pd.DatetimeIndex(expanding.index.get_level_values("valid_time_utc")).min()
    assert rolling_first == expanding_first


# ---------------------------------------------------------------------------
# DST persistence-baseline skip
# ---------------------------------------------------------------------------


def test_persistence_baseline_raises_on_spring_forward_changeover() -> None:
    # 2026-03-29 local midnight -> 2026-03-30 local midnight test day whose
    # D-1 (2026-03-29) is the spring-forward day itself -- local 02:00 never
    # happened on D-1.
    test_index = pd.date_range("2026-03-29T22:00:00Z", periods=23, freq="h", tz="UTC")
    label_flat = pd.Series(
        np.zeros(24 * 3), index=pd.date_range("2026-03-01", periods=24 * 3, freq="h", tz="UTC")
    )
    with pytest.raises(pytz.exceptions.InvalidTimeError):
        persistence_baseline_cf(label_flat, test_index)


def test_dst_day_is_skipped_and_logged_not_aborting_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_capacity_anchors(monkeypatch)
    # Span across the 2026-03-29 spring-forward changeover.
    days = _consecutive_days("2025-06-01", 5) + _consecutive_days("2026-03-25", 10)
    weather = _weather_for_days(days)
    target_hourly = _target_hourly_for_days(days)

    with caplog.at_level(logging.INFO):
        result = run_renewables_backtest(
            target_hourly, weather, window="rolling", train_span_days=5, refit_every=2
        )

    assert not result.empty  # the run completed rather than aborting
    assert any("baseline undefined" in message for message in caplog.messages)
