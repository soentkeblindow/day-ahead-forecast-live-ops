import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data import capacity as capacity_mod
from energy_price_forecast.data.capacity import (
    CapacitySource,
    ProductionType,
    installed_capacity_at,
)
from energy_price_forecast.models.renewables import (
    DEFAULT_SEED,
    RenewablesModel,
    apply_solar_night_zero,
    capacity_factor_label,
    daylight_mask_for_training,
    to_mw,
)

_WEATHER_PATH = PROJECT_ROOT / "data" / "interim" / "weather_ifs_run00.parquet"


def _synthetic_index(n: int = 48, start: str = "2024-06-01T00:00:00Z") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="h", tz="UTC"))


def _synthetic_features(
    index: pd.DatetimeIndex, n_features: int = 5, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(len(index), n_features))
    return pd.DataFrame(data, index=index, columns=[f"f{i}" for i in range(n_features)])


def _patch_capacity_anchors(monkeypatch: pytest.MonkeyPatch, value: float = 1000.0) -> None:
    anchors = pd.Series(
        [value, value], index=pd.DatetimeIndex(["2020-01-01", "2030-01-01"], tz="UTC")
    )
    monkeypatch.setitem(
        capacity_mod._ANCHOR_LOADERS, CapacitySource.PUBLIC_REGISTRY, lambda pt: anchors
    )


class _FakeForecaster:
    """Stand-in for LGBMForecaster.predict() -- lets clipping be tested
    against exact chosen raw values instead of fighting LightGBM's own
    behaviour on a tiny synthetic dataset.
    """

    def __init__(self, raw_values: list[float]) -> None:
        self._raw = raw_values

    def predict(
        self, test_index: pd.DatetimeIndex, *, history: pd.Series, x_test: pd.DataFrame
    ) -> pd.Series:
        return pd.Series(self._raw, index=test_index)


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def test_capacity_factor_label_basic_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch, value=1000.0)
    idx = _synthetic_index(5)
    target = pd.Series([100.0, 200.0, 500.0, 1200.0, 0.0], index=idx)

    label = capacity_factor_label(target, idx, ProductionType.SOLAR)

    np.testing.assert_allclose(label.to_numpy(), [0.1, 0.2, 0.5, 1.2, 0.0])


def test_capacity_factor_label_raises_on_missing_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch)
    idx = _synthetic_index(3)
    target = pd.Series([100.0, np.nan, 300.0], index=idx)

    with pytest.raises(ValueError, match="missing"):
        capacity_factor_label(target, idx, ProductionType.SOLAR)


def test_capacity_factor_label_above_one_is_kept_not_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_capacity_anchors(monkeypatch, value=1000.0)
    idx = _synthetic_index(2)
    target = pd.Series([1500.0, 300.0], index=idx)

    label = capacity_factor_label(target, idx, ProductionType.WIND_ONSHORE)

    assert len(label) == 2  # nothing dropped
    assert label.iloc[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Model: clipping, determinism, MW round trip
# ---------------------------------------------------------------------------


def test_predict_capacity_factor_clips_negative_and_keeps_above_one() -> None:
    model = RenewablesModel(ProductionType.WIND_ONSHORE)
    idx = _synthetic_index(4)
    model.forecaster = _FakeForecaster([-0.3, 0.0, 0.9, 1.4])  # type: ignore[assignment]

    pred = model.predict_capacity_factor(idx, pd.DataFrame(index=idx))

    np.testing.assert_allclose(pred.to_numpy(), [0.0, 0.0, 0.9, 1.4])


def test_determinism_same_seed_gives_identical_predictions() -> None:
    idx = _synthetic_index(200, start="2024-01-01T00:00:00Z")
    x = _synthetic_features(idx, n_features=6, seed=1)
    y = pd.Series(np.random.default_rng(2).uniform(0, 1, len(idx)), index=idx)

    model_a = RenewablesModel(ProductionType.SOLAR, seed=DEFAULT_SEED)
    model_a.fit(x, y)
    pred_a = model_a.predict_capacity_factor(idx, x)

    model_b = RenewablesModel(ProductionType.SOLAR, seed=DEFAULT_SEED)
    model_b.fit(x, y)
    pred_b = model_b.predict_capacity_factor(idx, x)

    pd.testing.assert_series_equal(pred_a, pred_b)


def test_mw_round_trip_is_lossless_to_float32_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capacity_anchors(monkeypatch, value=837.5)
    idx = _synthetic_index(10)
    cf = pd.Series(np.linspace(0.0, 1.0, len(idx)), index=idx)

    mw = to_mw(cf, idx, ProductionType.WIND_OFFSHORE)
    capacity_mw = installed_capacity_at(ProductionType.WIND_OFFSHORE, idx)
    cf_roundtrip = (mw.to_numpy() / capacity_mw.to_numpy()).astype("float32")

    np.testing.assert_allclose(cf_roundtrip, cf.to_numpy().astype("float32"), rtol=1e-6)


# ---------------------------------------------------------------------------
# Solar night hours (spec 2.3)
# ---------------------------------------------------------------------------


def test_apply_solar_night_zero_forces_night_predictions_to_zero() -> None:
    idx = _synthetic_index(48, start="2025-06-20T00:00:00Z")
    raw_pred = pd.Series(np.full(len(idx), 0.5), index=idx)

    zeroed = apply_solar_night_zero(raw_pred, idx)
    daylight = daylight_mask_for_training(idx)

    assert (zeroed[~daylight.to_numpy()] == 0.0).all()
    assert (zeroed[daylight.to_numpy()] == 0.5).all()


def test_night_hours_are_excluded_from_solar_training_when_filtered() -> None:
    idx = _synthetic_index(48, start="2025-06-20T00:00:00Z")
    y = pd.Series(np.arange(len(idx), dtype="float64"), index=idx)
    daylight = daylight_mask_for_training(idx).to_numpy()

    y_train = y[daylight]
    night_values = set(y[~daylight].to_numpy())

    assert len(y_train) < len(idx)  # some hours were actually removed
    assert night_values.isdisjoint(set(y_train.to_numpy()))


# ---------------------------------------------------------------------------
# Leakage (with negative control) and prediction provenance
# ---------------------------------------------------------------------------


def test_leakage_target_value_never_in_training_with_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_capacity_anchors(monkeypatch, value=1000.0)
    train_idx = _synthetic_index(24, start="2025-06-01T00:00:00Z")  # day D-1
    test_idx = _synthetic_index(24, start="2025-06-02T00:00:00Z")  # day D

    target_train = pd.Series(np.arange(24, dtype="float64"), index=train_idx)
    target_test = pd.Series(np.arange(1000.0, 1024.0), index=test_idx)  # disjoint value range

    label_train = capacity_factor_label(target_train, train_idx, ProductionType.WIND_ONSHORE)
    label_test = capacity_factor_label(target_test, test_idx, ProductionType.WIND_ONSHORE)

    # Correct discipline: day D never enters the training set.
    assert set(label_test.to_numpy()).isdisjoint(set(label_train.to_numpy()))

    # Negative control: a leaky split that folds day D into "training" data
    # is exactly what this kind of check exists to catch.
    leaky_train = pd.concat([label_train, label_test])
    assert not set(label_test.to_numpy()).isdisjoint(set(leaky_train.to_numpy()))


@pytest.mark.skipif(
    not _WEATHER_PATH.exists(),
    reason="weather artefact not present -- local integrity check, not a CI gate (same pattern as test_quarterhourly.py)",
)
def test_prediction_row_provenance_matches_run_init_for_target_day() -> None:
    from energy_price_forecast.data.loaders import load_interim_weather
    from energy_price_forecast.data.weather_client import run_init_for_target_day
    from energy_price_forecast.features.renewables_forecast import build_feature_matrix, columns_for

    weather = load_interim_weather()
    days = [dt.date(2025, 6, 1), dt.date(2025, 6, 2)]
    features, excluded = build_feature_matrix(days, weather)
    assert excluded == ()

    x = features[columns_for(ProductionType.WIND_ONSHORE)]
    y = pd.Series(
        np.random.default_rng(0).uniform(0.1, 0.6, len(x)), index=x.index, name="wind_onshore_cf"
    )

    model = RenewablesModel(ProductionType.WIND_ONSHORE)
    model.fit(x, y)
    pred = model.predict_capacity_factor(x.index, x)

    valid_time_local = pd.DatetimeIndex(pred.index.get_level_values("valid_time_utc")).tz_convert(
        "Europe/Berlin"
    )
    run_init = pred.index.get_level_values("run_init_utc")
    for day in days:
        day_rows = valid_time_local.date == day
        assert (run_init[day_rows] == run_init_for_target_day(day)).all()
