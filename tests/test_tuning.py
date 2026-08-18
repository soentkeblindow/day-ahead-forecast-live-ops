from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
import pytest

from energy_price_forecast.models.lgbm import LGBMForecaster
from energy_price_forecast.models.tuning import (
    _FORECASTER_KEYS,
    TuningResult,
    freeze_n_estimators,
    inner_tuning_folds,
    suggest_lgbm_params,
    tune_lgbm,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

_OUTER_TEST_START = pd.Timestamp("2021-06-01", tz="UTC")


def _make_xy(
    n_days: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic hourly feature matrix and price series (UTC index)."""
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(0, 1, (len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(50.0 + rng.normal(0, 10, len(idx)), index=idx, name="price")
    return x, y


def _make_learnable_xy(n_days: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data with a strong linear signal so early stopping fires early."""
    idx = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, len(idx))
    x = pd.DataFrame({"f0": f0, "f1": rng.normal(0, 1, len(idx))}, index=idx)
    y = pd.Series(50.0 + 20.0 * f0 + rng.normal(0, 0.5, len(idx)), index=idx, name="price")
    return x, y


def _small_tune_kwargs() -> dict:
    """Minimal tune_lgbm kwargs for fast tests."""
    return dict(
        outer_test_start=_OUTER_TEST_START,
        n_trials=3,
        patience=None,
        inner_window_days=20,
        es_val_days=10,
        es_rounds=5,
        n_estimators_ceiling=10,
        random_state=0,
        num_threads=1,
    )


# ---------------------------------------------------------------------------
# §7 — Leakage (core test)
# ---------------------------------------------------------------------------


def test_inner_tuning_folds_all_before_outer_test_start() -> None:
    """Every inner fold's train and test timestamps must be < outer_test_start."""
    x, y = _make_xy()
    folds = inner_tuning_folds(
        pd.DatetimeIndex(x.index),
        outer_test_start=_OUTER_TEST_START,
        inner_window_days=20,
    )
    assert len(folds) > 0
    for f in folds:
        assert f.train_index.max() < _OUTER_TEST_START, "train leaks into test window"
        assert f.test_index.max() < _OUTER_TEST_START, "test leaks into outer test window"


def test_inner_tuning_folds_pretest_filter_excludes_outer_rows() -> None:
    """The pre-test filter must exclude ALL timestamps >= outer_test_start."""
    x, _ = _make_xy()
    folds = inner_tuning_folds(
        pd.DatetimeIndex(x.index),
        outer_test_start=_OUTER_TEST_START,
        inner_window_days=20,
    )
    all_ts: list[pd.Timestamp] = []
    for f in folds:
        all_ts.extend(f.train_index.tolist())
        all_ts.extend(f.test_index.tolist())
    assert all(ts < _OUTER_TEST_START for ts in all_ts)


def test_inner_tuning_folds_raises_when_no_pretest_data() -> None:
    x, _ = _make_xy()
    with pytest.raises(ValueError, match="no pre-test data"):
        inner_tuning_folds(
            pd.DatetimeIndex(x.index),
            outer_test_start=pd.Timestamp("2020-01-01", tz="UTC"),
            inner_window_days=10,
        )


def test_inner_tuning_folds_raises_when_no_training_data() -> None:
    # Use 23 hours (00:00-22:00 UTC = 01:00-23:00 CET) so the pretest covers exactly
    # one local day. walk_forward_splits skips it (no prior day to train on) → folds = [].
    # Note: 24 hours would bleed into the next local day (23:00 UTC = 00:00 CET Jan 2).
    idx = pd.date_range("2021-01-01", periods=23, freq="h", tz="UTC")
    x = pd.DataFrame({"f": 1.0}, index=idx)
    with pytest.raises(ValueError, match="no folds"):
        inner_tuning_folds(
            pd.DatetimeIndex(x.index),
            outer_test_start=pd.Timestamp("2021-01-02", tz="UTC"),
            inner_window_days=5,
        )


# ---------------------------------------------------------------------------
# §7 — Search-space sanity
# ---------------------------------------------------------------------------


def test_suggest_lgbm_params_expected_keys_no_forbidden() -> None:
    """suggest_lgbm_params returns exactly the tunable structural keys."""
    expected = {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    }
    forbidden = {"subsample", "colsample_bytree", "n_estimators"}

    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_split_gain": 0.1,
        }
    )
    params = suggest_lgbm_params(trial)  # type: ignore[arg-type]

    assert set(params.keys()) == expected
    assert not (set(params.keys()) & forbidden)


def test_suggest_lgbm_params_values_within_bounds() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_split_gain": 0.1,
        }
    )
    params = suggest_lgbm_params(trial)  # type: ignore[arg-type]

    assert 0.03 <= params["learning_rate"] <= 0.3
    assert 15 <= params["num_leaves"] <= 255
    assert 3 <= params["max_depth"] <= 12
    assert 5 <= params["min_child_samples"] <= 100
    assert 1e-3 <= params["reg_alpha"] <= 10.0
    assert 1e-3 <= params["reg_lambda"] <= 10.0
    assert 0.0 <= params["min_split_gain"] <= 0.5


# ---------------------------------------------------------------------------
# §7 — Study reproducibility
# ---------------------------------------------------------------------------


def test_study_reproducible() -> None:
    """Two runs with the same random_state produce identical best params."""
    x, y = _make_xy()
    kwargs = _small_tune_kwargs()

    r1, _ = tune_lgbm(y, x, **kwargs)
    r2, _ = tune_lgbm(y, x, **kwargs)

    assert r1.params == r2.params
    assert r1.best_value == pytest.approx(r2.best_value)
    assert r1.n_trials_completed == r2.n_trials_completed


def test_study_reproducible_with_multiple_threads() -> None:
    """Two runs with num_threads=2 and the same random_state produce identical best params.

    deterministic=True + force_col_wise=True guarantee bit-identical results only when
    the thread count is fixed. This test verifies the contract holds at num_threads=2
    (CI-safe; physical-core count does not matter because the thread count is fixed).
    """
    x, y = _make_xy()
    kwargs = {**_small_tune_kwargs(), "num_threads": 2}

    r1, _ = tune_lgbm(y, x, **kwargs)
    r2, _ = tune_lgbm(y, x, **kwargs)

    assert r1.params == r2.params
    assert r1.best_value == pytest.approx(r2.best_value)
    assert r1.n_trials_completed == r2.n_trials_completed


# ---------------------------------------------------------------------------
# §7 — Convergence stop
# ---------------------------------------------------------------------------


def test_convergence_stopper_fires_before_budget() -> None:
    """With patience=2 and n_trials=20, the stopper fires before 20 trials on random data."""
    x, y = _make_xy()
    result, _ = tune_lgbm(
        y,
        x,
        outer_test_start=_OUTER_TEST_START,
        n_trials=20,
        patience=2,
        inner_window_days=20,
        es_val_days=10,
        es_rounds=5,
        n_estimators_ceiling=5,  # tiny models → near-constant loss → stopper fires early
        random_state=0,
    )
    assert result.n_trials_completed < result.n_trials


def test_patience_none_runs_full_budget() -> None:
    """With patience=None, the study always runs all n_trials."""
    x, y = _make_xy()
    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())
    assert result.n_trials_completed == result.n_trials


# ---------------------------------------------------------------------------
# §7 — Frozen dict form
# ---------------------------------------------------------------------------


def test_frozen_dict_contains_required_keys() -> None:
    """Frozen params dict must contain n_estimators and no-bagging flags."""
    x, y = _make_xy()
    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())

    assert "n_estimators" in result.params
    assert result.params["subsample"] == pytest.approx(1.0)
    assert result.params["colsample_bytree"] == pytest.approx(1.0)


def test_frozen_dict_contains_no_forecaster_keys() -> None:
    """Frozen params must NOT include keys that the LGBMForecaster sets itself."""
    x, y = _make_xy()
    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())

    overlap = set(result.params.keys()) & _FORECASTER_KEYS
    assert not overlap, f"frozen params contain forecaster-owned keys: {overlap}"


# ---------------------------------------------------------------------------
# §7 — Freeze/reload determinism
# ---------------------------------------------------------------------------


def test_freeze_reload_json_roundtrip() -> None:
    """Serialising and deserialising result.params via JSON gives the identical dict."""
    x, y = _make_xy()
    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())

    serialised = json.dumps(result.params)
    loaded: dict[str, object] = json.loads(serialised)

    # JSON round-trips float and int; compare values up to float precision.
    assert set(loaded.keys()) == set(result.params.keys())
    for k in result.params:
        assert loaded[k] == pytest.approx(result.params[k]), f"mismatch for key {k!r}"


def test_freeze_reload_identical_predictions() -> None:
    """LGBMForecaster with original params and with JSON-reloaded params must give identical predictions."""
    x, y = _make_xy()
    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())

    serialised = json.dumps(result.params)
    loaded: dict[str, object] = json.loads(serialised)

    n_train = 100 * 24
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train = y.iloc[:n_train]
    test_idx = pd.DatetimeIndex(x_test.index)

    m_orig = LGBMForecaster(params=dict(result.params), random_state=0)
    m_orig.fit(y_train, x_train)
    p_orig = m_orig.predict(test_idx, history=y_train, x_test=x_test)

    m_loaded = LGBMForecaster(params=loaded, random_state=0)
    m_loaded.fit(y_train, x_train)
    p_loaded = m_loaded.predict(test_idx, history=y_train, x_test=x_test)

    pd.testing.assert_series_equal(p_orig, p_loaded)


# ---------------------------------------------------------------------------
# §7 — Early stopping caps trees
# ---------------------------------------------------------------------------


def test_freeze_n_estimators_at_most_ceiling() -> None:
    """freeze_n_estimators always returns a value <= n_estimators_ceiling."""
    x, y = _make_xy()
    ceiling = 50
    best_iter = freeze_n_estimators(
        y,
        x,
        {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "min_split_gain": 0.0,
        },
        outer_test_start=_OUTER_TEST_START,
        random_state=0,
        n_estimators_ceiling=ceiling,
        es_val_days=10,
        es_rounds=5,
        num_threads=1,
    )
    assert best_iter <= ceiling


def test_freeze_n_estimators_fires_early_on_learnable_data() -> None:
    """On data with a strong linear signal, early stopping fires well before the ceiling."""
    x, y = _make_learnable_xy()
    ceiling = 200
    best_iter = freeze_n_estimators(
        y,
        x,
        {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "min_split_gain": 0.0,
        },
        outer_test_start=_OUTER_TEST_START,
        random_state=0,
        n_estimators_ceiling=ceiling,
        es_val_days=20,
        es_rounds=10,
        num_threads=1,
    )
    assert best_iter < ceiling


# ---------------------------------------------------------------------------
# §7 — End-to-end smoke
# ---------------------------------------------------------------------------


def test_tune_lgbm_smoke() -> None:
    """tune_lgbm runs through without error and returns a valid TuningResult."""
    x, y = _make_xy()
    result, study = tune_lgbm(y, x, **_small_tune_kwargs())

    assert isinstance(result, TuningResult)
    assert np.isfinite(result.best_value)
    assert "n_estimators" in result.params
    assert result.n_trials_completed > 0
    assert len(study.trials) == result.n_trials_completed


# ---------------------------------------------------------------------------
# §6 — All-NaN feature column does not crash (regression guard like 3.1)
# ---------------------------------------------------------------------------


def test_all_nan_feature_column_no_crash() -> None:
    """An all-NaN feature column must not cause a crash in the inner fitting."""
    x, y = _make_xy()
    x = x.copy()
    x["all_nan"] = float("nan")

    result, _ = tune_lgbm(y, x, **_small_tune_kwargs())
    assert np.isfinite(result.best_value)
