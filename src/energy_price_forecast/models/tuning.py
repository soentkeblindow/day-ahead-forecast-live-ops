from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ..evaluation.metrics import pinball
from ..evaluation.walkforward import Fold, walk_forward_splits

logger = logging.getLogger(__name__)

# n_estimators is intentionally NOT searched: it is capped per fit by early
# stopping and frozen afterwards. Row/column subsampling is intentionally NOT
# searched either -- step 3.1 froze subsample = colsample_bytree = 1.0 to keep
# predictions reproducible, and bagging is the main source of run-to-run
# randomness. Tuning only the deterministic structural knobs preserves the 3.1
# reproducibility guarantee.
_FORECASTER_KEYS = frozenset(
    {"objective", "alpha", "random_state", "deterministic", "force_col_wise", "verbose"}
)


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Optuna search space for the deterministic structural hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


def inner_tuning_folds(
    feature_index: pd.DatetimeIndex,
    *,
    outer_test_start: pd.Timestamp,
    inner_window_days: int,
    window: Literal["expanding", "rolling"] = "expanding",
    train_span_days: int | None = None,
) -> list[Fold]:
    """Walk-forward folds built PURELY on pre-test data (< outer_test_start).

    The inner evaluation window is the last `inner_window_days` local days before
    the outer test start; everything earlier is available for inner training. The
    pre-test index is hard-filtered to < outer_test_start, so no inner fold can
    ever reach the outer test window -- this is Sprint 3's second leakage axis.
    """
    pretest = feature_index[feature_index < outer_test_start]
    if len(pretest) == 0:
        raise ValueError("no pre-test data before outer_test_start to tune on")
    inner_test_start = outer_test_start - pd.DateOffset(days=inner_window_days)
    # walk_forward_splits localises a string to Europe/Berlin; passing a UTC-aware
    # Timestamp directly would raise ValueError ("Cannot pass tzinfo with tz=...").
    inner_test_start_str = inner_test_start.strftime("%Y-%m-%d")
    folds = list(
        walk_forward_splits(
            pretest,
            test_start=inner_test_start_str,
            test_end=None,  # runs to the end of pretest, which is < outer_test_start
            window=window,
            train_span_days=train_span_days,
        )
    )
    if not folds:
        raise ValueError(
            "inner walk-forward produced no folds; pre-test window too short "
            "for the requested inner_window_days (see spec section 6)"
        )
    # Belt-and-suspenders leakage guard (the index filter already enforces it):
    for f in folds:
        assert f.train_index.max() < outer_test_start
        assert f.test_index.max() < outer_test_start
    return folds


def _fit_score_inner_fold(
    y: pd.Series,
    x: pd.DataFrame,
    fold: Fold,
    params: dict[str, Any],
    *,
    alpha: float,
    random_state: int,
    n_estimators_ceiling: int,
    es_val_days: int,
    es_rounds: int,
    num_threads: int = 1,
) -> tuple[float, int]:
    """Fit one inner fold with early stopping, return (pinball@alpha, best_iter).

    The early-stopping validation set is the chronological TAIL of the fold's
    train window (its last `es_val_days` local days). That tail still lies before
    the fold's test window, so the leakage contract holds. The number of trees is
    capped by early stopping; we return best_iteration_ for the freeze step.
    """
    y_tr = y.loc[fold.train_index]
    x_tr = x.loc[fold.train_index]
    mask = y_tr.notna()
    y_tr, x_tr = y_tr.loc[mask], x_tr.loc[mask]

    cutoff = fold.train_index.max() - pd.DateOffset(days=es_val_days)
    is_val = x_tr.index >= cutoff
    x_fit, y_fit = x_tr.loc[~is_val], y_tr.loc[~is_val]
    x_val, y_val = x_tr.loc[is_val], y_tr.loc[is_val]

    if len(x_val) == 0:
        raise ValueError(
            f"early-stopping validation tail is empty after NaN masking; "
            f"consider increasing --es-val-days (currently {es_val_days})"
        )

    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=n_estimators_ceiling,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        n_jobs=num_threads,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        **params,
    )
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_val, y_val)],
        eval_metric="quantile",
        callbacks=[early_stopping(es_rounds, verbose=False), log_evaluation(0)],
    )
    raw_iter: int = model.best_iteration_
    if raw_iter <= 0:
        logger.warning(
            "early stopping did not produce a valid best_iteration_ (%d); "
            "falling back to n_estimators_ceiling=%d -- "
            "consider raising --es-rounds or --n-estimators-ceiling",
            raw_iter,
            n_estimators_ceiling,
        )
        raw_iter = n_estimators_ceiling
    best_iter = raw_iter
    preds = pd.Series(
        np.asarray(model.predict(x.loc[fold.test_index])),
        index=fold.test_index,
        name="y_pred",
    )
    return pinball(y.loc[fold.test_index], preds, alpha), best_iter


def _objective(
    trial: optuna.Trial,
    y: pd.Series,
    x: pd.DataFrame,
    inner_folds: list[Fold],
    *,
    random_state: int,
    n_estimators_ceiling: int,
    es_val_days: int,
    es_rounds: int,
    num_threads: int = 1,
) -> float:
    """Mean inner-fold pinball@0.5 -- the value optuna minimises (decision 2)."""
    params = suggest_lgbm_params(trial)
    losses = [
        _fit_score_inner_fold(
            y,
            x,
            fold,
            params,
            alpha=0.5,
            random_state=random_state,
            n_estimators_ceiling=n_estimators_ceiling,
            es_val_days=es_val_days,
            es_rounds=es_rounds,
            num_threads=num_threads,
        )[0]
        for fold in inner_folds
    ]
    return float(np.mean(losses))


@dataclass(frozen=True)
class TuningResult:
    """Frozen output of a tuning run, ready to be persisted and reused.

    `params` is the complete dict to inject into LGBMForecaster(params=...): the
    tuned structural knobs PLUS the frozen n_estimators PLUS the fixed no-bagging
    flags. It deliberately does NOT contain objective / alpha / random_state /
    deterministic / force_col_wise / verbose -- those are set by the forecaster.
    """

    params: dict[str, Any]
    best_value: float
    n_trials: int
    n_trials_completed: int
    patience: int | None
    inner_test_start: pd.Timestamp
    outer_test_start: pd.Timestamp
    n_estimators_ceiling: int
    es_val_days: int
    es_rounds: int
    random_state: int


class _ConvergenceStopper:
    """Stop the study once the best value has not improved for `patience` trials.

    Deterministic given a fixed TPESampler seed: the trial sequence is fixed, so
    the stop point is reproducible run to run. (A wall-clock `timeout` is NOT
    reproducible -- use it only for exploration, never for the final frozen run.)
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best: float | None = None
        self._best_trial = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = study.best_value
        if self._best is None or value < self._best:
            self._best, self._best_trial = value, trial.number
        elif trial.number - self._best_trial >= self.patience:
            study.stop()


def freeze_n_estimators(
    y: pd.Series,
    x: pd.DataFrame,
    best_params: dict[str, Any],
    *,
    outer_test_start: pd.Timestamp,
    random_state: int,
    n_estimators_ceiling: int,
    es_val_days: int,
    es_rounds: int,
    num_threads: int = 1,
) -> int:
    """Refit best_params ONCE on all pre-test data with an early-stopping tail;
    the chosen best_iteration is the frozen tree count for the outer runs.
    """
    pretest = pd.DatetimeIndex(x.index[x.index < outer_test_start])
    fold = Fold(
        delivery_day=outer_test_start,
        train_index=pretest,
        test_index=pretest[-1:],  # unused for the count; predict path is harmless
        gate_closure=outer_test_start,
    )
    _, best_iter = _fit_score_inner_fold(
        y,
        x,
        fold,
        best_params,
        alpha=0.5,
        random_state=random_state,
        n_estimators_ceiling=n_estimators_ceiling,
        es_val_days=es_val_days,
        es_rounds=es_rounds,
        num_threads=num_threads,
    )
    return best_iter


def tune_lgbm(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    outer_test_start: pd.Timestamp,
    n_trials: int = 50,
    patience: int | None = 10,
    timeout: float | None = None,
    inner_window_days: int = 90,
    es_val_days: int = 42,
    es_rounds: int = 50,
    n_estimators_ceiling: int = 1000,
    window: Literal["expanding", "rolling"] = "expanding",
    train_span_days: int | None = None,
    random_state: int = 0,
    num_threads: int = 8,
) -> tuple[TuningResult, optuna.Study]:
    """Run the optuna study on the inner walk-forward, then freeze the params.

    `n_trials` is the hard ceiling; `patience` stops earlier once the search has
    converged (no improvement for `patience` trials). Whichever fires first wins.
    `timeout` is an optional wall-clock cap and breaks reproducibility, so it is
    off by default and must never be used for the final frozen run.
    """
    t0 = time.monotonic()

    inner_folds = inner_tuning_folds(
        pd.DatetimeIndex(x.index),
        outer_test_start=outer_test_start,
        inner_window_days=inner_window_days,
        window=window,
        train_span_days=train_span_days,
    )
    logger.info(
        "built %d inner folds (pre-test %s..%s)",
        len(inner_folds),
        inner_folds[0].train_index.min().date(),
        inner_folds[-1].test_index.max().date(),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _trial_log_cb(s: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
        logger.info(
            "trial %d/%d: pinball=%.3f (best %.3f)",
            t.number + 1,
            n_trials,
            t.value if t.value is not None else float("nan"),
            s.best_value,
        )

    callbacks: list[Callable[..., None]] = [_trial_log_cb]
    if patience is not None:
        callbacks.append(_ConvergenceStopper(patience))

    study.optimize(
        lambda t: _objective(
            t,
            y,
            x,
            inner_folds,
            random_state=random_state,
            n_estimators_ceiling=n_estimators_ceiling,
            es_val_days=es_val_days,
            es_rounds=es_rounds,
            num_threads=num_threads,
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
        n_jobs=1,
    )

    logger.info("freezing n_estimators (refitting best params on full pre-test data)")
    n_est = freeze_n_estimators(
        y,
        x,
        study.best_params,
        outer_test_start=outer_test_start,
        random_state=random_state,
        n_estimators_ceiling=n_estimators_ceiling,
        es_val_days=es_val_days,
        es_rounds=es_rounds,
        num_threads=num_threads,
    )
    logger.info("n_estimators frozen to %d", n_est)

    frozen: dict[str, Any] = {
        **study.best_params,
        "n_estimators": n_est,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }

    result = TuningResult(
        params=frozen,
        best_value=float(study.best_value),
        n_trials=n_trials,
        n_trials_completed=len(study.trials),
        patience=patience,
        inner_test_start=outer_test_start - pd.DateOffset(days=inner_window_days),
        outer_test_start=outer_test_start,
        n_estimators_ceiling=n_estimators_ceiling,
        es_val_days=es_val_days,
        es_rounds=es_rounds,
        random_state=random_state,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "tuning finished in %.0fs (best pinball=%.4f, %d/%d trials completed)",
        elapsed,
        result.best_value,
        result.n_trials_completed,
        n_trials,
    )

    return result, study
