"""One-off ablation (not a spec step, not run by CI): how sensitive is the
`live` price model's performance to the walk-forward refit cadence?

Motivation: every documented production backtest (README Beleg D, 6.3
objective comparison, 6.4 bridge, 6.6 live-gate) fixes `refit_every=1`
(daily refit) as the inherited production default -- it has never been
measured against a coarser cadence for the PRICE model itself (only the
6.5.2 renewables model got a `refit_every=1` vs `=7` comparison, a
different model/target, not transferable). This is also a rough, NOT exact
proxy for "what happens if the last 1-3 training days become unusable for
some reason, but features for the target day are still there": with
`refit_every=N`, the model between refit points keeps using the training
window from its LAST refit, i.e. it is missing up to N-1 days of the most
recent price labels relative to a daily-refit model -- similar in spirit to
a stale/truncated training tail, though not identical (a truncated-window
ablation with refit_every=1 held fixed would be a more literal test of that
exact scenario, not run here).

Single feature set (`live`, build_feature_set_for_day, unchanged), swept
over refit_every in {1, 2, 3, 4, 7}. All five share the same feature matrix
and the same evaluable folds (built once) -- only the refit cadence differs.
DM test: each refit_every in {2,3,4,7} vs refit_every=1 (the production
default), both native quarter-hourly (hac_lag=192, horizon=96) and
daily-block (hac_lag=7, horizon=1) conventions, same as the other
ablation_*.py scripts in this repo. Arena baseline is included in the
output table for context only, not the focus of this comparison.

MLflow experiment: refit_cadence_sensitivity_for_live_system (separate from
feature_reduction_for_live_system -- a different axis of investigation).

Usage: python scripts/ablation_refit_every_sensitivity.py
       ABLATION_SMOKE=1 python scripts/ablation_refit_every_sensitivity.py   # 20 folds, smoke test
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import (
    load_interim_hourly,
    load_interim_quarterhourly,
    load_renewables_predictions,
)
from energy_price_forecast.evaluation.dm_test import DMResult, dm_test
from energy_price_forecast.evaluation.metrics import mae, rmse
from energy_price_forecast.evaluation.walkforward import walk_forward_splits
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation_refit")

_TZ = "Europe/Berlin"
_PRICE_COL = "day_ahead_price"
_TRAIN_SPAN_DAYS = 90
_RANDOM_STATE = 0
_SHAPE_WINDOW_DAYS = 28
_QH_HAC_LAG = 192
_QH_HORIZON = 96
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_REFIT_EVERY_VALUES = (1, 2, 3, 4, 7)
_REFERENCE_REFIT_EVERY = 1  # production default
_OUT_PATH = Path("data/processed/ablation_refit_every_sensitivity_predictions.parquet")
_MLFLOW_EXPERIMENT = "refit_cadence_sensitivity_for_live_system"


def _local_days(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.DatetimeIndex(index).tz_convert(tz).normalize().unique()
    ).sort_values()


def _build_matrix_over_days(days: pd.DatetimeIndex, builder) -> tuple[pd.DataFrame, set]:
    frames = []
    excluded: set = set()
    for day in days:
        try:
            frames.append(builder(day.date()))
        except IncompleteReconstructionError:
            excluded.add(day.date())
    if not frames:
        raise ValueError("no usable day")
    return pd.concat(frames).sort_index(), excluded


def _run_walkforward(
    refit_every: int,
    evaluable_folds: list,
    live_matrix: pd.DataFrame,
    y_hourly: pd.Series,
    prices_qh: pd.Series,
    qh_index: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Series]:
    """One full walk-forward pass at a given refit cadence. Returns
    {delivery_day: quarter-hourly prediction series}."""
    model = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
    preds_by_day: dict[pd.Timestamp, pd.Series] = {}
    for i, fold in enumerate(evaluable_folds):
        if i % refit_every == 0:
            y_train = y_hourly.reindex(fold.train_index)
            x_train = live_matrix.reindex(fold.train_index).dropna()
            model.fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[fold.train_index]
        x_test = live_matrix.loc[fold.test_index]
        hourly_pred = model.predict(fold.test_index, history=history, x_test=x_test)

        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )
        preds_by_day[fold.delivery_day] = expand_to_quarterhour(
            hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ
        )
    return preds_by_day


def main() -> None:
    t0 = time.monotonic()
    df = load_interim_hourly()
    renewables = load_renewables_predictions()
    prices_qh = load_interim_quarterhourly()[_PRICE_COL]

    hourly_index = pd.DatetimeIndex(df.index)
    renewables_valid_time = pd.DatetimeIndex(renewables.index.get_level_values("valid_time_utc"))
    renewables_days = _local_days(renewables_valid_time, _TZ)
    qh_local_days = _local_days(pd.DatetimeIndex(prices_qh.index), _TZ)

    if len(qh_local_days) <= _SHAPE_WINDOW_DAYS:
        raise ValueError(
            f"not enough quarter-hourly history ({len(qh_local_days)} local days) for a "
            f"{_SHAPE_WINDOW_DAYS}-day shape window."
        )
    min_start_day = renewables_days[0] + pd.Timedelta(days=_TRAIN_SPAN_DAYS)
    start_day = max(qh_local_days[_SHAPE_WINDOW_DAYS], min_start_day)
    end_day = min(qh_local_days[-1], renewables_days[-1])

    candidate_folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=start_day.strftime("%Y-%m-%d"),
            test_end=end_day.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=_TRAIN_SPAN_DAYS,
        )
    )
    if not candidate_folds:
        raise ValueError("no candidate folds -- check data coverage")

    first_train_day = candidate_folds[0].train_index.tz_convert(_TZ).normalize().min()
    all_days_needed = pd.date_range(first_train_day, end_day, freq="D", tz=_TZ)

    log.info(
        "building live matrix (NWP reconstruction) -- built once, shared across all refit_every values..."
    )
    live_matrix, live_excluded = _build_matrix_over_days(
        all_days_needed, lambda d: build_feature_set_for_day(d, df, renewables)
    )
    log.info("live matrix: %d rows (%d days excluded)", len(live_matrix), len(live_excluded))

    y_hourly = df[_PRICE_COL]
    qh_index = pd.DatetimeIndex(prices_qh.index)

    evaluable_folds = []
    for fold in candidate_folds:
        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        expected_hours = round((day_end - day_start) / pd.Timedelta(hours=1))
        if len(fold.test_index) != expected_hours:
            continue
        if fold.delivery_day.date() in live_excluded:
            continue
        qh_mask = (qh_index >= day_start) & (qh_index < day_end)
        if qh_mask.sum() != expected_hours * 4:
            continue
        try:
            persistence_forecast(prices_qh, fold.delivery_day, tz=_TZ)
        except ValueError:
            continue
        evaluable_folds.append(fold)

    if os.environ.get("ABLATION_SMOKE"):
        evaluable_folds = evaluable_folds[:20]

    n_folds = len(evaluable_folds)
    log.info("evaluable folds: %d (of %d candidates)", n_folds, len(candidate_folds))

    preds_by_refit: dict[int, dict[pd.Timestamp, pd.Series]] = {}
    for refit_every in _REFIT_EVERY_VALUES:
        t_run = time.monotonic()
        n_refits = len(range(0, n_folds, refit_every))
        log.info(
            "running walk-forward at refit_every=%d (%d refits over %d folds)...",
            refit_every,
            n_refits,
            n_folds,
        )
        preds_by_refit[refit_every] = _run_walkforward(
            refit_every, evaluable_folds, live_matrix, y_hourly, prices_qh, qh_index
        )
        log.info("refit_every=%d done in %.0fs", refit_every, time.monotonic() - t_run)

    records: list[pd.DataFrame] = []
    for fold in evaluable_folds:
        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()
        pred_baseline = persistence_forecast(prices_qh, fold.delivery_day, tz=_TZ)

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    **{
                        f"pred_refit{n}": preds_by_refit[n][fold.delivery_day]
                        .reindex(y_true_day.index)
                        .to_numpy()
                        for n in _REFIT_EVERY_VALUES
                    },
                    "pred_baseline": pred_baseline.reindex(y_true_day.index).to_numpy(),
                    "delivery_day": fold.delivery_day,
                    "n_slots_in_day": len(y_true_day),
                },
                index=y_true_day.index,
            )
        )

    predictions = pd.concat(records).sort_index()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(_OUT_PATH)

    all_columns = [f"refit{n}" for n in _REFIT_EVERY_VALUES] + ["baseline"]
    print(
        f"\nn_folds={n_folds}  train_span_days={_TRAIN_SPAN_DAYS}  shape_window_days={_SHAPE_WINDOW_DAYS}"
    )
    print(f"live_excluded_days={len(live_excluded)}\n")

    metrics = {}
    for name in all_columns:
        t, p = predictions["y_true"], predictions[f"pred_{name}"]
        metrics[name] = {"mae": mae(t, p), "rmse": rmse(t, p)}
        print(f"{name:12s}  MAE={metrics[name]['mae']:.4f}  RMSE={metrics[name]['rmse']:.4f}")

    def _run_dm(candidate: str, reference: str) -> tuple[DMResult, DMResult]:
        loss_c = (predictions[f"pred_{candidate}"] - predictions["y_true"]) ** 2
        loss_r = (predictions[f"pred_{reference}"] - predictions["y_true"]) ** 2
        native = dm_test(loss_c, loss_r, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
        daily_c = loss_c.groupby(predictions["delivery_day"]).mean()
        daily_r = loss_r.groupby(predictions["delivery_day"]).mean()
        daily = dm_test(daily_c, daily_r, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
        return native, daily

    print(f"\n--- refit_every=N vs refit_every={_REFERENCE_REFIT_EVERY} (production default) ---\n")
    dm_results: dict[int, dict[str, DMResult]] = {}
    for n in _REFIT_EVERY_VALUES:
        if n == _REFERENCE_REFIT_EVERY:
            continue
        native, daily = _run_dm(f"refit{n}", f"refit{_REFERENCE_REFIT_EVERY}")
        dm_results[n] = {"native": native, "daily": daily}
        sig = "significant (p<0.10)" if native.p_value < 0.10 else "not significant"
        verdict = (
            f"WORSE than refit_every={_REFERENCE_REFIT_EVERY}"
            if native.mean_loss_diff > 0
            else f"BETTER than refit_every={_REFERENCE_REFIT_EVERY}"
        )
        print(
            f"refit_every={n}: RMSE={metrics[f'refit{n}']['rmse']:.4f}  "
            f"native mean_loss_diff_sq={native.mean_loss_diff:.4f} p={native.p_value:.4f}  "
            f"daily mean_loss_diff_sq={daily.mean_loss_diff:.4f} p={daily.p_value:.4f}  -> {verdict}, {sig}"
        )

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")
    print(f"predictions written to {_OUT_PATH}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ablation_refit_every_sensitivity"):
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "train_span_days": _TRAIN_SPAN_DAYS,
                "shape_window_days": _SHAPE_WINDOW_DAYS,
                "live_excluded_days": len(live_excluded),
                "objective": "quantile",
                "alpha": 0.5,
                "random_state": _RANDOM_STATE,
                "refit_every_values": ",".join(str(n) for n in _REFIT_EVERY_VALUES),
                "reference_refit_every": _REFERENCE_REFIT_EVERY,
                **_DEFAULT_PARAMS,
            }
        )
        mlflow.set_tags(
            {
                "purpose": "one-off ablation -- refit cadence sensitivity of the live price model, "
                "rough proxy for a stale/truncated recent training tail",
                "not_a_spec_step": "true",
            }
        )
        log_metrics: dict[str, float] = {}
        for name in all_columns:
            log_metrics[f"{name}_mae"] = metrics[name]["mae"]
            log_metrics[f"{name}_rmse"] = metrics[name]["rmse"]
        for n, results in dm_results.items():
            log_metrics[f"refit{n}_vs_refit1_native_loss_diff"] = results["native"].mean_loss_diff
            log_metrics[f"refit{n}_vs_refit1_native_p"] = results["native"].p_value
            log_metrics[f"refit{n}_vs_refit1_daily_loss_diff"] = results["daily"].mean_loss_diff
            log_metrics[f"refit{n}_vs_refit1_daily_p"] = results["daily"].p_value
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(_OUT_PATH))
    print(f"\nlogged to mlflow experiment {_MLFLOW_EXPERIMENT!r}")


if __name__ == "__main__":
    main()
