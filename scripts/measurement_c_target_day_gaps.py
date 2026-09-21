"""Sprint 6.8, Messung C (docs/sprint6_step6_8_spec.md section 6): when the
last hours of a target-day input are missing, what is the best response --
and does it beat silence?

Reuses Messung A's `_prepare()` unchanged for window/folds/the `live` matrix
(only `live` is needed here -- basis per spec). Training is untouched by
this measurement (only TARGET-day inputs are degraded, never fold.train_index
-- that is Messung B's question), so the trained model is IDENTICAL across
every one of the 16 (group, h) configs at a given fold: one fit per
refit_every=14 checkpoint, queried many times with different degraded
x_test variants. This is far cheaper than a naive "retrain per config" loop
and is exactly what the spec means by "verglichen werden drei Antworten auf
dasselbe trainierte Modell".

Verbindlich (spec section 6, added after the 2026-09-21 real-world finding):
"Zieltag" is the Berlin local day as features.build.build_feature_set_for_day
actually forms it -- evaluation.walkforward.walk_forward_splits already
builds fold.test_index on that convention, so "the last h hours of the
target day" here just means fold.test_index[-h:], no separate local-day
arithmetic needed.

Four input groups (spec's own table), mapped to `live`'s real columns, with
NaN propagated to the two columns nwp_fundamentals.py derives by simple
arithmetic from them (residual_load_forecast_nwp = load - won - woff - solar;
renewable_share_forecast_nwp = (won+woff+solar)/load -- confirmed by reading
that module, not guessed):

  load_forecast  -- load_forecast_day_ahead (+ the two derived columns)
  price_lags     -- price_lag_24h (D-1's trailing hours land exactly here)
  actual_lags    -- load_actual_lag_48h, {load,wind_onshore,wind_offshore,solar}
                    _forecast_error_lag_48h (exactly 2026-09-20's real incident)
  nwp_residual   -- {wind_onshore,wind_offshore,solar}_forecast_nwp
                    (+ the two derived columns)

h in {1, 2, 3, 6} hours -- each hour of missing HOURLY input is 4 missing
quarter-hourly output slots after the bridge (spec's own point, automatically
true here since the gap is applied at the hourly feature-matrix level).

Three responses, each scored over the WHOLE delivery day (all 96 slots):
  ffill    -- forward-fill the gapped columns, model predicts every hour.
  partial  -- model predicts the (24-h) clean hours normally; the last h
              hours are replaced by the Arena persistence value (D-1, same
              slot) -- literally scripts.arena_baseline.persistence_forecast's
              own value for exactly those quarter-hours, not a separate
              computation.
  silence  -- the Arena persistence baseline for the entire day (the same
              `pred_baseline` Messung A/B already computed and logged).

refit_every=14 (spec explicitly permits this here: all three responses share
one trained model, so training freshness affects them identically).

Usage:
  python -m scripts.measurement_c_target_day_gaps --probe-folds 10
  python -m scripts.measurement_c_target_day_gaps
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.evaluation.dm_test import DMResult, dm_test
from energy_price_forecast.evaluation.metrics import rmse
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import LGBMForecaster
from scripts.measurement_a_candidate_intake import (
    _DAILY_HAC_LAG,
    _DAILY_HORIZON,
    _QH_HAC_LAG,
    _QH_HORIZON,
    _RANDOM_STATE,
    _SHAPE_WINDOW_DAYS,
    _TZ,
    FeatureConfig,
    _prepare,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("measurement_c")

_REFIT_EVERY = 14  # spec section 6: explicitly permitted, all responses share one model
_H_VALUES: tuple[int, ...] = (1, 2, 3, 6)
_MLFLOW_EXPERIMENT = "live_robustness_6_8"
_OUT_PATH = Path("data/processed/measurement_c_target_day_gaps_predictions.parquet")
_SUMMARY_PATH = Path("outputs/results/measurement_c_target_day_gaps_summary.csv")

_GROUPS: dict[str, tuple[str, ...]] = {
    "load_forecast": (
        "load_forecast_day_ahead",
        "residual_load_forecast_nwp",
        "renewable_share_forecast_nwp",
    ),
    "price_lags": ("price_lag_24h",),
    "actual_lags": (
        "load_actual_lag_48h",
        "load_forecast_error_lag_48h",
        "wind_onshore_forecast_error_lag_48h",
        "wind_offshore_forecast_error_lag_48h",
        "solar_forecast_error_lag_48h",
    ),
    "nwp_residual": (
        "wind_onshore_forecast_nwp",
        "wind_offshore_forecast_nwp",
        "solar_forecast_nwp",
        "residual_load_forecast_nwp",
        "renewable_share_forecast_nwp",
    ),
}


def _gap_variants(
    x_test: pd.DataFrame, group_cols: tuple[str, ...], h: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(x_test_gapped, x_test_ffilled) -- the last h rows of `group_cols`
    set to NaN, then the ffilled variant on top of that."""
    gapped = x_test.copy()
    gapped.iloc[-h:, gapped.columns.get_indexer(pd.Index(group_cols))] = float("nan")
    ffilled = gapped.ffill()
    return gapped, ffilled


def run_measurement_c(
    evaluable_folds: list,
    live_matrix: pd.DataFrame,
    y_hourly: pd.Series,
    prices_qh: pd.Series,
) -> pd.DataFrame:
    qh_index = pd.DatetimeIndex(prices_qh.index)
    model = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)

    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % _REFIT_EVERY == 0:
            y_train = y_hourly.reindex(fold.train_index)
            x_train = live_matrix.reindex(fold.train_index).dropna()
            model.fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )
        x_test = live_matrix.loc[fold.test_index]

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()
        baseline_qh = persistence_forecast(prices_qh, fold.delivery_day, tz=_TZ).reindex(
            y_true_day.index
        )

        for group_name, group_cols in _GROUPS.items():
            for h in _H_VALUES:
                x_gapped, x_ffilled = _gap_variants(x_test, group_cols, h)

                pred_gapped = model.predict(fold.test_index, history=history, x_test=x_gapped)
                pred_ffilled = model.predict(fold.test_index, history=history, x_test=x_ffilled)

                ffill_qh = expand_to_quarterhour(
                    pred_ffilled, profile, target_day=fold.delivery_day, tz=_TZ
                ).reindex(y_true_day.index)
                gapped_qh = expand_to_quarterhour(
                    pred_gapped, profile, target_day=fold.delivery_day, tz=_TZ
                ).reindex(y_true_day.index)

                partial_qh = gapped_qh.copy()
                partial_qh.iloc[-h * 4 :] = baseline_qh.iloc[-h * 4 :].to_numpy()

                records.append(
                    pd.DataFrame(
                        {
                            "y_true": y_true_day.to_numpy(),
                            "pred_ffill": ffill_qh.to_numpy(),
                            "pred_partial": partial_qh.to_numpy(),
                            "pred_silence": baseline_qh.to_numpy(),
                            "delivery_day": fold.delivery_day,
                            "group": group_name,
                            "h": h,
                        },
                        index=y_true_day.index,
                    )
                )
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records)


def _run_dm(
    loss_a: pd.Series, loss_b: pd.Series, delivery_day: pd.Series
) -> tuple[DMResult, DMResult]:
    native = dm_test(loss_a, loss_b, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
    daily_a = loss_a.groupby(delivery_day).mean()
    daily_b = loss_b.groupby(delivery_day).mean()
    daily = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    return native, daily


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-folds", type=int, default=None)
    args = p.parse_args()

    t0 = time.monotonic()
    cfg = FeatureConfig()
    y_hourly, matrices, prices_qh, evaluable_folds, end_day = _prepare(cfg)
    folds_to_run = evaluable_folds[: args.probe_folds] if args.probe_folds else evaluable_folds

    run_t0 = time.monotonic()
    out = run_measurement_c(folds_to_run, matrices["live"], y_hourly, prices_qh)
    run_elapsed = time.monotonic() - run_t0

    if args.probe_folds:
        per_fold = run_elapsed / len(folds_to_run)
        estimated_total = per_fold * len(evaluable_folds)
        log.info(
            "PROBE: %d folds took %.1fs (%.2fs/fold, all 16 configs per fold) -- extrapolated "
            "total for %d folds: %.0fs (%.1f min)",
            len(folds_to_run),
            run_elapsed,
            per_fold,
            len(evaluable_folds),
            estimated_total,
            estimated_total / 60,
        )
        return

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT_PATH)

    print(f"\nn_folds={len(folds_to_run)}  refit_every={_REFIT_EVERY}  window_end={end_day}\n")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    summary_rows: list[dict[str, object]] = []
    log_metrics: dict[str, float] = {}
    with mlflow.start_run(run_name="measurement_c_target_day_gaps"):
        mlflow.log_params({"n_folds": len(folds_to_run), "refit_every": _REFIT_EVERY})
        mlflow.set_tags({"spec": "sprint6_step6_8", "measurement": "C"})

        for group_name in _GROUPS:
            for h in _H_VALUES:
                sub = out.loc[(out["group"] == group_name) & (out["h"] == h)]
                loss_silence = (sub["pred_silence"] - sub["y_true"]) ** 2
                rmse_by_response: dict[str, float] = {
                    response: rmse(sub["y_true"], sub[f"pred_{response}"])
                    for response in ("ffill", "partial", "silence")
                }
                row: dict[str, object] = {"group": group_name, "h": h}
                for response, rmse_value in rmse_by_response.items():
                    row[f"rmse_{response}"] = rmse_value
                for response in ("ffill", "partial"):
                    loss_r = (sub[f"pred_{response}"] - sub["y_true"]) ** 2
                    native, daily = _run_dm(loss_r, loss_silence, sub["delivery_day"])
                    row[f"{response}_vs_silence_native_p"] = native.p_value
                    row[f"{response}_vs_silence_native_loss_diff"] = native.mean_loss_diff
                    row[f"{response}_vs_silence_daily_p"] = daily.p_value
                    beats_silence = native.mean_loss_diff < 0 and native.p_value < 0.10
                    row[f"{response}_beats_silence"] = beats_silence
                    label = f"{group_name}_h{h}_{response}"
                    log_metrics[f"{label}_rmse"] = rmse_by_response[response]
                    log_metrics[f"{label}_vs_silence_p"] = native.p_value
                    mlflow.set_tag(f"{label}_beats_silence", str(beats_silence))
                summary_rows.append(row)
                print(
                    f"{group_name:15s} h={h}  RMSE ffill={row['rmse_ffill']:.4f} "
                    f"partial={row['rmse_partial']:.4f} silence={row['rmse_silence']:.4f}  "
                    f"ffill_beats_silence={row['ffill_beats_silence']} (p={row['ffill_vs_silence_native_p']:.4f})  "
                    f"partial_beats_silence={row['partial_beats_silence']} (p={row['partial_vs_silence_native_p']:.4f})"
                )

        mlflow.log_metrics(log_metrics)
        summary = pd.DataFrame(summary_rows)
        _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(_SUMMARY_PATH, index=False)
        mlflow.log_artifact(str(_SUMMARY_PATH))
        mlflow.log_artifact(str(_OUT_PATH))

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")
    print(f"logged to mlflow experiment {_MLFLOW_EXPERIMENT!r}, summary at {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
