"""One-off ablation (not a spec step, not run by CI): a follow-up to
scripts/ablation_no_fundamentals.py's question, from the other direction.

Motivation: price history and the renewables/residual-load forecast track
are the two things that could plausibly be secured via a backup data source
(e.g. Energy-Charts) if ENTSO-E/the weather run were briefly unavailable.
Everything else in the live feature set -- realised-load lags, forecast-error
lags (which need realised generation), cross-border flow lags -- depends on
ENTSO-E actuals that a backup source wouldn't necessarily cover the same way.
So: how much of `live`'s performance survives on ONLY the securable core --
(lagged) prices, the renewables/residual-load-forecast track (the NWP
reconstruction, i.e. exactly the block ablation_no_fundamentals.py just
established is worth having), commodities, and calendar/regime? Three
commodity variants, since TTF gas and EUA CO2 are two independently
securable series:

  core_with_commodities  -- calendar + fundamentals(NWP) + price lags + TTF + EUA + eua_missing
  core_no_commodities    -- calendar + fundamentals(NWP) + price lags (no commodities at all)
  core_gas_only          -- calendar + fundamentals(NWP) + price lags + TTF only (no EUA)

"fundamentals(NWP)" is literally features.nwp_fundamentals.build_nwp_forecast_fundamentals
-- the same function build_feature_set_for_day (the `live` candidate) calls,
reused unchanged, not reimplemented. Dropped relative to `live`: actual-load
lags, forecast-error lags, cross-border lags.

Compared against `live` (full feature set, for context) and `baseline` (Arena
persistence replica) with the same methodology, gate criterion, and DM-test
conventions as ablation_no_fundamentals.py -- see that script's docstring for
the full rationale (native quarter-hourly hac_lag=192/horizon=96, daily-block
hac_lag=7/horizon=1, gate = RMSE(candidate) < RMSE(baseline) AND native p<0.10).
MLflow: same experiment as ablation_no_fundamentals.py
(feature_reduction_for_live_system), separate run.

Usage: python scripts/ablation_core_minimal_feature_set.py
       ABLATION_SMOKE=1 python scripts/ablation_core_minimal_feature_set.py   # 20 folds, smoke test
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
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
from energy_price_forecast.features.availability import Feature, build_matrix
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig
from energy_price_forecast.features.fundamentals import build_commodity_features
from energy_price_forecast.features.lags import build_price_lags
from energy_price_forecast.features.nwp_fundamentals import (
    IncompleteReconstructionError,
    build_nwp_forecast_fundamentals,
)
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster
from energy_price_forecast.ops.windows import local_day_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation_core")

_TZ = "Europe/Berlin"
_PRICE_COL = "day_ahead_price"
_TRAIN_SPAN_DAYS = 90
_REFIT_EVERY = 14
_RANDOM_STATE = 0
_SHAPE_WINDOW_DAYS = 28
_QH_HAC_LAG = 192
_QH_HORIZON = 96
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_GATE_P_THRESHOLD = 0.10
_OUT_PATH = Path("data/processed/ablation_core_minimal_feature_set_predictions.parquet")
_MLFLOW_EXPERIMENT = "feature_reduction_for_live_system"

_CORE_CANDIDATES = ("core_with_commodities", "core_no_commodities", "core_gas_only")


@dataclass
class GateResult:
    native_vs_baseline: DMResult
    daily_vs_baseline: DMResult
    gate_pass: bool
    native_vs_live: DMResult | None = None
    daily_vs_live: DMResult | None = None


def _hourly_index_for_local_day(target_day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def build_core_for_day(
    target_day: dt.date,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    commodities: str,
) -> pd.DataFrame:
    """calendar + NWP renewables/residual-load-forecast track + price lags,
    plus commodities per `commodities` ("full" / "gas_only" / "none")."""
    target_index = _hourly_index_for_local_day(target_day)
    fundamentals = build_nwp_forecast_fundamentals(df, renewables, target_index)
    price = build_price_lags(df, target_index, cfg)

    commodity_feats: list[Feature]
    if commodities == "full":
        commodity_feats = build_commodity_features(df, target_index, cfg)
    elif commodities == "gas_only":
        commodity_feats = build_commodity_features(df, target_index, cfg)[:1]  # ttf_gas only
    elif commodities == "none":
        commodity_feats = []
    else:
        raise ValueError(f"unknown commodities mode {commodities!r}")

    features = [
        *build_calendar_features(target_index, cfg),
        *fundamentals,
        *commodity_feats,
        *price,
    ]
    return build_matrix(features)


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


def main() -> None:
    t0 = time.monotonic()
    cfg = FeatureConfig()
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

    log.info("building live matrix (NWP reconstruction)...")
    live_matrix, live_excluded = _build_matrix_over_days(
        all_days_needed, lambda d: build_feature_set_for_day(d, df, renewables)
    )
    log.info("building core_with_commodities matrix...")
    core_full_matrix, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_core_for_day(d, df, renewables, cfg, commodities="full")
    )
    log.info("building core_no_commodities matrix...")
    core_none_matrix, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_core_for_day(d, df, renewables, cfg, commodities="none")
    )
    log.info("building core_gas_only matrix...")
    core_gas_matrix, _ = _build_matrix_over_days(
        all_days_needed,
        lambda d: build_core_for_day(d, df, renewables, cfg, commodities="gas_only"),
    )
    log.info(
        "matrices built: live=%d rows (%d days excluded), core_full=%d, core_none=%d, core_gas=%d",
        len(live_matrix),
        len(live_excluded),
        len(core_full_matrix),
        len(core_none_matrix),
        len(core_gas_matrix),
    )

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

    candidates = {
        "core_with_commodities": core_full_matrix,
        "core_no_commodities": core_none_matrix,
        "core_gas_only": core_gas_matrix,
        "live": live_matrix,
    }
    models = {
        name: LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
        for name in candidates
    }

    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % _REFIT_EVERY == 0:
            y_train = y_hourly.reindex(fold.train_index)
            for name, matrix in candidates.items():
                x_train = matrix.reindex(fold.train_index).dropna()
                models[name].fit(y_train.loc[x_train.index], x_train)
            log.info("refit at fold %d/%d (%s)", i + 1, n_folds, fold.delivery_day.date())

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        preds_qh = {}
        for name, matrix in candidates.items():
            x_test = matrix.loc[fold.test_index]
            hourly_pred = models[name].predict(fold.test_index, history=history, x_test=x_test)
            preds_qh[name] = expand_to_quarterhour(
                hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ
            )
        preds_qh["baseline"] = persistence_forecast(prices_qh, fold.delivery_day, tz=_TZ)

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    **{
                        f"pred_{name}": preds_qh[name].reindex(y_true_day.index).to_numpy()
                        for name in (*candidates, "baseline")
                    },
                    "delivery_day": fold.delivery_day,
                    "n_slots_in_day": len(y_true_day),
                },
                index=y_true_day.index,
            )
        )
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("progress: %d/%d folds", i + 1, n_folds)

    predictions = pd.concat(records).sort_index()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(_OUT_PATH)

    all_candidates = (*candidates, "baseline")
    print(
        f"\nn_folds={n_folds}  refit_every={_REFIT_EVERY}  train_span_days={_TRAIN_SPAN_DAYS}  "
        f"shape_window_days={_SHAPE_WINDOW_DAYS}"
    )
    print(f"live_excluded_days={len(live_excluded)}\n")

    metrics = {}
    for name in all_candidates:
        t, p = predictions["y_true"], predictions[f"pred_{name}"]
        metrics[name] = {"mae": mae(t, p), "rmse": rmse(t, p)}
        print(f"{name:22s}  MAE={metrics[name]['mae']:.4f}  RMSE={metrics[name]['rmse']:.4f}")

    def _run_dm(candidate: str, reference: str) -> tuple[DMResult, DMResult]:
        loss_c = (predictions[f"pred_{candidate}"] - predictions["y_true"]) ** 2
        loss_r = (predictions[f"pred_{reference}"] - predictions["y_true"]) ** 2
        native = dm_test(loss_c, loss_r, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
        daily_c = loss_c.groupby(predictions["delivery_day"]).mean()
        daily_r = loss_r.groupby(predictions["delivery_day"]).mean()
        daily = dm_test(daily_c, daily_r, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
        return native, daily

    print("\n--- gate check: is the securable-core feature set still worth submitting? ---")
    print(
        "(spec 6.6 Entscheidung 24 criterion: RMSE(candidate) < RMSE(baseline) AND native-QH DM p<0.10)\n"
    )
    gate_results: dict[str, GateResult] = {}
    for name in _CORE_CANDIDATES:
        native_vs_base, daily_vs_base = _run_dm(name, "baseline")
        rmse_ok = metrics[name]["rmse"] < metrics["baseline"]["rmse"]
        dm_ok = native_vs_base.mean_loss_diff < 0 and native_vs_base.p_value < _GATE_P_THRESHOLD
        gate_pass = rmse_ok and dm_ok
        verdict = (
            "PASS -- still worth submitting" if gate_pass else "FAIL -- baseline wins, don't submit"
        )
        gate_results[name] = GateResult(
            native_vs_baseline=native_vs_base,
            daily_vs_baseline=daily_vs_base,
            gate_pass=gate_pass,
        )
        print(f"{name} vs baseline:")
        print(
            f"  native (hac={_QH_HAC_LAG},h={_QH_HORIZON}): mean_loss_diff_sq={native_vs_base.mean_loss_diff:.4f}  "
            f"p={native_vs_base.p_value:.4f}  n={native_vs_base.n_obs}"
        )
        print(
            f"  daily  (hac={_DAILY_HAC_LAG},h={_DAILY_HORIZON}): mean_loss_diff_sq={daily_vs_base.mean_loss_diff:.4f}  "
            f"p={daily_vs_base.p_value:.4f}  n={daily_vs_base.n_obs}"
        )
        print(
            f"  RMSE(candidate)={metrics[name]['rmse']:.4f} vs RMSE(baseline)={metrics['baseline']['rmse']:.4f}"
        )
        print(f"  -> {verdict}\n")

    print("--- for context: vs the current live candidate (full feature set) ---\n")
    for name in _CORE_CANDIDATES:
        native_vs_live, daily_vs_live = _run_dm(name, "live")
        gate_results[name].native_vs_live = native_vs_live
        gate_results[name].daily_vs_live = daily_vs_live
        sig = (
            "significant (p<0.10)"
            if native_vs_live.p_value < _GATE_P_THRESHOLD
            else "not significant"
        )
        verdict = "WORSE than live" if native_vs_live.mean_loss_diff > 0 else "BETTER than live"
        print(
            f"{name:22s} vs live (native): mean_loss_diff_sq={native_vs_live.mean_loss_diff:.4f}  "
            f"p={native_vs_live.p_value:.4f}  -> {verdict}, {sig}"
        )

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")
    print(f"predictions written to {_OUT_PATH}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ablation_core_minimal_feature_set"):
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "refit_every": _REFIT_EVERY,
                "train_span_days": _TRAIN_SPAN_DAYS,
                "shape_window_days": _SHAPE_WINDOW_DAYS,
                "live_excluded_days": len(live_excluded),
                "objective": "quantile",
                "alpha": 0.5,
                "random_state": _RANDOM_STATE,
                **_DEFAULT_PARAMS,
            }
        )
        mlflow.set_tags(
            {
                "purpose": "one-off ablation -- securable-core feature set (price lags + NWP "
                "renewables/residual-load-forecast + commodities variants + calendar) vs "
                "live and the Arena baseline",
                "candidates": ",".join(all_candidates),
                "not_a_spec_step": "true",
            }
        )
        log_metrics: dict[str, float] = {}
        for name in all_candidates:
            log_metrics[f"{name}_mae"] = metrics[name]["mae"]
            log_metrics[f"{name}_rmse"] = metrics[name]["rmse"]
        for name in _CORE_CANDIDATES:
            gr = gate_results[name]
            log_metrics[f"{name}_vs_baseline_native_loss_diff"] = (
                gr.native_vs_baseline.mean_loss_diff
            )
            log_metrics[f"{name}_vs_baseline_native_p"] = gr.native_vs_baseline.p_value
            log_metrics[f"{name}_vs_baseline_daily_loss_diff"] = gr.daily_vs_baseline.mean_loss_diff
            log_metrics[f"{name}_vs_baseline_daily_p"] = gr.daily_vs_baseline.p_value
            assert gr.native_vs_live is not None and gr.daily_vs_live is not None
            log_metrics[f"{name}_vs_live_native_loss_diff"] = gr.native_vs_live.mean_loss_diff
            log_metrics[f"{name}_vs_live_native_p"] = gr.native_vs_live.p_value
            log_metrics[f"{name}_vs_live_daily_loss_diff"] = gr.daily_vs_live.mean_loss_diff
            log_metrics[f"{name}_vs_live_daily_p"] = gr.daily_vs_live.p_value
            mlflow.set_tag(f"{name}_gate_verdict", "PASS" if gr.gate_pass else "FAIL")
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(_OUT_PATH))
    print(f"\nlogged to mlflow experiment {_MLFLOW_EXPERIMENT!r}")


if __name__ == "__main__":
    main()
