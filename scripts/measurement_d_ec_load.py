"""Sprint 6.8, Messung D (docs/sprint6_step6_8_spec.md section 7): when the
ENTSO-E day-ahead load forecast is missing, how is the Energy-Charts load
forecast best used -- patch the existing residual-load track, or train a
wholly separate model on it?

Reuses, unchanged: scripts.measurement_a_candidate_intake._prepare (window,
folds, the `core_no_commodities`/`floor_core` matrices and D0's own
predictions), scripts.ablation_energy_charts_residual_load's
build_ec_candidate_for_day/_read_ec_hourly (the `core_ec_load` candidate --
load from Energy-Charts, renewables from the own NWP reconstruction, exactly
D1's definition), and features.nwp_fundamentals.build_residual_load_nwp/
build_renewable_share_nwp (the same arithmetic, only fed a patched load
Feature for D2/D3's target-day row).

Four arms (spec's own table), refit_every=1 (section 2, D shares this with
A/B):
  D0  ENTSO-E load, train and target      == core_no_commodities (Messung A)
  D1  Energy-Charts load, train and target == core_ec_load (separate model)
  D2  ENTSO-E load in training; Energy-Charts load patched in at the target
      day only (raw patch, no offset) -- D0's own trained model, D2's own
      target-day row
  D3  same as D2, but the patched load gets a bias correction estimated
      ROLLING from the fold's own training window (median of ENTSO-E-EC over
      fold.train_index only -- never the whole-window value from the earlier
      Teil 4a screening, which would leak future days into a backtest fold)

D2/D3 treat every test day as an ENTSO-E-load outage day -- these numbers
describe outage-day quality, not average system quality (spec's own point).

Multi-day-outage arm set (the actual scenario of interest):
  D3-k3  D3's target-day patch, PLUS the last 3 training days dropped
         entirely (no ENTSO-E load forecast during a real 3-day outage --
         same "edge" gap-removal Messung B already used, reapplied here to
         core_no_commodities' own training window; the D3 offset for this
         arm is estimated from the SAME reduced window, consistent with "no
         ENTSO-E load available for those days at all")
  D1-k3  == D1 itself, unchanged, reported not recomputed: Energy-Charts
         covers the outage days too, so there is no gap to remove.

`floor_core` (Messung A) is the no-residual-load floor for comparison.

Mandatory caveat (spec section 7, verbatim, written into the results CSV):
Energy-Charts load is, per the probe data so far, only available shortly
before the LAST submission slot (n small) -- these numbers only apply to
slots where the probe demonstrates availability.

Usage:
  python -m scripts.measurement_d_ec_load --probe-folds 10
  python -m scripts.measurement_d_ec_load
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_renewables_predictions
from energy_price_forecast.evaluation.dm_test import DMResult, dm_test
from energy_price_forecast.evaluation.metrics import mae, rmse
from energy_price_forecast.features.availability import Feature, build_matrix
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.fundamentals import build_commodity_features
from energy_price_forecast.features.lags import build_price_lags
from energy_price_forecast.features.nwp_fundamentals import (
    build_nwp_forecast_fundamentals,
    build_renewable_share_nwp,
    build_residual_load_nwp,
)
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import LGBMForecaster
from scripts.ablation_energy_charts_residual_load import (
    _hourly_index_for_local_day,
    _read_ec_hourly,
    build_ec_candidate_for_day,
    build_ec_fundamentals,
)
from scripts.measurement_a_candidate_intake import (
    _DAILY_HAC_LAG,
    _DAILY_HORIZON,
    _GATE_P_THRESHOLD,
    _QH_HAC_LAG,
    _QH_HORIZON,
    _RANDOM_STATE,
    _REFIT_EVERY,
    _SHAPE_WINDOW_DAYS,
    _TZ,
    FeatureConfig,
    _build_matrix_over_days,
    _prepare,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("measurement_d")

_MLFLOW_EXPERIMENT = "live_robustness_6_8"
_OUT_PATH = Path("data/processed/measurement_d_ec_load_predictions.parquet")
_SUMMARY_PATH = Path("outputs/results/measurement_d_ec_load_summary.csv")
_REFERENCE_PATH = Path("data/processed/measurement_a_candidate_intake_predictions.parquet")

# Spec 6.9 section 2.2: D1 (EC load, training and target) on a core_gas_only
# basis -- the follow-up measurement closing "Messung D lief gasfrei" before
# core_gas_ec (rank 2 of the fallback ladder) can go live. Kept in separate
# output files from the original D0-D3_k3 arms above (untouched by this
# addition) rather than folded into them.
_D1_GAS_OUT_PATH = Path("data/processed/measurement_d1_gas_ec_load_predictions.parquet")
_D1_GAS_SUMMARY_PATH = Path("outputs/results/measurement_d1_gas_ec_load_summary.csv")

_CAVEAT = (
    "# CAVEAT (spec section 7, verbatim): Energy-Charts load is, per the probe data so far, "
    "only available shortly before the LAST submission slot (n small). These numbers apply "
    "only to slots where the probe demonstrates availability. See "
    "docs/sprint6_auftrag_ec_load_sonde_fein.md for the dedicated fine-grained probe."
)


def _renamed(feature: Feature, name: str) -> Feature:
    return replace(feature, name=name, values=feature.values.rename(name))


def build_patched_target_row(
    target_day,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    ec_load_hourly: pd.Series,
    *,
    offset: float,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """core_no_commodities' own target-day row, with the ENTSO-E load
    forecast replaced by (Energy-Charts load + offset) -- offset=0.0 is D2's
    raw patch, a rolling median-diff is D3's corrected patch. Renewables stay
    the own NWP reconstruction (D2/D3's spec definition), same as D0/D1."""
    target_index = _hourly_index_for_local_day(target_day)
    nwp = build_nwp_forecast_fundamentals(df, renewables, target_index)
    won, woff, solar = nwp[1], nwp[2], nwp[3]

    patched_values = ec_load_hourly.reindex(target_index).to_numpy() + offset
    load = Feature(
        "load_forecast_day_ahead",
        pd.Series(patched_values, index=target_index, name="load_forecast_day_ahead"),
        pd.Series(gate_closure_for_index(target_index), index=target_index),
    )
    residual = _renamed(
        build_residual_load_nwp(load, won, woff, solar), "residual_load_forecast_nwp"
    )
    share = _renamed(
        build_renewable_share_nwp(load, won, woff, solar), "renewable_share_forecast_nwp"
    )
    price = build_price_lags(df, target_index, cfg)
    features = [
        *build_calendar_features(target_index, cfg),
        load,
        won,
        woff,
        solar,
        residual,
        share,
        *price,
    ]
    return build_matrix(features)


def _rolling_offset(
    entsoe_load: pd.Series, ec_load: pd.Series, train_index: pd.DatetimeIndex
) -> float:
    """Median(ENTSO-E - EC) over `train_index` only -- never the whole
    window (spec's explicit leakage trap: the -547 MW from the earlier
    screening used the whole window and must not be reused here)."""
    diff = (entsoe_load.reindex(train_index) - ec_load.reindex(train_index)).dropna()
    return float(diff.median())


def _last_k_days_mask(train_index: pd.DatetimeIndex, k: int, tz: str) -> pd.Series:
    local_days = pd.DatetimeIndex(train_index).tz_convert(tz).normalize()
    cutoff_days = set(pd.DatetimeIndex(local_days.unique()).sort_values()[-k:].date)
    return ~pd.Series(local_days.date, index=train_index).isin(cutoff_days)


_K3 = 3


def _predict_qh(
    model: LGBMForecaster, x_test: pd.DataFrame, fold, history: pd.Series, profile
) -> pd.Series:
    hourly_pred = model.predict(fold.test_index, history=history, x_test=x_test)
    return expand_to_quarterhour(hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ)


def build_ec_gas_for_day(
    target_day,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    ec: dict[str, pd.Series],
    cfg,
) -> pd.DataFrame:
    """Spec 6.9 section 2.2 / section 2.1's rank-2 row (`core_gas_ec`):
    core_ec_load's own fundamentals (EC load, own NWP renewables
    reconstruction -- build_ec_fundamentals's "core_ec_load" branch,
    unchanged) plus a TTF-gas-only commodity feature, the same way
    scripts.ablation_core_minimal_feature_set.build_core_for_day and
    scripts.measurement_a_candidate_intake.build_floor_for_day already add
    gas on top of their own shared blocks -- not a new pattern, the one
    missing repetition identified in step 1 (docs/sprint6_step6_9_log.md
    section 3.1 point 1: build_ec_candidate_for_day never calls
    build_commodity_features at all)."""
    target_index = _hourly_index_for_local_day(target_day)
    fundamentals = build_ec_fundamentals("core_ec_load", target_index, df, renewables, ec)
    gas = build_commodity_features(df, target_index, cfg)[:1]  # ttf_gas only, no EUA
    price = build_price_lags(df, target_index, cfg)
    features = [*build_calendar_features(target_index, cfg), *fundamentals, *gas, *price]
    return build_matrix(features)


def run_d1_gas_measurement(
    evaluable_folds: list,
    ec_gas_matrix: pd.DataFrame,
    y_hourly: pd.Series,
    prices_qh: pd.Series,
) -> pd.DataFrame:
    """Spec 6.9 section 2.2: D1 (EC load, training and target) on a
    core_gas_only basis.

    No separate D1-k3 arm is fitted here -- and an earlier revision of this
    function that did (dropping the training window's last _K3 local days
    via _last_k_days_mask, mirroring D3_k3) was a real modelling mistake,
    caught by its own "muss = D1 sein" plausibility check failing loudly
    (max abs diff ~59 EUR/MWh on a real run, 2026-09-23): _last_k_days_mask
    simulates an ENTSO-E load-forecast outage, the one and only thing D3_k3
    exists to test. None of D1_gas's own inputs (Energy-Charts load, TTF
    gas, the NWP renewables reconstruction) come from ENTSO-E's load
    forecast at all, so that scenario removes nothing from D1_gas's
    training data -- there is no gap of any kind to simulate, exactly the
    same reasoning that already made the original (gas-free) D1-k3 "reported
    not recomputed" (spec 6.8 section 7) rather than a second fit. Fitting a
    smaller, differently-windowed model and comparing it against D1_gas was
    never a meaningful plausibility check for this candidate -- it just
    measured "does removing 3 unrelated training days change the fit"
    (yes, unsurprisingly, for any model). D1-gas-k3 is therefore identical
    to D1_gas by construction, same as D1-k3 was to D1."""
    qh_index = pd.DatetimeIndex(prices_qh.index)
    model_d1 = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)

    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % _REFIT_EVERY == 0:
            y_train = y_hourly.reindex(fold.train_index)
            x_train = ec_gas_matrix.reindex(fold.train_index).dropna()
            model_d1.fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        x_test = ec_gas_matrix.loc[fold.test_index]
        preds = {"D1_gas": _predict_qh(model_d1, x_test, fold, history, profile)}

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    **{
                        f"pred_{name}": series.reindex(y_true_day.index).to_numpy()
                        for name, series in preds.items()
                    },
                    "delivery_day": fold.delivery_day,
                },
                index=y_true_day.index,
            )
        )
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("D1_gas progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records).sort_index()


def run_measurement_d(
    evaluable_folds: list,
    core_matrix: pd.DataFrame,
    ec_matrix: pd.DataFrame,
    y_hourly: pd.Series,
    prices_qh: pd.Series,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    ec_load_hourly: pd.Series,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    qh_index = pd.DatetimeIndex(prices_qh.index)
    entsoe_load = df["load_forecast_day_ahead"]
    columns = core_matrix.columns

    model_d0 = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
    model_d1 = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
    model_d3k3 = LGBMForecaster(
        objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1
    )

    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        gapped_train_index = fold.train_index[
            _last_k_days_mask(fold.train_index, _K3, _TZ).to_numpy()
        ]
        if i % _REFIT_EVERY == 0:
            y_train = y_hourly.reindex(fold.train_index)
            x_train_d0 = core_matrix.reindex(fold.train_index).dropna()
            model_d0.fit(y_train.loc[x_train_d0.index], x_train_d0)

            x_train_d1 = ec_matrix.reindex(fold.train_index).dropna()
            model_d1.fit(y_train.loc[x_train_d1.index], x_train_d1)

            x_train_d3k3 = core_matrix.reindex(gapped_train_index).dropna()
            y_train_gapped = y_hourly.reindex(gapped_train_index)
            model_d3k3.fit(y_train_gapped.loc[x_train_d3k3.index], x_train_d3k3)

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        target_day = fold.delivery_day.date()
        offset_d3 = _rolling_offset(entsoe_load, ec_load_hourly, fold.train_index)
        offset_d3k3 = _rolling_offset(entsoe_load, ec_load_hourly, gapped_train_index)

        x_test_d0 = core_matrix.loc[fold.test_index]
        x_test_d1 = ec_matrix.loc[fold.test_index]
        x_test_d2 = build_patched_target_row(
            target_day, df, renewables, ec_load_hourly, offset=0.0, cfg=cfg
        )[columns]
        x_test_d3 = build_patched_target_row(
            target_day, df, renewables, ec_load_hourly, offset=offset_d3, cfg=cfg
        )[columns]
        x_test_d3k3 = build_patched_target_row(
            target_day, df, renewables, ec_load_hourly, offset=offset_d3k3, cfg=cfg
        )[columns]

        preds = {
            "D0": _predict_qh(model_d0, x_test_d0, fold, history, profile),
            "D1": _predict_qh(model_d1, x_test_d1, fold, history, profile),
            "D2": _predict_qh(model_d0, x_test_d2, fold, history, profile),
            "D3": _predict_qh(model_d0, x_test_d3, fold, history, profile),
            "D3_k3": _predict_qh(model_d3k3, x_test_d3k3, fold, history, profile),
        }

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    **{
                        f"pred_{name}": series.reindex(y_true_day.index).to_numpy()
                        for name, series in preds.items()
                    },
                    "delivery_day": fold.delivery_day,
                    "offset_d3": offset_d3,
                },
                index=y_true_day.index,
            )
        )
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records).sort_index()


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
    core_matrix = matrices["core_no_commodities"]

    df = load_interim_hourly()
    renewables = load_renewables_predictions()
    ec = {
        production_type: _read_ec_hourly(production_type)
        for production_type in ("load", "solar", "wind_onshore", "wind_offshore")
    }

    core_index_local = pd.DatetimeIndex(core_matrix.index).tz_convert(_TZ)
    all_days_needed = pd.date_range(core_index_local.min(), core_index_local.max(), freq="D")
    log.info("building core_ec_load matrix...")
    ec_matrix, ec_excluded = _build_matrix_over_days(
        all_days_needed,
        lambda d: build_ec_candidate_for_day("core_ec_load", d, df, renewables, ec, cfg),
    )
    log.info("core_ec_load matrix: %d rows, %d days excluded", len(ec_matrix), len(ec_excluded))

    test_index = pd.DatetimeIndex(
        sorted({ts for fold in evaluable_folds for ts in fold.test_index})
    )
    if not pd.DatetimeIndex(ec_matrix.index).intersection(test_index).equals(test_index):
        raise AssertionError(
            "core_ec_load matrix does not cover every evaluable fold's test index -- "
            "fold lists are not identical across Messung D's own candidates (section 2)"
        )

    log.info("building core_gas_ec matrix (spec 6.9 section 2.2)...")
    ec_gas_matrix, ec_gas_excluded = _build_matrix_over_days(
        all_days_needed,
        lambda d: build_ec_gas_for_day(d, df, renewables, ec, cfg),
    )
    log.info(
        "core_gas_ec matrix: %d rows, %d days excluded", len(ec_gas_matrix), len(ec_gas_excluded)
    )
    if not pd.DatetimeIndex(ec_gas_matrix.index).intersection(test_index).equals(test_index):
        raise AssertionError(
            "core_gas_ec matrix does not cover every evaluable fold's test index -- "
            "fold lists are not identical (spec 6.9 section 2.2)"
        )

    folds_to_run = evaluable_folds[: args.probe_folds] if args.probe_folds else evaluable_folds

    run_t0 = time.monotonic()
    out = run_measurement_d(
        folds_to_run, core_matrix, ec_matrix, y_hourly, prices_qh, df, renewables, ec["load"], cfg
    )
    run_elapsed = time.monotonic() - run_t0

    gas_t0 = time.monotonic()
    out_gas = run_d1_gas_measurement(folds_to_run, ec_gas_matrix, y_hourly, prices_qh)
    gas_elapsed = time.monotonic() - gas_t0

    if args.probe_folds:
        per_fold = run_elapsed / len(folds_to_run)
        estimated_total = per_fold * len(evaluable_folds)
        log.info(
            "PROBE (D0-D3_k3): %d folds took %.1fs (%.2fs/fold) -- extrapolated total for %d "
            "folds: %.0fs (%.1f min)",
            len(folds_to_run),
            run_elapsed,
            per_fold,
            len(evaluable_folds),
            estimated_total,
            estimated_total / 60,
        )
        per_fold_gas = gas_elapsed / len(folds_to_run)
        estimated_total_gas = per_fold_gas * len(evaluable_folds)
        log.info(
            "PROBE (D1_gas): %d folds took %.1fs (%.2fs/fold) -- extrapolated total "
            "for %d folds: %.0fs (%.1f min)",
            len(folds_to_run),
            gas_elapsed,
            per_fold_gas,
            len(evaluable_folds),
            estimated_total_gas,
            estimated_total_gas / 60,
        )
        return

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT_PATH)

    reference = pd.read_parquet(_REFERENCE_PATH)
    ref = reference.loc[out.index]
    out["pred_floor_core"] = ref["pred_floor_core"].to_numpy()
    out["pred_baseline"] = ref["pred_baseline"].to_numpy()

    d0_check = (out["pred_D0"] - ref["pred_core_no_commodities"]).abs().max()
    log.info(
        "sanity: D0 vs Messung A's own core_no_commodities predictions, max abs diff=%.6g", d0_check
    )

    print(f"\nn_folds={len(folds_to_run)}  refit_every={_REFIT_EVERY}  window_end={end_day}")
    print(f"D0 vs Messung A core_no_commodities max abs diff: {d0_check:.6g}\n")

    arms = ("D0", "D1", "D2", "D3", "D3_k3")
    metrics: dict[str, dict[str, float]] = {}
    for name in (*arms, "floor_core", "baseline"):
        col = f"pred_{name}"
        metrics[name] = {"mae": mae(out["y_true"], out[col]), "rmse": rmse(out["y_true"], out[col])}
        print(f"{name:12s}  MAE={metrics[name]['mae']:.4f}  RMSE={metrics[name]['rmse']:.4f}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    summary_rows: list[dict[str, object]] = []
    log_metrics: dict[str, float] = {}
    with mlflow.start_run(run_name="measurement_d_ec_load"):
        mlflow.log_params({"n_folds": len(folds_to_run), "refit_every": _REFIT_EVERY, "k3": _K3})
        mlflow.set_tags({"spec": "sprint6_step6_8", "measurement": "D"})

        print("\n--- DM vs D0, floor_core, baseline ---\n")
        for arm in ("D1", "D2", "D3", "D3_k3"):
            print(f"{arm}:")
            for reference_name in ("D0", "floor_core", "baseline"):
                loss_a = (out[f"pred_{arm}"] - out["y_true"]) ** 2
                loss_b = (out[f"pred_{reference_name}"] - out["y_true"]) ** 2
                native, daily = _run_dm(loss_a, loss_b, out["delivery_day"])
                better = "better" if native.mean_loss_diff < 0 else "worse"
                gate_pass = (
                    metrics[arm]["rmse"] < metrics["baseline"]["rmse"]
                    if reference_name == "baseline"
                    else None
                )
                print(
                    f"  vs {reference_name:12s} loss_diff={native.mean_loss_diff:+.4f} "
                    f"p={native.p_value:.4f} ({better})"
                    + (f"  gate={'PASS' if gate_pass else 'FAIL'}" if gate_pass is not None else "")
                )
                summary_rows.append(
                    {
                        "arm": arm,
                        "reference": reference_name,
                        "rmse_arm": metrics[arm]["rmse"],
                        "rmse_reference": metrics[reference_name]["rmse"],
                        "native_loss_diff": native.mean_loss_diff,
                        "native_p": native.p_value,
                        "daily_loss_diff": daily.mean_loss_diff,
                        "daily_p": daily.p_value,
                    }
                )
                log_metrics[f"{arm}_vs_{reference_name}_native_p"] = native.p_value
                log_metrics[f"{arm}_vs_{reference_name}_native_loss_diff"] = native.mean_loss_diff
            print()
        for name, m in metrics.items():
            log_metrics[f"{name}_rmse"] = m["rmse"]
            log_metrics[f"{name}_mae"] = m["mae"]

        mlflow.log_metrics(log_metrics)
        summary = pd.DataFrame(summary_rows)
        _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SUMMARY_PATH.open("w", newline="\n") as f:
            f.write(_CAVEAT + "\n")
            summary.to_csv(f, index=False, lineterminator="\n")
        mlflow.log_artifact(str(_SUMMARY_PATH))
        mlflow.log_artifact(str(_OUT_PATH))

    print("D1_k3: identical to D1 -- Energy-Charts load covers the outage days, no gap to remove.")

    # --- Spec 6.9 section 2.2: D1 on a core_gas_only basis -----------------
    _D1_GAS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_gas.to_parquet(_D1_GAS_OUT_PATH)

    ref_gas = reference.loc[out_gas.index]
    out_gas["pred_baseline"] = ref_gas["pred_baseline"].to_numpy()
    out_gas["pred_core_gas_only"] = ref_gas["pred_core_gas_only"].to_numpy()

    print(
        "\nD1_gas_k3: identical to D1_gas by construction -- none of D1_gas's own inputs "
        "(Energy-Charts load, TTF gas, the NWP renewables reconstruction) come from ENTSO-E's "
        "load forecast, so an ENTSO-E load-forecast outage (the one scenario _last_k_days_mask "
        "simulates) removes nothing from its training data. Same reasoning as the original "
        "D1-k3 (spec 6.8 section 7). See run_d1_gas_measurement's own docstring for the real "
        "run (2026-09-23) that found fitting a separate, differently-windowed model here was a "
        "modelling mistake, not a meaningful plausibility check."
    )

    gas_metrics: dict[str, dict[str, float]] = {}
    for name in ("D1_gas", "core_gas_only", "baseline"):
        col = f"pred_{name}"
        gas_metrics[name] = {
            "mae": mae(out_gas["y_true"], out_gas[col]),
            "rmse": rmse(out_gas["y_true"], out_gas[col]),
        }
        print(
            f"{name:15s}  MAE={gas_metrics[name]['mae']:.4f}  RMSE={gas_metrics[name]['rmse']:.4f}"
        )

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    gas_summary_rows: list[dict[str, object]] = []
    gas_log_metrics: dict[str, float] = {}
    gate_pass = gas_metrics["D1_gas"]["rmse"] < gas_metrics["baseline"]["rmse"]
    with mlflow.start_run(run_name="measurement_d1_gas_ec_load_6_9"):
        mlflow.log_params(
            {"n_folds": len(folds_to_run), "refit_every": _REFIT_EVERY, "k3": _K3}
        )
        mlflow.set_tags({"spec": "sprint6_step6_9", "measurement": "D1_gas (section 2.2)"})

        print("\n--- DM vs baseline (criterion), vs core_gas_only (informative) ---\n")
        native_gate = None
        for reference_name in ("baseline", "core_gas_only"):
            loss_a = (out_gas["pred_D1_gas"] - out_gas["y_true"]) ** 2
            loss_b = (out_gas[f"pred_{reference_name}"] - out_gas["y_true"]) ** 2
            native, daily = _run_dm(loss_a, loss_b, out_gas["delivery_day"])
            if reference_name == "baseline":
                native_gate = native
            better = "better" if native.mean_loss_diff < 0 else "worse"
            gate_label = (
                f"  gate={'PASS' if gate_pass and native.p_value < _GATE_P_THRESHOLD else 'FAIL'}"
                if reference_name == "baseline"
                else ""
            )
            print(
                f"  vs {reference_name:15s} loss_diff={native.mean_loss_diff:+.4f} "
                f"p={native.p_value:.4f} ({better}){gate_label}"
            )
            gas_summary_rows.append(
                {
                    "arm": "D1_gas",
                    "reference": reference_name,
                    "rmse_arm": gas_metrics["D1_gas"]["rmse"],
                    "rmse_reference": gas_metrics[reference_name]["rmse"],
                    "native_loss_diff": native.mean_loss_diff,
                    "native_p": native.p_value,
                    "daily_loss_diff": daily.mean_loss_diff,
                    "daily_p": daily.p_value,
                }
            )
            gas_log_metrics[f"D1_gas_vs_{reference_name}_native_p"] = native.p_value
            gas_log_metrics[f"D1_gas_vs_{reference_name}_native_loss_diff"] = native.mean_loss_diff
        for name, m in gas_metrics.items():
            gas_log_metrics[f"{name}_rmse"] = m["rmse"]
            gas_log_metrics[f"{name}_mae"] = m["mae"]

        gate_final = bool(
            gate_pass and native_gate is not None and native_gate.p_value < _GATE_P_THRESHOLD
        )
        mlflow.log_metrics(gas_log_metrics)
        mlflow.set_tag("gate_pass", str(gate_final))
        gas_summary = pd.DataFrame(gas_summary_rows)
        _D1_GAS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _D1_GAS_SUMMARY_PATH.open("w", newline="\n") as f:
            f.write(_CAVEAT + "\n")
            f.write(
                "# D1_gas_k3: identical to D1_gas by construction (no ENTSO-E-load-forecast "
                "input to outage-simulate) -- not a separate row, same as spec 6.8's own D1-k3.\n"
            )
            gas_summary.to_csv(f, index=False, lineterminator="\n")
        mlflow.log_artifact(str(_D1_GAS_SUMMARY_PATH))
        mlflow.log_artifact(str(_D1_GAS_OUT_PATH))

    if gate_final:
        print(
            f"\nGATE PASS: D1_gas (core_gas_ec candidate) RMSE={gas_metrics['D1_gas']['rmse']:.4f} "
            "-- replaces the interpolated \"~26.5\" placeholder in the Zielbild (spec 6.9 "
            "section 2.2)."
        )
    else:
        print(
            "\nGATE FAIL -- do not enter D1_gas into the fallback-ladder table as measured. "
            "Rueckfrage per spec 6.9 section 2.2/10.9."
        )

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")
    print(f"logged to mlflow experiment {_MLFLOW_EXPERIMENT!r}, summaries at {_SUMMARY_PATH} "
          f"and {_D1_GAS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
