"""Energy-Charts backup Auftrag (docs/sprint6_auftrag_energy_charts_backup_2.md,
Teil 3): residual_load_forecast_ec built from the Energy-Charts day-ahead
series (Teil 1's historical pull), measured against the two already-measured
reference points from docs/feature_reduction_for_live_system.md -- NOT the
baseline (Auftrag section 4): `core_no_commodities` (28.90 EUR/MWh, own NWP
reconstruction) as the upper bound, `no_fundamentals` (42.28 EUR/MWh, no
residual-load track at all) as the floor.

Reuses `core_no_commodities`'s / `no_fundamentals`'s own predictions and
evaluable-fold set (both already on disk from
scripts/ablation_core_minimal_feature_set.py / scripts/ablation_no_fundamentals.py)
rather than recomputing them -- identical config (same window, refit_every,
train_span_days, shape window, random_state) provably yields the same fold
set, checked by an index-equality assertion below, not just assumed.

The residual-load formula is reused, not rebuilt (Auftrag section 4): calls
features/nwp_fundamentals.py's build_residual_load_nwp / build_renewable_share_nwp
directly with Energy-Charts-sourced Features instead of NWP-reconstruction
Features -- same function, same formula, only the inputs differ.

Three candidates, varying only where the two residual-load halves come from:

  core_ec_load        load forecast: Energy-Charts        renewables: own NWP reconstruction
  core_ec_renewables  load forecast: ENTSO-E (DA_FORECAST) renewables: Energy-Charts
  core_ec_full        load forecast: Energy-Charts         renewables: Energy-Charts

Energy-Charts values are treated as known at gate closure of their own
delivery day -- the same knowledge-time rule as a DA_FORECAST column
(Availability.DA_FORECAST) -- WITHOUT registering a new column in
features/availability.py (Auftrag section 5: no registry entry). This is a
deliberate, disclosed upper-bound assumption: real availability before gate
closure is not measured by this Auftrag -- see the mandatory caveat written
into the results CSV and docs/feature_reduction_for_live_system.md's Teil 3.

Usage: python scripts/ablation_energy_charts_residual_load.py
       ABLATION_SMOKE=1 python scripts/ablation_energy_charts_residual_load.py   # 20 folds
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import replace
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
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig
from energy_price_forecast.features.lags import build_price_lags
from energy_price_forecast.features.nwp_fundamentals import (
    IncompleteReconstructionError,
    build_nwp_forecast_fundamentals,
    build_renewable_share_nwp,
    build_residual_load_nwp,
)
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster
from energy_price_forecast.ops.windows import local_day_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation_energy_charts")

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
_OUT_PATH = Path("data/processed/ablation_energy_charts_residual_load_predictions.parquet")
_REPORT_PATH = Path("outputs/results/energy_charts_residual_load_ablation.csv")
_MLFLOW_EXPERIMENT = "feature_reduction_for_live_system"

# Mandatory caveat (Auftrag section 4), pattern matching the 6.4 "Bestfall-Hinweis"
# CSV comment line (see e.g. outputs/results/arena_bridge_backtest.csv).
_UPPER_BOUND_CAVEAT = (
    "# UPPER BOUND: these numbers use Energy-Charts values whose availability "
    "before gate closure is NOT measured (see docs/sprint6_auftrag_energy_charts_backup_2.md "
    "section 4 and the separate knowledge-time probe, scripts/probe_energy_charts_forecast.py). "
    "Reachable only if a later knowledge-time measurement confirms timely availability. "
    "mean_loss_diff_sq = loss(candidate) - loss(reference); negative means the candidate is better."
)

_CORE_REF_PATH = Path("data/processed/ablation_core_minimal_feature_set_predictions.parquet")
_NO_FUNDAMENTALS_REF_PATH = Path("data/processed/ablation_no_fundamentals_predictions.parquet")

_EC_DIR = Path("data/raw/energy_charts/public_power_forecast")
_EC_CANDIDATES = ("core_ec_load", "core_ec_renewables", "core_ec_full")


def _read_ec_hourly(production_type: str, *, root: Path = _EC_DIR) -> pd.Series:
    """Energy-Charts history for one series (Teil 1's monthly files),
    resampled hourly by time-average -- the same MW-column aggregation rule
    data/normalize.py::to_hourly applies to every other raw MW series in
    this project, so this series enters the model on the identical grid as
    the ENTSO-E-sourced fundamentals it stands in for."""
    files = sorted(root.glob(f"{production_type}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no Energy-Charts history files for {production_type!r} under {root} -- "
            "run scripts/fetch_energy_charts_forecast_history.py first"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    series = frame[production_type]
    return series.resample("h").mean()


def _ec_forecast_feature(name: str, series: pd.Series, target_index: pd.DatetimeIndex) -> Feature:
    """An Energy-Charts value, treated as known at gate closure of its own
    delivery day -- the same knowledge-time rule
    features/availability.py::forecast_for_target applies to a registered
    DA_FORECAST column, reproduced here directly (rather than calling
    forecast_for_target) because doing so would require registering a new
    column in features/availability.py's _RAW_AVAILABILITY, which Auftrag
    section 5 explicitly rules out for this screening experiment."""
    values = pd.Series(series.reindex(target_index).to_numpy(), index=target_index, name=name)
    kt = pd.Series(gate_closure_for_index(target_index), index=target_index)
    return Feature(name, values, kt)


def _renamed(feature: Feature, name: str) -> Feature:
    return replace(feature, name=name, values=feature.values.rename(name))


def _hourly_index_for_local_day(target_day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def build_ec_fundamentals(
    candidate: str,
    target_index: pd.DatetimeIndex,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    ec: dict[str, pd.Series],
) -> list[Feature]:
    """The fundamentals block for one of the three EC candidates -- same
    shape as build_nwp_forecast_fundamentals's own six-feature output
    (load, won, woff, solar, residual, share), varying only which half
    (load vs. renewables) is Energy-Charts-sourced vs. NWP/ENTSO-E-sourced."""
    if candidate == "core_ec_renewables":
        # load stays ENTSO-E (DA_FORECAST); only the renewables half moves to
        # Energy-Charts. Reuses build_nwp_forecast_fundamentals for the load
        # feature and its own coverage check, discarding its NWP renewables.
        nwp = build_nwp_forecast_fundamentals(df, renewables, target_index)
        load = nwp[0]
        won = _ec_forecast_feature("wind_onshore_forecast_ec", ec["wind_onshore"], target_index)
        woff = _ec_forecast_feature("wind_offshore_forecast_ec", ec["wind_offshore"], target_index)
        solar = _ec_forecast_feature("solar_forecast_ec", ec["solar"], target_index)
    elif candidate == "core_ec_load":
        # renewables stay the own NWP reconstruction; only load moves to
        # Energy-Charts. Still needs build_nwp_forecast_fundamentals's own
        # coverage check on the reconstruction artefact (whole-day-out
        # policy, spec 6.5.3) even though its load feature is discarded.
        nwp = build_nwp_forecast_fundamentals(df, renewables, target_index)
        load = _ec_forecast_feature("load_forecast_day_ahead_ec", ec["load"], target_index)
        won, woff, solar = nwp[1], nwp[2], nwp[3]
    elif candidate == "core_ec_full":
        load = _ec_forecast_feature("load_forecast_day_ahead_ec", ec["load"], target_index)
        won = _ec_forecast_feature("wind_onshore_forecast_ec", ec["wind_onshore"], target_index)
        woff = _ec_forecast_feature("wind_offshore_forecast_ec", ec["wind_offshore"], target_index)
        solar = _ec_forecast_feature("solar_forecast_ec", ec["solar"], target_index)
    else:
        raise ValueError(f"unknown EC candidate {candidate!r}")

    residual = _renamed(
        build_residual_load_nwp(load, won, woff, solar), "residual_load_forecast_ec"
    )
    share = _renamed(
        build_renewable_share_nwp(load, won, woff, solar), "renewable_share_forecast_ec"
    )
    return [load, won, woff, solar, residual, share]


def build_ec_candidate_for_day(
    candidate: str,
    target_day: dt.date,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    ec: dict[str, pd.Series],
    cfg: FeatureConfig,
) -> pd.DataFrame:
    target_index = _hourly_index_for_local_day(target_day)
    fundamentals = build_ec_fundamentals(candidate, target_index, df, renewables, ec)
    price = build_price_lags(df, target_index, cfg)
    features = [*build_calendar_features(target_index, cfg), *fundamentals, *price]
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
    ec = {
        production_type: _read_ec_hourly(production_type)
        for production_type in ("load", "solar", "wind_onshore", "wind_offshore")
    }

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

    matrices: dict[str, pd.DataFrame] = {}
    excluded_by_candidate: dict[str, set] = {}
    for candidate in _EC_CANDIDATES:
        log.info("building %s matrix...", candidate)
        matrices[candidate], excluded_by_candidate[candidate] = _build_matrix_over_days(
            all_days_needed,
            lambda d, c=candidate: build_ec_candidate_for_day(c, d, df, renewables, ec, cfg),
        )
    all_excluded: set = set().union(*excluded_by_candidate.values())
    log.info(
        "matrices built: %s (union excluded days=%d)",
        {k: len(v) for k, v in matrices.items()},
        len(all_excluded),
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
        if fold.delivery_day.date() in all_excluded:
            continue
        qh_mask = (qh_index >= day_start) & (qh_index < day_end)
        if qh_mask.sum() != expected_hours * 4:
            continue
        evaluable_folds.append(fold)

    if os.environ.get("ABLATION_SMOKE"):
        evaluable_folds = evaluable_folds[:20]

    n_folds = len(evaluable_folds)
    log.info("evaluable folds: %d (of %d candidates)", n_folds, len(candidate_folds))

    models = {
        name: LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
        for name in _EC_CANDIDATES
    }

    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % _REFIT_EVERY == 0:
            y_train = y_hourly.reindex(fold.train_index)
            for name in _EC_CANDIDATES:
                x_train = matrices[name].reindex(fold.train_index).dropna()
                models[name].fit(y_train.loc[x_train.index], x_train)
            log.info("refit at fold %d/%d (%s)", i + 1, n_folds, fold.delivery_day.date())

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        preds_qh = {}
        for name in _EC_CANDIDATES:
            x_test = matrices[name].loc[fold.test_index]
            hourly_pred = models[name].predict(fold.test_index, history=history, x_test=x_test)
            preds_qh[name] = expand_to_quarterhour(
                hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ
            )

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    **{
                        f"pred_{name}": preds_qh[name].reindex(y_true_day.index).to_numpy()
                        for name in _EC_CANDIDATES
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

    smoke = bool(os.environ.get("ABLATION_SMOKE"))
    ref_core = pd.read_parquet(_CORE_REF_PATH)
    ref_nf = pd.read_parquet(_NO_FUNDAMENTALS_REF_PATH)
    if not smoke:
        if not predictions.index.equals(ref_core.index) or not predictions.index.equals(
            ref_nf.index
        ):
            raise ValueError(
                "fold index mismatch against the reference ablations -- config drifted; "
                "the Auftrag requires an identical fold set (section 6 checklist), not a "
                "reproduction of it"
            )
        max_price_diff = (predictions["y_true"] - ref_core["y_true"]).abs().max()
        if max_price_diff > 1e-9:
            raise ValueError(
                f"y_true mismatch against reference ablation (max diff {max_price_diff})"
            )

    reference_cols = {
        "baseline": ref_core["pred_baseline"],
        "core_no_commodities": ref_core["pred_core_no_commodities"],
        "no_fundamentals": ref_nf["pred_no_fundamentals"],
    }
    if smoke:
        # Reference frames weren't built on the smoke subset -- align on the
        # intersection only, purely so the script exercises its own logic
        # end to end; not a real measurement (section 6 checklist requires
        # the full, non-smoke run for the real comparison).
        common = predictions.index.intersection(ref_core.index).intersection(ref_nf.index)
        predictions = predictions.loc[common]
        reference_cols = {k: v.loc[common] for k, v in reference_cols.items()}
    for name, series in reference_cols.items():
        predictions[f"pred_{name}"] = series.reindex(predictions.index).to_numpy()

    all_candidates = (*_EC_CANDIDATES, "core_no_commodities", "no_fundamentals", "baseline")
    print(
        f"\nn_folds={len(predictions['delivery_day'].unique())}  refit_every={_REFIT_EVERY}  "
        f"train_span_days={_TRAIN_SPAN_DAYS}  shape_window_days={_SHAPE_WINDOW_DAYS}"
    )
    print(f"excluded_days={len(all_excluded)}\n")

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

    print(
        "\n--- Teil 3: vs. the two already-measured reference points "
        "(NOT the baseline, Auftrag section 4) ---\n"
    )
    log_metrics: dict[str, float] = {}
    for name in all_candidates:
        log_metrics[f"{name}_mae"] = metrics[name]["mae"]
        log_metrics[f"{name}_rmse"] = metrics[name]["rmse"]

    for candidate in _EC_CANDIDATES:
        print(f"{candidate}:")
        for reference in ("baseline", "core_no_commodities", "no_fundamentals"):
            native, daily = _run_dm(candidate, reference)
            log_metrics[f"{candidate}_vs_{reference}_native_loss_diff"] = native.mean_loss_diff
            log_metrics[f"{candidate}_vs_{reference}_native_p"] = native.p_value
            log_metrics[f"{candidate}_vs_{reference}_daily_loss_diff"] = daily.mean_loss_diff
            log_metrics[f"{candidate}_vs_{reference}_daily_p"] = daily.p_value
            better = "better" if native.mean_loss_diff < 0 else "worse"
            sig = "significant" if native.p_value < _GATE_P_THRESHOLD else "not significant"
            print(
                f"  vs {reference:20s} RMSE(ref)={metrics[reference]['rmse']:.2f}  "
                f"native mean_loss_diff_sq={native.mean_loss_diff:+.4f} p={native.p_value:.4f} "
                f"({better}, {sig})"
            )
        print()

    report_rows: list[dict[str, object]] = []
    for name in all_candidates:
        report_rows.append(
            {
                "row_type": "metric",
                "candidate": name,
                "reference": "",
                "convention": "",
                "rmse": metrics[name]["rmse"],
                "mae": metrics[name]["mae"],
                "mean_loss_diff_sq": "",
                "p_value": "",
                "n_obs": "",
                "hac_lag": "",
                "horizon": "",
            }
        )
    for candidate in _EC_CANDIDATES:
        for reference in ("baseline", "core_no_commodities", "no_fundamentals"):
            native, daily = _run_dm(candidate, reference)
            for convention, result, hac_lag, horizon in (
                ("native", native, _QH_HAC_LAG, _QH_HORIZON),
                ("daily", daily, _DAILY_HAC_LAG, _DAILY_HORIZON),
            ):
                report_rows.append(
                    {
                        "row_type": "dm_test",
                        "candidate": candidate,
                        "reference": reference,
                        "convention": convention,
                        "rmse": "",
                        "mae": "",
                        "mean_loss_diff_sq": result.mean_loss_diff,
                        "p_value": result.p_value,
                        "n_obs": result.n_obs,
                        "hac_lag": hac_lag,
                        "horizon": horizon,
                    }
                )
    report = pd.DataFrame(report_rows)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _REPORT_PATH.open("w", newline="\n") as f:
        f.write(_UPPER_BOUND_CAVEAT + "\n")
        report.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", _REPORT_PATH, len(report))

    elapsed = time.monotonic() - t0
    print(f"elapsed: {elapsed:.0f}s")
    print(f"predictions written to {_OUT_PATH}")
    print(f"report written to {_REPORT_PATH}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ablation_energy_charts_residual_load"):
        mlflow.log_params(
            {
                "n_folds": len(predictions["delivery_day"].unique()),
                "refit_every": _REFIT_EVERY,
                "train_span_days": _TRAIN_SPAN_DAYS,
                "shape_window_days": _SHAPE_WINDOW_DAYS,
                "excluded_days": len(all_excluded),
                "objective": "quantile",
                "alpha": 0.5,
                "random_state": _RANDOM_STATE,
                **_DEFAULT_PARAMS,
            }
        )
        mlflow.set_tags(
            {
                "purpose": "Energy-Charts backup Auftrag Teil 3 -- residual_load_forecast_ec "
                "candidates vs. core_no_commodities (upper bound) / no_fundamentals (floor)",
                "candidates": ",".join(_EC_CANDIDATES),
                "not_a_spec_step": "true",
                "upper_bound_caveat": "Energy-Charts pre-gate-closure availability is unmeasured",
            }
        )
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(_OUT_PATH))
    print(f"\nlogged to mlflow experiment {_MLFLOW_EXPERIMENT!r}")


if __name__ == "__main__":
    main()
