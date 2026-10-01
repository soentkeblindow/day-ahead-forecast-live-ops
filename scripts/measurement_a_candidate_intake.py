"""Sprint 6.8, Messung A (docs/sprint6_step6_8_spec.md section 4): does any
future fallback candidate row pass the go-live criterion (Entscheidung 24) in
PRODUCTION configuration (refit_every=1, the config candidates actually run
under live -- the prior sounding numbers 28.90/31.01/42.28 used refit_every=14
and, per the spec's own words, "berechtigen keine Zeile zum Eintritt")?

Six candidates, one shared window and fold list (section 2 -- identical
across every 6.8 measurement, enforced by an index-equality check here, not
just asserted in prose):

  live                 -- production feature set (features.build.build_feature_set_for_day)
  core_no_commodities  -- calendar + NWP residual-load fundamentals + price lags
  core_gas_only        -- as above, + TTF gas lag
  no_fundamentals      -- live minus the residual-load-forecast track
  floor_core           -- calendar + price lags only
  floor_core_gas       -- as above, + TTF gas lag

`core_no_commodities`/`core_gas_only` reuse scripts/ablation_core_minimal_
feature_set.py::build_core_for_day unchanged; `no_fundamentals` reuses
scripts/ablation_no_fundamentals.py::build_no_fundamentals_for_day unchanged
(both imported, not copied -- "keine zweite Walk-Forward-Implementierung").
`floor_core`/`floor_core_gas` are new here: the same calendar+price-lags
core, minus the NWP fundamentals block entirely (the "boden, der auch ohne
Ist-Werte trägt" role the spec asks for).

Window: fixed start 2025-10-29 (spec's own literal date, not derived), end =
the last local day with complete data across the frozen interim/renewables
artifacts at the time this was run (see log for the resolved date -- the
frozen backtest artifacts this script reads, per CLAUDE.md, are NOT the live
store and stop short of "today").

refit_every=1 throughout (the live production cadence) -- unlike the three
prior one-off ablation_*.py scripts in this repo, all of which used 14
purely for runtime and are explicitly disqualified by the spec.

Usage:
  python scripts/measurement_a_candidate_intake.py --probe-folds 10   # timing probe only, no MLflow log
  python scripts/measurement_a_candidate_intake.py                    # full run
  python scripts/measurement_a_candidate_intake.py --check-determinism  # section 2's once-only determinism check
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from collections.abc import Callable
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
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster
from energy_price_forecast.ops.windows import local_day_bounds
from scripts.ablation_core_minimal_feature_set import build_core_for_day
from scripts.ablation_no_fundamentals import build_no_fundamentals_for_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("measurement_a")

_TZ = "Europe/Berlin"
_PRICE_COL = "day_ahead_price"
_TRAIN_SPAN_DAYS = 90
_REFIT_EVERY = (
    1  # section 2: the production cadence, unlike the three prior refit_every=14 ablations
)
_RANDOM_STATE = 0
_SHAPE_WINDOW_DAYS = 28
_QH_HAC_LAG = 192
_QH_HORIZON = 96
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_GATE_P_THRESHOLD = 0.10
_WINDOW_START = dt.date(2025, 10, 29)  # spec section 2: fixed, not derived
_OUT_DIR = Path("data/processed")
_MLFLOW_EXPERIMENT = "live_robustness_6_8"

_FALLBACK_CANDIDATES = (
    "core_no_commodities",
    "core_gas_only",
    "no_fundamentals",
    "floor_core",
    "floor_core_gas",
)
_ALL_CANDIDATES = ("live", *_FALLBACK_CANDIDATES)
_SUMMARY_PATH = Path("outputs/results/measurement_a_candidate_intake_summary.csv")


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


def build_floor_for_day(
    target_day: dt.date, df: pd.DataFrame, cfg: FeatureConfig, *, commodities: str
) -> pd.DataFrame:
    """calendar + price lags only, plus commodities per `commodities`
    ("none" / "gas_only") -- the "boden, der auch ohne Ist-Werte trägt" role
    (spec section 4): no NWP residual-load fundamentals at all, unlike
    build_core_for_day."""
    target_index = _hourly_index_for_local_day(target_day)
    price = build_price_lags(df, target_index, cfg)

    commodity_feats: list[Feature]
    if commodities == "gas_only":
        commodity_feats = build_commodity_features(df, target_index, cfg)[:1]  # ttf_gas only
    elif commodities == "none":
        commodity_feats = []
    else:
        raise ValueError(f"unknown commodities mode {commodities!r}")

    features = [*build_calendar_features(target_index, cfg), *commodity_feats, *price]
    return build_matrix(features)


def _local_days(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.DatetimeIndex(index).tz_convert(tz).normalize().unique()
    ).sort_values()


def _build_matrix_over_days(
    days: pd.DatetimeIndex, builder: Callable[[dt.date], pd.DataFrame]
) -> tuple[pd.DataFrame, set[dt.date]]:
    frames = []
    excluded: set[dt.date] = set()
    for day in days:
        try:
            frames.append(builder(day.date()))
        except IncompleteReconstructionError:
            excluded.add(day.date())
    if not frames:
        raise ValueError("no usable day")
    return pd.concat(frames).sort_index(), excluded


def build_candidate_matrices(
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    cfg: FeatureConfig,
    all_days_needed: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], set[dt.date]]:
    log.info("building live matrix (NWP reconstruction)...")
    live_matrix, live_excluded = _build_matrix_over_days(
        all_days_needed, lambda d: build_feature_set_for_day(d, df, renewables)
    )
    log.info("building core_no_commodities matrix...")
    core_none, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_core_for_day(d, df, renewables, cfg, commodities="none")
    )
    log.info("building core_gas_only matrix...")
    core_gas, _ = _build_matrix_over_days(
        all_days_needed,
        lambda d: build_core_for_day(d, df, renewables, cfg, commodities="gas_only"),
    )
    log.info("building no_fundamentals matrix...")
    no_fund, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_no_fundamentals_for_day(d, df, cfg)
    )
    log.info("building floor_core matrix...")
    floor_none, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_floor_for_day(d, df, cfg, commodities="none")
    )
    log.info("building floor_core_gas matrix...")
    floor_gas, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_floor_for_day(d, df, cfg, commodities="gas_only")
    )
    matrices = {
        "live": live_matrix,
        "core_no_commodities": core_none,
        "core_gas_only": core_gas,
        "no_fundamentals": no_fund,
        "floor_core": floor_none,
        "floor_core_gas": floor_gas,
    }
    # Section 2: identical fold list across all candidates -- enforced, not claimed.
    # floor_core/no_fundamentals/etc. never raise IncompleteReconstructionError
    # themselves (they don't touch the NWP block), so `live_excluded` is the
    # only exclusion source; applying it uniformly (done in the caller's
    # evaluable_folds filter) is what makes the six index sets comparable.
    return matrices, live_excluded


def build_evaluable_folds(
    candidate_folds: list, live_excluded: set[dt.date], prices_qh: pd.Series
) -> list:
    qh_index = pd.DatetimeIndex(prices_qh.index)
    evaluable = []
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
        evaluable.append(fold)
    return evaluable


def _assert_identical_index(matrices: dict[str, pd.DataFrame], evaluable_folds: list) -> None:
    """Section 2's own requirement: identical fold list across all
    candidates, enforced by an actual index-equality check, not prose."""
    test_index = pd.DatetimeIndex(
        sorted({ts for fold in evaluable_folds for ts in fold.test_index})
    )
    for name, matrix in matrices.items():
        available = pd.DatetimeIndex(matrix.index).intersection(test_index)
        if not available.equals(test_index):
            missing = test_index.difference(available)
            raise AssertionError(
                f"{name}: {len(missing)} test timestamp(s) missing from its own matrix -- "
                f"fold lists are not identical across candidates (first missing: {missing.min()})"
            )


def run_folds(
    evaluable_folds: list,
    matrices: dict[str, pd.DataFrame],
    y_hourly: pd.Series,
    prices_qh: pd.Series,
    *,
    refit_every: int,
    random_state: int,
) -> pd.DataFrame:
    qh_index = pd.DatetimeIndex(prices_qh.index)
    models = {
        name: LGBMForecaster(objective="quantile", alpha=0.5, random_state=random_state, n_jobs=1)
        for name in matrices
    }
    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % refit_every == 0:
            y_train = y_hourly.reindex(fold.train_index)
            for name, matrix in matrices.items():
                x_train = matrix.reindex(fold.train_index).dropna()
                models[name].fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        preds_qh = {}
        for name, matrix in matrices.items():
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
                        for name in (*matrices, "baseline")
                    },
                    "delivery_day": fold.delivery_day,
                },
                index=y_true_day.index,
            )
        )
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records).sort_index()


def _run_dm(predictions: pd.DataFrame, candidate: str, reference: str) -> tuple[DMResult, DMResult]:
    loss_c = (predictions[f"pred_{candidate}"] - predictions["y_true"]) ** 2
    loss_r = (predictions[f"pred_{reference}"] - predictions["y_true"]) ** 2
    native = dm_test(loss_c, loss_r, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
    daily_c = loss_c.groupby(predictions["delivery_day"]).mean()
    daily_r = loss_r.groupby(predictions["delivery_day"]).mean()
    daily = dm_test(daily_c, daily_r, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    return native, daily


def _prepare(
    cfg: FeatureConfig,
) -> tuple[pd.Series, dict[str, pd.DataFrame], pd.Series, list, dt.date]:
    df = load_interim_hourly()
    renewables = load_renewables_predictions()
    prices_qh = load_interim_quarterhourly()[_PRICE_COL]

    hourly_index = pd.DatetimeIndex(df.index)
    renewables_valid_time = pd.DatetimeIndex(renewables.index.get_level_values("valid_time_utc"))
    renewables_days = _local_days(renewables_valid_time, _TZ)
    qh_local_days = _local_days(pd.DatetimeIndex(prices_qh.index), _TZ)
    end_day_ts = min(qh_local_days[-1], renewables_days[-1])
    end_day = end_day_ts.date()

    candidate_folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=_WINDOW_START.isoformat(),
            test_end=end_day.isoformat(),
            window="rolling",
            train_span_days=_TRAIN_SPAN_DAYS,
        )
    )
    if not candidate_folds:
        raise ValueError("no candidate folds -- check data coverage")

    first_train_day = candidate_folds[0].train_index.tz_convert(_TZ).normalize().min()
    all_days_needed = pd.date_range(first_train_day, end_day_ts, freq="D", tz=_TZ)

    matrices, live_excluded = build_candidate_matrices(df, renewables, cfg, all_days_needed)
    evaluable_folds = build_evaluable_folds(candidate_folds, live_excluded, prices_qh)
    _assert_identical_index(matrices, evaluable_folds)

    log.info(
        "window: %s..%s, %d evaluable folds (of %d candidate folds), %d days excluded (NWP)",
        _WINDOW_START,
        end_day,
        len(evaluable_folds),
        len(candidate_folds),
        len(live_excluded),
    )
    return df[_PRICE_COL], matrices, prices_qh, evaluable_folds, end_day


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probe-folds",
        type=int,
        default=None,
        help="run only the first N folds, print an extrapolated total runtime, and exit "
        "without logging to MLflow (spec section 2: measure before committing to the full run)",
    )
    p.add_argument(
        "--check-determinism",
        action="store_true",
        help="section 2's once-only determinism check: run floor_core (cheapest candidate) "
        "twice with different random_state over a small fold subset and compare predictions",
    )
    p.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        help="skip the fit/predict walk-forward entirely and recompute metrics plus "
        "outputs/results/measurement_a_candidate_intake_summary.csv from an existing "
        "predictions parquet (e.g. the one this script's own full run already wrote) -- "
        "for regenerating the committed summary without redoing an expensive refit",
    )
    return p.parse_args()


def _determinism_check(y_hourly: pd.Series, matrix: pd.DataFrame, folds: list) -> None:
    preds = {}
    for seed in (0, 1):
        out = run_folds(
            folds[:10],
            {"floor_core": matrix},
            y_hourly,
            load_interim_quarterhourly()[_PRICE_COL],
            refit_every=1,
            random_state=seed,
        )
        preds[seed] = out["pred_floor_core"]
    identical = preds[0].equals(preds[1])
    max_abs_diff = (preds[0] - preds[1]).abs().max()
    log.info(
        "determinism check (floor_core, seeds 0 vs 1, 10 folds): identical=%s max_abs_diff=%.6g",
        identical,
        max_abs_diff,
    )
    if not identical and max_abs_diff > 1e-6:
        log.warning(
            "predictions differ by more than a plausible threading/rounding effect -- "
            "flag the seed question to the owner per section 2, do not silently proceed"
        )


def _compute_metrics_and_gates(
    predictions: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, GateResult]]:
    metrics: dict[str, dict[str, float]] = {}
    for name in (*_ALL_CANDIDATES, "baseline"):
        t, p = predictions["y_true"], predictions[f"pred_{name}"]
        metrics[name] = {"mae": mae(t, p), "rmse": rmse(t, p)}
        print(f"{name:22s}  MAE={metrics[name]['mae']:.4f}  RMSE={metrics[name]['rmse']:.4f}")

    print("\n--- gate check vs baseline (Entscheidung 24) ---\n")
    gate_results: dict[str, GateResult] = {}
    for name in _ALL_CANDIDATES:
        native_vs_base, daily_vs_base = _run_dm(predictions, name, "baseline")
        rmse_ok = metrics[name]["rmse"] < metrics["baseline"]["rmse"]
        dm_ok = native_vs_base.mean_loss_diff < 0 and native_vs_base.p_value < _GATE_P_THRESHOLD
        gate_pass = rmse_ok and dm_ok
        gate_results[name] = GateResult(
            native_vs_baseline=native_vs_base,
            daily_vs_baseline=daily_vs_base,
            gate_pass=gate_pass,
        )
        verdict = "PASS" if gate_pass else "FAIL"
        print(
            f"{name:22s} native p={native_vs_base.p_value:.4f} loss_diff={native_vs_base.mean_loss_diff:+.4f} "
            f"| daily p={daily_vs_base.p_value:.4f} loss_diff={daily_vs_base.mean_loss_diff:+.4f} -> {verdict}"
        )

    print("\n--- informative: vs live ---\n")
    for name in _FALLBACK_CANDIDATES:
        native_vs_live, daily_vs_live = _run_dm(predictions, name, "live")
        gate_results[name].native_vs_live = native_vs_live
        gate_results[name].daily_vs_live = daily_vs_live
        print(
            f"{name:22s} native p={native_vs_live.p_value:.4f} loss_diff={native_vs_live.mean_loss_diff:+.4f} "
            f"(negative = better than live)"
        )

    print(
        "\nNOTE (spec section 4 Pflichthinweis): the shared window contains no gas crisis "
        "(TTF CV in-window vs. full history is materially lower) -- an advantage for the "
        "no-gas variants here is not evidence against gas in a crisis scenario. The choice "
        "between core_*/floor_core* with vs. without gas is an owner decision, not settled here."
    )
    return metrics, gate_results


def _write_summary_csv(
    metrics: dict[str, dict[str, float]], gate_results: dict[str, GateResult]
) -> None:
    """Committed counterpart to the mlflow metrics below (outputs/results/,
    not the gitignored mlruns/) -- the only place a README number citing this
    measurement can actually be traced back to, per publication_spec.md
    section 4's "every README number needs a repo source" rule."""
    rows = []
    for name in _ALL_CANDIDATES:
        gr = gate_results[name]
        row = {
            "candidate": name,
            "rmse": metrics[name]["rmse"],
            "mae": metrics[name]["mae"],
            "rmse_baseline": metrics["baseline"]["rmse"],
            "vs_baseline_native_loss_diff": gr.native_vs_baseline.mean_loss_diff,
            "vs_baseline_native_p": gr.native_vs_baseline.p_value,
            "vs_baseline_daily_loss_diff": gr.daily_vs_baseline.mean_loss_diff,
            "vs_baseline_daily_p": gr.daily_vs_baseline.p_value,
            "gate_pass": gr.gate_pass,
            "vs_live_native_loss_diff": (
                gr.native_vs_live.mean_loss_diff if gr.native_vs_live is not None else None
            ),
            "vs_live_native_p": gr.native_vs_live.p_value
            if gr.native_vs_live is not None
            else None,
            "vs_live_daily_loss_diff": (
                gr.daily_vs_live.mean_loss_diff if gr.daily_vs_live is not None else None
            ),
            "vs_live_daily_p": gr.daily_vs_live.p_value if gr.daily_vs_live is not None else None,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SUMMARY_PATH.open("w", newline="\n") as f:
        f.write(
            "# refit_every=1 (production cadence), train_span_days=90, quantile/alpha=0.5, "
            "shape_window_days=28 -- see this script's own module docstring for the six "
            "candidates' feature sets.\n"
        )
        summary.to_csv(f, index=False, lineterminator="\n")
    print(f"\nwrote {_SUMMARY_PATH}")


def main() -> None:
    t0 = time.monotonic()
    args = _parse_args()

    if args.from_predictions is not None:
        print(f"\nrecomputing metrics from {args.from_predictions} (no refit)")
        predictions = pd.read_parquet(args.from_predictions)
        metrics, gate_results = _compute_metrics_and_gates(predictions)
        _write_summary_csv(metrics, gate_results)
        return

    cfg = FeatureConfig()

    y_hourly, matrices, prices_qh, evaluable_folds, end_day = _prepare(cfg)

    if args.check_determinism:
        _determinism_check(y_hourly, matrices["floor_core"], evaluable_folds)
        return

    folds_to_run = evaluable_folds[: args.probe_folds] if args.probe_folds else evaluable_folds
    n_folds = len(folds_to_run)

    run_t0 = time.monotonic()
    predictions = run_folds(
        folds_to_run,
        matrices,
        y_hourly,
        prices_qh,
        refit_every=_REFIT_EVERY,
        random_state=_RANDOM_STATE,
    )
    run_elapsed = time.monotonic() - run_t0

    if args.probe_folds:
        per_fold = run_elapsed / n_folds
        estimated_total = per_fold * len(evaluable_folds)
        log.info(
            "PROBE: %d folds took %.1fs (%.2fs/fold) -- extrapolated total for all %d folds: "
            "%.0fs (%.1f min, %.2f h)",
            n_folds,
            run_elapsed,
            per_fold,
            len(evaluable_folds),
            estimated_total,
            estimated_total / 60,
            estimated_total / 3600,
        )
        return

    out_path = _OUT_DIR / "measurement_a_candidate_intake_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_path)

    print(
        f"\nn_folds={n_folds}  refit_every={_REFIT_EVERY}  train_span_days={_TRAIN_SPAN_DAYS}  window_end={end_day}"
    )

    metrics, gate_results = _compute_metrics_and_gates(predictions)
    _write_summary_csv(metrics, gate_results)

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="measurement_a_candidate_intake"):
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "refit_every": _REFIT_EVERY,
                "train_span_days": _TRAIN_SPAN_DAYS,
                "shape_window_days": _SHAPE_WINDOW_DAYS,
                "window_start": _WINDOW_START.isoformat(),
                "window_end": end_day.isoformat(),
                "objective": "quantile",
                "alpha": 0.5,
                "random_state": _RANDOM_STATE,
                **_DEFAULT_PARAMS,
            }
        )
        mlflow.set_tags(
            {
                "spec": "sprint6_step6_8",
                "measurement": "A",
                "candidates": ",".join(_ALL_CANDIDATES),
            }
        )
        log_metrics: dict[str, float] = {}
        for name in (*_ALL_CANDIDATES, "baseline"):
            log_metrics[f"{name}_mae"] = metrics[name]["mae"]
            log_metrics[f"{name}_rmse"] = metrics[name]["rmse"]
        for name in _ALL_CANDIDATES:
            gr = gate_results[name]
            log_metrics[f"{name}_vs_baseline_native_loss_diff"] = (
                gr.native_vs_baseline.mean_loss_diff
            )
            log_metrics[f"{name}_vs_baseline_native_p"] = gr.native_vs_baseline.p_value
            log_metrics[f"{name}_vs_baseline_daily_loss_diff"] = gr.daily_vs_baseline.mean_loss_diff
            log_metrics[f"{name}_vs_baseline_daily_p"] = gr.daily_vs_baseline.p_value
            mlflow.set_tag(f"{name}_gate_verdict", "PASS" if gr.gate_pass else "FAIL")
            if name in _FALLBACK_CANDIDATES:
                assert gr.native_vs_live is not None and gr.daily_vs_live is not None
                log_metrics[f"{name}_vs_live_native_loss_diff"] = gr.native_vs_live.mean_loss_diff
                log_metrics[f"{name}_vs_live_native_p"] = gr.native_vs_live.p_value
                log_metrics[f"{name}_vs_live_daily_loss_diff"] = gr.daily_vs_live.mean_loss_diff
                log_metrics[f"{name}_vs_live_daily_p"] = gr.daily_vs_live.p_value
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(out_path))
    print(f"\nlogged to mlflow experiment {_MLFLOW_EXPERIMENT!r}")


if __name__ == "__main__":
    main()
