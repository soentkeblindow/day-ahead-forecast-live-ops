"""Sprint 6 / Step 6.4 + 6.6: run the Energy-Arena backtests.

--candidates bridge (default, unchanged from 6.4): thin CLI/MLflow wrapper
around evaluation.arena_walkforward.run_arena_backtest (spec 6.4 section
5.4/section 6). Fixed, production model configuration (LightGBM,
objective="quantile", alpha=0.5, untuned, full feature set) -- this mode
measures the bridge architecture itself, not model tuning, so those knobs
are not exposed on the CLI. UPPER BOUND: this run uses the full engineered
feature set, including wind/solar/residual-load FORECAST columns that the
6.2 availability audit found are NOT available at gate closure. Every
metric this script logs or persists is therefore a best case, not a
live-capability claim -- see the sprint 6 abstract, section 9, and step 6.6
for the real gate.

--candidates live-gate (spec 6.6 section 6): runs
evaluation.arena_walkforward.run_live_gate_backtest instead -- two
independently trained hourly models (one per feature set, built per day
via features.build.build_feature_set_for_day /
build_original_feature_set_for_day, not the precomputed features.parquet,
spec 6.6 section 5.1), --resolution quarterhourly (the binding run) or
hourly (secondary, not binding).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import (
    load_interim_hourly,
    load_interim_quarterhourly,
    load_processed_features,
    load_renewables_predictions,
)
from energy_price_forecast.evaluation.arena_walkforward import (
    run_arena_backtest,
    run_live_gate_backtest,
)
from energy_price_forecast.evaluation.metrics import summarise
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster

_PRICE_COL = "day_ahead_price"
_BRIDGE_CANDIDATES = ("bridge_shape", "bridge_flat", "baseline")
_LIVE_GATE_CANDIDATES = ("original", "live", "baseline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Energy-Arena backtests: the 6.4 bridge-vs-baseline architecture gate, "
        "or the 6.6 live-gate (live vs. original vs. baseline)."
    )
    p.add_argument(
        "--candidates",
        choices=("bridge", "live-gate"),
        default="bridge",
        help="'bridge' (default, spec 6.4, UPPER BOUND) or 'live-gate' (spec 6.6).",
    )
    p.add_argument(
        "--resolution",
        choices=("quarterhourly", "hourly"),
        default="quarterhourly",
        help="--candidates live-gate only: quarterhourly is the binding run (spec 6.6 "
        "section 2.2), hourly is secondary/non-binding (spec 6.6 section 3.3).",
    )
    p.add_argument(
        "--shape-window-days",
        type=int,
        default=None,
        help="Shape profile window N (spec 2.3: 28 is the preregistered headline; "
        "7/14/56 are sensitivity-only, never the headline). Required unless "
        "--candidates live-gate --resolution hourly.",
    )
    p.add_argument("--train-span-days", type=int, default=90)
    p.add_argument("--refit-every", type=int, default=1)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--experiment", default="arena_models")
    p.add_argument("--study", default="bridge_gate")
    p.add_argument("--note", default="")
    p.add_argument("--hourly-path", default=None, type=Path)
    p.add_argument("--features-path", default=None, type=Path)
    p.add_argument("--quarterhourly-path", default=None, type=Path)
    p.add_argument("--renewables-path", default=None, type=Path)
    p.add_argument("--out", default=None, type=Path)
    args = p.parse_args()
    if args.candidates == "bridge" and args.shape_window_days is None:
        p.error("--shape-window-days is required for --candidates bridge")
    if (
        args.candidates == "live-gate"
        and args.resolution == "quarterhourly"
        and args.shape_window_days is None
    ):
        p.error(
            "--shape-window-days is required for --candidates live-gate --resolution quarterhourly"
        )
    if args.out is None:
        default_name = (
            "preds_arena.parquet"
            if args.candidates == "bridge"
            else (
                "preds_arena_live.parquet"
                if args.resolution == "quarterhourly"
                else "preds_arena_live_hourly.parquet"
            )
        )
        args.out = Path("data/processed") / default_name
    return args


def _candidate_summary(predictions: pd.DataFrame, candidate: str) -> dict[str, float]:
    frame = predictions[["y_true", "delivery_day"]].assign(y_pred=predictions[f"pred_{candidate}"])
    return summarise(frame)


def _run_bridge(args: argparse.Namespace, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    log.warning(
        "UPPER BOUND run: full feature set, including forecast columns not available at "
        "gate closure (spec 6.4 section 4 rule 1; sprint 6 abstract section 9)."
    )
    hourly = load_interim_hourly(args.hourly_path) if args.hourly_path else load_interim_hourly()
    features = (
        load_processed_features(args.features_path)
        if args.features_path
        else load_processed_features()
    )
    quarterhourly = (
        load_interim_quarterhourly(args.quarterhourly_path)
        if args.quarterhourly_path
        else load_interim_quarterhourly()
    )

    y_hourly = hourly[_PRICE_COL].reindex(features.index)
    model = LGBMForecaster(
        objective="quantile", alpha=0.5, random_state=args.random_state, n_jobs=args.n_jobs
    )

    predictions = run_arena_backtest(
        y_hourly,
        features,
        quarterhourly[_PRICE_COL],
        model,
        shape_window_days=args.shape_window_days,
        train_span_days=args.train_span_days,
        refit_every=args.refit_every,
    )
    extra_tags = {"claim": "upper_bound"}
    return predictions, extra_tags


def _run_live_gate(args: argparse.Namespace, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    hourly = load_interim_hourly(args.hourly_path) if args.hourly_path else load_interim_hourly()
    renewables = (
        load_renewables_predictions(args.renewables_path)
        if args.renewables_path
        else load_renewables_predictions()
    )
    prices_qh = None
    if args.resolution == "quarterhourly":
        quarterhourly = (
            load_interim_quarterhourly(args.quarterhourly_path)
            if args.quarterhourly_path
            else load_interim_quarterhourly()
        )
        prices_qh = quarterhourly[_PRICE_COL]

    model_live = LGBMForecaster(
        objective="quantile", alpha=0.5, random_state=args.random_state, n_jobs=args.n_jobs
    )
    model_original = LGBMForecaster(
        objective="quantile", alpha=0.5, random_state=args.random_state, n_jobs=args.n_jobs
    )

    predictions = run_live_gate_backtest(
        hourly,
        renewables,
        model_live,
        model_original,
        resolution=args.resolution,
        prices_qh=prices_qh,
        shape_window_days=args.shape_window_days,
        train_span_days=args.train_span_days,
        refit_every=args.refit_every,
    )
    extra_tags = {"claim": "live_gate", "resolution": args.resolution}
    return predictions, extra_tags


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    t0 = time.monotonic()
    args = _parse_args()

    if args.candidates == "bridge":
        predictions, extra_tags = _run_bridge(args, log)
        candidates = _BRIDGE_CANDIDATES
    else:
        predictions, extra_tags = _run_live_gate(args, log)
        candidates = _LIVE_GATE_CANDIDATES

    summaries = {candidate: _candidate_summary(predictions, candidate) for candidate in candidates}

    log_params: dict[str, object] = {
        "candidates": args.candidates,
        "resolution": args.resolution,
        "shape_window_days": args.shape_window_days,
        "train_span_days": args.train_span_days,
        "refit_every": args.refit_every,
        "objective": "quantile",
        "alpha": 0.5,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        **_DEFAULT_PARAMS,
    }
    log_tags: dict[str, str] = {
        "study": args.study,
        "note": args.note,
        "n_folds": str(predictions["delivery_day"].nunique()),
        **extra_tags,
    }
    log_metrics: dict[str, float] = {
        f"{candidate}_{metric}": value
        for candidate, metric_dict in summaries.items()
        for metric, value in metric_dict.items()
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    run_name = "bridge_arena_backtest" if args.candidates == "bridge" else "live_gate_backtest"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics(log_metrics)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.out)
        mlflow.log_artifact(str(args.out))

    elapsed = time.monotonic() - t0
    log.info(
        "arena backtest (%s) finished in %.0fs (%d folds, %d rows)",
        args.candidates,
        elapsed,
        predictions["delivery_day"].nunique(),
        len(predictions),
    )
    claim = (
        "UPPER BOUND -- full feature set, not a live-capability claim"
        if args.candidates == "bridge"
        else "live gate"
    )
    print(f"Summary ({claim}):")
    for candidate, metric_dict in summaries.items():
        print(f"  {candidate}:")
        for k, v in metric_dict.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
