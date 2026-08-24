"""Sprint 6 / Step 6.4: run the quarter-hourly bridge-vs-baseline backtest.

Thin CLI/MLflow wrapper around evaluation.arena_walkforward.run_arena_backtest
(spec 6.4 section 5.4/section 6). Fixed, production model configuration
(LightGBM, objective="quantile", alpha=0.5, untuned, full feature set) --
this step measures the bridge architecture itself, not model tuning, so
those knobs are not exposed on the CLI.

UPPER BOUND: this run uses the full engineered feature set, including
wind/solar/residual-load FORECAST columns that the 6.2 availability audit
found are NOT available at gate closure. Every metric this script logs or
persists is therefore a best case, not a live-capability claim -- see the
sprint 6 abstract, section 9, and step 6.6 for the real gate.
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
)
from energy_price_forecast.evaluation.arena_walkforward import run_arena_backtest
from energy_price_forecast.evaluation.metrics import summarise
from energy_price_forecast.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster

_PRICE_COL = "day_ahead_price"
_CANDIDATES = ("bridge_shape", "bridge_flat", "baseline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quarter-hourly Energy-Arena architecture gate: bridge vs baseline "
        "(spec 6.4 section 6). UPPER BOUND run -- full feature set, not a live-capability claim."
    )
    p.add_argument(
        "--shape-window-days",
        type=int,
        required=True,
        help="Shape profile window N (spec 2.3: 28 is the preregistered headline; "
        "7/14/56 are sensitivity-only, never the headline).",
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
    p.add_argument("--out", default=Path("data/processed/preds_arena.parquet"), type=Path)
    return p.parse_args()


def _candidate_summary(predictions: pd.DataFrame, candidate: str) -> dict[str, float]:
    frame = predictions[["y_true", "delivery_day"]].assign(y_pred=predictions[f"pred_{candidate}"])
    return summarise(frame)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    log.warning(
        "UPPER BOUND run: full feature set, including forecast columns not available at "
        "gate closure (spec 6.4 section 4 rule 1; sprint 6 abstract section 9)."
    )
    t0 = time.monotonic()
    args = _parse_args()

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
    x_hourly = features
    prices_qh = quarterhourly[_PRICE_COL]

    model = LGBMForecaster(
        objective="quantile",
        alpha=0.5,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    predictions = run_arena_backtest(
        y_hourly,
        x_hourly,
        prices_qh,
        model,
        shape_window_days=args.shape_window_days,
        train_span_days=args.train_span_days,
        refit_every=args.refit_every,
    )

    summaries = {candidate: _candidate_summary(predictions, candidate) for candidate in _CANDIDATES}

    log_params: dict[str, object] = {
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
        "claim": "upper_bound",
        "n_folds": str(predictions["delivery_day"].nunique()),
    }
    log_metrics: dict[str, float] = {
        f"{candidate}_{metric}": value
        for candidate, metric_dict in summaries.items()
        for metric, value in metric_dict.items()
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="bridge_arena_backtest"):
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics(log_metrics)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.out)
        mlflow.log_artifact(str(args.out))

    elapsed = time.monotonic() - t0
    log.info(
        "arena backtest finished in %.0fs (%d folds, %d rows)",
        elapsed,
        predictions["delivery_day"].nunique(),
        len(predictions),
    )
    print("Summary (UPPER BOUND -- full feature set, not a live-capability claim):")
    for candidate, metric_dict in summaries.items():
        print(f"  {candidate}:")
        for k, v in metric_dict.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
