"""Tune LightGBM hyperparameters via Optuna inner walk-forward; persist frozen params."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm
import mlflow
import optuna
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.evaluation.config import SPRINT3_EXPERIMENT_NAME
from energy_price_forecast.models.tuning import tune_lgbm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune LightGBM hyperparameters via Optuna inner walk-forward."
    )
    p.add_argument(
        "--test-start",
        default="2022-01-01",
        help="Outer test start date. Pre-test data is used for tuning. "
        "Must leave enough pre-test data for inner_window_days (see spec §6).",
    )
    p.add_argument("--n-trials", type=int, default=50, help="Hard trial budget ceiling.")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="LightGBM threads per fit (not Optuna trial parallelism — Optuna stays single-threaded).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many non-improving trials. Reproducible. "
        "Pass 0 to disable (runs full n_trials budget).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Wall-clock budget in seconds. NOT reproducible -- exploration only.",
    )
    p.add_argument("--inner-window-days", type=int, default=90)
    p.add_argument("--es-val-days", type=int, default=42)
    p.add_argument("--es-rounds", type=int, default=50)
    p.add_argument("--n-estimators-ceiling", type=int, default=1000)
    p.add_argument("--window", default="expanding", choices=["expanding", "rolling"])
    p.add_argument("--train-span-days", type=int, default=None)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--features-path", default=None, type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument(
        "--out",
        default=Path("data/processed/lgbm_frozen_params.json"),
        type=Path,
        help="Output path for the frozen params JSON artifact.",
    )
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    # --patience 0 → disable convergence stop
    patience: int | None = args.patience if args.patience > 0 else None

    t0 = time.monotonic()

    features_path = args.features_path or Path("data/processed/features.parquet")
    data_path = args.data_path or Path("data/interim/hourly.parquet")

    df = load_interim_hourly(data_path)
    features = load_processed_features(features_path)
    price = df["day_ahead_price"].rename("day_ahead_price")
    y = price.reindex(features.index)
    x = features

    # Convert --test-start to UTC-aware timestamp (feature index is UTC;
    # a tz-naive comparison would raise TypeError).
    outer_test_start = pd.Timestamp(args.test_start, tz="UTC")
    log.info(
        "outer_test_start=%s  pre-test rows=%d",
        outer_test_start,
        (x.index < outer_test_start).sum(),
    )

    result, study = tune_lgbm(
        y,
        x,
        outer_test_start=outer_test_start,
        n_trials=args.n_trials,
        patience=patience,
        timeout=args.timeout,
        inner_window_days=args.inner_window_days,
        es_val_days=args.es_val_days,
        es_rounds=args.es_rounds,
        n_estimators_ceiling=args.n_estimators_ceiling,
        window=args.window,
        train_span_days=args.train_span_days,
        random_state=args.random_state,
        num_threads=args.n_jobs,
    )

    # Persist frozen params as a self-documenting JSON artifact
    artifact: dict[str, object] = {
        "params": result.params,
        "best_value": result.best_value,
        "outer_test_start": str(result.outer_test_start),
        "inner_test_start": str(result.inner_test_start),
        "n_trials": result.n_trials,
        "n_trials_completed": result.n_trials_completed,
        "patience": result.patience,
        "n_estimators_ceiling": result.n_estimators_ceiling,
        "es_val_days": result.es_val_days,
        "es_rounds": result.es_rounds,
        "random_state": result.random_state,
        "optuna_version": optuna.__version__,
        "lightgbm_version": lightgbm.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    log.info("frozen params written to %s", args.out)

    trials_path = args.out.with_suffix(".trials.parquet")
    study.trials_dataframe().to_parquet(trials_path)
    log.info("trials table written to %s", trials_path)

    features_fingerprint = f"{features.index.min()}_{features.index.max()}_{features.shape}"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(SPRINT3_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lgbm_tuning") as run:
        mlflow.log_params(
            {
                "n_trials": args.n_trials,
                "patience": patience,
                "timeout": args.timeout,
                "inner_window_days": args.inner_window_days,
                "es_val_days": args.es_val_days,
                "es_rounds": args.es_rounds,
                "n_estimators_ceiling": args.n_estimators_ceiling,
                "window": args.window,
                "train_span_days": str(args.train_span_days),
                "random_state": args.random_state,
                "num_threads": args.n_jobs,
                "n_trials_completed": result.n_trials_completed,
                **{f"best_{k}": v for k, v in result.params.items()},
            }
        )
        mlflow.set_tags(
            {
                "study": "tuning",
                "note": args.note,
                "features_fingerprint": features_fingerprint,
            }
        )
        mlflow.log_metric("best_pinball_0.50", result.best_value)
        mlflow.log_artifact(str(args.out))
        mlflow.log_artifact(str(trials_path))

        run_id = run.info.run_id

    elapsed = time.monotonic() - t0
    log.info("tune.py finished in %.0fs", elapsed)
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
