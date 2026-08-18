"""Load three quantile backtest predictions and log a calibration run to MLflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.config import SPRINT3_EXPERIMENT_NAME
from energy_price_forecast.evaluation.metrics import summarise_quantiles


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate calibration of the three LGBM quantile forecasts."
    )
    p.add_argument("--pred-q05", required=True, type=Path)
    p.add_argument("--pred-q50", required=True, type=Path)
    p.add_argument("--pred-q95", required=True, type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument(
        "--out",
        default=Path("data/processed/lgbm_quantile_calibration.csv"),
        type=Path,
    )
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    p.add_argument(
        "--study",
        default="lgbm_quantiles",
        help="MLflow tag: logical study grouping (default: lgbm_quantiles).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state used in the backtest runs (logged for traceability).",
    )
    return p.parse_args()


def _load_pred(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    return df["y_pred"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    preds = {
        0.05: _load_pred(args.pred_q05),
        0.5: _load_pred(args.pred_q50),
        0.95: _load_pred(args.pred_q95),
    }

    indices = [s.index for s in preds.values()]
    if not all(indices[0].equals(idx) for idx in indices[1:]):
        raise ValueError(
            "The three prediction files have different time indices. "
            "Make sure all three backtest runs used identical --test-start / --test-end / --n-jobs."
        )

    common_index = indices[0]
    if len(common_index) == 0:
        raise ValueError("Prediction index is empty after loading.")

    data_path = args.data_path or Path("data/interim/hourly.parquet")
    y = load_interim_hourly(data_path)["day_ahead_price"].reindex(common_index)

    if y.notna().sum() == 0:
        raise ValueError(
            "No valid y_true values after reindex — check data path and index alignment."
        )

    kpis = summarise_quantiles(y, preds)
    log.info("calibration KPIs: %s", {k: f"{v:.4f}" for k, v in kpis.items()})

    test_start = str(common_index.min().date())
    test_end = str(common_index.max().date())

    calib_table = pd.DataFrame(
        {
            "pinball": {a: kpis[f"pinball_{a:.2f}"] for a in (0.05, 0.5, 0.95)},
            "coverage": {a: kpis[f"coverage_{a:.2f}"] for a in (0.05, 0.5, 0.95)},
        }
    )
    calib_table.index.name = "alpha"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calib_table.to_csv(args.out)
    log.info("calibration table written to %s", args.out)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(SPRINT3_EXPERIMENT_NAME)

    run_name = f"{args.study}_calibration"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pred_q05": str(args.pred_q05),
                "pred_q50": str(args.pred_q50),
                "pred_q95": str(args.pred_q95),
                "levels": "(0.05, 0.5, 0.95)",
                "test_start": test_start,
                "test_end": test_end,
                "random_state": args.random_state,
            }
        )
        mlflow.set_tags({"study": args.study, "note": args.note})
        mlflow.log_metrics(kpis)
        mlflow.log_artifact(str(args.out))

    log.info("MLflow calibration run logged.")


if __name__ == "__main__":
    main()
