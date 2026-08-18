"""Load the extended quantile-grid predictions and log a reliability run to MLflow.

Thin glue only: argparse -> load parquets + target -> call reliability_curve /
band_metrics -> MLflow run with the two tidy tables as artefacts. All
diagnostic logic lives in energy_price_forecast.evaluation.reliability.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.config import CALIBRATION_EXPERIMENT_NAME
from energy_price_forecast.evaluation.reliability import (
    FORECAST_LEVEL_BUCKETS,
    NESTED_BANDS,
    band_metrics,
    reliability_curve,
)

LGBM_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ARIMAX_LEVELS: tuple[float, ...] = (0.05, 0.50, 0.95)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reliability curve and nested-band metrics for the extended "
        "LightGBM quantile grid (Sprint 4.3a), with an optional ARIMAX reference."
    )
    for a in LGBM_LEVELS:
        p.add_argument(f"--pred-q{int(round(a * 100)):02d}", required=True, type=Path)
    for a in ARIMAX_LEVELS:
        p.add_argument(f"--arimax-q{int(round(a * 100)):02d}", default=None, type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument(
        "--out-reliability", default=Path("data/processed/reliability_curve.csv"), type=Path
    )
    p.add_argument("--out-bands", default=Path("data/processed/reliability_bands.csv"), type=Path)
    p.add_argument("--experiment", default=CALIBRATION_EXPERIMENT_NAME)
    p.add_argument("--study", default="lgbm_quantile_grid")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def _load_pred(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "y_pred" not in df.columns:
        raise ValueError(f"{path} is missing the required 'y_pred' column (got {list(df.columns)})")
    return df["y_pred"]


def _fingerprint(y: pd.Series) -> str:
    return f"{y.index.min()}_{y.index.max()}_{y.shape}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    preds_lgbm = {
        a: _load_pred(getattr(args, f"pred_q{int(round(a * 100)):02d}")) for a in LGBM_LEVELS
    }

    indices = [s.index for s in preds_lgbm.values()]
    if not all(indices[0].equals(idx) for idx in indices[1:]):
        raise ValueError(
            "The seven LightGBM prediction files have different time indices. "
            "Make sure all seven backtest runs used identical --test-start / "
            "--test-end / --n-jobs."
        )
    common_index = indices[0]
    if len(common_index) == 0:
        raise ValueError("Prediction index is empty after loading.")

    data_path = args.data_path or Path("data/interim/hourly.parquet")
    y_lgbm = load_interim_hourly(data_path)["day_ahead_price"].reindex(common_index)

    curve_tables = [
        reliability_curve(y_lgbm, preds_lgbm, buckets=FORECAST_LEVEL_BUCKETS).assign(
            model="lightgbm"
        )
    ]
    band_tables = [
        band_metrics(y_lgbm, preds_lgbm, buckets=FORECAST_LEVEL_BUCKETS).assign(model="lightgbm")
    ]

    arimax_paths = {a: getattr(args, f"arimax_q{int(round(a * 100)):02d}") for a in ARIMAX_LEVELS}
    n_arimax_given = sum(p is not None for p in arimax_paths.values())
    if n_arimax_given not in (0, len(ARIMAX_LEVELS)):
        raise ValueError(
            "Provide all three --arimax-q05/--arimax-q50/--arimax-q95 or none of them."
        )
    if n_arimax_given == len(ARIMAX_LEVELS):
        preds_arimax = {a: _load_pred(path) for a, path in arimax_paths.items() if path is not None}
        arimax_index = preds_arimax[0.50].index
        y_arimax = load_interim_hourly(data_path)["day_ahead_price"].reindex(arimax_index)
        curve_tables.append(reliability_curve(y_arimax, preds_arimax).assign(model="arimax"))
        band_tables.append(band_metrics(y_arimax, preds_arimax).assign(model="arimax"))

    reliability_table = pd.concat(curve_tables, ignore_index=True)
    bands_table = pd.concat(band_tables, ignore_index=True)

    args.out_reliability.parent.mkdir(parents=True, exist_ok=True)
    args.out_bands.parent.mkdir(parents=True, exist_ok=True)
    reliability_table.to_csv(args.out_reliability, index=False)
    bands_table.to_csv(args.out_bands, index=False)
    log.info("reliability curve written to %s", args.out_reliability)
    log.info("band metrics written to %s", args.out_bands)

    overall_curve = reliability_table[
        (reliability_table["model"] == "lightgbm") & (reliability_table["bucket"] == "overall")
    ]
    overall_bands = bands_table[
        (bands_table["model"] == "lightgbm") & (bands_table["bucket"] == "overall")
    ]
    metrics = {
        f"coverage_{level:.2f}": float(row["coverage"])
        for level, row in overall_curve.set_index("level").iterrows()
    }
    for name, _low, _high in NESTED_BANDS:
        band_row = overall_bands.loc[overall_bands["band"] == name]
        if band_row.empty:
            continue
        metrics[f"interval_coverage_{name}"] = float(band_row["coverage"].iloc[0])
        metrics[f"interval_width_{name}"] = float(band_row["width"].iloc[0])

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="reliability_eval"):
        mlflow.log_params(
            {
                "levels": ",".join(f"{a:.2f}" for a in LGBM_LEVELS),
                "test_start": str(common_index.min().date()),
                "test_end": str(common_index.max().date()),
                "arimax_reference": n_arimax_given == len(ARIMAX_LEVELS),
            }
        )
        mlflow.set_tags(
            {
                "study": args.study,
                "note": args.note,
                "features_fingerprint": _fingerprint(y_lgbm),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.out_reliability))
        mlflow.log_artifact(str(args.out_bands))

    log.info("MLflow reliability run logged.")


if __name__ == "__main__":
    main()
