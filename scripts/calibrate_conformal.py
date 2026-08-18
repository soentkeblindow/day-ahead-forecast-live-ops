"""Apply scaled conformal recalibration to the extended quantile grid and log
a raw / calibrated / calibrated_sorted MLflow run.

Thin glue only: argparse -> load the seven parquets + target -> call
scaled_conformal_calibrate -> rearrange_quantiles on its output -> reliability
via the UNCHANGED 4.3a functions (reliability_curve, band_metrics) on all
three artifact generations -> MLflow run with tables + all three prediction
sets as artefacts. All diagnostic/calibration/rearrangement logic lives in
energy_price_forecast.evaluation.conformal / .rearrangement / .reliability.

conformal_diagnostics.csv (crossing_rate, Q, sigma) always describes the
UNSORTED calibrated set -- rearrangement enforces monotonicity afterwards but
must not hide the noise signal that crossing_rate reports.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.config import (
    CALIBRATION_EXPERIMENT_NAME,
    QUANTILE_GRID,
    ConformalConfig,
)
from energy_price_forecast.evaluation.conformal import scaled_conformal_calibrate
from energy_price_forecast.evaluation.rearrangement import rearrange_quantiles
from energy_price_forecast.evaluation.reliability import (
    FORECAST_LEVEL_BUCKETS,
    NESTED_BANDS,
    band_metrics,
    reliability_curve,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scaled per-quantile conformal recalibration for the extended "
        "LightGBM quantile grid (Sprint 4.3b)."
    )
    p.add_argument("--calibration-window", type=int, default=None)
    p.add_argument("--preds-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--data-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--study", default="conformal_scaled")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def _load_pred(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "y_pred" not in df.columns:
        raise ValueError(f"{path} is missing the required 'y_pred' column (got {list(df.columns)})")
    return df["y_pred"]


def _pred_path(preds_dir: Path, level: float) -> Path:
    return preds_dir / f"preds_lgbm_q{int(round(level * 100)):02d}.parquet"


def _calibrated_path(out_dir: Path, level: float) -> Path:
    return out_dir / f"preds_lgbm_q{int(round(level * 100)):02d}_calibrated.parquet"


def _sorted_path(out_dir: Path, level: float) -> Path:
    return out_dir / f"preds_lgbm_q{int(round(level * 100)):02d}_calibrated_sorted.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    preds = {a: _load_pred(_pred_path(args.preds_dir, a)) for a in QUANTILE_GRID}

    indices = [s.index for s in preds.values()]
    if not all(indices[0].equals(idx) for idx in indices[1:]):
        raise ValueError(
            "The seven LightGBM prediction files have different time indices. "
            "Make sure all seven backtest runs used identical --test-start / "
            "--test-end / --n-jobs (they are not coherently calibratable otherwise)."
        )
    common_index = indices[0]
    if len(common_index) == 0:
        raise ValueError("Prediction index is empty after loading.")

    y = load_interim_hourly(args.data_path)["day_ahead_price"].reindex(common_index)

    config = ConformalConfig()
    calibrated, diagnostics = scaled_conformal_calibrate(
        y, preds, config=config, window_days=args.calibration_window
    )
    calibrated_sorted = rearrange_quantiles(calibrated)

    reliability_raw = reliability_curve(y, preds, buckets=FORECAST_LEVEL_BUCKETS)
    reliability_calibrated = reliability_curve(y, calibrated, buckets=FORECAST_LEVEL_BUCKETS)
    reliability_sorted = reliability_curve(y, calibrated_sorted, buckets=FORECAST_LEVEL_BUCKETS)
    bands_raw = band_metrics(y, preds, buckets=FORECAST_LEVEL_BUCKETS)
    bands_calibrated = band_metrics(y, calibrated, buckets=FORECAST_LEVEL_BUCKETS)
    bands_sorted = band_metrics(y, calibrated_sorted, buckets=FORECAST_LEVEL_BUCKETS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "reliability_raw": args.out_dir / "conformal_reliability_raw.csv",
        "reliability_calibrated": args.out_dir / "conformal_reliability_calibrated.csv",
        "reliability_sorted": args.out_dir / "conformal_reliability_sorted.csv",
        "bands_raw": args.out_dir / "conformal_bands_raw.csv",
        "bands_calibrated": args.out_dir / "conformal_bands_calibrated.csv",
        "bands_sorted": args.out_dir / "conformal_bands_sorted.csv",
        "diagnostics": args.out_dir / "conformal_diagnostics.csv",
    }
    reliability_raw.to_csv(out_paths["reliability_raw"], index=False)
    reliability_calibrated.to_csv(out_paths["reliability_calibrated"], index=False)
    reliability_sorted.to_csv(out_paths["reliability_sorted"], index=False)
    bands_raw.to_csv(out_paths["bands_raw"], index=False)
    bands_calibrated.to_csv(out_paths["bands_calibrated"], index=False)
    bands_sorted.to_csv(out_paths["bands_sorted"], index=False)
    # diagnostics describes the UNSORTED calibrated set on purpose: crossing_rate
    # is a real signal about noise in Q_alpha, not a defect rearrangement should hide.
    diagnostics.to_csv(out_paths["diagnostics"], index=False)
    for name, path in out_paths.items():
        log.info("%s written to %s", name, path)

    calibrated_paths = {a: _calibrated_path(args.out_dir, a) for a in QUANTILE_GRID}
    for a, path in calibrated_paths.items():
        calibrated[a].to_frame("y_pred").to_parquet(path)
        log.info("calibrated q%.2f written to %s", a, path)

    sorted_paths = {a: _sorted_path(args.out_dir, a) for a in QUANTILE_GRID}
    for a, path in sorted_paths.items():
        calibrated_sorted[a].to_frame("y_pred").to_parquet(path)
        log.info("calibrated_sorted q%.2f written to %s", a, path)

    overall_raw = reliability_raw[reliability_raw["bucket"] == "overall"]
    overall_calibrated = reliability_calibrated[reliability_calibrated["bucket"] == "overall"]
    overall_sorted = reliability_sorted[reliability_sorted["bucket"] == "overall"]
    metrics: dict[str, float] = {
        f"coverage_before_{level:.2f}": float(row["coverage"])
        for level, row in overall_raw.set_index("level").iterrows()
    }
    metrics.update(
        {
            f"coverage_after_{level:.2f}": float(row["coverage"])
            for level, row in overall_calibrated.set_index("level").iterrows()
        }
    )
    metrics.update(
        {
            f"coverage_sorted_{level:.2f}": float(row["coverage"])
            for level, row in overall_sorted.set_index("level").iterrows()
        }
    )

    bands_raw_overall = bands_raw[bands_raw["bucket"] == "overall"]
    bands_calibrated_overall = bands_calibrated[bands_calibrated["bucket"] == "overall"]
    bands_sorted_overall = bands_sorted[bands_sorted["bucket"] == "overall"]
    for name, _low, _high in NESTED_BANDS:
        raw_row = bands_raw_overall.loc[bands_raw_overall["band"] == name]
        if not raw_row.empty:
            metrics[f"interval_coverage_before_{name}"] = float(raw_row["coverage"].iloc[0])
        calibrated_row = bands_calibrated_overall.loc[bands_calibrated_overall["band"] == name]
        if not calibrated_row.empty:
            metrics[f"interval_coverage_after_{name}"] = float(calibrated_row["coverage"].iloc[0])
        sorted_row = bands_sorted_overall.loc[bands_sorted_overall["band"] == name]
        if not sorted_row.empty:
            metrics[f"interval_coverage_sorted_{name}"] = float(sorted_row["coverage"].iloc[0])
            metrics[f"interval_width_sorted_{name}"] = float(sorted_row["width"].iloc[0])

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CALIBRATION_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="conformal_scaled_calibration"):
        mlflow.log_params(
            {
                "window_days": args.calibration_window or config.window_days,
                "embargo_days": config.embargo_days,
                "n_neighbors": config.n_neighbors,
                "sigma_floor_fraction": config.sigma_floor_fraction,
                "recompute_cadence_days": config.recompute_cadence_days,
                "min_calibration_hours": config.min_calibration_hours,
                "test_start": str(common_index.min().date()),
                "test_end": str(common_index.max().date()),
            }
        )
        mlflow.set_tags({"study": args.study, "note": args.note})
        mlflow.log_metrics(metrics)
        for path in out_paths.values():
            mlflow.log_artifact(str(path))
        for path in calibrated_paths.values():
            mlflow.log_artifact(str(path))
        for path in sorted_paths.values():
            mlflow.log_artifact(str(path))

    log.info("MLflow conformal calibration run logged.")


if __name__ == "__main__":
    main()
