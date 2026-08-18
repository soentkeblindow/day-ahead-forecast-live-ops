"""Sprint 4.4b Nachtrag 2, part A: block-days sensitivity aggregation.

Purely aggregating -- reads the already-written per-K outputs of
scripts/backtest_risk.py (one run per BacktestConfig.block_days_grid value)
and compares them. Calls NO evaluation/ function, runs NO bootstrap, fits NO
model (spec Nachtrag 2, 3.4): every number here is already sitting in the
per-K backtest_coverage[_block{K}].csv / backtest_variant_contrast[_block{K}].csv
files, this script only reshapes and contrasts them against the block_days=1
headline.

variant='raw' rows are excluded: they carry no bootstrap CI (4.4a spec 2.3),
so there is nothing here for block_days to be sensitive to.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.evaluation.config import CALIBRATION_EXPERIMENT_NAME, BacktestConfig

_VALIDATED_VARIANTS: tuple[str, ...] = ("calibrated", "fhs")
_COVERAGE_KEY: tuple[str, ...] = ("variant", "side", "subset")
_CONTRAST_KEY: tuple[str, ...] = ("side", "subset")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 4.4b Nachtrag 2, part A: block-days bootstrap sensitivity "
        "aggregation. Reads the per-K backtest_risk.py outputs and compares them; does "
        "not call evaluation/ or run any bootstrap itself."
    )
    p.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    p.add_argument(
        "--block-days-grid",
        default=None,
        type=int,
        nargs="+",
        help="Overrides BacktestConfig.block_days_grid.",
    )
    p.add_argument("--study", default="block_sensitivity")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def _coverage_path(out_dir: Path, block_days: int) -> Path:
    if block_days == 1:
        return out_dir / "backtest_coverage.csv"
    return out_dir / f"backtest_coverage_block{block_days}.csv"


def _contrast_path(out_dir: Path, block_days: int) -> Path:
    if block_days == 1:
        return out_dir / "backtest_variant_contrast.csv"
    return out_dir / f"backtest_variant_contrast_block{block_days}.csv"


def _block_days_cli_flag(block_days: int) -> str:
    return "" if block_days == 1 else f" --tag block{block_days}"


def _load_grid(
    path_fn: Callable[[Path, int], Path], out_dir: Path, grid: tuple[int, ...]
) -> pd.DataFrame:
    """Fail-fast concat of one CSV per K in `grid`, trusting each file's OWN block_days column."""
    frames: list[pd.DataFrame] = []
    for k in grid:
        path = path_fn(out_dir, k)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- generate it first with: "
                f"'uv run python scripts/backtest_risk.py --block-days {k}"
                f"{_block_days_cli_flag(k)}'"
            )
        frame = pd.read_csv(path)
        if "block_days" not in frame.columns:
            raise ValueError(
                f"{path} has no block_days column -- was it produced before Nachtrag 2, "
                "part A? Re-run scripts/backtest_risk.py."
            )
        mismatched = sorted(frame.loc[frame["block_days"] != k, "block_days"].unique())
        if mismatched:
            raise ValueError(
                f"{path} was expected to hold block_days={k} rows (it is the grid's K={k} "
                f"file), but its block_days column contains {mismatched} -- wrong "
                "--block-days was used to produce this file."
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _verdict_flipped(current: pd.Series, base: pd.Series) -> pd.Series:
    """current != base as nullable boolean; NA (not False) wherever either side is unknown."""
    known = current.notna() & base.notna()
    flipped = pd.Series(pd.NA, index=current.index, dtype="boolean")
    flipped[known] = current[known].astype(bool).to_numpy() != base[known].astype(bool).to_numpy()
    return flipped


def _assert_point_estimate_invariant(frame: pd.DataFrame, *, key: list[str], col: str) -> None:
    spread = frame.groupby(key)[col].nunique(dropna=False)
    bad = spread[spread > 1]
    if not bad.empty:
        raise ValueError(
            f"{col} varies with block_days for cells:\n{bad.index.to_list()}\n"
            "The point estimate must depend only on the realised data, never on the "
            "resampling regime (spec Nachtrag 2, section 9) -- this is a bug upstream, "
            "not a sensitivity finding."
        )


def build_coverage_sensitivity(coverage: pd.DataFrame) -> pd.DataFrame:
    key = list(_COVERAGE_KEY)
    validated = coverage[coverage["variant"].isin(_VALIDATED_VARIANTS)].copy()

    _assert_point_estimate_invariant(validated, key=key, col="breach_rate")
    _assert_point_estimate_invariant(validated, key=key, col="z1")

    validated["ci_width"] = validated["breach_rate_ci_high"] - validated["breach_rate_ci_low"]
    validated["_z1_ci_width"] = validated["z1_ci_high"] - validated["z1_ci_low"]

    base_cols = [
        *key,
        "ci_width",
        "_z1_ci_width",
        "breach_rate_ci_excludes_alpha",
        "z1_excludes_zero",
    ]
    base = validated.loc[validated["block_days"] == 1, base_cols].rename(
        columns={
            "ci_width": "_base_ci_width",
            "_z1_ci_width": "_base_z1_ci_width",
            "breach_rate_ci_excludes_alpha": "_base_ci_excludes_alpha",
            "z1_excludes_zero": "_base_z1_excludes_zero",
        }
    )
    merged = validated.merge(base, on=key, how="left", validate="many_to_one")

    merged["ci_width_ratio"] = merged["ci_width"] / merged["_base_ci_width"]
    merged["z1_ci_width_ratio"] = merged["_z1_ci_width"] / merged["_base_z1_ci_width"]
    merged["ci_excludes_alpha"] = merged["breach_rate_ci_excludes_alpha"]
    merged["verdict_flipped"] = _verdict_flipped(
        merged["ci_excludes_alpha"], merged["_base_ci_excludes_alpha"]
    )
    merged["z1_verdict_flipped"] = _verdict_flipped(
        merged["z1_excludes_zero"], merged["_base_z1_excludes_zero"]
    )

    columns = [
        "variant",
        "side",
        "subset",
        "block_days",
        "breach_rate",
        "breach_rate_ci_low",
        "breach_rate_ci_high",
        "ci_width",
        "ci_width_ratio",
        "ci_excludes_alpha",
        "verdict_flipped",
        "z1",
        "z1_ci_low",
        "z1_ci_high",
        "z1_ci_width_ratio",
        "z1_excludes_zero",
        "z1_verdict_flipped",
        "low_support",
        "converged",
        "n_bootstrap_used",
    ]
    return merged[columns].sort_values(key + ["block_days"]).reset_index(drop=True)


def build_contrast_sensitivity(contrast: pd.DataFrame) -> pd.DataFrame:
    key = list(_CONTRAST_KEY)
    frame = contrast.copy()

    _assert_point_estimate_invariant(frame, key=key, col="d_es_ratio")
    _assert_point_estimate_invariant(frame, key=key, col="d_breach_rate")

    frame["d_es_ratio_ci_width"] = frame["d_es_ratio_ci_high"] - frame["d_es_ratio_ci_low"]
    frame["d_breach_rate_ci_width"] = frame["d_breach_rate_ci_high"] - frame["d_breach_rate_ci_low"]

    base_cols = [
        *key,
        "d_es_ratio_ci_width",
        "d_breach_rate_ci_width",
        "separates",
    ]
    base = frame.loc[frame["block_days"] == 1, base_cols].rename(
        columns={
            "d_es_ratio_ci_width": "_base_d_es_ratio_ci_width",
            "d_breach_rate_ci_width": "_base_d_breach_rate_ci_width",
            "separates": "_base_separates",
        }
    )
    merged = frame.merge(base, on=key, how="left", validate="many_to_one")

    merged["d_es_ratio_ci_width_ratio"] = (
        merged["d_es_ratio_ci_width"] / merged["_base_d_es_ratio_ci_width"]
    )
    merged["d_breach_rate_ci_width_ratio"] = (
        merged["d_breach_rate_ci_width"] / merged["_base_d_breach_rate_ci_width"]
    )
    merged["separates_flipped"] = _verdict_flipped(merged["separates"], merged["_base_separates"])

    columns = [
        "side",
        "subset",
        "block_days",
        "d_es_ratio",
        "d_es_ratio_ci_low",
        "d_es_ratio_ci_high",
        "d_es_ratio_ci_width",
        "d_es_ratio_ci_width_ratio",
        "d_breach_rate",
        "d_breach_rate_ci_low",
        "d_breach_rate_ci_high",
        "d_breach_rate_ci_width",
        "d_breach_rate_ci_width_ratio",
        "separates",
        "separates_flipped",
        "low_support",
    ]
    return merged[columns].sort_values(key + ["block_days"]).reset_index(drop=True)


def _true_count(series: pd.Series) -> float:
    """Count of True entries; NA/unknown entries count as 0 (nullable-boolean sum semantics)."""
    return float(series.astype("boolean").sum())


def _grid_metrics(coverage_sens: pd.DataFrame, block_days: int) -> dict[str, float]:
    at_k = coverage_sens[coverage_sens["block_days"] == block_days]
    return {
        f"n_ci_rejections_block{block_days}": _true_count(at_k["ci_excludes_alpha"]),
        f"median_ci_width_ratio_block{block_days}": float(at_k["ci_width_ratio"].median()),
        f"max_ci_width_ratio_block{block_days}": float(at_k["ci_width_ratio"].max()),
        f"n_verdict_flips_block{block_days}": _true_count(at_k["verdict_flipped"]),
        f"n_z1_verdict_flips_block{block_days}": _true_count(at_k["z1_verdict_flipped"]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()
    config = BacktestConfig()
    if args.block_days_grid is not None:
        config = dataclasses.replace(config, block_days_grid=tuple(args.block_days_grid))
    grid = config.block_days_grid

    coverage = _load_grid(_coverage_path, args.out_dir, grid)
    contrast = _load_grid(_contrast_path, args.out_dir, grid)

    coverage_sens = build_coverage_sensitivity(coverage)
    contrast_sens = build_contrast_sensitivity(contrast)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coverage_sens_path = args.out_dir / "backtest_block_sensitivity_coverage.csv"
    contrast_sens_path = args.out_dir / "backtest_block_sensitivity_contrast.csv"
    coverage_sens.to_csv(coverage_sens_path, index=False)
    contrast_sens.to_csv(contrast_sens_path, index=False)
    for path in (coverage_sens_path, contrast_sens_path):
        log.info("written to %s", path)

    ramp_short = contrast_sens[
        (contrast_sens["side"] == "short") & (contrast_sens["subset"] == "evening_ramp")
    ].set_index("block_days")["separates"]

    metrics: dict[str, float] = {}
    console_rows: list[dict[str, object]] = []
    for k in grid:
        metrics.update(_grid_metrics(coverage_sens, k))
        metrics[f"ramp_short_separates_block{k}"] = float(bool(ramp_short.get(k, False)))
        console_rows.append(
            {
                "block_days": k,
                "n_ci_rejections": metrics[f"n_ci_rejections_block{k}"],
                "median_ci_width_ratio": metrics[f"median_ci_width_ratio_block{k}"],
                "n_verdict_flips": metrics[f"n_verdict_flips_block{k}"],
                "ramp_short_separates": bool(ramp_short.get(k, False)),
            }
        )

    log.info(
        "block-days sensitivity summary:\n%s", pd.DataFrame(console_rows).to_string(index=False)
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CALIBRATION_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="backtest_block_sensitivity"):
        mlflow.set_tags({"study": "block_sensitivity", "note": args.note})
        mlflow.log_params({"block_days_grid": list(grid)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(coverage_sens_path))
        mlflow.log_artifact(str(contrast_sens_path))

    log.info("MLflow backtest_block_sensitivity run logged.")


if __name__ == "__main__":
    main()
