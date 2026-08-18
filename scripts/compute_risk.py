"""Thin wiring for Sprint 4.4a: trading-book P&L, VaR and Expected Shortfall.

Loads the raw and calibrated_sorted quantile parquets plus the target,
builds the residual pool (energy_price_forecast.evaluation.residuals),
computes risk_measures for three threshold sources x two book sides
(energy_price_forecast.evaluation.risk), checks the FHS price quantiles
against the UNCHANGED 4.3a reliability_curve, and logs an MLflow run.

All logic lives in evaluation.residuals / evaluation.risk / evaluation.
reliability; this script only loads data, wires the calls together, and
writes artefacts out.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.config import (
    CALIBRATION_EXPERIMENT_NAME,
    QUANTILE_GRID,
    ConformalConfig,
)
from energy_price_forecast.evaluation.conformal import hourly_sigma
from energy_price_forecast.evaluation.reliability import FORECAST_LEVEL_BUCKETS, reliability_curve
from energy_price_forecast.evaluation.residuals import (
    pool_by_day,
    standardised_residuals,
    tail_quantile,
)
from energy_price_forecast.evaluation.risk import (
    QUANTITY_MWH,
    Side,
    Variant,
    book_pnl,
    es_ratio_conditional,
    realised_es,
    risk_measures,
    sdac_limits_asof,
)

_SIDES: tuple[Side, ...] = ("long", "short")
_VARIANTS: tuple[Variant, ...] = ("raw", "calibrated", "fhs")
_LEVEL = 0.95
_LONG_ALPHA, _SHORT_ALPHA = 1.0 - _LEVEL, _LEVEL


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 4.4a: trading-book VaR/ES from three threshold sources."
    )
    p.add_argument("--preds-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--data-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--study", default="risk_hourly")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def _load_pred(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "y_pred" not in df.columns:
        raise ValueError(f"{path} is missing the required 'y_pred' column (got {list(df.columns)})")
    return df["y_pred"]


def _raw_path(preds_dir: Path, level: float) -> Path:
    return preds_dir / f"preds_lgbm_q{int(round(level * 100)):02d}.parquet"


def _calibrated_sorted_path(preds_dir: Path, level: float) -> Path:
    return preds_dir / f"preds_lgbm_q{int(round(level * 100)):02d}_calibrated_sorted.parquet"


def _check_index_consistency(named_series: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Fail-fast, exact index equality across every loaded series (spec 3.3)."""
    names = list(named_series)
    reference = pd.DatetimeIndex(named_series[names[0]].index)
    if not reference.is_monotonic_increasing:
        raise ValueError(f"Index of {names[0]!r} is not ascending sorted.")
    for name in names[1:]:
        idx = pd.DatetimeIndex(named_series[name].index)
        if not reference.equals(idx):
            first_diff = next(
                ((a, b) for a, b in zip(reference, idx, strict=False) if a != b),
                "different lengths",
            )
            raise ValueError(
                f"Index mismatch between {names[0]!r} and {name!r}: first divergence "
                f"{first_diff}. All prediction files and the target must share an "
                "identical, ascending, hourly UTC index."
            )
    return reference


def _fhs_price_quantiles(
    common_index: pd.DatetimeIndex,
    mark: pd.Series,
    sigma: pd.Series,
    pool: dict,
) -> dict[float, pd.Series]:
    """FHS price quantiles at the extended grid, for the reliability check
    below only -- not part of the reusable module API (spec 2.4).
    """
    local_days = common_index.tz_convert("Europe/Berlin").normalize()
    mark_np, sigma_np = mark.to_numpy(), sigma.to_numpy()
    fhs_preds: dict[float, pd.Series] = {}
    for level in QUANTILE_GRID:
        values = np.full(len(common_index), np.nan)
        for day_ts in pd.unique(local_days):
            day_pool = pool.get(pd.Timestamp(day_ts).date(), np.array([], dtype=float))
            if day_pool.size == 0:
                continue
            mask = local_days == day_ts
            values[mask] = mark_np[mask] + sigma_np[mask] * tail_quantile(day_pool, level)
        fhs_preds[level] = pd.Series(values, index=common_index)
    return fhs_preds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    raw = {lvl: _load_pred(_raw_path(args.preds_dir, lvl)) for lvl in (0.05, 0.5, 0.95)}
    calibrated_sorted = {
        lvl: _load_pred(_calibrated_sorted_path(args.preds_dir, lvl)) for lvl in (0.05, 0.5, 0.95)
    }
    named = {f"raw_q{lvl:.2f}": s for lvl, s in raw.items()}
    named.update({f"calibrated_sorted_q{lvl:.2f}": s for lvl, s in calibrated_sorted.items()})
    common_index = _check_index_consistency(named)
    if len(common_index) == 0:
        raise ValueError("Prediction index is empty after loading.")

    # price/y_true is reduced to exactly this index -- not the full history
    # in load_interim_hourly() -- so hourly_sigma reproduces the SAME
    # calibration windows that shaped calibrated_sorted (spec 3.3).
    price = load_interim_hourly(args.data_path)["day_ahead_price"].reindex(common_index)

    config = ConformalConfig()
    sigma = hourly_sigma(price, raw[0.5], config=config)

    mark = calibrated_sorted[0.5]
    r = standardised_residuals(price, mark, sigma)
    pool = pool_by_day(r, embargo_days=config.embargo_days)

    floor, cap = sdac_limits_asof(common_index)
    book = pd.DataFrame(
        {
            "price": price,
            "mark": mark,
            "sigma": sigma,
            "r": r,
            "pnl_long": book_pnl(price, mark, side="long"),
            "pnl_short": book_pnl(price, mark, side="short"),
            "sdac_floor": floor,
            "sdac_cap": cap,
        }
    )

    risk_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for variant in _VARIANTS:
        for side in _SIDES:
            threshold_quantile: pd.Series | None = None
            if variant == "raw":
                threshold_quantile = raw[0.05] if side == "long" else raw[0.95]
            elif variant == "calibrated":
                threshold_quantile = (
                    calibrated_sorted[0.05] if side == "long" else calibrated_sorted[0.95]
                )

            result = risk_measures(
                price=price,
                mark=mark,
                sigma=sigma,
                pool=pool,
                variant=variant,
                side=side,
                level=_LEVEL,
                threshold_quantile=threshold_quantile,
            )
            tagged = result.copy()
            tagged.insert(0, "side", side)
            tagged.insert(0, "variant", variant)
            tagged.insert(0, "timestamp", common_index)
            risk_rows.append(tagged)

            pnl = book["pnl_long"] if side == "long" else book["pnl_short"]
            realised = realised_es(pnl, level=_LEVEL)
            ratio = es_ratio_conditional(pnl, result)

            summary_rows.append(
                {
                    "variant": variant,
                    "side": side,
                    "mean_var": float(result["var"].mean()),
                    "mean_es": float(result["es"].mean()),
                    "mean_es_unclipped": float(result["es_unclipped"].mean()),
                    "breach_rate": float(result["breach"].mean()),
                    "mean_pi": float(result["pi"].mean()),
                    "clip_bind_rate": float((result["n_clipped"] > 0).mean()),
                    "mean_clip_impact_es": float(result["clip_impact_es"].mean()),
                    "es_ratio_conditional": ratio,
                    "realised_var_unconditional": realised["var"],
                    "realised_es_unconditional": realised["es"],
                    "n": float(len(result)),
                }
            )

    risk_hourly = pd.concat(risk_rows, ignore_index=True)
    risk_hourly["variant"] = risk_hourly["variant"].astype("category")
    risk_hourly["side"] = risk_hourly["side"].astype("category")
    risk_summary = pd.DataFrame(summary_rows)

    # "raw" is a COUNTERFACTUAL, not a risk number (spec 2.3): a raw quantile
    # grid has no mass beyond its own boundary, so its ES borrows the tail
    # mean from the pool behind an uncalibrated threshold. Flagged here, in
    # the run's own summary, not just in the risk_measures docstring.
    log.warning(
        "variant='raw' rows in risk_summary.csv are a COUNTERFACTUAL, not a "
        "risk number: 'what would a manager using the raw q05/q95 as their "
        "threshold have reported, with the pool's tail behind it?' (spec 2.3)."
    )

    fhs_preds = _fhs_price_quantiles(common_index, mark, sigma, pool)
    risk_reliability_fhs = reliability_curve(price, fhs_preds, buckets=FORECAST_LEVEL_BUCKETS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    book_path = args.out_dir / "book_hourly.parquet"
    risk_hourly_path = args.out_dir / "risk_hourly.parquet"
    risk_summary_path = args.out_dir / "risk_summary.csv"
    risk_reliability_path = args.out_dir / "risk_reliability_fhs.csv"

    book.to_parquet(book_path)
    risk_hourly.to_parquet(risk_hourly_path)
    risk_summary.to_csv(risk_summary_path, index=False)
    risk_reliability_fhs.to_csv(risk_reliability_path, index=False)
    for path in (book_path, risk_hourly_path, risk_summary_path, risk_reliability_path):
        log.info("written to %s", path)

    def _row(variant: str, side: str) -> pd.Series:
        match = risk_summary[(risk_summary["variant"] == variant) & (risk_summary["side"] == side)]
        return match.iloc[0]

    buffer_factor = {
        side: float(_row("calibrated", side)["mean_var"]) / float(_row("raw", side)["mean_var"])
        for side in _SIDES
    }
    skew_ratio = float(_row("fhs", "short")["mean_es_unclipped"]) / float(
        _row("fhs", "long")["mean_es_unclipped"]
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CALIBRATION_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="risk_measures"):
        mlflow.set_tags({"study": args.study, "note": args.note})
        mlflow.log_params(
            {
                "quantity_mwh": QUANTITY_MWH,
                "level": _LEVEL,
                "test_start": str(common_index.min().date()),
                "test_end": str(common_index.max().date()),
            }
        )
        metrics: dict[str, float] = {}
        for row in summary_rows:
            key = f"{row['variant']}_{row['side']}"
            for name in (
                "mean_var",
                "mean_es",
                "breach_rate",
                "es_ratio_conditional",
                "clip_bind_rate",
            ):
                metrics[f"{key}_{name}"] = float(row[name])  # type: ignore[arg-type]
        for side in _SIDES:
            metrics[f"buffer_factor_{side}"] = buffer_factor[side]
        metrics["skew_ratio"] = skew_ratio
        mlflow.log_metrics(metrics)
        for path in (book_path, risk_hourly_path, risk_summary_path, risk_reliability_path):
            mlflow.log_artifact(str(path))

    log.info("MLflow risk_measures run logged.")


if __name__ == "__main__":
    main()
