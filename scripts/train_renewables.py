"""Sprint 6 / Step 6.5.2: run one of the three renewables walk-forward variants.

Thin CLI/MLflow wrapper around evaluation.renewables_walkforward.run_renewables_backtest
(spec 6.5.2 section 5.7/section 6). The CLI only accepts the three named,
preregistered variants (spec section 5.7 table) -- not free-form window x
objective combinations -- so "the fourth combination is never run" is
enforced by the tool, not just by convention.

Metrics: MAE and RMSE per target, in capacity factor and MW, against the
persistence baseline (spec 5.7), plus a coarse daylight/dark and calendar-
season breakdown computed inline here (no new module -- evaluation/metrics.py
stays unmodified, spec 3.3). RMSE is the primary metric (spec 2.10); MAE is
reported alongside, never decisive.
"""

from __future__ import annotations

import argparse
import logging
import time

import mlflow
import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
)
from energy_price_forecast.data.loaders import load_interim_hourly, load_interim_weather
from energy_price_forecast.evaluation.metrics import mae, rmse
from energy_price_forecast.evaluation.renewables_walkforward import (
    TARGET_COLUMNS,
    persistence_baseline_cf,
    run_renewables_backtest,
)
from energy_price_forecast.models.renewables import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_SEED,
    capacity_factor_label,
)

logger = logging.getLogger(__name__)

_OUT_DIR = PROJECT_ROOT / "data" / "processed"
_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "renewables_backtest.csv"

# spec 6.5.2 section 5.7 -- exactly these three, the fourth combination
# (expanding + quantile) is deliberately never run.
_VARIANTS: dict[str, dict[str, object]] = {
    "rolling365_l2": {
        "window": "rolling",
        "train_span_days": 365,
        "min_history_days": None,
        "objective": "l2",
    },
    "expanding_l2": {
        "window": "expanding",
        "train_span_days": None,
        "min_history_days": 365,
        "objective": "l2",
    },
    "rolling365_quantile": {
        "window": "rolling",
        "train_span_days": 365,
        "min_history_days": None,
        "objective": "quantile",
    },
}

_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}  # fmt: skip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Renewables capacity-factor walk-forward (spec 6.5.2 section 6)."
    )
    p.add_argument("--variant", required=True, choices=sorted(_VARIANTS))
    p.add_argument("--refit-every", type=int, default=7)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--source",
        default=CapacitySource.PUBLIC_REGISTRY.value,
        choices=[s.value for s in CapacitySource],
    )
    p.add_argument(
        "--method",
        default=CapacityExtrapolation.LAST_INCREMENT.value,
        choices=[m.value for m in CapacityExtrapolation],
    )
    p.add_argument("--experiment", default="arena_models")
    p.add_argument("--note", default="")
    p.add_argument(
        "--start", default=None, help="ISO date, restricts target_hourly for a quick probe run"
    )
    p.add_argument(
        "--end", default=None, help="ISO date, restricts target_hourly for a quick probe run"
    )
    return p.parse_args()


def _metrics_for_target(
    predictions: pd.DataFrame, target: ProductionType, full_label_flat: pd.Series
) -> list[dict[str, object]]:
    """``full_label_flat`` is the capacity-factor label over the *entire*
    original target_hourly range (not just the walk-forward output), so a
    D-1 lookup for the output's very first evaluated day -- whose D-1 was
    only ever training data, never itself a row in ``predictions`` -- still
    resolves.
    """
    rows: list[dict[str, object]] = []
    cf_actual = predictions[f"{target.value}_cf_actual"]
    cf_pred = predictions[f"{target.value}_cf_pred"]
    mw_actual = predictions[f"{target.value}_mw_actual"]
    mw_pred = predictions[f"{target.value}_mw_pred"]

    valid_time = pd.DatetimeIndex(predictions.index.get_level_values("valid_time_utc"))
    baseline_cf = persistence_baseline_cf(full_label_flat, valid_time)
    baseline_mw = baseline_cf.to_numpy() * predictions[f"{target.value}_capacity_mw"].to_numpy()

    def _add(label: str, mask: pd.Series | None) -> None:
        idx = slice(None) if mask is None else mask.to_numpy()
        rows.append(
            {
                "target": target.value,
                "breakdown": label,
                "n": int(len(cf_actual[idx])),
                "rmse_cf_model": rmse(cf_actual[idx], cf_pred[idx]),
                "mae_cf_model": mae(cf_actual[idx], cf_pred[idx]),
                "rmse_mw_model": rmse(mw_actual[idx], mw_pred[idx]),
                "mae_mw_model": mae(mw_actual[idx], mw_pred[idx]),
                "rmse_cf_baseline": rmse(
                    cf_actual[idx],
                    pd.Series(baseline_cf.to_numpy()[idx], index=cf_actual[idx].index),
                ),
                "mae_cf_baseline": mae(
                    cf_actual[idx],
                    pd.Series(baseline_cf.to_numpy()[idx], index=cf_actual[idx].index),
                ),
                "rmse_mw_baseline": rmse(
                    mw_actual[idx], pd.Series(baseline_mw[idx], index=mw_actual[idx].index)
                ),
                "mae_mw_baseline": mae(
                    mw_actual[idx], pd.Series(baseline_mw[idx], index=mw_actual[idx].index)
                ),
            }
        )

    _add("overall", None)

    daylight = predictions["is_daylight_hour"]
    _add("daylight", daylight)
    _add("dark", ~daylight)

    season = pd.Series(
        valid_time.tz_convert("Europe/Berlin").month.map(_SEASON_BY_MONTH), index=predictions.index
    )
    for season_name in ("winter", "spring", "summer", "autumn"):
        _add(f"season_{season_name}", season == season_name)

    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    args = _parse_args()
    variant = _VARIANTS[args.variant]

    hourly = load_interim_hourly()
    target_hourly = hourly[list(TARGET_COLUMNS.values())]
    if args.start:
        target_hourly = target_hourly.loc[args.start :]
    if args.end:
        target_hourly = target_hourly.loc[: args.end]
    weather = load_interim_weather()

    logger.info(
        "running variant=%s refit_every=%d seed=%d", args.variant, args.refit_every, args.seed
    )

    predictions = run_renewables_backtest(
        target_hourly,
        weather,
        window=variant["window"],  # type: ignore[arg-type]
        train_span_days=variant["train_span_days"],  # type: ignore[arg-type]
        min_history_days=variant["min_history_days"],  # type: ignore[arg-type]
        refit_every=args.refit_every,
        objective=variant["objective"],  # type: ignore[arg-type]
        seed=args.seed,
        source=CapacitySource(args.source),
        method=CapacityExtrapolation(args.method),
    )

    elapsed = time.monotonic() - t0
    logger.info("walk-forward finished in %.0fs, %d rows", elapsed, len(predictions))

    metric_rows: list[dict[str, object]] = []
    for target in ProductionType:
        raw = target_hourly[TARGET_COLUMNS[target]]
        available = raw.notna()
        full_label = capacity_factor_label(
            raw[available],
            pd.DatetimeIndex(raw.index[available]),
            target,
            source=CapacitySource(args.source),
            method=CapacityExtrapolation(args.method),
        )
        full_label_flat = pd.Series(
            full_label.to_numpy(), index=pd.DatetimeIndex(raw.index[available])
        )
        metric_rows.extend(_metrics_for_target(predictions, target, full_label_flat))
    metrics_df = pd.DataFrame(metric_rows)

    logger.info(
        "PRIMARY METRIC: RMSE (spec 2.10). MAE is a secondary metric, reported but not decisive."
    )
    overall = metrics_df[metrics_df["breakdown"] == "overall"]
    for row in overall.itertuples():
        logger.info(
            "%-14s RMSE(cf) model=%.4f baseline=%.4f | RMSE(MW) model=%.1f baseline=%.1f | MAE(cf) model=%.4f baseline=%.4f",
            row.target,
            row.rmse_cf_model,
            row.rmse_cf_baseline,
            row.rmse_mw_model,
            row.rmse_mw_baseline,
            row.mae_cf_model,
            row.mae_cf_baseline,
        )

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not _RESULTS_PATH.exists()
    metrics_df.insert(0, "variant", args.variant)
    metrics_df.to_csv(_RESULTS_PATH, mode="a" if not header else "w", header=header, index=False)
    logger.info("appended %d metric rows to %s", len(metrics_df), _RESULTS_PATH)

    out_path = _OUT_DIR / f"renewables_forecast_{args.variant}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.astype("float32", errors="ignore").to_parquet(out_path)
    logger.info("wrote %d rows to %s", len(predictions), out_path)

    log_params: dict[str, object] = {
        "variant": args.variant,
        "window": variant["window"],
        "train_span_days": variant["train_span_days"],
        "min_history_days": variant["min_history_days"],
        "objective": variant["objective"],
        "refit_every": args.refit_every,
        "seed": args.seed,
        "source": args.source,
        "method": args.method,
        **DEFAULT_LGBM_PARAMS,
    }
    log_tags: dict[str, str] = {"study": f"renewables_{args.variant}", "note": args.note}
    log_metrics: dict[str, float] = {
        f"{row.target}_{metric}": getattr(row, metric)
        for row in overall.itertuples()
        for metric in (
            "rmse_cf_model",
            "rmse_mw_model",
            "mae_cf_model",
            "mae_mw_model",
            "rmse_cf_baseline",
            "rmse_mw_baseline",
        )  # fmt: skip
    }
    log_metrics["elapsed_seconds"] = elapsed

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"renewables_{args.variant}"):
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(out_path))
        mlflow.log_artifact(str(_RESULTS_PATH))

    logger.info("done in %.0fs", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
