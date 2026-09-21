"""Sprint 6.8, Messung B (docs/sprint6_step6_8_spec.md section 5): what does
it cost when individual training days are missing -- mid-window (a past
missing weather run, today an unconditional 90-day lockout) or at the right
edge (an ENTSO-E outage -- the owner's own proposed "train through the last
complete day and apply anyway" delay track)?

Reuses Messung A's window/fold-building and matrices unchanged (imported,
not reimplemented -- "keine zweite Walk-Forward-Implementierung"): the
gap-free reference predictions for `live`/`core_no_commodities` are read
straight from Messung A's own saved parquet
(data/processed/measurement_a_candidate_intake_predictions.parquet) rather
than recomputed, since the code and window are identical and a second
gap-free run would just reproduce the same numbers at the cost of another
~10 minutes.

Nine configurations (spec's own table):
  position=random, k in {1,3,7},      basis=live
  position=edge,   k in {1,3,7,14},   basis=live
  position=random, k=7,               basis=core_no_commodities
  position=edge,   k=7,               basis=core_no_commodities

"random" draws k local calendar days out of the fold's own 90-day rolling
training window, per fold, via a single fixed/logged seed (spec: "mit
festem, protokolliertem Seed gezogen") -- np.random.default_rng(_GAP_SEED),
advanced sequentially across folds in fold order, never reseeded per fold
(that would make every fold drop the identical relative days). "edge" drops
the last k calendar days of the training window (deterministic, no RNG).
The test window and refit_every=1 (section 2, same as Messung A) are
unaffected -- only the rows entering .fit() shrink.

refit_every=1 throughout, same window/folds as Messung A (2025-10-29..the
frozen artifacts' last complete day) -- so B's "lückenlos" comparator really
is Messung A's own live/core_no_commodities run, not a separately-drawn one.

Usage:
  python -m scripts.measurement_b_training_gaps --probe-folds 10   # timing probe
  python -m scripts.measurement_b_training_gaps                    # full run
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from energy_price_forecast.evaluation.dm_test import DMResult, dm_test
from energy_price_forecast.evaluation.metrics import rmse
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import LGBMForecaster
from scripts.measurement_a_candidate_intake import (
    _DAILY_HAC_LAG,
    _DAILY_HORIZON,
    _GATE_P_THRESHOLD,
    _QH_HAC_LAG,
    _QH_HORIZON,
    _RANDOM_STATE,
    _REFIT_EVERY,
    _SHAPE_WINDOW_DAYS,
    _TZ,
    FeatureConfig,
    _prepare,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("measurement_b")

_GAP_SEED = 20260929  # fixed, logged (spec section 5)
_MLFLOW_EXPERIMENT = "live_robustness_6_8"
_REFERENCE_PATH = Path("data/processed/measurement_a_candidate_intake_predictions.parquet")
_OUT_PATH = Path("data/processed/measurement_b_training_gaps_predictions.parquet")

_CONFIGS: tuple[tuple[str, str, int], ...] = (
    ("live", "random", 1),
    ("live", "random", 3),
    ("live", "random", 7),
    ("live", "edge", 1),
    ("live", "edge", 3),
    ("live", "edge", 7),
    ("live", "edge", 14),
    ("core_no_commodities", "random", 7),
    ("core_no_commodities", "edge", 7),
)


def _local_calendar_days(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.DatetimeIndex(index).tz_convert(tz).normalize().unique()
    ).sort_values()


def _days_to_drop(
    train_index: pd.DatetimeIndex, position: str, k: int, rng: np.random.Generator | None
) -> set[dt.date]:
    local_days = _local_calendar_days(train_index, _TZ)
    if position == "edge":
        return {d.date() for d in local_days[-k:]}
    if position == "random":
        assert rng is not None
        chosen = rng.choice(len(local_days), size=k, replace=False)
        return {local_days[i].date() for i in chosen}
    raise ValueError(f"unknown position {position!r}")


def run_gap_config(
    evaluable_folds: list,
    matrix: pd.DataFrame,
    y_hourly: pd.Series,
    prices_qh: pd.Series,
    *,
    position: str,
    k: int,
    seed: int | None,
) -> pd.DataFrame:
    qh_index = pd.DatetimeIndex(prices_qh.index)
    model = LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
    rng = np.random.default_rng(seed) if position == "random" else None

    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        drop_days = _days_to_drop(fold.train_index, position, k, rng)
        train_local_days = pd.DatetimeIndex(fold.train_index).tz_convert(_TZ).normalize()
        keep_mask = ~pd.Series(train_local_days.date, index=fold.train_index).isin(drop_days)
        gapped_train_index = fold.train_index[keep_mask.to_numpy()]

        y_train = y_hourly.reindex(gapped_train_index)
        x_train = matrix.reindex(gapped_train_index).dropna()
        model.fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[gapped_train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        x_test = matrix.loc[fold.test_index]
        hourly_pred = model.predict(fold.test_index, history=history, x_test=x_test)
        pred_qh = expand_to_quarterhour(hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ)

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    "pred": pred_qh.reindex(y_true_day.index).to_numpy(),
                    "delivery_day": fold.delivery_day,
                    "n_dropped_days": len(drop_days),
                },
                index=y_true_day.index,
            )
        )
        if (i + 1) % max(1, n_folds // 5) == 0 or i == n_folds - 1:
            log.info("  progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records).sort_index()


def _run_dm(
    loss_a: pd.Series, loss_b: pd.Series, delivery_day: pd.Series
) -> tuple[DMResult, DMResult]:
    native = dm_test(loss_a, loss_b, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
    daily_a = loss_a.groupby(delivery_day).mean()
    daily_b = loss_b.groupby(delivery_day).mean()
    daily = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    return native, daily


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-folds", type=int, default=None)
    return p.parse_args()


def main() -> None:
    t0 = time.monotonic()
    args = _parse_args()
    cfg = FeatureConfig()

    y_hourly, matrices, prices_qh, evaluable_folds, end_day = _prepare(cfg)
    reference = pd.read_parquet(_REFERENCE_PATH)
    reference_days = set(pd.DatetimeIndex(reference["delivery_day"]).unique())
    fold_days = {fold.delivery_day for fold in evaluable_folds}
    if not fold_days.issubset(reference_days):
        raise ValueError(
            "Messung A's saved reference predictions do not cover Messung B's own fold set -- "
            "re-run scripts/measurement_a_candidate_intake.py first (section 2: identical window "
            "across all 6.8 measurements)"
        )

    folds_to_run = evaluable_folds[: args.probe_folds] if args.probe_folds else evaluable_folds
    log.info(
        "window end (frozen artifacts): %s, %d folds to run per config", end_day, len(folds_to_run)
    )

    results: dict[str, pd.DataFrame] = {}
    per_config_seconds: dict[str, float] = {}
    for basis, position, k in _CONFIGS:
        label = f"{basis}__{position}_{k}"
        log.info("running config %s ...", label)
        cfg_t0 = time.monotonic()
        seed = _GAP_SEED if position == "random" else None
        results[label] = run_gap_config(
            folds_to_run, matrices[basis], y_hourly, prices_qh, position=position, k=k, seed=seed
        )
        per_config_seconds[label] = time.monotonic() - cfg_t0

    if args.probe_folds:
        total_probe_s = sum(per_config_seconds.values())
        per_fold_s = total_probe_s / (len(folds_to_run) * len(_CONFIGS))
        estimated_total = per_fold_s * len(evaluable_folds) * len(_CONFIGS)
        log.info(
            "PROBE: %d folds x %d configs took %.1fs total -- extrapolated full run (%d folds x "
            "%d configs): %.0fs (%.1f min, %.2f h)",
            len(folds_to_run),
            len(_CONFIGS),
            total_probe_s,
            len(evaluable_folds),
            len(_CONFIGS),
            estimated_total,
            estimated_total / 60,
            estimated_total / 3600,
        )
        return

    out = pd.concat(results, names=["config"])
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT_PATH)

    print(f"\ngap seed={_GAP_SEED}  n_folds={len(folds_to_run)}  refit_every={_REFIT_EVERY}\n")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    log_metrics: dict[str, float] = {}
    summary_rows: list[dict[str, object]] = []
    with mlflow.start_run(run_name="measurement_b_training_gaps"):
        mlflow.log_params(
            {"gap_seed": _GAP_SEED, "n_folds": len(folds_to_run), "refit_every": _REFIT_EVERY}
        )
        mlflow.set_tags({"spec": "sprint6_step6_8", "measurement": "B"})

        for basis, position, k in _CONFIGS:
            label = f"{basis}__{position}_{k}"
            variant = results[label]
            ref = reference.loc[variant.index]

            loss_variant = (variant["pred"] - variant["y_true"]) ** 2
            loss_gapfree = (ref[f"pred_{basis}"] - ref["y_true"]) ** 2
            loss_baseline = (ref["pred_baseline"] - ref["y_true"]) ** 2

            rmse_variant = rmse(variant["y_true"], variant["pred"])
            rmse_gapfree = rmse(ref["y_true"], ref[f"pred_{basis}"])
            rmse_baseline = rmse(ref["y_true"], ref["pred_baseline"])

            native_vs_gapfree, daily_vs_gapfree = _run_dm(
                loss_variant, loss_gapfree, variant["delivery_day"]
            )
            native_vs_baseline, daily_vs_baseline = _run_dm(
                loss_variant, loss_baseline, variant["delivery_day"]
            )

            gate_pass = (
                rmse_variant < rmse_baseline
                and native_vs_baseline.mean_loss_diff < 0
                and native_vs_baseline.p_value < _GATE_P_THRESHOLD
            )
            print(
                f"{label:32s} RMSE={rmse_variant:8.4f} (gapfree={rmse_gapfree:.4f}, "
                f"diff={rmse_variant - rmse_gapfree:+.4f})  vs-gapfree p={native_vs_gapfree.p_value:.4f}  "
                f"vs-baseline p={native_vs_baseline.p_value:.4f}  gate={'PASS' if gate_pass else 'FAIL'}"
            )
            summary_rows.append(
                {
                    "config": label,
                    "basis": basis,
                    "position": position,
                    "k": k,
                    "rmse_variant": rmse_variant,
                    "rmse_gapfree": rmse_gapfree,
                    "rmse_diff": rmse_variant - rmse_gapfree,
                    "native_vs_gapfree_p": native_vs_gapfree.p_value,
                    "native_vs_gapfree_loss_diff": native_vs_gapfree.mean_loss_diff,
                    "daily_vs_gapfree_p": daily_vs_gapfree.p_value,
                    "native_vs_baseline_p": native_vs_baseline.p_value,
                    "native_vs_baseline_loss_diff": native_vs_baseline.mean_loss_diff,
                    "daily_vs_baseline_p": daily_vs_baseline.p_value,
                    "gate_pass": gate_pass,
                }
            )
            log_metrics[f"{label}_rmse"] = rmse_variant
            log_metrics[f"{label}_rmse_diff_vs_gapfree"] = rmse_variant - rmse_gapfree
            log_metrics[f"{label}_vs_gapfree_native_p"] = native_vs_gapfree.p_value
            log_metrics[f"{label}_vs_gapfree_native_loss_diff"] = native_vs_gapfree.mean_loss_diff
            log_metrics[f"{label}_vs_baseline_native_p"] = native_vs_baseline.p_value
            mlflow.set_tag(f"{label}_gate_verdict", "PASS" if gate_pass else "FAIL")

        mlflow.log_metrics(log_metrics)
        summary_path = Path("outputs/results/measurement_b_training_gaps_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(_OUT_PATH))

    elapsed = time.monotonic() - t0
    print(f"\nelapsed: {elapsed:.0f}s")
    print(f"logged to mlflow experiment {_MLFLOW_EXPERIMENT!r}, summary at {summary_path}")


if __name__ == "__main__":
    main()
