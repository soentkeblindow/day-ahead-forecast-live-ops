"""Sprint 6 / Step 6.4: bridge-vs-baseline comparison and significance tests.

Thin I/O and glue layer (spec 6.4 section 5.5) -- all metric logic lives in
the existing, UNCHANGED evaluation.metrics.summarise and evaluation.dm_test
modules. Loads one persisted preds_arena.parquet (written by
scripts/backtest_arena.py) and writes two CSVs:

- outputs/results/arena_bridge_backtest.csv: one row per (period, day_type,
  candidate) with MAE/RMSE/WAPE and the per-day MAE distribution, for the
  three candidates (bridge_shape, bridge_flat, baseline), two periods
  (full, post_changeover) and each day type actually present
  (normal/spring_dst/fall_dst, derived from n_slots_in_day -- not hard-coded
  to 96/92, spec section 11: the 100-slot day is not in today's evaluable
  window but is handled the same way once it is).
- outputs/results/dm_test_bridge.csv: Diebold-Mariano test on the SQUARED-
  error loss differential, daily-block primary + native quarter-hourly
  resolution as a robustness check, for both comparisons this step asks
  (bridge_shape vs baseline, bridge_shape vs bridge_flat), per period. Sign
  convention: mean_loss_diff_sq_eur2_mwh2 = loss(candidate) - loss(reference);
  a NEGATIVE value means the candidate has the lower expected loss.

Daily-block loss is aggregated by the ``delivery_day`` column preds_arena.py
already carries (the correct Europe/Berlin local calendar day for that fold)
rather than dm_test.daily_mean_loss's UTC-midnight normalize, which would
misalign daily boundaries against a UTC-indexed quarter-hourly series -- a
deliberate, contained deviation from the exact compare_objectives.py helper
call, not from the module itself (dm_test.py is not modified).

No model runs here, no MLflow queries -- reads only the persisted parquet.
Fails fast on missing files or malformed columns.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from energy_price_forecast.evaluation.dm_test import dm_test
from energy_price_forecast.evaluation.metrics import summarise

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "y_true",
    "pred_bridge_shape",
    "pred_bridge_flat",
    "pred_baseline",
    "delivery_day",
    "n_slots_in_day",
)
_CANDIDATES = ("bridge_shape", "bridge_flat", "baseline")
_DAY_TYPE_LABELS = {96: "normal", 92: "spring_dst", 100: "fall_dst"}
_METRIC_KEYS = (
    "mae",
    "rmse",
    "wape",
    "mae_per_day_mean",
    "mae_per_day_std",
    "mae_per_day_p05",
    "mae_per_day_p50",
    "mae_per_day_p95",
)

# Daily-block: same lag/horizon as 6.3's daily variant (compare_objectives.py).
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
# Native quarter-hourly resolution robustness check: 2-day lag / 1-day
# horizon at 15-minute granularity, the direct analogue of 6.3's hourly
# robustness check (48-hour lag / 24-hour horizon at hourly granularity).
_QH_HAC_LAG = 192
_QH_HORIZON = 96

_UPPER_BOUND_COMMENT = (
    "# UPPER BOUND: run with the full feature set, including DA_FORECAST series that are "
    "NOT available at gate closure (see sprint 6 abstract, section 9). Not a live-capability "
    "claim.\n"
)
_DM_SIGN_COMMENT = (
    "# mean_loss_diff_sq_eur2_mwh2 = loss(candidate) - loss(reference); "
    "a NEGATIVE value means the candidate is better.\n"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bridge-vs-baseline comparison and DM tests (spec 6.4 section 5.5). "
        "UPPER BOUND -- full feature set, not a live-capability claim."
    )
    p.add_argument("--preds-path", default=Path("data/processed/preds_arena.parquet"), type=Path)
    p.add_argument(
        "--changeover-start",
        default="2026-01-01",
        help="First delivery day whose entire training window post-dates the switch to the "
        "quarter-hourly auction product (spec 6.4 section 11); start of the post_changeover "
        "period.",
    )
    p.add_argument("--tz", default="Europe/Berlin")
    p.add_argument("--out", default=Path("outputs/results/arena_bridge_backtest.csv"), type=Path)
    p.add_argument("--dm-out", default=Path("outputs/results/dm_test_bridge.csv"), type=Path)
    return p.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/backtest_arena.py first "
            "(see spec 6.4 section 6 for the exact invocation)"
        )
    frame = pd.read_parquet(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing} (got {list(frame.columns)})"
        )
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError(f"{path}: index must be a UTC tz-aware DatetimeIndex, got {index!r}")
    return frame


def _day_type_label(n_slots: int) -> str:
    return _DAY_TYPE_LABELS.get(n_slots, f"n{n_slots}")


def _metrics_row(
    frame: pd.DataFrame, candidate: str, period: str, day_type: str
) -> dict[str, object]:
    cand_frame = frame[["y_true", "delivery_day"]].assign(y_pred=frame[f"pred_{candidate}"])
    metrics = summarise(cand_frame)
    return {
        "period": period,
        "day_type": day_type,
        "candidate": candidate,
        "n": len(frame),
        **{k: metrics[k] for k in _METRIC_KEYS},
    }


def _period_frames(
    predictions: pd.DataFrame, changeover_start: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    full = predictions
    post = predictions.loc[predictions["delivery_day"] >= changeover_start]
    return {"full": full, "post_changeover": post}


def _comparison_table(predictions: pd.DataFrame, changeover_start: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, period_frame in _period_frames(predictions, changeover_start).items():
        if period_frame.empty:
            log.warning(
                "period=%s has no rows (changeover_start=%s) -- skipping",
                period,
                changeover_start,
            )
            continue
        for candidate in _CANDIDATES:
            rows.append(_metrics_row(period_frame, candidate, period, "overall"))
        for n_slots in sorted(period_frame["n_slots_in_day"].unique().tolist()):
            day_type_frame = period_frame.loc[period_frame["n_slots_in_day"] == n_slots]
            label = _day_type_label(int(n_slots))
            for candidate in _CANDIDATES:
                rows.append(_metrics_row(day_type_frame, candidate, period, label))
    return pd.DataFrame(rows)


def _daily_loss(frame: pd.DataFrame, candidate: str) -> pd.Series:
    loss = (frame[f"pred_{candidate}"] - frame["y_true"]) ** 2
    return loss.groupby(frame["delivery_day"]).mean()


def _dm_rows_for_period(period_frame: pd.DataFrame, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = (
        ("bridge_shape_vs_baseline", "bridge_shape", "baseline"),
        ("bridge_shape_vs_bridge_flat", "bridge_shape", "bridge_flat"),
    )
    for comparison, candidate, reference in comparisons:
        loss_candidate_qh = (period_frame[f"pred_{candidate}"] - period_frame["y_true"]) ** 2
        loss_reference_qh = (period_frame[f"pred_{reference}"] - period_frame["y_true"]) ** 2
        daily_candidate = _daily_loss(period_frame, candidate)
        daily_reference = _daily_loss(period_frame, reference)

        for variant, loss_a, loss_b, hac_lag, horizon in (
            ("quarter_hourly", loss_candidate_qh, loss_reference_qh, _QH_HAC_LAG, _QH_HORIZON),
            ("daily", daily_candidate, daily_reference, _DAILY_HAC_LAG, _DAILY_HORIZON),
        ):
            try:
                result = dm_test(loss_a, loss_b, hac_lag=hac_lag, horizon=horizon)
            except ValueError as exc:
                log.warning("skipping DM test %s/%s/%s: %s", comparison, period, variant, exc)
                continue
            rows.append(
                {"comparison": comparison, "period": period, "variant": variant, **asdict(result)}
            )
    return rows


def _log_overall_rows(table: pd.DataFrame, period: str) -> None:
    overall = table.loc[(table["period"] == period) & (table["day_type"] == "overall")]
    for _, row in overall.iterrows():
        metrics_str = ", ".join(f"{m}={row[m]:.4f}" for m in _METRIC_KEYS)
        log.info("period=%s candidate=%s overall: %s", period, row["candidate"], metrics_str)


def _run(args: argparse.Namespace) -> None:
    predictions = _load_predictions(args.preds_path)
    changeover_start = pd.Timestamp(args.changeover_start, tz=args.tz)

    table = _comparison_table(predictions, changeover_start)
    for period in table["period"].unique():
        _log_overall_rows(table, period)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as f:
        f.write(_UPPER_BOUND_COMMENT)
        table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.out, len(table))

    dm_rows: list[dict[str, object]] = []
    for period, period_frame in _period_frames(predictions, changeover_start).items():
        if period_frame.empty:
            continue
        dm_rows += _dm_rows_for_period(period_frame, period)
    dm_table = pd.DataFrame(dm_rows).rename(
        columns={"mean_loss_diff": "mean_loss_diff_sq_eur2_mwh2"}
    )
    dm_columns = [
        "comparison",
        "period",
        "variant",
        "n_obs",
        "hac_lag",
        "horizon",
        "mean_loss_diff_sq_eur2_mwh2",
        "dm_stat",
        "p_value",
    ]
    dm_table = dm_table[dm_columns] if not dm_table.empty else pd.DataFrame(columns=dm_columns)

    args.dm_out.parent.mkdir(parents=True, exist_ok=True)
    with args.dm_out.open("w", newline="\n") as f:
        f.write(_UPPER_BOUND_COMMENT)
        f.write(_DM_SIGN_COMMENT)
        dm_table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.dm_out, len(dm_table))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run(_parse_args())


if __name__ == "__main__":
    main()
