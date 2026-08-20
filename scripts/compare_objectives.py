"""Sprint 6 / Step 6.3: RMSE-alignment comparison between the inherited
quantile/alpha=0.5 (median) LGBM objective and the new l2 (mean) objective.

Thin I/O and glue layer -- all metric/regime/DM-test logic lives in the
existing, UNCHANGED `evaluation.breakdown`, `evaluation.regimes`, and
`evaluation.dm_test` modules. This script only loads two persisted
prediction parquets, slices them by regime and by time period, and writes
two CSVs:

- outputs/results/objective_comparison.csv: one row per (period, regime),
  the breakdown_point metrics for both runs side by side plus a delta
  column. Convention: delta = l2 - median; a NEGATIVE delta means the l2
  (mean) objective is better on that metric/slice.
- outputs/results/dm_test_objective.csv: Diebold-Mariano significance test
  on the SQUARED-error loss differential (l2 vs. median), daily-block
  primary + hourly robustness check, same two variants as Sprint 5.6's
  run_dm_test.py. Same sign convention: negative mean_loss_diff means l2
  has the lower expected (squared-error) loss.

Both periods (`full` = the whole common index, `recent` = the last
`--recent-months` months of it) are reported, per Step 6.3 §2 point 4: an
effect that only shows up in `full` may be an artefact of the 2022 crisis
and would not help the Arena, which scores forecasts *now*.

No model runs here, no MLflow queries -- reads only persisted parquet and
the interim hourly frame. Fails fast on missing files, malformed columns,
or a median/mean index mismatch (a comparison over different time windows
would be meaningless).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.breakdown import breakdown_point
from energy_price_forecast.evaluation.config import RegimeConfig
from energy_price_forecast.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.evaluation.regimes import tag_regimes

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("y_true", "y_pred", "delivery_day")

# Same two DM-test variants and lag/horizon choices as Sprint 5.6's
# run_dm_test.py, so the two studies are directly comparable.
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

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

_COMPARISON_COLUMNS = ["comparison", "period", "variant", "n_obs", "hac_lag", "horizon"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime x period comparison of the LGBM quantile/median and l2/mean objectives."
    )
    p.add_argument(
        "--median-path", default=Path("data/processed/preds_lgbm_q50.parquet"), type=Path
    )
    p.add_argument("--mean-path", default=Path("data/processed/preds_lgbm_mean.parquet"), type=Path)
    p.add_argument("--interim-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--recent-months", type=int, default=12)
    p.add_argument("--out", default=Path("outputs/results/objective_comparison.csv"), type=Path)
    p.add_argument("--dm-out", default=Path("outputs/results/dm_test_objective.csv"), type=Path)
    return p.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/backtest.py first "
            "(see Step 6.3 §6 for the exact invocation)"
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


def _recent_index(index: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    cutoff = index.max() - pd.DateOffset(months=months)
    return index[index > cutoff]


def _breakdown_for_period(
    median: pd.DataFrame,
    mean: pd.DataFrame,
    flags: pd.DataFrame,
    period: str,
    period_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    median_bd = breakdown_point(median.loc[period_index], flags)
    mean_bd = breakdown_point(mean.loc[period_index], flags)

    rows: list[dict[str, object]] = []
    for regime in median_bd.index:
        row: dict[str, object] = {
            "period": period,
            "regime": regime,
            "axis": median_bd.loc[regime, "axis"],
            "n": median_bd.loc[regime, "n"],
        }
        for metric in _METRIC_KEYS:
            m_median = float(median_bd.loc[regime, metric])
            m_l2 = float(mean_bd.loc[regime, metric])
            row[f"{metric}_median"] = m_median
            row[f"{metric}_l2"] = m_l2
            row[f"{metric}_delta"] = m_l2 - m_median
        rows.append(row)
    return pd.DataFrame(rows)


def _dm_rows_for_period(
    loss_l2: pd.Series, loss_median: pd.Series, period: str
) -> list[dict[str, object]]:
    comparison = "lgbm_mean_vs_lgbm_q50"
    rows: list[dict[str, object]] = []

    daily_l2, n_incomplete_l2 = daily_mean_loss(loss_l2)
    daily_median, n_incomplete_median = daily_mean_loss(loss_median)
    log.info(
        "%s (%s): %d incomplete day(s) for l2, %d for median",
        comparison,
        period,
        n_incomplete_l2,
        n_incomplete_median,
    )
    daily_result = dm_test(daily_l2, daily_median, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    rows.append(
        {"comparison": comparison, "period": period, "variant": "daily", **asdict(daily_result)}
    )

    hourly_result = dm_test(loss_l2, loss_median, hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON)
    rows.append(
        {"comparison": comparison, "period": period, "variant": "hourly", **asdict(hourly_result)}
    )
    return rows


def _log_overall_rows(period_table: pd.DataFrame, period: str) -> None:
    overall = period_table.loc[period_table["regime"] == "overall"].iloc[0]
    for variant, suffix in (("median", "_median"), ("l2", "_l2")):
        metrics_str = ", ".join(f"{m}={overall[f'{m}{suffix}']:.4f}" for m in _METRIC_KEYS)
        log.info("period=%s variant=%s overall: %s", period, variant, metrics_str)


def _run(args: argparse.Namespace) -> None:
    median = _load_predictions(args.median_path)
    mean = _load_predictions(args.mean_path)

    if not median.index.equals(mean.index):
        raise ValueError(
            "median and mean prediction indices differ -- the two runs must cover "
            "an identical index for the comparison to be meaningful "
            f"(median: {len(median)} rows, mean: {len(mean)} rows)"
        )

    interim = load_interim_hourly(args.interim_path)
    flags = tag_regimes(interim, RegimeConfig())

    full_index = pd.DatetimeIndex(median.index)
    periods: dict[str, pd.DatetimeIndex] = {
        "full": full_index,
        "recent": _recent_index(full_index, args.recent_months),
    }

    comparison_tables: list[pd.DataFrame] = []
    dm_rows: list[dict[str, object]] = []
    for period, period_index in periods.items():
        period_table = _breakdown_for_period(median, mean, flags, period, period_index)
        comparison_tables.append(period_table)
        _log_overall_rows(period_table, period)

        loss_median = (median.loc[period_index, "y_pred"] - median.loc[period_index, "y_true"]) ** 2
        loss_l2 = (mean.loc[period_index, "y_pred"] - mean.loc[period_index, "y_true"]) ** 2
        dm_rows += _dm_rows_for_period(loss_l2, loss_median, period)

    table = pd.concat(comparison_tables, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as f:
        f.write(
            "# delta = l2 - median; a NEGATIVE delta means the l2 (mean) objective is better.\n"
        )
        table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.out, len(table))

    dm_table = pd.DataFrame(dm_rows).rename(
        columns={"mean_loss_diff": "mean_loss_diff_sq_eur2_mwh2"}
    )
    dm_table = dm_table[[*_COMPARISON_COLUMNS, "mean_loss_diff_sq_eur2_mwh2", "dm_stat", "p_value"]]
    args.dm_out.parent.mkdir(parents=True, exist_ok=True)
    dm_table.to_csv(args.dm_out, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.dm_out, len(dm_table))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run(_parse_args())


if __name__ == "__main__":
    main()
