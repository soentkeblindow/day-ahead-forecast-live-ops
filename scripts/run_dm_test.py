"""Sprint 5.6: reproducible Diebold-Mariano significance test on persisted predictions.

Thin I/O layer only -- all statistics live in `energy_price_forecast.evaluation.dm_test`
(pure functions, tested against hand computations in `tests/test_dm_test.py`). This
script loads already PERSISTED Sprint 2/3 predictions, computes MAE loss differentials
(LightGBM vs. Lasso, LightGBM vs. ARIMAX), and writes `outputs/results/dm_test.csv`.

Reads ONLY persisted files -- no model runs, no MLflow queries at runtime. If an
input is missing, fails fast and loud, naming the command that produces it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.dm_test import daily_mean_loss, dm_test

log = logging.getLogger(__name__)

_MIN_INTERSECTION_FRACTION = 0.99

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

_COLUMNS: tuple[str, ...] = (
    "comparison",
    "variant",
    "n_obs",
    "hac_lag",
    "horizon",
    "mean_loss_diff_eur_mwh",
    "dm_stat",
    "p_value",
)

# (filename, producing command) for each required prediction input.
_INPUTS: dict[str, tuple[str, str]] = {
    "lgbm": (
        "preds_lgbm_q50.parquet",
        "uv run python scripts/backtest.py --model lgbm --alpha 0.5",
    ),
    "lasso": (
        "backtest_lasso.parquet",
        "uv run python scripts/backtest.py --model lasso",
    ),
    "arimax": (
        "preds_arimax_v2_q50.parquet",
        "uv run python scripts/backtest.py --model arimax --study arimax_benchmark_v2",
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diebold-Mariano test on persisted point predictions "
            "(LightGBM vs. Lasso, LightGBM vs. ARIMAX)."
        )
    )
    p.add_argument("--price-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--pred-lgbm", default=Path("data/processed/preds_lgbm_q50.parquet"), type=Path)
    p.add_argument("--pred-lasso", default=Path("data/processed/backtest_lasso.parquet"), type=Path)
    p.add_argument(
        "--pred-arimax", default=Path("data/processed/preds_arimax_v2_q50.parquet"), type=Path
    )
    p.add_argument("--out", default=Path("outputs/results/dm_test.csv"), type=Path)
    return p.parse_args()


def _require_inputs(paths: dict[str, Path]) -> None:
    missing: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            filename, producer = _INPUTS[name]
            missing.append(f"  {name} ({filename}): {path} (run: {producer})")
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))


def _load_pred(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if "y_pred" not in frame.columns:
        raise ValueError(f"{path}: missing expected column 'y_pred' (got {list(frame.columns)})")
    return frame["y_pred"]


def _run_dm_pair(
    name_a: str, loss_a: pd.Series, name_b: str, loss_b: pd.Series
) -> list[dict[str, object]]:
    comparison = f"{name_a}_vs_{name_b}"
    rows: list[dict[str, object]] = []

    daily_a, n_incomplete_a = daily_mean_loss(loss_a)
    daily_b, n_incomplete_b = daily_mean_loss(loss_b)
    log.info(
        "%s: %d incomplete day(s) for %s, %d for %s",
        comparison,
        n_incomplete_a,
        name_a,
        n_incomplete_b,
        name_b,
    )
    daily_result = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    rows.append({"comparison": comparison, "variant": "daily", **asdict(daily_result)})

    hourly_result = dm_test(loss_a, loss_b, hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON)
    rows.append({"comparison": comparison, "variant": "hourly", **asdict(hourly_result)})

    for row in rows:
        mean_loss_diff = float(row["mean_loss_diff"])  # type: ignore[arg-type]
        if mean_loss_diff > 0:
            log.warning(
                "%s (%s): mean_loss_diff is POSITIVE (%.4f) -- %s does NOT have the lower "
                "expected loss here; reporting unchanged.",
                comparison,
                row["variant"],
                mean_loss_diff,
                name_a,
            )
    return rows


def _run(args: argparse.Namespace) -> None:
    pred_paths = {"lgbm": args.pred_lgbm, "lasso": args.pred_lasso, "arimax": args.pred_arimax}
    _require_inputs(pred_paths)

    preds = {name: _load_pred(path) for name, path in pred_paths.items()}

    common_index = preds["lgbm"].index
    for series in preds.values():
        common_index = common_index.intersection(series.index)
    if len(common_index) == 0:
        raise ValueError("Empty index intersection across the three prediction series.")

    smallest = min(len(s) for s in preds.values())
    if len(common_index) < _MIN_INTERSECTION_FRACTION * smallest:
        log.warning(
            "Index intersection (%d rows) is below %.0f%% of the smallest input series (%d rows).",
            len(common_index),
            _MIN_INTERSECTION_FRACTION * 100,
            smallest,
        )
    for name, series in preds.items():
        log.info("%s: %d rows (intersection: %d)", name, len(series), len(common_index))

    price = load_interim_hourly(args.price_path)["day_ahead_price"].reindex(common_index)
    losses = {name: (series.reindex(common_index) - price).abs() for name, series in preds.items()}

    rows: list[dict[str, object]] = []
    rows += _run_dm_pair("lightgbm", losses["lgbm"], "lasso", losses["lasso"])
    rows += _run_dm_pair("lightgbm", losses["lgbm"], "arimax", losses["arimax"])

    table = pd.DataFrame(rows).rename(columns={"mean_loss_diff": "mean_loss_diff_eur_mwh"})
    table = table[list(_COLUMNS)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.out, len(table))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run(_parse_args())


if __name__ == "__main__":
    main()
