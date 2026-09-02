"""Seasonal-error check for the renewables headline run (spec 6.5.2a, section 5.1).

Pure evaluation of already-committed results -- no new walk-forward run, no
model code, no change to any input file. Checks expectation 4 from spec
6.5.2 section 11: the absolute error is larger for wind in winter, and for
solar in summer, than in the respective opposite season.

Two inputs, both already produced by the 6.5.2 headline run and read as-is:

- ``outputs/results/renewables_backtest.csv`` for the MAE-in-MW figures
  themselves (the committed, already-verified numbers -- not recomputed
  here, to avoid a second, potentially diverging implementation of the same
  metric).
- ``data/processed/renewables_forecast_rolling365_l2.parquet`` (the
  headline variant's own per-timestamp predictions/actuals, written by
  ``scripts/train_renewables.py``) for mean actual generation per season --
  context the aggregated CSV does not carry, without which a seasonal MAE
  difference is not interpretable (a season with a much higher mean output
  will tend to have a much higher absolute error even absent any true
  win-in-winter/summer effect).

Only the ``rolling365_l2`` variant (spec 6.5.2a section 5.1): expectation 4
is a statement about the model, not a variant comparison -- that comparison
has its own preregistered place, the DM test in 6.5.4.

Not a CI test, same reasoning as ``check_capacity_extrapolation.py``: the
numbers move with every new walk-forward run.

A failing verdict is not a reason to change the model (spec 6.5.2a section
5.1) -- it is a finding to hand to 6.5.4, logged here, not acted on.
"""

from __future__ import annotations

import logging

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.capacity import ProductionType

logger = logging.getLogger(__name__)

_VARIANT = "rolling365_l2"
_TZ = "Europe/Berlin"

_BACKTEST_PATH = PROJECT_ROOT / "outputs" / "results" / "renewables_backtest.csv"
_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / f"renewables_forecast_{_VARIANT}.parquet"
_OUT_PATH = PROJECT_ROOT / "outputs" / "results" / "renewables_seasonal_error.csv"

_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}  # fmt: skip
_SEASONS = ("winter", "spring", "summer", "autumn")

# spec 6.5.2 section 11, expectation 4: which season's absolute error is
# expected to be larger, per target, and against which opposite season.
_EXPECTED_HIGHER_SEASON: dict[ProductionType, str] = {
    ProductionType.SOLAR: "summer",
    ProductionType.WIND_ONSHORE: "winter",
    ProductionType.WIND_OFFSHORE: "winter",
}
_OPPOSITE_SEASON = {"winter": "summer", "summer": "winter"}


def _load_seasonal_mae(target: ProductionType) -> dict[str, tuple[int, float]]:
    """(n, mae_mw_model) per season, for one target, from the committed backtest CSV."""
    backtest = pd.read_csv(_BACKTEST_PATH, comment="#")
    subset = backtest.loc[(backtest["variant"] == _VARIANT) & (backtest["target"] == target.value)]
    result: dict[str, tuple[int, float]] = {}
    for season in _SEASONS:
        row = subset.loc[subset["breakdown"] == f"season_{season}"]
        if len(row) != 1:
            raise ValueError(
                f"expected exactly one row for variant={_VARIANT!r} target={target.value!r} "
                f"breakdown='season_{season}' in {_BACKTEST_PATH}, found {len(row)}"
            )
        result[season] = (int(row["n"].iloc[0]), float(row["mae_mw_model"].iloc[0]))
    return result


def _load_mean_actual_generation() -> dict[ProductionType, dict[str, float]]:
    """Mean actual generation in MW per target and season, from the headline
    variant's own per-timestamp predictions/actuals parquet."""
    predictions = pd.read_parquet(_PREDICTIONS_PATH)
    valid_time = pd.DatetimeIndex(predictions.index.get_level_values("valid_time_utc"))
    season = pd.Series(
        valid_time.tz_convert(_TZ).month.map(_SEASON_BY_MONTH), index=predictions.index
    )

    result: dict[ProductionType, dict[str, float]] = {}
    for target in ProductionType:
        actual = predictions[f"{target.value}_mw_actual"]
        result[target] = {s: float(actual[(season == s).to_numpy()].mean()) for s in _SEASONS}
    return result


def _build_report() -> pd.DataFrame:
    mean_actual = _load_mean_actual_generation()

    rows: list[dict[str, object]] = []
    for target in ProductionType:
        seasonal_mae = _load_seasonal_mae(target)
        higher = _EXPECTED_HIGHER_SEASON[target]
        lower = _OPPOSITE_SEASON[higher]
        expectation = f"mae_mw[{higher}] > mae_mw[{lower}]"
        verdict = "pass" if seasonal_mae[higher][1] > seasonal_mae[lower][1] else "fail"

        for season in _SEASONS:
            n, mae_mw = seasonal_mae[season]
            rows.append(
                {
                    "target": target.value,
                    "season": season,
                    "n": n,
                    "mae_mw": mae_mw,
                    "mean_actual_mw": mean_actual[target][season],
                    "expectation": expectation,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def _log_report(report: pd.DataFrame) -> None:
    logger.info(
        "Seasonal error check (expectation 4, spec 6.5.2 section 11), variant=%s:", _VARIANT
    )
    for target in ProductionType:
        subset = report.loc[report["target"] == target.value]
        expectation = str(subset["expectation"].iloc[0])
        verdict = str(subset["verdict"].iloc[0])
        logger.info("  %s -- expected: %s -- verdict: %s", target.value, expectation, verdict)
        for row in subset.itertuples():
            logger.info(
                "    %-8s n=%-6d mae=%8.2f MW  mean_actual=%8.2f MW",
                row.season,
                row.n,
                row.mae_mw,
                row.mean_actual_mw,
            )

    failed = [
        t.value
        for t in ProductionType
        if report.loc[report["target"] == t.value, "verdict"].iloc[0] == "fail"
    ]
    if failed:
        logger.warning(
            "Expectation 4 did NOT hold for: %s. This is a finding to document and hand to "
            "6.5.4, not a reason to change the model (spec 6.5.2a section 5.1).",
            ", ".join(failed),
        )
    else:
        logger.info("Expectation 4 held for all three targets.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = _build_report()

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(_OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(report), _OUT_PATH)

    _log_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
