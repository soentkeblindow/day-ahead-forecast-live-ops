"""Sprint 6 / Step 6.6 section 5.4: the MW-level number from the discontinued
6.5.4 (spec Entscheidung 14) -- this project's own residual-load
reconstruction against the TSO original, in MW rather than capacity factor,
so the external write-up has a physical-unit number, not just capacity
factors.

Pure evaluation, no model code (same pattern as scripts/check_nwp_coverage.py
from 6.5.3, also untested for the same reason -- a thin comparison script,
not a decision the live gate depends on).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_renewables_predictions

log = logging.getLogger(__name__)

_PAIRS = (
    ("residual_load_forecast", "residual_load_forecast_nwp"),
    ("wind_onshore_forecast", "wind_onshore_forecast_nwp"),
    ("wind_offshore_forecast", "wind_offshore_forecast_nwp"),
    ("solar_forecast", "solar_forecast_nwp"),
)
_MW_PRED_COLUMNS = {
    "wind_onshore_forecast_nwp": "wind_onshore_mw_pred",
    "wind_offshore_forecast_nwp": "wind_offshore_mw_pred",
    "solar_forecast_nwp": "solar_mw_pred",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hourly-path", default=None, type=Path)
    p.add_argument("--renewables-path", default=None, type=Path)
    p.add_argument("--tz", default="Europe/Berlin")
    p.add_argument(
        "--out", default=Path("outputs/results/residual_load_reconstruction_mw.csv"), type=Path
    )
    return p.parse_args()


def _build_comparison_frame(hourly: pd.DataFrame, renewables: pd.DataFrame) -> pd.DataFrame:
    """Original (TSO) and NWP-reconstructed values, aligned on the common
    hourly window both series cover -- reconstructed values indexed by
    valid_time_utc, not run_init_utc: a physical comparison at delivery
    time, not at forecast-issue time (spec 6.6 section 5.4)."""
    valid_time = pd.DatetimeIndex(renewables.index.get_level_values("valid_time_utc"))
    nwp = renewables.set_axis(valid_time)[list(_MW_PRED_COLUMNS.values())]
    nwp = nwp[~nwp.index.duplicated(keep="first")].sort_index()

    common_index = hourly.index.intersection(nwp.index)
    hourly_c = hourly.loc[common_index]
    nwp_c = nwp.loc[common_index]

    frame = pd.DataFrame(index=common_index)
    frame["wind_onshore_forecast"] = hourly_c["wind_onshore_forecast"]
    frame["wind_onshore_forecast_nwp"] = nwp_c["wind_onshore_mw_pred"]
    frame["wind_offshore_forecast"] = hourly_c["wind_offshore_forecast"]
    frame["wind_offshore_forecast_nwp"] = nwp_c["wind_offshore_mw_pred"]
    frame["solar_forecast"] = hourly_c["solar_forecast"]
    frame["solar_forecast_nwp"] = nwp_c["solar_mw_pred"]

    load = hourly_c["load_forecast_day_ahead"]
    frame["residual_load_forecast"] = (
        load
        - frame["wind_onshore_forecast"]
        - frame["wind_offshore_forecast"]
        - frame["solar_forecast"]
    )
    frame["residual_load_forecast_nwp"] = (
        load
        - frame["wind_onshore_forecast_nwp"]
        - frame["wind_offshore_forecast_nwp"]
        - frame["solar_forecast_nwp"]
    )
    return frame.dropna()


def _error_metrics(frame: pd.DataFrame, original: str, nwp: str) -> dict[str, float]:
    error = frame[nwp] - frame[original]
    return {
        "mae": float(error.abs().mean()),
        "rmse": float((error**2).mean() ** 0.5),
        "me": float(error.mean()),
        "n": int(len(frame)),
    }


def _daytime_label(hour: int) -> str:
    return "day" if 6 <= hour < 18 else "night"


def _comparison_table(frame: pd.DataFrame, tz: str) -> pd.DataFrame:
    local = pd.DatetimeIndex(frame.index).tz_convert(tz)
    daytime = pd.Series([_daytime_label(h) for h in local.hour], index=frame.index)
    month = pd.Series(local.month, index=frame.index)

    rows: list[dict[str, object]] = []
    for original, nwp in _PAIRS:
        rows.append(
            {
                "pair": original,
                "breakdown": "overall",
                "group": "all",
                **_error_metrics(frame, original, nwp),
            }
        )
        for label, sub in frame.groupby(daytime):
            rows.append(
                {
                    "pair": original,
                    "breakdown": "daytime",
                    "group": label,
                    **_error_metrics(sub, original, nwp),
                }
            )
        for label_month, sub in frame.groupby(month):
            rows.append(
                {
                    "pair": original,
                    "breakdown": "month",
                    "group": f"{label_month:02d}",
                    **_error_metrics(sub, original, nwp),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    hourly = load_interim_hourly(args.hourly_path) if args.hourly_path else load_interim_hourly()
    renewables = (
        load_renewables_predictions(args.renewables_path)
        if args.renewables_path
        else load_renewables_predictions()
    )

    frame = _build_comparison_frame(hourly, renewables)
    log.info(
        "comparison window: %s -> %s (%d rows)", frame.index.min(), frame.index.max(), len(frame)
    )

    table = _comparison_table(frame, args.tz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as f:
        f.write("# me = mean(nwp - original); positive means the reconstruction over-predicts.\n")
        table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.out, len(table))

    overall = table.loc[table["breakdown"] == "overall"]
    for _, row in overall.iterrows():
        log.info(
            "pair=%s overall: mae=%.1f rmse=%.1f me=%+.1f n=%d",
            row["pair"],
            row["mae"],
            row["rmse"],
            row["me"],
            row["n"],
        )


if __name__ == "__main__":
    main()
