"""Energy-Charts backup Auftrag (docs/sprint6_auftrag_energy_charts_backup_2.md,
Teil 2): value-agreement check between the four Energy-Charts
``public_power_forecast`` day-ahead series (Teil 1's historical pull,
``scripts/fetch_energy_charts_forecast_history.py``) and this project's own
store-cached ENTSO-E series, over their common window.

Grid alignment: none needed. Both sides were measured (not assumed) to be
native quarter-hourly over the entire compared window -- the store's own
``load``/``wind_solar`` cache directories carry distinct (non-repeated)
values on every 15-minute slot back to 2020, unlike ``day_ahead_price``,
whose quarter-hourly era only starts 2025-09-30 (data/quarterhourly.py::
QUARTERHOUR_START). The two series are compared directly on their
timestamp intersection, no resampling in either direction.

Extended for Sprint 6.8, Schritt 0 (docs/sprint6_step6_8_spec.md section 3):
also compares Energy-Charts day-ahead prices (scripts/fetch_energy_charts_
price_history.py) against the store's own ``day_ahead_price``. Same
intersection-only comparison, no special handling needed for the 2025-09-30
resolution transition -- both sides are hourly before it and quarter-hourly
after it, so the timestamp intersection is well-defined throughout. The
whole 6.8 outage-response plan assumes "EC price == ENTSO-E price"; this is
where that assumption gets measured rather than taken on faith (spec: "mit
diesem Abgleich ist sie gemessen statt angenommen").

Pure evaluation script, no model code, not run in CI and not gating the
live path (same class as scripts/compare_residual_load_mw.py /
scripts/check_nwp_coverage.py) -- untested for the same reason.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.config import DATA_RAW, PROJECT_ROOT
from energy_price_forecast.data.energy_charts_probe import SERIES
from energy_price_forecast.ops.store import read_cached_range
from energy_price_forecast.ops.store_sources import ENTSOE_SOURCES
from energy_price_forecast.ops.windows import LOCAL_TZ

log = logging.getLogger(__name__)

_EC_DIR = DATA_RAW / "energy_charts" / "public_power_forecast"
_EC_PRICE_DIR = DATA_RAW / "energy_charts" / "price"
_EC_PRICE_COLUMN = "day_ahead_price_ec"
_PRICE_LABEL = "day_ahead_price"
_PRICE_SOURCE = next(s for s in ENTSOE_SOURCES if s.name == "day_ahead_price")

# (Energy-Charts production_type) -> (store cache_dir, store column). "load"
# and "wind_solar" are two distinct ENTSO-E fetch groups sharing one cache
# dir each (ops/store_sources.py::ENTSOE_SOURCES) -- wind_onshore/_offshore/
# solar all read the same "wind_solar" directory.
_STORE_BINDINGS: dict[str, tuple[Path, str]] = {
    "load": (DATA_RAW / "entsoe" / "load", "load_forecast_day_ahead"),
    "solar": (DATA_RAW / "entsoe" / "wind_solar", "solar_forecast"),
    "wind_onshore": (DATA_RAW / "entsoe" / "wind_solar", "wind_onshore_forecast"),
    "wind_offshore": (DATA_RAW / "entsoe" / "wind_solar", "wind_offshore_forecast"),
}


def read_ec_series(production_type: str, *, root: Path = _EC_DIR) -> pd.Series:
    """Concatenate Teil 1's monthly Parquet files for one series back into a
    single UTC-indexed series. Raises if the on-disk files themselves
    contain duplicate timestamps -- a silent overlap between two monthly
    files would be a Teil 1 bug, not something to paper over here."""
    files = sorted(root.glob(f"{production_type}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no Energy-Charts history files for {production_type!r} under {root} -- "
            "run scripts/fetch_energy_charts_forecast_history.py first"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(
            f"{production_type}: {dupe_count} duplicate timestamp(s) across monthly EC files"
        )
    return frame[production_type]


def _missing_calendar_days(
    have: pd.DatetimeIndex, window_start: pd.Timestamp, window_end: pd.Timestamp, tz: str
) -> int:
    """Local calendar days within [window_start, window_end] with zero
    timestamps in ``have`` -- a day "without overlap" per the Auftrag's own
    wording, not a per-slot gap count. The window is the two sources'
    shared candidate range (EC's own archive start through the earlier of
    the two maxima); the store's 2020-2024 pre-archive history is not "EC
    coverage missing" by any meaningful definition, since Teil 1 never asked
    EC for that range at all."""
    have_days = set(pd.DatetimeIndex(have).tz_convert(tz).date)
    local_start = window_start.tz_convert(tz)
    local_end = window_end.tz_convert(tz)
    all_days = pd.date_range(local_start, local_end, freq="D").date
    return len(set(all_days) - have_days)


def _diff_stats(ec: pd.Series, store: pd.Series) -> dict[str, float]:
    diff = ec - store
    return {
        "n": int(len(diff)),
        "exact_fraction": float((ec == store).mean()) if len(diff) else float("nan"),
        "diff_median": float(diff.median()) if len(diff) else float("nan"),
        "diff_p95_abs": float(diff.abs().quantile(0.95)) if len(diff) else float("nan"),
        "diff_max_abs": float(diff.abs().max()) if len(diff) else float("nan"),
        "corr": float(ec.corr(store)) if len(diff) > 1 else float("nan"),
    }


def _agreement_rows(label: str, ec: pd.Series, store: pd.Series) -> list[dict[str, object]]:
    common = ec.index.intersection(store.index)
    pair = pd.DataFrame({"ec": ec.loc[common], "store": store.loc[common]}).dropna()

    window_start = ec.index.min()
    window_end = min(ec.index.max(), store.index.max())
    rows: list[dict[str, object]] = [
        {
            "series": label,
            "breakdown": "overall",
            "group": "all",
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_days_missing_in_store": _missing_calendar_days(
                pd.DatetimeIndex(store.index), window_start, window_end, LOCAL_TZ
            ),
            "n_days_missing_in_ec": _missing_calendar_days(
                pd.DatetimeIndex(ec.index), window_start, window_end, LOCAL_TZ
            ),
            **_diff_stats(pair["ec"], pair["store"]),
        }
    ]

    year = pd.DatetimeIndex(pair.index).tz_convert(LOCAL_TZ).year
    for group_label, sub in pair.groupby(year):
        rows.append(
            {
                "series": label,
                "breakdown": "year",
                "group": str(group_label),
                "n_days_missing_in_store": None,
                "n_days_missing_in_ec": None,
                **_diff_stats(sub["ec"], sub["store"]),
            }
        )
    return rows


def _series_report(production_type: str) -> list[dict[str, object]]:
    store_dir, store_col = _STORE_BINDINGS[production_type]
    ec = read_ec_series(production_type)
    store = read_cached_range(store_dir)[store_col]
    return _agreement_rows(production_type, ec, store)


def read_ec_price_series(*, root: Path = _EC_PRICE_DIR) -> pd.Series:
    """Concatenate scripts/fetch_energy_charts_price_history.py's monthly
    Parquet files back into a single UTC-indexed series -- same duplicate
    -timestamp guard as read_ec_series (a silent overlap between two
    monthly files would be a fetch-script bug, not something to paper over
    here)."""
    files = sorted(root.glob("DE_LU_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no Energy-Charts price history files under {root} -- run "
            "scripts/fetch_energy_charts_price_history.py first"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if frame.index.has_duplicates:
        dupe_count = int(frame.index.duplicated().sum())
        raise ValueError(f"price: {dupe_count} duplicate timestamp(s) across monthly EC files")
    return frame[_EC_PRICE_COLUMN]


def _price_report() -> list[dict[str, object]]:
    ec = read_ec_price_series()
    store = read_cached_range(_PRICE_SOURCE.cache_dir)[_PRICE_LABEL]
    return _agreement_rows(_PRICE_LABEL, ec, store)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default=PROJECT_ROOT / "outputs" / "results" / "energy_charts_forecast_agreement.csv",
        type=Path,
    )
    p.add_argument(
        "--price",
        action="store_true",
        help="also compare Energy-Charts day-ahead prices against the store's day_ahead_price "
        "(Sprint 6.8, Schritt 0), written to --price-out instead of --out",
    )
    p.add_argument(
        "--price-out",
        default=PROJECT_ROOT / "outputs" / "results" / "energy_charts_price_agreement.csv",
        type=Path,
    )
    return p.parse_args()


def _write_report(table: pd.DataFrame, out: Path, *, header_comment: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="\n") as f:
        f.write(header_comment)
        table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", out, len(table))

    overall = table.loc[table["breakdown"] == "overall"]
    for _, row in overall.iterrows():
        log.info(
            "series=%s overall: n=%d exact_fraction=%.4f diff_median=%+.2f "
            "diff_p95_abs=%.2f diff_max_abs=%.2f corr=%.6f days_missing_in_store=%s "
            "days_missing_in_ec=%s",
            row["series"],
            row["n"],
            row["exact_fraction"],
            row["diff_median"],
            row["diff_p95_abs"],
            row["diff_max_abs"],
            row["corr"],
            row["n_days_missing_in_store"],
            row["n_days_missing_in_ec"],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    rows: list[dict[str, object]] = []
    for production_type in SERIES:
        rows.extend(_series_report(production_type))
    table = pd.DataFrame(rows)
    _write_report(
        table,
        args.out,
        header_comment=(
            "# diff = ec - store; both sides native quarter-hourly, compared "
            "on their raw timestamp intersection, no resampling.\n"
        ),
    )

    if args.price:
        price_table = pd.DataFrame(_price_report())
        _write_report(
            price_table,
            args.price_out,
            header_comment=(
                "# diff = ec - store; day_ahead_price, hourly before 2025-09-30 and "
                "quarter-hourly after on both sides, compared on their raw timestamp "
                "intersection, no resampling.\n"
            ),
        )


if __name__ == "__main__":
    main()
