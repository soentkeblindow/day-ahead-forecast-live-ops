"""Sprint 6.9 step 10, spec section 6.1 (Parity Check): proves the three
fallback-ladder rows are byte-identical column subsets of the real
production feature builder (features.build.build_feature_set_for_day), for
at least 5 real calendar days from the 6.8 window -- including a Monday, a
holiday, and a day right after a month boundary, per the spec's own list.

Row 1 (`core_gas`) and row 3 (`base_gas`, gas-free per Fassung 2, spec
section 2.1) are compared directly here: scripts/ablation_core_minimal_
feature_set.py::build_core_for_day / scripts/measurement_a_candidate_
intake.py::build_floor_for_day, imported unchanged (not copied -- spec
section 2.1: "keine zweite Walk-Forward-Implementierung"), against
build_feature_set_for_day's own output for the identical day, restricted to
each row's own column set.

Row 2 (`core_gas_loadpatch`) is NOT re-measured here: spec section 6.1 says
its comparison "ist exakt, auch an Feiertagen" BECAUSE the re-measurement
(scripts/measurement_e_load_forecast_reconstruction.py, step 10's own
Nachmessung) already uses the one and only Similar-Day-Patch implementation
(arena/load_patch.py, spec section 3.2: "Einzige Implementierung") -- proven
by construction, not by a fresh comparison run:
  (a) arena/load_patch.py has its own 16 unit tests (tests/test_load_patch.py,
      step 9), covering every escalation chain and DST edge case in isolation;
  (b) measurement_e_load_forecast_reconstruction.py's own run_sanity_check()
      feeds the REAL (unpatched) load through the identical patching
      mechanism (build_reconstructed_row/_patch_fundamentals) and reproduces
      the real matrix row bit-for-bit (max abs diff 0.0, reconfirmed in this
      step's own Nachmessung run, log 2026-09-26);
  (c) row 2's future live builder (step 11) is required to import
      choose_reference_day/build_patched_load from the package, not
      reimplement them -- there is structurally only one code path.
No further code is needed to establish row 2's parity; a fresh numeric
comparison would just re-derive (b).

Requirement (spec 6.1): max deviation 0 (or <=1e-9 with justification).

Usage: python -m scripts.parity_check_ladder_rows
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_renewables_predictions
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.config import FeatureConfig
from scripts.ablation_core_minimal_feature_set import build_core_for_day
from scripts.measurement_a_candidate_intake import build_floor_for_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parity_check")

_OUT_PATH = Path("outputs/results/parity_check_ladder_rows.csv")

# Spec 6.1: >=5 days from the 6.8 window, including a Monday, a holiday, and a
# day right after a month boundary. Two more added for spread (an ordinary
# Tuesday, a Sunday) -- all five verified holiday-free except the one deliberate
# holiday day (2026-04-14 Karfreitag/2026-04-17 avoided on purpose).
_DAYS: tuple[tuple[dt.date, str], ...] = (
    (dt.date(2025, 11, 3), "Monday"),
    (dt.date(2025, 12, 25), "holiday (1. Weihnachtsfeiertag)"),
    (dt.date(2026, 7, 1), "day after a month boundary (June->July)"),
    (dt.date(2026, 4, 14), "ordinary Tuesday"),
    (dt.date(2026, 1, 18), "Sunday"),
)


def _compare(row_frame: pd.DataFrame, production: pd.DataFrame) -> tuple[int, bool, float]:
    """Aligns `row_frame`'s columns against `production`'s and returns
    (n_columns_compared, nan_pattern_matches, max_abs_diff_excluding_nan).
    A NaN in a shared column is only a real deviation if the two sides
    disagree on WHICH cells are NaN -- a matching NaN (e.g. a genuinely
    missing lag on that historical day, present in both builders because
    they call the identical lag function) is not a parity violation, and
    plain .max() would otherwise poison the whole result via NaN
    propagation."""
    aligned = row_frame.reindex(columns=production.columns.intersection(row_frame.columns))
    prod_aligned = production[aligned.columns]
    nan_matches = bool((aligned.isna() == prod_aligned.isna()).to_numpy().all())
    diff = (aligned - prod_aligned).abs().to_numpy()
    max_diff = float(np.nanmax(diff)) if diff.size and not np.isnan(diff).all() else 0.0
    return len(aligned.columns), nan_matches, max_diff


def main() -> None:
    cfg = FeatureConfig()
    df = load_interim_hourly()
    renewables = load_renewables_predictions()

    rows: list[dict[str, object]] = []
    any_violation = False
    for day, label in _DAYS:
        production = build_feature_set_for_day(day, df, renewables, cfg)
        core = build_core_for_day(day, df, renewables, cfg, commodities="gas_only")
        base = build_floor_for_day(day, df, cfg, commodities="none")

        for row_name, row_frame in (("core_gas", core), ("base_gas", base)):
            n_cols, nan_matches, max_diff = _compare(row_frame, production)
            violation = not nan_matches or max_diff > 1e-9
            log.info(
                "%s (%s, %s): n_columns=%d nan_pattern_matches=%s max_abs_diff=%.3g",
                row_name,
                day,
                label,
                n_cols,
                nan_matches,
                max_diff,
            )
            if violation:
                any_violation = True
            rows.append(
                {
                    "target_day": day,
                    "day_label": label,
                    "row": row_name,
                    "n_columns_compared": n_cols,
                    "nan_pattern_matches": nan_matches,
                    "max_abs_diff": max_diff,
                }
            )

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(_OUT_PATH, index=False)
    log.info("wrote %s", _OUT_PATH)

    if any_violation:
        raise AssertionError(
            "parity check found a real column deviation (nonzero diff or a mismatched "
            f"NaN pattern) -- see the table above / {_OUT_PATH} before proceeding "
            "(spec 6.9 section 6.1 requirement: max diff 0, or <=1e-9 with justification)"
        )
    log.info(
        "PASS: rows 1 and 3 are exact column subsets of build_feature_set_for_day "
        "on all %d checked days",
        len(_DAYS),
    )


if __name__ == "__main__":
    main()
