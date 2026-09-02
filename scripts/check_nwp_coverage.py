"""Coverage-exclusion check for the NWP reconstruction feature (spec 6.5.3,
section 3.3).

Not a CI test: the numbers move whenever the renewables artefact is
refreshed. Run manually and review the output -- same pattern as
scripts/check_capacity_extrapolation.py and scripts/check_seasonal_error.py.

Loops over every local calendar day whose full 23/24/25-hour range sits
inside the renewables prediction artefact's valid_time_utc coverage, and
counts how many are excluded by the whole-day NaN policy
(features/nwp_fundamentals.py's IncompleteReconstructionError -- the same
check the real feature builder uses, not a reimplementation). Reports the
exclusion rate against the spec's preregistered 2% Rueckfrage threshold.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.loaders import load_renewables_predictions
from energy_price_forecast.features.nwp_fundamentals import (
    IncompleteReconstructionError,
    _check_reconstruction_coverage,
)
from energy_price_forecast.ops.windows import local_day_bounds

logger = logging.getLogger(__name__)

_OUT_PATH = PROJECT_ROOT / "outputs" / "results" / "nwp_coverage_exclusions.csv"
EXCLUSION_RATE_HALT_THRESHOLD = 0.02  # spec 6.5.3 section 3.3: Rueckfrage above this


def _local_hourly_index(target_day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _evaluable_days(predictions: pd.DataFrame) -> list[dt.date]:
    """Local calendar days whose full hourly range is inside the artefact's
    own valid_time_utc coverage -- a day only partially inside the artefact
    isn't a fair test of the NaN policy, it's just an artefact-boundary
    effect."""
    valid_time = pd.DatetimeIndex(predictions.index.get_level_values("valid_time_utc"))
    first, last = valid_time.min(), valid_time.max()
    local_dates = sorted({d.date() for d in valid_time.tz_convert("Europe/Berlin")})
    days: list[dt.date] = []
    for d in local_dates:
        idx = _local_hourly_index(d)
        if idx.min() >= first and idx.max() <= last:
            days.append(d)
    return days


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    predictions = load_renewables_predictions()
    days = _evaluable_days(predictions)
    logger.info("checking %d evaluable local days (%s to %s)", len(days), days[0], days[-1])

    rows: list[dict[str, object]] = []
    for d in days:
        target_index = _local_hourly_index(d)
        try:
            _check_reconstruction_coverage(predictions, target_index)
            rows.append({"target_day": d.isoformat(), "excluded": False, "reason": ""})
        except IncompleteReconstructionError as exc:
            rows.append({"target_day": d.isoformat(), "excluded": True, "reason": str(exc)})

    detail = pd.DataFrame(rows)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(_OUT_PATH, index=False)
    logger.info("wrote %d rows to %s", len(detail), _OUT_PATH)

    n_excluded = int(detail["excluded"].sum())
    rate = n_excluded / len(detail) if len(detail) else 0.0
    logger.info("evaluable days: %d, excluded: %d (%.2f%%)", len(detail), n_excluded, rate * 100)
    for row in detail.loc[detail["excluded"]].itertuples():
        logger.info("  excluded %s: %s", row.target_day, row.reason)

    if rate > EXCLUSION_RATE_HALT_THRESHOLD:
        logger.warning(
            "RUECKFRAGE: exclusion rate %.2f%% exceeds the %.0f%% threshold (spec 6.5.3 section 3.3)",
            rate * 100,
            EXCLUSION_RATE_HALT_THRESHOLD * 100,
        )
        return 1
    logger.info(
        "Exclusion rate is within the %.0f%% threshold.", EXCLUSION_RATE_HALT_THRESHOLD * 100
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
