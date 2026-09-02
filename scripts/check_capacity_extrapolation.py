"""Leave-one-out extrapolation-error diagnostic for installed-capacity anchors.

Not a CI test (spec 6.5.2, section 5.8): the numbers move every time a new
anchor appears in the registry table, so a test with a fixed threshold would
regularly go red for the wrong reason. Run manually and review the output.

For every anchor from the third one onward, the value is predicted as if
only the anchors strictly before ``anchor_date`` minus a horizon of
``horizon_intervals`` known intervals were known (spec 6.5.2a, section 5.2),
using exactly the same extrapolation rate function `installed_capacity_at`
itself uses. Measured for horizon_intervals in {1, 2, 3, 4, 6} -- h=1 is the
original one-interval diagnostic from 6.5.2 step 6 and must reproduce those
numbers exactly; h=4 is what 6.5.2a proposes raising
`MAX_EXTRAPOLATION_INTERVALS` to. Compared against "hold the last known
value" (the naive baseline) and reported per source, production type, rule,
and horizon.

Why leave-one-out at horizon h measures the live case, not an optimistic
proxy for it: for anchor k, only anchors strictly before k-h+1 are used --
under the live discard rule (`TRAILING_ANCHORS_DISCARDED`), those are
themselves anchors at least that many intervals old and therefore already
past the registry's own revision window, exactly like the anchors
`installed_capacity_at` extrapolates from in production. The measured
percentages are not more optimistic than reality because of this.

Writes outputs/results/capacity_extrapolation_error.csv (one row per
source/production_type/rule/horizon/anchor) and logs three summaries: the
signed error at the interval end (GW and %), max/mean absolute error over
the whole interval, and the last five anchors individually -- plus a
dedicated Haltepunkt summary (spec 6.5.2a, section 5.2) isolating the one
row that actually decides `MAX_EXTRAPOLATION_INTERVALS`: interval-end error
at h=4, rule=LAST_INCREMENT (the rule `installed_capacity_at` actually uses
in production), per production type. A relative error above 5% at the
interval end for any production type/horizon/rule combination is logged as
a warning (spec 6.5.2, section 2.5) -- this script's job is to measure and
report, not to halt the caller -- but the h=4/LAST_INCREMENT row is the one
that gates whether the constant may be set (spec 6.5.2a, section 2.3):
above 5% there means Rueckfrage to the owner, not a unilateral lower value.
"""

from __future__ import annotations

import logging
from typing import Final, cast

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.data.capacity import (
    _ANCHOR_LOADERS,
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
    _days_since_epoch,
    _extrapolation_rate_per_day,
)

logger = logging.getLogger(__name__)

_OUT_PATH = PROJECT_ROOT / "outputs" / "results" / "capacity_extrapolation_error.csv"

_REGISTERED_METHODS = (
    CapacityExtrapolation.LAST_INCREMENT,
    CapacityExtrapolation.MEAN_LAST_THREE_INCREMENTS,
    CapacityExtrapolation.LINEAR_REGRESSION_LAST_THREE,
)
# Not a real CapacityExtrapolation member -- the naive comparison column the
# spec requires alongside the three registered rules (spec 6.5.2, section 2.5).
_HOLD_LAST_VALUE = "hold_last_value"
_ALL_RULES = (*_REGISTERED_METHODS, _HOLD_LAST_VALUE)

_MIN_KNOWN_ANCHORS = {
    CapacityExtrapolation.LAST_INCREMENT: 2,
    CapacityExtrapolation.MEAN_LAST_THREE_INCREMENTS: 4,
    CapacityExtrapolation.LINEAR_REGRESSION_LAST_THREE: 3,
    _HOLD_LAST_VALUE: 1,
}

RELATIVE_ERROR_HALT_THRESHOLD = 0.05  # spec 6.5.2 section 2.5: Rueckfrage above this

# spec 6.5.2a section 5.2: the horizons the diagnostic is measured at. h=1
# reproduces the original 6.5.2-step-6 numbers; h=4 is the proposed new
# MAX_EXTRAPOLATION_INTERVALS (spec 6.5.2a section 2.3).
_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6)

# The horizon/rule combination that actually gates the Haltepunkt decision
# (spec 6.5.2a section 5.2): LAST_INCREMENT is the only rule
# `installed_capacity_at` uses in production; the other two registered rules
# and the naive baseline exist only for comparison within this diagnostic.
_HALTEPUNKT_HORIZON = 4
_HALTEPUNKT_RULE = CapacityExtrapolation.LAST_INCREMENT


def _predict(known: pd.Series, rule: CapacityExtrapolation | str, delta_days: float) -> float:
    if rule == _HOLD_LAST_VALUE:
        return float(known.iloc[-1])
    # rule is provably CapacityExtrapolation here (the only other member of
    # _ALL_RULES), but it's typed as a union to hold the plain-string
    # _HOLD_LAST_VALUE sentinel above.
    rate_per_day = _extrapolation_rate_per_day(known, cast(CapacityExtrapolation, rule))
    return float(known.iloc[-1] + rate_per_day * delta_days)


def _leave_one_out(
    source: CapacitySource, production_type: ProductionType, horizon_intervals: int
) -> pd.DataFrame:
    """One row per (rule, anchor) predicted from anchors known up to
    ``horizon_intervals`` intervals before it (spec 6.5.2a section 5.2).

    horizon_intervals=1 predicts anchor n from anchors strictly before n --
    the original 6.5.2 diagnostic -- and must reproduce those numbers
    exactly (this function's n_known reduces to n for horizon_intervals=1).
    """
    anchors = _ANCHOR_LOADERS[source](production_type)
    days = _days_since_epoch(pd.DatetimeIndex(anchors.index))
    values = anchors.to_numpy(dtype="float64")

    rows: list[dict[str, object]] = []
    for rule in _ALL_RULES:
        min_known = _MIN_KNOWN_ANCHORS[rule]
        for n in range(2, len(anchors)):  # "ab der dritten Stuetzstelle" -- 0-indexed third anchor
            n_known = n - horizon_intervals + 1  # anchors 0..n_known-1 are "known"
            if n_known < 1:
                continue
            known = anchors.iloc[:n_known]
            if len(known) < min_known:
                continue
            last_known_idx = n_known - 1
            delta_days = days[n] - days[last_known_idx]
            predicted = _predict(known, rule, delta_days)
            actual = float(values[n])
            error_mw = predicted - actual
            rows.append(
                {
                    "source": source.value,
                    "production_type": production_type.value,
                    "rule": str(rule),
                    "horizon_intervals": horizon_intervals,
                    "anchor_date": anchors.index[n].date().isoformat(),
                    "actual_mw": actual,
                    "predicted_mw": predicted,
                    "error_mw": error_mw,
                    "error_gw": error_mw / 1000.0,
                    "error_pct": error_mw / actual * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _log_interval_end_summary(
    detail: pd.DataFrame,
) -> list[tuple[str, str, str, int, float]]:
    """Returns (source, production_type, rule, horizon_intervals, relative_error_pct)
    for the Rueckfrage check."""
    flagged: list[tuple[str, str, str, int, float]] = []
    logger.info("Interval-end signed error (last held-out anchor), GW and %%:")
    for key, group in detail.groupby(["source", "production_type", "rule", "horizon_intervals"]):
        source, ptype, rule, horizon = cast(tuple[str, str, str, int], key)
        last = group.iloc[-1]
        logger.info(
            "  %-14s %-13s %-26s  h=%d  anchor=%s  error=%+.4f GW (%+.2f%%)",
            source,
            ptype,
            rule,
            horizon,
            last["anchor_date"],
            last["error_gw"],
            last["error_pct"],
        )
        if abs(last["error_pct"]) > RELATIVE_ERROR_HALT_THRESHOLD * 100.0:
            flagged.append(
                (str(source), str(ptype), str(rule), int(horizon), float(last["error_pct"]))
            )
    return flagged


def _log_max_mean_summary(detail: pd.DataFrame) -> None:
    logger.info("Max / mean absolute error over the whole leave-one-out interval, GW:")
    abs_gw = detail["error_gw"].abs()
    summary = (
        detail.assign(abs_error_gw=abs_gw)
        .groupby(["source", "production_type", "rule", "horizon_intervals"])["abs_error_gw"]
        .agg(["max", "mean"])
        .reset_index()
    )
    for row in summary.itertuples():
        logger.info(
            "  %-14s %-13s %-26s  h=%d  max=%.4f GW  mean=%.4f GW",
            row.source,
            row.production_type,
            row.rule,
            row.horizon_intervals,
            row.max,
            row.mean,
        )


def _log_last_five_anchors(detail: pd.DataFrame) -> None:
    logger.info("Error at the last five anchors individually, GW:")
    for (source, ptype, rule, horizon), group in detail.groupby(
        ["source", "production_type", "rule", "horizon_intervals"]
    ):
        tail = group.tail(5)
        values_str = ", ".join(
            f"{row.anchor_date}={row.error_gw:+.4f}" for row in tail.itertuples()
        )
        logger.info("  %-14s %-13s %-26s  h=%d  %s", source, ptype, rule, horizon, values_str)


def _log_haltepunkt_summary(detail: pd.DataFrame) -> None:
    """The one number that actually decides MAX_EXTRAPOLATION_INTERVALS
    (spec 6.5.2a section 5.2): interval-end error at h=4, rule=LAST_INCREMENT,
    per production type -- pulled out of the full table for the owner review."""
    logger.info(
        "HALTEPUNKT (spec 6.5.2a section 5.2): interval-end error at h=%d, rule=%s, per production type:",
        _HALTEPUNKT_HORIZON,
        _HALTEPUNKT_RULE,
    )
    subset = detail.loc[
        (detail["horizon_intervals"] == _HALTEPUNKT_HORIZON)
        & (detail["rule"] == str(_HALTEPUNKT_RULE))
    ]
    for (source, ptype), group in subset.groupby(["source", "production_type"]):
        last = group.iloc[-1]
        over = abs(last["error_pct"]) > RELATIVE_ERROR_HALT_THRESHOLD * 100.0
        logger.info(
            "  %-14s %-13s  anchor=%s  error=%+.4f GW (%+.2f%%)  %s",
            source,
            ptype,
            last["anchor_date"],
            last["error_gw"],
            last["error_pct"],
            "OVER 5%% -- RUECKFRAGE" if over else "within 5%%",
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames: list[pd.DataFrame] = []
    for horizon in _HORIZONS:
        for source in CapacitySource:
            for production_type in ProductionType:
                try:
                    frames.append(_leave_one_out(source, production_type, horizon))
                except NotImplementedError as exc:
                    logger.info(
                        "Skipping source=%s production_type=%s horizon=%d: %s",
                        source.value,
                        production_type.value,
                        horizon,
                        exc,
                    )

    detail = pd.concat(frames, ignore_index=True)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(_OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(detail), _OUT_PATH)

    flagged = _log_interval_end_summary(detail)
    _log_max_mean_summary(detail)
    _log_last_five_anchors(detail)
    _log_haltepunkt_summary(detail)

    if flagged:
        logger.warning(
            "RUECKFRAGE: relative error at the interval end exceeds %.0f%% for: %s",
            RELATIVE_ERROR_HALT_THRESHOLD * 100.0,
            "; ".join(f"{s}/{p}/{r}/h={h} ({pct:+.2f}%)" for s, p, r, h, pct in flagged),
        )
        return 1

    logger.info(
        "No production type exceeds the %.0f%% Rueckfrage threshold.",
        RELATIVE_ERROR_HALT_THRESHOLD * 100.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
