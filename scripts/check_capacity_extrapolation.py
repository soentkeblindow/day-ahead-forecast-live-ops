"""Leave-one-out extrapolation-error diagnostic for installed-capacity anchors.

Not a CI test (spec 6.5.2, section 5.8): the numbers move every time a new
anchor appears in the registry table, so a test with a fixed threshold would
regularly go red for the wrong reason. Run manually and review the output.

For every anchor from the third one onward, the value is predicted as if
only the anchors strictly before it were known, using exactly the same
extrapolation rate function `installed_capacity_at` itself uses. Compared
against "hold the last known value" (the naive baseline) and reported per
source, production type, and rule.

Writes outputs/results/capacity_extrapolation_error.csv (one row per
source/production_type/rule/anchor) and logs three summaries: the signed
error at the interval end (GW and %), max/mean absolute error over the
whole interval, and the last five anchors individually. A relative error
above 5% at the interval end for any production type is the spec's
Rueckfrage trigger (section 2.5) -- logged as a warning, not raised, since
this script's job is to measure and report, not to halt the caller.
"""

from __future__ import annotations

import logging
from typing import cast

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


def _predict(known: pd.Series, rule: CapacityExtrapolation | str, delta_days: float) -> float:
    if rule == _HOLD_LAST_VALUE:
        return float(known.iloc[-1])
    # rule is provably CapacityExtrapolation here (the only other member of
    # _ALL_RULES), but it's typed as a union to hold the plain-string
    # _HOLD_LAST_VALUE sentinel above.
    rate_per_day = _extrapolation_rate_per_day(known, cast(CapacityExtrapolation, rule))
    return float(known.iloc[-1] + rate_per_day * delta_days)


def _leave_one_out(source: CapacitySource, production_type: ProductionType) -> pd.DataFrame:
    anchors = _ANCHOR_LOADERS[source](production_type)
    days = _days_since_epoch(pd.DatetimeIndex(anchors.index))
    values = anchors.to_numpy(dtype="float64")

    rows: list[dict[str, object]] = []
    for rule in _ALL_RULES:
        min_known = _MIN_KNOWN_ANCHORS[rule]
        for n in range(2, len(anchors)):  # "ab der dritten Stuetzstelle" -- 0-indexed third anchor
            known = anchors.iloc[:n]
            if len(known) < min_known:
                continue
            delta_days = days[n] - days[n - 1]
            predicted = _predict(known, rule, delta_days)
            actual = float(values[n])
            error_mw = predicted - actual
            rows.append(
                {
                    "source": source.value,
                    "production_type": production_type.value,
                    "rule": str(rule),
                    "anchor_date": anchors.index[n].date().isoformat(),
                    "actual_mw": actual,
                    "predicted_mw": predicted,
                    "error_mw": error_mw,
                    "error_gw": error_mw / 1000.0,
                    "error_pct": error_mw / actual * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _log_interval_end_summary(detail: pd.DataFrame) -> list[tuple[str, str, str, float]]:
    """Returns (source, production_type, rule, relative_error_pct) for the Rueckfrage check."""
    flagged: list[tuple[str, str, str, float]] = []
    logger.info("Interval-end signed error (last held-out anchor), GW and %%:")
    for (source, ptype, rule), group in detail.groupby(["source", "production_type", "rule"]):
        last = group.iloc[-1]
        logger.info(
            "  %-14s %-13s %-26s  anchor=%s  error=%+.4f GW (%+.2f%%)",
            source,
            ptype,
            rule,
            last["anchor_date"],
            last["error_gw"],
            last["error_pct"],
        )
        if abs(last["error_pct"]) > RELATIVE_ERROR_HALT_THRESHOLD * 100.0:
            flagged.append((str(source), str(ptype), str(rule), float(last["error_pct"])))
    return flagged


def _log_max_mean_summary(detail: pd.DataFrame) -> None:
    logger.info("Max / mean absolute error over the whole leave-one-out interval, GW:")
    abs_gw = detail["error_gw"].abs()
    summary = (
        detail.assign(abs_error_gw=abs_gw)
        .groupby(["source", "production_type", "rule"])["abs_error_gw"]
        .agg(["max", "mean"])
        .reset_index()
    )
    for row in summary.itertuples():
        logger.info(
            "  %-14s %-13s %-26s  max=%.4f GW  mean=%.4f GW",
            row.source,
            row.production_type,
            row.rule,
            row.max,
            row.mean,
        )


def _log_last_five_anchors(detail: pd.DataFrame) -> None:
    logger.info("Error at the last five anchors individually, GW:")
    for (source, ptype, rule), group in detail.groupby(["source", "production_type", "rule"]):
        tail = group.tail(5)
        values_str = ", ".join(
            f"{row.anchor_date}={row.error_gw:+.4f}" for row in tail.itertuples()
        )
        logger.info("  %-14s %-13s %-26s  %s", source, ptype, rule, values_str)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames: list[pd.DataFrame] = []
    for source in CapacitySource:
        for production_type in ProductionType:
            try:
                frames.append(_leave_one_out(source, production_type))
            except NotImplementedError as exc:
                logger.info(
                    "Skipping source=%s production_type=%s: %s",
                    source.value,
                    production_type.value,
                    exc,
                )

    detail = pd.concat(frames, ignore_index=True)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(_OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(detail), _OUT_PATH)

    flagged = _log_interval_end_summary(detail)
    _log_max_mean_summary(detail)
    _log_last_five_anchors(detail)

    if flagged:
        logger.warning(
            "RUECKFRAGE: relative error at the interval end exceeds %.0f%% for: %s",
            RELATIVE_ERROR_HALT_THRESHOLD * 100.0,
            "; ".join(f"{s}/{p}/{r} ({pct:+.2f}%)" for s, p, r, pct in flagged),
        )
        return 1

    logger.info(
        "No production type exceeds the %.0f%% Rueckfrage threshold.",
        RELATIVE_ERROR_HALT_THRESHOLD * 100.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
