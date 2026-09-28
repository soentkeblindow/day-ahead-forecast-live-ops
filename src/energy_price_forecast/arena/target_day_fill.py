"""Target-day gap handling for the fallback ladder (spec 6.9, section 2.8 /
5.6).

``arena.candidates.GapPolicy`` was declared in step 11 but every group's
target-day viability was still checked STRICT regardless of its own declared
policy (see that module's docstring) -- this module is what actually makes
``FILL_ONLY``/``FILL_THEN_PARTIAL``/``FILL_THEN_FAIL`` differ from ``STRICT``
at runtime. Order per spec section 2.8: "erst planen, dann auswählen, dann
füllen" -- ``plan_target_day_fill`` only inspects the ungapped data and
returns either a failure or a plan; the caller decides which candidate row
wins; only the winning row is ever actually patched via
``apply_forward_fill``/``apply_partial_fill``. Never used in training (spec:
"Nie im Training. Die Füllung ist eine Politik der Vorhersage-Zeile") -- both
apply functions only ever touch the hours a plan names, never a training row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from energy_price_forecast.arena.candidates import (
    FFILL_MAX_HOURS,
    MAX_GAP_HOURS,
    Candidate,
    GapPolicy,
)
from energy_price_forecast.arena.preflight import PreflightResult


@dataclass(frozen=True)
class GroupFill:
    """One feature group's target-day fill action (spec section 5.6).

    ``hours`` are the UTC hourly timestamps of the gap (``target_rows``'s own
    index convention, spec 6.9's usual 23/24/25-hour local-day index) --
    never a count alone, so ``apply_forward_fill``/``apply_partial_fill`` can
    address exactly these hours and nothing else.
    """

    group: str
    action: Literal["forward_fill", "partial_fill"]
    columns: frozenset[str]
    hours: tuple[pd.Timestamp, ...]


def plan_target_day_fill(
    target_rows: pd.DataFrame, candidate: Candidate
) -> PreflightResult | tuple[GroupFill, ...]:
    """Assess ``candidate``'s target-day rows group by group, on the
    ungapped data (spec section 2.8's own "erst planen" step) -- never
    mutates anything.

    Every group's failures are collected before returning (not just the
    first) -- a candidate can fail for several groups at once. ``STRICT``
    fails on any gap hour at all; the three fill-capable policies each have
    their own hour-count bracket (spec section 2.8's table):

    - ``FILL_ONLY``: h <= MAX_GAP_HOURS -> forward_fill, else fail.
    - ``FILL_THEN_FAIL``: h <= FFILL_MAX_HOURS -> forward_fill, else fail.
    - ``FILL_THEN_PARTIAL``: h <= FFILL_MAX_HOURS -> forward_fill;
      FFILL_MAX_HOURS < h <= MAX_GAP_HOURS -> partial_fill; else fail.

    If any hour would be forward-filled in one group and partial-filled in
    another, it is partial-filled (spec: "wird sie teil-gefüllt") -- dropped
    from the forward-filling group's own hours, since that hour's payload
    slots are replaced by persistence regardless of what any other group's
    feature cell holds (LightGBM tolerates a NaN feature cell natively, no
    crash risk from leaving it unfilled).
    """
    failures: list[str] = []
    missing_features: list[str] = []
    pending: list[
        tuple[str, Literal["forward_fill", "partial_fill"], frozenset[str], pd.DatetimeIndex]
    ] = []
    partial_hours: set[pd.Timestamp] = set()

    for group in candidate.groups:
        if not group.columns:
            continue
        columns = sorted(group.columns)
        gap_mask = target_rows[columns].isna().any(axis=1)
        gap_hours = pd.DatetimeIndex(target_rows.index[gap_mask])
        h = len(gap_hours)
        if h == 0:
            continue

        fail_message: str | None = None
        if group.policy is GapPolicy.STRICT:
            fail_message = f"{h} hour(s) missing or NaN (STRICT)"
        elif group.policy is GapPolicy.FILL_ONLY:
            if h > MAX_GAP_HOURS:
                fail_message = f"{h} hour(s) exceed MAX_GAP_HOURS={MAX_GAP_HOURS}"
            else:
                pending.append((group.name, "forward_fill", group.columns, gap_hours))
        elif group.policy is GapPolicy.FILL_THEN_FAIL:
            if h > FFILL_MAX_HOURS:
                fail_message = f"{h} hour(s) exceed FFILL_MAX_HOURS={FFILL_MAX_HOURS}"
            else:
                pending.append((group.name, "forward_fill", group.columns, gap_hours))
        elif group.policy is GapPolicy.FILL_THEN_PARTIAL:
            if h <= FFILL_MAX_HOURS:
                pending.append((group.name, "forward_fill", group.columns, gap_hours))
            elif h <= MAX_GAP_HOURS:
                pending.append((group.name, "partial_fill", group.columns, gap_hours))
                partial_hours.update(gap_hours)
            else:
                fail_message = f"{h} hour(s) exceed MAX_GAP_HOURS={MAX_GAP_HOURS}"

        if fail_message is not None:
            failures.append(f"{group.name}: {fail_message}")
            for column in columns:
                if target_rows[column].isna().any() and column not in missing_features:
                    missing_features.append(column)

    if failures:
        return PreflightResult(
            ok=False, reasons=tuple(failures), missing_features=tuple(missing_features)
        )

    fills: list[GroupFill] = []
    for name, action, columns, hours in pending:
        if action == "forward_fill":
            hours = pd.DatetimeIndex([h for h in hours if h not in partial_hours])
            if len(hours) == 0:
                continue
        fills.append(GroupFill(group=name, action=action, columns=columns, hours=tuple(hours)))
    return tuple(fills)


def apply_forward_fill(matrix: pd.DataFrame, fills: Sequence[GroupFill]) -> pd.DataFrame:
    """Forward-fills exactly the declared gap hours of every ``forward_fill``
    plan entry (spec section 2.8) -- reaches back into ``matrix``'s own
    earlier rows (the training window, or D-1 for a gap at the start of D),
    never the other way around, and never touches any row outside the
    declared hours (training rows already passed ``check_training_window``
    and are therefore never gapped in the first place, but this makes that
    explicit rather than incidental).
    """
    out = matrix.copy()
    for fill in fills:
        if fill.action != "forward_fill" or not fill.hours:
            continue
        columns = sorted(fill.columns)
        filled = out[columns].ffill()
        hours = pd.DatetimeIndex(fill.hours)
        out.loc[hours, columns] = filled.loc[hours, columns]
    return out


def apply_partial_fill(
    payload: pd.Series, persistence: pd.Series, fills: Sequence[GroupFill]
) -> pd.Series:
    """Replaces every ``partial_fill`` plan entry's quarter-hourly payload
    slots with the persistence value for that slot (spec section 2.8,
    Measurement C's own mechanic: predict with the NaN feature cell left in
    place, then overwrite the trailing slots on the payload afterward).

    One hourly gap hour maps to exactly its own four 15-minute slots
    (``models/bridge.py::expand_to_quarterhour`` expands each UTC hourly
    timestamp onto its own four quarter-hour slots starting at that same
    timestamp) -- "eine fehlende Viertelstunde betrifft die ganze Stunde,
    also vier Slots" (spec section 7).
    """
    out = payload.copy()
    for fill in fills:
        if fill.action != "partial_fill":
            continue
        for hour in fill.hours:
            quarter_hours = pd.date_range(hour, periods=4, freq="15min").intersection(
                pd.DatetimeIndex(out.index)
            )
            out.loc[quarter_hours] = persistence.loc[quarter_hours]
    return out
