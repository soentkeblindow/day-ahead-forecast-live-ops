"""Declarative candidate table for the daily submission job (spec 6.7.2,
sections 2.10, 5.5).

Phase 1 has exactly one row (Entscheidung 25) -- but the selection code
doesn't know that; it's a loop over a table, so 6.9 adding rows with
feature subsets is a data change to this table, not a code change. An
``if features_complete: submit else: skip`` shape is deliberately not
used anywhere in this module (spec section 2.10): a provisional branch
like that hardens, and rewiring a live submission path in production is
exactly the change nobody wants to make.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from energy_price_forecast.arena.preflight import PreflightResult, check_target_row


@dataclass(frozen=True)
class Candidate:
    """One row of the candidate table.

    ``required_features`` declares what this candidate needs to fire --
    checked against the built feature row via
    arena.preflight.check_target_row, never a separately maintained list
    (the same "the built row *is* the check" discipline as Check B
    itself, spec section 2.2).
    """

    name: str
    required_features: frozenset[str]


@dataclass(frozen=True)
class CandidateSelection:
    """Outcome of walking the candidate table for one target day.

    ``candidate`` is None when no row was viable -- the caller reads
    ``result.missing_features``/``result.reasons`` from the *last*
    attempted candidate for the skip reason (spec section 5.5: "Trägt
    keiner: Schweigen mit Grund").
    """

    candidate: Candidate | None
    result: PreflightResult


def select_candidate(
    candidates: tuple[Candidate, ...],
    features: pd.DataFrame,
    target_day: dt.date,
) -> CandidateSelection:
    """Walk candidates **in order** and return the first whose required
    features are all present and non-NaN in the built row for
    ``target_day`` (spec section 5.5).

    An empty ``candidates`` tuple or an all-unviable table both resolve
    to ``candidate=None`` with the last (or a synthetic "no candidates")
    result -- never an exception; a day with no viable candidate is a
    silent day, an expected operating state (spec section 2.7).
    """
    last_result = PreflightResult(ok=False, reasons=("no candidates configured",))
    for candidate in candidates:
        result = check_target_row(features, target_day, candidate.required_features)
        if result.ok:
            return CandidateSelection(candidate=candidate, result=result)
        last_result = result
    return CandidateSelection(candidate=None, result=last_result)


def full_live_set_candidate_table(feature_columns: frozenset[str]) -> tuple[Candidate, ...]:
    """Phase 1's one-row candidate table (spec section 2.10): the single
    candidate requires every column the live feature builder actually
    produced for the training window -- derived from the built matrix's
    own columns, never a hand-maintained list (the same discipline
    6.7.1a already applied to the store's checked-column set).
    """
    return (Candidate(name="full_live_set", required_features=frozenset(feature_columns)),)
