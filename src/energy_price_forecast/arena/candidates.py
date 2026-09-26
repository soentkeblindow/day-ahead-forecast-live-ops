"""Declarative fallback-ladder table for the daily submission job (spec
6.9, sections 2.1, 5.2).

Phase 1 (spec 6.7.2) had exactly one row (Entscheidung 25). Spec 6.9
replaces "ein Modell oder Schweigen" with three gemessen rows, each a real,
distinct role (rank is role order, not quality order, spec section 2.1):

  rank 1 ``core_gas``            -- calendar + NWP residual-load fundamentals
                                     + price lags + TTF gas, load from ENTSO-E.
                                     Measured as Messung A ``core_gas_only``
                                     (RMSE 28.97).
  rank 2 ``core_gas_loadpatch``  -- the SAME model/features as rank 1; only
                                     target_day's own load forecast is
                                     replaced by the Similar-Day-Patch
                                     (arena/load_patch.py). Measured as
                                     Messung E Variante A on ``core_gas_only``,
                                     re-measured with the Endregel in spec
                                     6.9 step 10 (RMSE 29.01, PASS).
  rank 3 ``base``                -- calendar + price lags only, no NWP, no
                                     gas (Owner 2026-09-24, Fassung 2: gas-
                                     free so a dead TTF feed can no longer
                                     take down the whole system, spec section
                                     2.1). Measured as Messung A
                                     ``floor_core`` (RMSE 38.39).

Old -> new naming (spec section 2.1): ``live`` -> retired entirely (no
replacement row -- spec section 2.1: "full (bisher live) ... fliegen
raus"), ``core_gas_only`` -> ``core_gas``, Messung E Variante A ->
``core_gas_loadpatch``, ``floor_core`` -> ``base``.

This module stays purely declarative (FeatureGroup/GapPolicy/Candidate
dataclasses plus the concrete three-row table) -- it does not know how to
build a row's features, determine row 2's reference day, or patch the
load forecast. That orchestration lives in scripts/run_daily_submission.py,
which has the df/renewables/config dependencies this module deliberately
does not take (spec section 5.2 gives Candidate a data shape, not a
build-a-row method).

``GapPolicy`` is declared here in full (spec section 5.2's own contract),
but for THIS step (6.9 Schritt 11) every group's effective viability check
is STRICT regardless of its declared policy: the forward-fill/partial-fill
mechanics spec section 2.8 describes are explicitly Schritt 12's own job
(``arena/target_day_fill.py``, not yet built). Declaring the real intended
policy now, ahead of Schritt 12 actually consuming it, avoids a second table
rewrite later -- the same "scaffold now, populate later" shape already used
for the protocol_version 4 fields in Schritt 3/8. Until Schritt 12 lands,
every row's target-day viability is checked via
arena.preflight.check_target_row against ``Candidate.required_columns()``
(the union of every group's own columns) -- the same zero-tolerance check
Phase 1 already used, just per-row instead of table-wide.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class GapPolicy(Enum):
    """Spec 6.9 section 5.2 -- the four target-day gap policies. Only
    declared here for now (see module docstring): every group's real
    selection-time behaviour in Schritt 11 is STRICT regardless of which
    policy it carries. Schritt 12's arena/target_day_fill.py is what will
    make FILL_ONLY/FILL_THEN_PARTIAL/FILL_THEN_FAIL actually differ from
    STRICT at runtime.
    """

    STRICT = "strict"  # any NaN hour fails the candidate
    FILL_ONLY = "fill_only"  # forward fill up to MAX_GAP_HOURS
    FILL_THEN_PARTIAL = "fill_then_partial"  # ffill <= FFILL_MAX_HOURS, partial fill beyond
    FILL_THEN_FAIL = "fill_then_fail"  # ffill <= FFILL_MAX_HOURS, otherwise the candidate fails


# spec 6.9 section 5.2: FFILL_MAX_HOURS is the ffill ceiling for FILL_THEN_PARTIAL/
# FILL_THEN_FAIL groups; MAX_GAP_HOURS (Messung C) is FILL_ONLY's own ceiling for
# price_lags. Both are constants Schritt 12 will actually apply -- declared here since
# they are named in the same spec section as GapPolicy/FeatureGroup.
FFILL_MAX_HOURS = 3
MAX_GAP_HOURS = 6


@dataclass(frozen=True)
class FeatureGroup:
    """One named group of columns sharing one gap policy (spec section 5.2).

    The real per-row group membership (which columns belong to
    "calendar"/"nwp_residual"/"load_forecast"/"price_lags"/"gas") is
    documented in docs/sprint6_step6_9_log.md's own "Befunde nach §3.1"
    point 1 table -- derived from the real builder functions
    (scripts/ablation_core_minimal_feature_set.py::build_core_for_day /
    scripts/measurement_a_candidate_intake.py::build_floor_for_day), not
    hand-enumerated here, so the group's ``columns`` can never silently
    drift from what the builder actually produces (see
    scripts/run_daily_submission.py::three_row_ladder_for_day, which
    constructs these FeatureGroup instances from the real built columns).
    """

    name: str
    columns: frozenset[str]
    policy: GapPolicy


@dataclass(frozen=True)
class Candidate:
    """One row of the fallback ladder (spec section 5.2). ``rank`` is role
    order, not quality order (spec section 2.1: "Rang ist Rolle, nicht
    Güte").

    ``load_source`` is None for rank 3 (``base``), which reads no load
    forecast column at all -- neither ENTSO-E's own nor the Similar-Day-
    Patch's output.
    """

    name: str
    rank: int
    groups: tuple[FeatureGroup, ...]
    requires_nwp: bool
    load_source: Literal["entsoe", "similar_day"] | None
    measured_as: str

    def required_columns(self) -> frozenset[str]:
        """The union of every group's own columns -- what
        arena.preflight.check_target_row checks against for this row
        (Schritt 11's STRICT-only viability check, see module docstring)."""
        out: frozenset[str] = frozenset()
        for group in self.groups:
            out |= group.columns
        return out


def three_row_ladder(
    *,
    calendar_columns: frozenset[str],
    nwp_residual_columns: frozenset[str],
    load_forecast_columns: frozenset[str],
    price_lag_columns: frozenset[str],
    gas_columns: frozenset[str],
) -> tuple[Candidate, ...]:
    """The three fallback-ladder rows (spec section 2.1/5.2), in rank order.

    The five column sets are each group's real columns, as actually
    produced by the real builders (spec section 2.1: "die Zeilen-Builder
    ... sind echte Spalten-Teilmengen von build_feature_set_for_day",
    Parity-Check-proven in Schritt 10) -- passed in rather than hand-
    enumerated here, so a group's columns can never silently drift from
    what the builder produces. See
    scripts/run_daily_submission.py::three_row_ladder_for_day for where
    these five sets are actually derived from the real built matrices.

    Row 2 (``core_gas_loadpatch``) shares row 1's exact groups: "gleiche
    Features, gleiches Training" (spec section 2.1) -- only the
    ``load_source``/``measured_as`` differ, and only target_day's own row
    (not required_columns) is ever patched (arena/load_patch.py, applied by
    the caller before Check B, not by this table). Row 3 (``base``) has no
    ``nwp_residual``/``load_forecast``/``gas`` groups at all -- gas-free per
    Fassung 2 (Owner 2026-09-24, spec section 2.1).
    """
    core_groups = (
        FeatureGroup("calendar", calendar_columns, GapPolicy.STRICT),
        FeatureGroup("nwp_residual", nwp_residual_columns, GapPolicy.FILL_THEN_PARTIAL),
        FeatureGroup("load_forecast", load_forecast_columns, GapPolicy.FILL_THEN_FAIL),
        FeatureGroup("price_lags", price_lag_columns, GapPolicy.FILL_ONLY),
        FeatureGroup("gas", gas_columns, GapPolicy.STRICT),
    )
    base_groups = (
        FeatureGroup("calendar", calendar_columns, GapPolicy.STRICT),
        FeatureGroup("price_lags", price_lag_columns, GapPolicy.FILL_ONLY),
    )

    return (
        Candidate(
            name="core_gas",
            rank=1,
            groups=core_groups,
            requires_nwp=True,
            load_source="entsoe",
            measured_as="Messung A core_gas_only (RMSE 28.97)",
        ),
        Candidate(
            name="core_gas_loadpatch",
            rank=2,
            groups=core_groups,
            requires_nwp=True,
            load_source="similar_day",
            measured_as=(
                "Messung E Variante A auf core_gas_only, "
                "Nachmessung mit der Endregel (RMSE 29.01, PASS, spec 6.9 step 10)"
            ),
        ),
        Candidate(
            name="base",
            rank=3,
            groups=base_groups,
            requires_nwp=False,
            load_source=None,
            measured_as="Messung A floor_core (RMSE 38.39)",
        ),
    )


def best_accepted_rank(
    protocol_rows: Iterable[Mapping[str, object]],
    target_day: object,
    ladder: tuple[Candidate, ...],
) -> int | None:
    """Spec 6.9 section 2.9 (Downgrade-Schutz): the best (lowest) rank among
    already-accepted LIVE submissions for ``target_day``. Rejected
    submissions and smoke-mode rows never count (spec: "Abgelehnte
    Einreichungen und Smoke-Einreichungen zählen nicht").

    ``target_day`` is compared via ``.isoformat()`` against each row's own
    ``target_day`` string (ops.protocol.SubmissionRecord's own on-disk
    shape) -- typed as ``object`` rather than ``datetime.date`` so this
    module does not need a ``datetime`` import just for this one comparison.
    ``ladder`` maps each row's ``candidate_selected`` name back to its rank;
    a name not present in the current ladder (e.g. a retired candidate from
    before a future re-measurement) is silently ignored, never an error --
    an old record naming a since-removed row must not crash a live run.
    """
    name_to_rank = {c.name: c.rank for c in ladder}
    target_day_str = target_day.isoformat() if hasattr(target_day, "isoformat") else str(target_day)
    best: int | None = None
    for row in protocol_rows:
        if row.get("target_day") != target_day_str:
            continue
        if row.get("submitted") is not True or row.get("submission_mode") != "live":
            continue
        selected_name = row.get("candidate_selected")
        rank = name_to_rank.get(selected_name) if isinstance(selected_name, str) else None
        if rank is not None and (best is None or rank < best):
            best = rank
    return best


def may_submit(own_rank: int, best_rank: int | None) -> bool:
    """Spec 6.9 section 2.9: send only at equal-or-better rank. Equal rank
    still sends -- "die Daten sind neuer" (a later run for the same rank has
    fresher inputs, so it should still overwrite)."""
    return best_rank is None or own_rank <= best_rank
