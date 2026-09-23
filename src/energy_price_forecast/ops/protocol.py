"""JSON-lines operational protocol for the daily submission job (spec
6.7.2, sections 2.8/5.7).

``logs/submissions.jsonl``: one JSON line per run, fixed key order,
``protocol_version`` in every line. Not CSV, for two reasons spelled out
in the spec and both already evidenced in this repo:

1. The schema is expected to grow (6.9 adds a candidate column,
   Entscheidung 25) -- a new field on a JSON line changes nothing about
   old lines, since there is no header to migrate. ``logs/store_sync.csv``
   hit exactly this failure mode for real (6.7.1a, 2026-09-11): a new
   per-source column left an already-committed file's header narrower
   than the next row, making the file unparseable
   (``pd.read_csv`` -> ``ParserError``).
2. The data is not flat -- per-source ages, per-feature missing markers,
   payload statistics. In a table that either explodes the column count
   (``logs/store_sync.csv`` already needs 42 columns for 9 flat sources)
   or collapses the structure into a delimited string. Neither is what
   6.8 needs to read back out.

``logs/audit_runs.csv``, ``logs/availability.csv``, ``logs/store_sync.csv``
stay CSV on purpose (spec section 2.8) -- flat, stable, with real history;
converting them would be churn without benefit. The rule going forward:
flat and stable -> CSV, nested or growing -> JSON lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

PROTOCOL_VERSION: Final[int] = 4

# Fixed so every appended line writes its keys in the same order (spec
# section 2.8: "sonst sind die Git-Diffs verrauscht"). New fields are
# appended at the end of this tuple, never inserted in the middle --
# doing so would reorder every future line relative to old ones, the
# exact diff noise this ordering exists to prevent.
_KEY_ORDER: Final[tuple[str, ...]] = (
    "protocol_version",
    "run_timestamp_utc",
    "run_id",
    "run_url",
    "code_sha",
    "target_day",
    "nominal_slot",
    "gate_closure_ok",
    "source_ages",
    "candidate_selected",
    "skip_reason",
    "missing_features",
    "n_values",
    "payload_min",
    "payload_mean",
    "payload_max",
    "submitted",
    "api_status",
    "api_message",
    "runtime_seconds",
    "n_training_rows",
    "n_training_labels",
    # protocol_version 3 (spec 6.7.3, section 5.6) -- appended, not inserted,
    # so every already-written line stays diff-stable.
    "submission_mode",
    "submission_id",
    "api_response_received_utc",
    "confirmed_via_query",
    "smoke_baseline_source_day",
    # protocol_version 4 (spec 6.9, section 5.7) -- appended, same rule.
    # Step 3 (docs/sprint6_step6_9_log.md) only adds the schema; every field
    # here is None/empty until the step that actually computes it lands
    # (candidate_rank/candidates_evaluated/downgrade_blocked/
    # best_accepted_rank_before in step 10, nwp_available/training_* in
    # step 8, target_day_fills in step 11, load_forecast_source/
    # price_provenance/price_source_conflicts in step 7,
    # rnw_label_edge_age_days in step 8, capacity_anchor_days_left in step 8).
    "candidate_rank",
    "candidates_evaluated",
    "nwp_available",
    "nwp_unavailable_reason",
    "price_provenance",
    "price_source_conflicts",
    "load_forecast_source",
    "n_training_days",
    "age_of_last_complete_day",
    "training_tolerance_used",
    "training_missing_days",
    "known_weather_defect_days",
    "rnw_label_edge_age_days",
    "target_day_fills",
    "downgrade_blocked",
    "best_accepted_rank_before",
    "capacity_anchor_days_left",
)


@dataclass(frozen=True)
class SubmissionRecord:
    """One run of the daily submission job (spec section 5.7's field table).

    ``source_ages``: hours since each store source's own ``covered_end_utc``,
    at ``as_of`` -- logged unconditionally, never gates a run (spec section
    2.3: staleness is measured where it bites, in the built feature row,
    not approximated per source).

    ``candidate_selected`` xor ``skip_reason``: exactly one of the two is
    non-None for any run that reached the candidate-selection step; a run
    that failed Check A never reaches it, so both may be None there --
    Check A's own reason belongs in ``skip_reason`` too.

    ``missing_features``: named individually (spec section 2.5) -- the
    input 6.8 needs to prioritise real fallback candidates, not a bare
    "incomplete" flag.

    ``n_values``/``payload_min``/``payload_mean``/``payload_max`` are read
    off the payload actually built, never off what the code intended to
    build (spec section 5.7 -- the generalised lesson from 6.6's
    ``_log_gate_verdict`` resolution-mixup finding).

    ``submitted``/``api_status``/``api_message`` stay ``False``/``"dry_run"``/
    ``None`` throughout 6.7.2 -- ``live=True`` is never passed to
    ``arena.submit.submit()`` in this step. From protocol_version 3 (spec
    6.7.3) onward they carry real values once the live switch is on.

    protocol_version 3 fields (spec 6.7.3 section 5.6), all optional and
    additive -- an older reader ignores them via ``.get(...)``, an older
    written line simply lacks them:

    - ``submission_mode``: ``"dry_run"``/``"live"``/``"smoke"``, purely
      informative (no code path branches on it, spec section 2.5).
    - ``submission_id``: from the platform, if the POST returned one.
    - ``api_response_received_utc``: when the POST's response (success or
      failure) was actually received.
    - ``confirmed_via_query``: ``True``/``False`` if the own-submissions
      query endpoint was queried after an accepted POST, else ``None``.
    - ``smoke_baseline_source_day``: the price day the smoke mode's
      baseline was built from -- only set in ``submission_mode="smoke"``.

    ``n_training_rows``/``n_training_labels`` (protocol_version 2, Restarbeit
    Teil A): the size of the training matrix actually passed to
    ``LGBMForecaster.fit()`` and how many of those rows had a real,
    non-NaN ``day_ahead_price`` label -- read off the exact ``x_train``/
    ``y_train`` objects the fit call itself receives
    (``scripts/run_daily_submission.py::fit_predict_expand``), never a
    separately re-derived count. Observability only, never a gate (Restarbeit
    Teil A.4): a row-present-but-label-missing training day silently
    shrinks the effective training window since
    ``docs/sprint6_fix_partial_today.md`` stopped dropping such rows
    upstream, and 6.8 needs to tell "the model was worse" apart from "the
    model had less data" -- but this repo sets no guessed thresholds, so
    whether a gap here should ever block a run is a decision for 6.8, made
    from real logged numbers, not this step. Both ``None`` for any run that
    never reached the fit step (Check A/extent/Check B skip).

    protocol_version 4 fields (spec 6.9 section 5.7), all optional and
    additive, schema-only as of this step (docs/sprint6_step6_9_log.md,
    step 3) -- every one of them is still its default here, filled in by
    later 6.9 steps once the corresponding logic exists:

    - ``candidate_rank``: the selected fallback-ladder row's rank (1-3).
    - ``candidates_evaluated``: one entry per row tried, each
      ``{"name": ..., "rank": ..., "outcome": "selected"/"failed"/
      "not_reached", "reason": ...}``.
    - ``nwp_available``/``nwp_unavailable_reason``: Check A's split NWP
      verdict (spec section 2.3) -- whether weather/capacity allowed a
      renewables walk-forward at all this run, independent of which row
      the ladder ultimately selected.
    - ``price_provenance``: per price-consumer (``training_labels``,
      ``price_lags``, ``shape_window``, ``persistence``) which source fed
      it -- ``"entsoe"`` or ``"energy_charts"`` (spec section 5.3).
    - ``price_source_conflicts``: count of cells where both ENTSO-E and
      Energy-Charts had a value and they disagreed (spec section 5.3) --
      purely informative, never gates, never warns.
    - ``load_forecast_source``: ``"entsoe"``/``"energy_charts"``/``None``,
      whichever fed the selected row's load-forecast fundamentals.
    - ``n_training_days``/``age_of_last_complete_day``/
      ``training_tolerance_used``/``training_missing_days``: the measured-
      tolerance training-window report (spec section 2.6) that replaces
      the old exact-row-count ``check_training_extent``.
    - ``known_weather_defect_days``: training days excluded whose cause is
      an already-documented, permanent weather defect (spec section 2.6) --
      reporting only, no longer a tolerance input
      (``known_defect_tolerance_hours`` is removed in this spec).
    - ``rnw_label_edge_age_days``: how many days before the training
      window's own right edge the last complete 14.1.D renewables label
      lies (spec section 2.7).
    - ``target_day_fills``: one entry per target-day gap handled, each
      ``{"group": ..., "n_hours": ..., "hours": [...], "action":
      "forward_fill"/"partial_fill"}`` (spec section 2.8).
    - ``downgrade_blocked``/``best_accepted_rank_before``: whether this
      run's own candidate would have overwritten an already-accepted,
      better-ranked submission for the same target day, and what that
      rank was (spec section 2.9).
    - ``capacity_anchor_days_left``: days until the capacity anchor table's
      own validity boundary, at this run's ``as_of`` (spec section 2.3).
    """

    run_timestamp_utc: str
    run_id: str
    run_url: str
    code_sha: str
    target_day: str
    nominal_slot: str
    gate_closure_ok: bool
    source_ages: dict[str, float | None] = field(default_factory=dict)
    candidate_selected: str | None = None
    skip_reason: str | None = None
    missing_features: list[str] = field(default_factory=list)
    n_values: int | None = None
    payload_min: float | None = None
    payload_mean: float | None = None
    payload_max: float | None = None
    submitted: bool = False
    api_status: str | None = None
    api_message: str | None = None
    runtime_seconds: float | None = None
    n_training_rows: int | None = None
    n_training_labels: int | None = None
    submission_mode: str | None = None
    submission_id: int | None = None
    api_response_received_utc: str | None = None
    confirmed_via_query: bool | None = None
    smoke_baseline_source_day: str | None = None
    candidate_rank: int | None = None
    candidates_evaluated: list[dict[str, Any]] = field(default_factory=list)
    nwp_available: bool | None = None
    nwp_unavailable_reason: str | None = None
    price_provenance: dict[str, str] = field(default_factory=dict)
    price_source_conflicts: int | None = None
    load_forecast_source: str | None = None
    n_training_days: int | None = None
    age_of_last_complete_day: int | None = None
    training_tolerance_used: bool | None = None
    training_missing_days: list[str] = field(default_factory=list)
    known_weather_defect_days: list[str] = field(default_factory=list)
    rnw_label_edge_age_days: int | None = None
    target_day_fills: list[dict[str, Any]] = field(default_factory=list)
    downgrade_blocked: bool | None = None
    best_accepted_rank_before: int | None = None
    capacity_anchor_days_left: int | None = None
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _KEY_ORDER}


def append_submission_record(path: Path, record: SubmissionRecord) -> None:
    """Append one JSON line to path, in fixed key order.

    No header to write or migrate (unlike a CSV log) -- that is the whole
    point of this format (spec section 2.8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), sort_keys=False) + "\n")


def read_submission_records(path: Path) -> list[dict[str, Any]]:
    """Read every line as a plain dict, tolerant of both directions of
    schema drift: a line with a field this reader doesn't know about is
    just an extra dict key (ignored by anyone not looking for it), and a
    line from an older ``protocol_version`` missing a field a newer
    reader wants is simply absent from that line's dict -- callers use
    ``.get(...)``, never direct indexing, for exactly this reason.

    Returns ``[]`` if path does not exist yet (no runs logged so far is a
    valid, not exceptional, state).
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
