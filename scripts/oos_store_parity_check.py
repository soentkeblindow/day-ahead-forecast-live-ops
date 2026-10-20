"""OOS-Store parity check against real submitted payloads (spec 7.1,
section 2.4).

For every target day since the 3-row fallback ladder went live
(2026-09-26, commit ``94cec35`` -- earlier real payloads used a since-
retired single-candidate design with no equivalent ladder row, so
comparing against them would measure a model mismatch, not a store bug;
confirmed with the owner, not re-litigated here), compares the OOS-
Store's forecast for the ladder row that was ACTUALLY submitted against
the real payload archived in ``logs/submission_payloads/``.

Haltepunkt (spec section 2.4): more than a handful of days (named here as
``HALTEPUNKT_MIN_OFFENDING_DAYS``) with a mean absolute difference over
``HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH`` means the store does not
reproduce what actually happened live -- stop and ask before any Teil B
(quantile track) work begins; this script only measures and reports, it
never decides to proceed on its own.

Reads only already-committed repo files (``logs/oos_store.csv``,
``logs/submissions.jsonl``, ``logs/submission_payloads/*.json``) -- no
store download, no model fit, no network call.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.ops.oos_store import OOS_STORE_PATH, load_oos_store
from energy_price_forecast.ops.protocol import read_submission_records

# scripts/ itself on sys.path, not the repo root -- same fix
# run_daily_submission.py/oos_store_job.py already need.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_daily_submission import PAYLOADS_DIR, SUBMISSIONS_LOG  # noqa: E402

# Ladder went live 2026-09-26 (commit 94cec35, "three-row fallback ladder
# replaces single-candidate live model") -- real payloads before this date
# used a since-retired candidate the OOS-Store has no equivalent row for.
PARITY_CHECK_START_DAY: Final[dt.date] = dt.date(2026, 9, 26)

HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH: Final[float] = 1.0
# "mehr als eine Handvoll Tage" (spec section 2.4) -- named explicitly
# rather than left as an implicit literal in the comparison below.
HALTEPUNKT_MIN_OFFENDING_DAYS: Final[int] = 5

OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "logs" / "oos_store_parity_check.csv"

_LOCAL_TZ = "Europe/Berlin"


@dataclass(frozen=True)
class ParityDayResult:
    """One (target_day, ladder_row) comparison -- always one row per real
    submitted day, never per ladder row in general (only the row that was
    actually submitted is ever compared, spec section 2.4)."""

    target_day: dt.date
    ladder_row: str
    n_matched_timestamps: int
    mean_abs_diff: float | None
    max_abs_diff: float | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_day": self.target_day.isoformat(),
            "ladder_row": self.ladder_row,
            "n_matched_timestamps": self.n_matched_timestamps,
            "mean_abs_diff": self.mean_abs_diff,
            "max_abs_diff": self.max_abs_diff,
            "note": self.note,
        }


def real_submitted_rows(records: list[dict[str, Any]], *, start_day: dt.date) -> dict[dt.date, str]:
    """target_day -> candidate_selected, for every real, accepted LIVE
    submission on or after ``start_day`` -- the same field-reading
    pattern as ``arena.candidates.best_accepted_rank`` (``.get(...)``,
    never direct indexing, so an older/newer protocol_version's missing
    field is simply absent, not an error). The chronologically LAST
    accepted record for a given day wins if more than one slot
    submitted that day (append order in the JSON-lines file IS
    chronological order) -- consistent with the downgrade-protection
    contract (``may_submit``) that a later slot never accepts a worse
    rank, so the last accepted name is always the best-or-equal one.
    """
    start_str = start_day.isoformat()
    result: dict[dt.date, str] = {}
    for row in records:
        target_day_str = row.get("target_day")
        if not isinstance(target_day_str, str) or target_day_str < start_str:
            continue
        if row.get("submitted") is not True or row.get("submission_mode") != "live":
            continue
        candidate = row.get("candidate_selected")
        if not isinstance(candidate, str):
            continue
        result[dt.date.fromisoformat(target_day_str)] = candidate
    return result


def _expected_quarterhour_index(target_day: dt.date) -> pd.DatetimeIndex:
    """The UTC quarter-hourly index a payload's own ``values`` implicitly
    carry (92/96/100 entries across DST) -- the exact same construction
    ``models/bridge.py::expand_to_quarterhour`` uses to build the series a
    payload's ``values`` are read off, so a payload's i-th value always
    zips onto this index's i-th timestamp regardless of DST."""
    day_start = pd.Timestamp(target_day, tz=_LOCAL_TZ)
    day_end = pd.Timestamp(target_day + dt.timedelta(days=1), tz=_LOCAL_TZ)
    return pd.date_range(day_start, day_end, freq="15min", inclusive="left").tz_convert("UTC")


def compare_day(
    oos_store: pd.DataFrame, payload: dict[str, Any], target_day: dt.date, ladder_row: str
) -> ParityDayResult:
    values = payload.get("values")
    if not isinstance(values, list):
        return ParityDayResult(target_day, ladder_row, 0, None, None, "payload has no values")

    expected_index = _expected_quarterhour_index(target_day)
    if len(values) != len(expected_index):
        return ParityDayResult(
            target_day,
            ladder_row,
            0,
            None,
            None,
            f"payload has {len(values)} values, expected {len(expected_index)}",
        )
    payload_series = pd.Series([float(v) for v in values], index=expected_index, dtype=float)

    store_rows = oos_store.loc[
        (oos_store["target_day"] == target_day) & (oos_store["ladder_row"] == ladder_row)
    ]
    if store_rows.empty:
        return ParityDayResult(
            target_day, ladder_row, 0, None, None, "no OOS-store row for this ladder_row/day"
        )
    store_series = store_rows.set_index("timestamp_utc")["forecast"].astype(float)

    common = payload_series.index.intersection(store_series.index)
    if len(common) == 0:
        return ParityDayResult(
            target_day, ladder_row, 0, None, None, "no overlapping timestamps with the OOS-store"
        )

    diff = (payload_series.loc[common] - store_series.loc[common]).abs()
    note = (
        ""
        if len(common) == len(expected_index)
        else f"only {len(common)}/{len(expected_index)} timestamps matched"
    )
    return ParityDayResult(
        target_day,
        ladder_row,
        len(common),
        float(diff.mean()),
        float(diff.max()),
        note,
    )


def run_parity_check(
    *,
    submissions_log: Path = SUBMISSIONS_LOG,
    payloads_dir: Path = PAYLOADS_DIR,
    oos_store: pd.DataFrame,
    start_day: dt.date = PARITY_CHECK_START_DAY,
) -> tuple[list[ParityDayResult], bool]:
    """Returns (per-day results, haltepunkt_triggered). A day with no real
    payload file or no matching OOS-store row is reported (via its own
    ``note``) but excluded from the Haltepunkt's offending-day count --
    a genuine gap is a different failure mode than a disagreement and
    must not silently move that count either way."""
    records = read_submission_records(submissions_log)
    submitted = real_submitted_rows(records, start_day=start_day)

    results: list[ParityDayResult] = []
    for target_day in sorted(submitted):
        ladder_row = submitted[target_day]
        payload_path = payloads_dir / f"{target_day.isoformat()}.json"
        if not payload_path.exists():
            results.append(
                ParityDayResult(target_day, ladder_row, 0, None, None, "no real payload file")
            )
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        results.append(compare_day(oos_store, payload, target_day, ladder_row))

    offending_days = sum(
        1
        for r in results
        if r.mean_abs_diff is not None and r.mean_abs_diff > HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH
    )
    haltepunkt_triggered = offending_days > HALTEPUNKT_MIN_OFFENDING_DAYS
    return results, haltepunkt_triggered


def _write_report(results: list[ParityDayResult], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(path, index=False)


def main() -> int:
    oos_store = load_oos_store(OOS_STORE_PATH)
    results, haltepunkt_triggered = run_parity_check(oos_store=oos_store)
    _write_report(results)

    for r in results:
        mad = f"{r.mean_abs_diff:.3f}" if r.mean_abs_diff is not None else "n/a"
        print(f"{r.target_day}  {r.ladder_row:<20}  mean_abs_diff={mad}  {r.note}")

    n_compared = sum(1 for r in results if r.mean_abs_diff is not None)
    n_offending = sum(
        1
        for r in results
        if r.mean_abs_diff is not None and r.mean_abs_diff > HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH
    )
    print(
        f"\n{len(results)} day(s) checked, {n_compared} compared, "
        f"{n_offending} offending (> {HALTEPUNKT_MAD_THRESHOLD_EUR_PER_MWH} EUR/MWh)."
    )
    if haltepunkt_triggered:
        print(
            "HALTEPUNKT: more than "
            f"{HALTEPUNKT_MIN_OFFENDING_DAYS} offending day(s) -- "
            "the store does not reliably reproduce the real submissions. "
            "Stop and ask before any Teil B work (spec section 2.4)."
        )
        return 1

    print("Parity check passed -- no Haltepunkt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
