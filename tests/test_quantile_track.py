"""Unit/integration tests for Challenge 8 (quantile track) wiring in
scripts/run_daily_submission.py (Sprint 7.1 Teil B -> Probephase).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import QuantileChallengeSpec
from energy_price_forecast.arena.quantile_payload import (
    validate_payload as quantile_validate_payload,
)
from energy_price_forecast.arena.submit import SubmissionResult
from energy_price_forecast.ops import protocol, store
from scripts.run_daily_submission import (
    QuantileSubmissionOutcome,
    SubmissionOutcome,
    build_quantile_submission_record,
    run_daily_submission,
    run_quantile_track_for_day,
)

_TARGET_DAY = dt.date(2026, 7, 15)
_GATE_CLOSURE_UTC = pd.Timestamp("2026-07-14T10:00:00", tz="UTC")
_BEFORE_GATE_CLOSURE = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)

_QUANTILE_CHALLENGE = QuantileChallengeSpec(
    challenge_id="8",
    name="Day-Ahead Prices | Germany-Luxembourg | Quantile Forecast (5 quantiles)",
    timezone="Europe/Berlin",
    resolution_minutes=15,
    quantile_levels=(0.025, 0.25, 0.5, 0.75, 0.975),
    precision_decimals=2,
    allow_negative=True,
    max_forecast_points=None,
    raw={},
)

_PRICES_QH = pd.DataFrame({"day_ahead_price": []}, dtype=float)


def _point_outcome(
    *, candidate_selected: str | None = "core_gas", candidate_rank: int | None = 1
) -> SubmissionOutcome:
    payload = {
        "challenge_id": "2",
        "target_start": "2026-07-15T00:00:00+02:00",
        "values": [50.0] * 96,
    }
    return SubmissionOutcome(
        candidate_selected=candidate_selected,
        candidate_rank=candidate_rank,
        skip_reason=None if candidate_selected else "Check A: no weather run",
        payload=payload if candidate_selected else None,
    )


def _quantiles_dict() -> dict[float, pd.Series]:
    idx = pd.RangeIndex(96)
    return {a: pd.Series([50.0 + 10 * a] * 96, index=idx) for a in (0.025, 0.25, 0.5, 0.75, 0.975)}


MODULE = "scripts.run_daily_submission"


# ---------------------------------------------------------------------------
# run_quantile_track_for_day
# ---------------------------------------------------------------------------


def test_skips_when_challenge_2_had_no_point_median() -> None:
    outcome = run_quantile_track_for_day(
        _point_outcome(candidate_selected=None, candidate_rank=None),
        _PRICES_QH,
        _TARGET_DAY,
    )
    assert outcome.skip_reason == "no point median (Challenge 2 silent)"
    assert outcome.candidate_selected is None
    assert outcome.payload is None


def test_skips_when_the_fhs_pool_is_unusable() -> None:
    with patch(f"{MODULE}.run_quantile_step", return_value=None) as mock_step:
        outcome = run_quantile_track_for_day(_point_outcome(), _PRICES_QH, _TARGET_DAY)

    mock_step.assert_called_once()
    assert outcome.skip_reason == "FHS pool/sigma unusable for this day"
    assert outcome.candidate_selected == "core_gas"
    assert outcome.candidate_rank == 1
    assert outcome.payload is None


def test_builds_and_dry_run_submits_when_everything_is_ok() -> None:
    with (
        patch(f"{MODULE}.run_quantile_step", return_value=_quantiles_dict()),
        patch(f"{MODULE}.get_quantile_challenge", return_value=_QUANTILE_CHALLENGE),
        patch(f"{MODULE}.best_accepted_rank", return_value=None),
        patch(
            f"{MODULE}.submit",
            return_value=SubmissionResult(sent=False, challenge_id="8"),
        ) as mock_submit,
    ):
        outcome = run_quantile_track_for_day(
            _point_outcome(),
            _PRICES_QH,
            _TARGET_DAY,
            live=False,
            now=lambda: _BEFORE_GATE_CLOSURE,
        )

    assert outcome.skip_reason is None
    assert outcome.payload is not None
    assert outcome.payload["challenge_id"] == "8"
    assert len(outcome.payload["values"]) == 96
    assert all(len(row) == 5 for row in outcome.payload["values"])
    mock_submit.assert_called_once()
    assert mock_submit.call_args.kwargs["live"] is False
    # the real bug mypy found while wiring this: submit() must be told to
    # use the quantile (nested-aware) validator, not its own default.
    assert mock_submit.call_args.kwargs["validate"] is quantile_validate_payload


def test_skips_past_gate_closure() -> None:
    with (
        patch(f"{MODULE}.run_quantile_step", return_value=_quantiles_dict()),
        patch(f"{MODULE}.get_quantile_challenge", return_value=_QUANTILE_CHALLENGE),
        patch(f"{MODULE}.submit") as mock_submit,
    ):
        outcome = run_quantile_track_for_day(
            _point_outcome(),
            _PRICES_QH,
            _TARGET_DAY,
            now=lambda: _GATE_CLOSURE_UTC,
        )

    assert outcome.skip_reason == "gate closure passed before POST"
    assert outcome.payload is not None  # archived regardless, same as Challenge 2
    mock_submit.assert_not_called()


def test_downgrade_blocked_when_a_better_rank_is_already_accepted() -> None:
    with (
        patch(f"{MODULE}.run_quantile_step", return_value=_quantiles_dict()),
        patch(f"{MODULE}.get_quantile_challenge", return_value=_QUANTILE_CHALLENGE),
        patch(f"{MODULE}.best_accepted_rank", return_value=1) as mock_best_rank,
        patch(f"{MODULE}.submit") as mock_submit,
    ):
        outcome = run_quantile_track_for_day(
            _point_outcome(candidate_selected="base", candidate_rank=3),
            _PRICES_QH,
            _TARGET_DAY,
            now=lambda: _BEFORE_GATE_CLOSURE,
        )

    assert mock_best_rank.call_args.kwargs["challenge_id"] == "8"
    assert outcome.downgrade_blocked is True
    assert outcome.best_accepted_rank_before == 1
    mock_submit.assert_not_called()


# ---------------------------------------------------------------------------
# build_quantile_submission_record
# ---------------------------------------------------------------------------


def test_record_carries_challenge_id_8() -> None:
    outcome = QuantileSubmissionOutcome(candidate_selected="core_gas", candidate_rank=1)
    record = build_quantile_submission_record(
        outcome,
        target_day=_TARGET_DAY,
        nominal_slot="10:40",
        as_of=_BEFORE_GATE_CLOSURE,
    )
    assert record.challenge_id == "8"
    assert record.candidate_selected == "core_gas"
    assert record.candidate_rank == 1


def test_record_payload_stats_are_flattened_across_all_levels() -> None:
    payload = {
        "challenge_id": "8",
        "target_start": "x",
        "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }
    outcome = QuantileSubmissionOutcome(payload=payload)
    record = build_quantile_submission_record(
        outcome, target_day=_TARGET_DAY, nominal_slot="10:40", as_of=_BEFORE_GATE_CLOSURE
    )
    assert record.n_values == 2  # two timesteps, not six flat values
    assert record.payload_min == 1.0
    assert record.payload_max == 6.0
    assert record.payload_mean == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Independence: a Challenge-8 failure must never affect Challenge 2's own
# already-written outcome (spec section 3.4's explicit requirement).
# ---------------------------------------------------------------------------


def _manifest() -> store.Manifest:
    return store.Manifest(
        store_format_version=1,
        created_at_utc="2026-07-14T09:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc123",
        sources={},
    )


def test_a_quantile_track_exception_does_not_affect_challenge_2s_own_record(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "submissions.jsonl"
    payloads_dir = tmp_path / "payloads"
    point_payload = {"challenge_id": "2", "target_start": "x", "values": [1.0, 2.0]}
    fake_outcome = SubmissionOutcome(
        candidate_selected="core_gas", candidate_rank=1, skip_reason=None, payload=point_payload
    )

    with (
        patch(f"{MODULE}.SUBMISSIONS_LOG", log_path),
        patch(f"{MODULE}.PAYLOADS_DIR", payloads_dir),
        patch(f"{MODULE}.store") as mock_store,
        patch(f"{MODULE}.assemble_price_model_inputs", return_value=pd.DataFrame()),
        patch(f"{MODULE}.read_weather_runs", return_value=pd.DataFrame()),
        patch(f"{MODULE}.read_quarterhourly_prices", return_value=pd.DataFrame()),
        patch(f"{MODULE}.run_submission_for_day", return_value=fake_outcome),
        patch(f"{MODULE}.price_provenance_report", return_value=({}, None)),
        patch(
            f"{MODULE}.run_quantile_track_for_day",
            side_effect=RuntimeError("Challenge 8 is broken"),
        ) as mock_quantile_track,
    ):
        mock_store.load_store.return_value.manifest = _manifest()
        exit_code = run_daily_submission(
            _BEFORE_GATE_CLOSURE, nominal_slot="10:40", is_last_slot_of_day=True
        )

    mock_quantile_track.assert_called_once()
    # Challenge 2's own outcome is completely unaffected by the exception.
    assert exit_code == 0
    records = protocol.read_submission_records(log_path)
    assert len(records) == 1
    assert records[0]["candidate_selected"] == "core_gas"
    assert records[0].get("challenge_id", "2") == "2"
    written_payload_path = payloads_dir / f"{_TARGET_DAY.isoformat()}.json"
    assert written_payload_path.exists()
    assert json.loads(written_payload_path.read_text(encoding="utf-8")) == point_payload
