"""Unit tests for ops/protocol.py -- the JSON-lines submission log (spec
6.7.2, sections 2.8/5.7/7).
"""

from __future__ import annotations

import json
from pathlib import Path

from energy_price_forecast.ops.protocol import (
    PROTOCOL_VERSION,
    SubmissionRecord,
    append_submission_record,
    read_submission_records,
)


def _sample_record(**overrides: object) -> SubmissionRecord:
    defaults: dict[str, object] = {
        "run_timestamp_utc": "2026-09-12T10:40:00+00:00",
        "run_id": "1",
        "run_url": "",
        "code_sha": "abc123",
        "target_day": "2026-09-13",
        "nominal_slot": "10:40",
        "gate_closure_ok": True,
    }
    defaults.update(overrides)
    return SubmissionRecord(**defaults)  # type: ignore[arg-type]


_EXPECTED_KEY_ORDER = [
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
]


def test_appended_line_carries_protocol_version_and_fixed_key_order(tmp_path: Path) -> None:
    path = tmp_path / "submissions.jsonl"
    append_submission_record(path, _sample_record())

    raw_line = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw_line)
    assert parsed["protocol_version"] == PROTOCOL_VERSION

    # Fixed key order: the keys appear in the raw JSON text in exactly the
    # declared order, not just after re-parsing into a dict (Python dicts
    # preserve insertion order, so a parsed-dict check alone wouldn't catch
    # a writer that reordered keys before serialising).
    pairs = json.loads(raw_line, object_pairs_hook=list)
    keys_in_text_order = [k for k, _ in pairs]
    assert keys_in_text_order == _EXPECTED_KEY_ORDER


def test_two_appended_records_are_two_lines(tmp_path: Path) -> None:
    path = tmp_path / "submissions.jsonl"
    append_submission_record(path, _sample_record(run_id="1"))
    append_submission_record(path, _sample_record(run_id="2"))

    records = read_submission_records(path)
    assert [r["run_id"] for r in records] == ["1", "2"]


def test_reader_tolerates_a_line_with_an_unknown_extra_field(tmp_path: Path) -> None:
    """spec section 7: 'eine Zeile mit einem zusaetzlichen Feld wird von
    einem Leser ohne dieses Feld gelesen, ohne zu werfen' -- the forward-
    compatibility property JSON lines were chosen for."""
    path = tmp_path / "submissions.jsonl"
    future_line = _sample_record().to_dict()
    future_line["candidate_column"] = "phase_2_fallback"  # a field from a future protocol_version
    path.write_text(json.dumps(future_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0]["candidate_column"] == "phase_2_fallback"
    assert records[0]["run_id"] == "1"


def test_reader_tolerates_an_old_line_missing_a_newer_field(tmp_path: Path) -> None:
    """The reverse direction: a line written under an older protocol_version
    (missing a field a newer reader might look for) must not raise --
    callers read via .get(...), never direct indexing."""
    path = tmp_path / "submissions.jsonl"
    old_line = _sample_record().to_dict()
    del old_line["missing_features"]  # simulate a field added after this line was written
    path.write_text(json.dumps(old_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0].get("missing_features") is None
    assert "missing_features" not in records[0]


def test_reader_tolerates_a_real_protocol_version_1_line_without_training_label_fields(
    tmp_path: Path,
) -> None:
    """Restarbeit Teil A.5: the first real, not synthetic, practical case of
    the compatibility guarantee above -- protocol_version 2 (Restarbeit
    Teil A) added n_training_rows/n_training_labels; every line written
    under protocol_version 1 genuinely lacks them, not just as a test
    fixture but as a real fact about logs/submissions.jsonl's own history."""
    path = tmp_path / "submissions.jsonl"
    v1_line = _sample_record().to_dict()
    v1_line["protocol_version"] = 1
    del v1_line["n_training_rows"]
    del v1_line["n_training_labels"]
    path.write_text(json.dumps(v1_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0]["protocol_version"] == 1
    assert records[0].get("n_training_rows") is None
    assert records[0].get("n_training_labels") is None


def test_missing_features_names_each_feature_individually() -> None:
    record = _sample_record(
        candidate_selected=None,
        skip_reason="incomplete target row",
        missing_features=["price_lag_24h", "renewable_share_forecast_nwp"],
    )
    d = record.to_dict()
    assert d["missing_features"] == ["price_lag_24h", "renewable_share_forecast_nwp"]


def test_read_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert read_submission_records(tmp_path / "does_not_exist.jsonl") == []


def test_candidate_selected_and_skip_reason_are_mutually_exclusive_in_practice() -> None:
    selected = _sample_record(candidate_selected="full_live_set")
    assert selected.candidate_selected is not None
    assert selected.skip_reason is None

    silent = _sample_record(skip_reason="Check A: weather run unavailable")
    assert silent.skip_reason is not None
    assert silent.candidate_selected is None


def test_submitted_defaults_false_and_api_fields_default_none() -> None:
    record = _sample_record()
    assert record.submitted is False
    assert record.api_status is None
    assert record.api_message is None


def test_training_label_fields_default_none_and_round_trip(tmp_path: Path) -> None:
    """Restarbeit Teil A: both None for a run that never reached the fit
    step; both real ints round-trip through a written-then-read line
    unchanged when a run did."""
    default_record = _sample_record()
    assert default_record.n_training_rows is None
    assert default_record.n_training_labels is None

    path = tmp_path / "submissions.jsonl"
    append_submission_record(path, _sample_record(n_training_rows=2160, n_training_labels=2136))
    records = read_submission_records(path)
    assert records[0]["n_training_rows"] == 2160
    assert records[0]["n_training_labels"] == 2136
