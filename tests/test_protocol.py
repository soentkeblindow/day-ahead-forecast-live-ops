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
    "submission_mode",
    "submission_id",
    "api_response_received_utc",
    "confirmed_via_query",
    "smoke_baseline_source_day",
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
    "load_patch_reference_day",
    "load_patch_weeks_back",
    "load_patch_skipped",
]

_SECTION_5_7_FIELDS = (
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


def test_reader_tolerates_a_real_protocol_version_2_line_without_section_5_6_fields(
    tmp_path: Path,
) -> None:
    """The same compatibility guarantee, this time for the 6.7.3 jump to
    protocol_version 3 (spec section 5.6): every line written under
    protocol_version 2 genuinely lacks the five new fields."""
    path = tmp_path / "submissions.jsonl"
    v2_line = _sample_record().to_dict()
    v2_line["protocol_version"] = 2
    for field_name in (
        "submission_mode",
        "submission_id",
        "api_response_received_utc",
        "confirmed_via_query",
        "smoke_baseline_source_day",
    ):
        del v2_line[field_name]
    path.write_text(json.dumps(v2_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0]["protocol_version"] == 2
    assert records[0].get("submission_mode") is None
    assert records[0].get("submission_id") is None
    assert records[0].get("api_response_received_utc") is None
    assert records[0].get("confirmed_via_query") is None
    assert records[0].get("smoke_baseline_source_day") is None


def test_section_5_6_fields_default_none_and_round_trip(tmp_path: Path) -> None:
    default_record = _sample_record()
    assert default_record.submission_mode is None
    assert default_record.submission_id is None
    assert default_record.api_response_received_utc is None
    assert default_record.confirmed_via_query is None
    assert default_record.smoke_baseline_source_day is None

    path = tmp_path / "submissions.jsonl"
    append_submission_record(
        path,
        _sample_record(
            submission_mode="live",
            submission_id=42,
            api_response_received_utc="2026-09-19T10:41:03+00:00",
            confirmed_via_query=True,
            smoke_baseline_source_day=None,
        ),
    )
    records = read_submission_records(path)
    assert records[0]["submission_mode"] == "live"
    assert records[0]["submission_id"] == 42
    assert records[0]["api_response_received_utc"] == "2026-09-19T10:41:03+00:00"
    assert records[0]["confirmed_via_query"] is True


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


def test_reader_tolerates_a_real_protocol_version_3_line_without_section_5_7_fields(
    tmp_path: Path,
) -> None:
    """Same compatibility guarantee as the v1->v2 and v2->v3 tests above,
    this time for the 6.9 jump to protocol_version 4 (spec section 5.7):
    every line written under protocol_version 3 genuinely lacks the
    seventeen new fields."""
    path = tmp_path / "submissions.jsonl"
    v3_line = _sample_record().to_dict()
    v3_line["protocol_version"] = 3
    for field_name in _SECTION_5_7_FIELDS:
        del v3_line[field_name]
    path.write_text(json.dumps(v3_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0]["protocol_version"] == 3
    for field_name in _SECTION_5_7_FIELDS:
        assert records[0].get(field_name) is None


def test_section_5_7_fields_default_and_round_trip(tmp_path: Path) -> None:
    """Spec 6.9 section 5.7 -- step 3 only adds the schema (docs/
    sprint6_step6_9_log.md), so every field defaults to None/empty here;
    later 6.9 steps are what actually compute real values. Both directions
    checked: the defaults, and that real values round-trip unchanged."""
    default_record = _sample_record()
    assert default_record.candidate_rank is None
    assert default_record.candidates_evaluated == []
    assert default_record.nwp_available is None
    assert default_record.nwp_unavailable_reason is None
    assert default_record.price_provenance == {}
    assert default_record.price_source_conflicts is None
    assert default_record.load_forecast_source is None
    assert default_record.n_training_days is None
    assert default_record.age_of_last_complete_day is None
    assert default_record.training_tolerance_used is None
    assert default_record.training_missing_days == []
    assert default_record.known_weather_defect_days == []
    assert default_record.rnw_label_edge_age_days is None
    assert default_record.target_day_fills == []
    assert default_record.downgrade_blocked is None
    assert default_record.best_accepted_rank_before is None
    assert default_record.capacity_anchor_days_left is None

    path = tmp_path / "submissions.jsonl"
    append_submission_record(
        path,
        _sample_record(
            candidate_rank=2,
            candidates_evaluated=[
                {"name": "core_gas", "rank": 1, "outcome": "failed", "reason": "load missing"},
                {"name": "core_gas_ec", "rank": 2, "outcome": "selected", "reason": None},
            ],
            nwp_available=True,
            price_provenance={"training_labels": "entsoe", "price_lags": "energy_charts"},
            price_source_conflicts=0,
            load_forecast_source="energy_charts",
            n_training_days=87,
            age_of_last_complete_day=1,
            training_tolerance_used=True,
            training_missing_days=["2026-09-20"],
            known_weather_defect_days=["2026-06-23"],
            rnw_label_edge_age_days=2,
            target_day_fills=[
                {"group": "nwp_residual", "n_hours": 2, "hours": [22, 23], "action": "forward_fill"}
            ],
            downgrade_blocked=False,
            best_accepted_rank_before=None,
            capacity_anchor_days_left=35,
        ),
    )
    records = read_submission_records(path)
    r = records[0]
    assert r["candidate_rank"] == 2
    assert r["candidates_evaluated"][1]["name"] == "core_gas_ec"
    assert r["nwp_available"] is True
    assert r["price_provenance"] == {"training_labels": "entsoe", "price_lags": "energy_charts"}
    assert r["price_source_conflicts"] == 0
    assert r["load_forecast_source"] == "energy_charts"
    assert r["n_training_days"] == 87
    assert r["age_of_last_complete_day"] == 1
    assert r["training_tolerance_used"] is True
    assert r["training_missing_days"] == ["2026-09-20"]
    assert r["known_weather_defect_days"] == ["2026-06-23"]
    assert r["rnw_label_edge_age_days"] == 2
    assert r["target_day_fills"][0]["group"] == "nwp_residual"
    assert r["downgrade_blocked"] is False
    assert r["best_accepted_rank_before"] is None
    assert r["capacity_anchor_days_left"] == 35


def test_reader_tolerates_a_real_protocol_version_4_line_without_v5_fields(
    tmp_path: Path,
) -> None:
    """Same compatibility guarantee as the earlier version-jump tests, for
    the 6.9 Schritt 11 jump from protocol_version 4 to 5."""
    path = tmp_path / "submissions.jsonl"
    v4_line = _sample_record().to_dict()
    v4_line["protocol_version"] = 4
    for field_name in ("load_patch_reference_day", "load_patch_weeks_back", "load_patch_skipped"):
        del v4_line[field_name]
    path.write_text(json.dumps(v4_line) + "\n", encoding="utf-8")

    records = read_submission_records(path)
    assert len(records) == 1
    assert records[0]["protocol_version"] == 4
    assert records[0].get("load_patch_reference_day") is None
    assert records[0].get("load_patch_weeks_back") is None
    assert records[0].get("load_patch_skipped") is None


def test_v5_load_patch_fields_default_and_round_trip(tmp_path: Path) -> None:
    """spec 6.9 section 5.7 -- defaults first, then a real value round-trip,
    same discipline as test_section_5_7_fields_default_and_round_trip."""
    default_record = _sample_record()
    assert default_record.load_patch_reference_day is None
    assert default_record.load_patch_weeks_back is None
    assert default_record.load_patch_skipped == []

    path = tmp_path / "submissions.jsonl"
    append_submission_record(
        path,
        _sample_record(
            load_patch_reference_day="2026-10-01",
            load_patch_weeks_back=1,
            load_patch_skipped=[["2026-10-08", "holiday"]],
        ),
    )
    records = read_submission_records(path)
    r = records[0]
    assert r["load_patch_reference_day"] == "2026-10-01"
    assert r["load_patch_weeks_back"] == 1
    assert r["load_patch_skipped"] == [["2026-10-08", "holiday"]]


def test_protocol_version_is_5() -> None:
    """spec 6.9 section 5.7, Schritt 11: v5 adds load_patch_reference_day/
    weeks_back/skipped, appended to the end of _KEY_ORDER."""
    assert PROTOCOL_VERSION == 5
