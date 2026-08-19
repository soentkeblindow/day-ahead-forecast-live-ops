"""Unit tests for the raw-series availability audit. No network access.

All fetch calls are mocked via ops.availability_audit._FETCH_FUNCTIONS and
arena.catalog.get_challenge; synthetic frames stand in for real ENTSO-E/Yahoo
responses.
"""

import datetime as dt
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.config import PROJECT_ROOT
from energy_price_forecast.features.availability import Availability
from energy_price_forecast.ops import availability_audit as audit

MODULE = "energy_price_forecast.ops.availability_audit"

CHALLENGE = ChallengeSpec(
    challenge_id="2",
    name="Day-Ahead Prices | Germany-Luxembourg | Point Forecast",
    timezone="Europe/Berlin",
    resolution_minutes=15,
    precision_decimals=2,
    allow_negative=True,
    max_forecast_points=None,
    raw={
        "submission_window": {"allow_multiple": True, "selection_policy": "latest_before_deadline"},
    },
)


def _series(
    start_local_iso: str, periods: int, freq_minutes: int, value: float = 100.0
) -> pd.Series:
    naive_start = pd.Timestamp(start_local_iso).tz_localize(None)
    idx = pd.date_range(naive_start, periods=periods, freq=f"{freq_minutes}min", tz="Europe/Berlin")
    return pd.Series(value, index=idx.tz_convert("UTC"))


def _day_ahead_price_frame(
    start_local_iso: str, periods: int, freq_minutes: int = 60
) -> pd.DataFrame:
    return _series(start_local_iso, periods, freq_minutes).rename("day_ahead_price").to_frame()


@pytest.fixture(autouse=True)
def _no_network_challenge_call() -> Any:
    with patch(f"{MODULE}.get_challenge", return_value=CHALLENGE) as p:
        yield p


# ---------------------------------------------------------------------------
# Window derivation per availability class, incl. a DST day
# ---------------------------------------------------------------------------


def test_critical_windows_per_availability_class() -> None:
    d = dt.date(2026, 8, 20)

    assert audit._critical_eval_window(Availability.DA_FORECAST, d) == audit.local_day_bounds(d)
    assert audit._critical_fetch_window(Availability.DA_FORECAST, d) == (
        audit.local_day_bounds(d - dt.timedelta(days=1))[0],
        audit.local_day_bounds(d)[1],
    )

    assert audit._critical_eval_window(Availability.DA_FIXED, d) == (
        audit.local_day_bounds(d - dt.timedelta(days=1))[0],
        audit.local_day_bounds(d)[0],
    )
    assert audit._critical_eval_window(Availability.DA_FIXED, d) == audit._critical_fetch_window(
        Availability.DA_FIXED, d
    )

    assert audit._critical_eval_window(Availability.RT_ACTUAL, d) == (
        audit.local_day_bounds(d - dt.timedelta(days=2))[0],
        audit.local_day_bounds(d - dt.timedelta(days=1))[0],
    )


def test_rt_actual_critical_window_spans_a_dst_transition_day() -> None:
    # D=2026-10-27 -> D-2..D-1 = local midnight 2026-10-25 to local midnight
    # 2026-10-26, which *is* the 25h fall-back day: the window must be 25h,
    # not a flat 24h.
    d = dt.date(2026, 10, 27)
    start, end = audit._critical_eval_window(Availability.RT_ACTUAL, d)
    assert end - start == pd.Timedelta(hours=25)


# ---------------------------------------------------------------------------
# Checklist derivation from two sources
# ---------------------------------------------------------------------------


def test_checklist_includes_base_registry_and_cross_border_by_prefix() -> None:
    checklist = {spec.column: spec for spec in audit.build_checklist()}
    assert checklist["day_ahead_price"].availability_class is Availability.DA_FIXED
    assert checklist["load_forecast_day_ahead"].availability_class is Availability.DA_FORECAST
    assert checklist["scheduled_net_de_to_fr"].availability_class is Availability.DA_FIXED
    assert checklist["physical_net_de_to_fr"].availability_class is Availability.RT_ACTUAL


def test_checklist_picks_up_a_new_neighbor_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "NEIGHBORS", [*audit.NEIGHBORS, "ZZ"])
    checklist = {spec.column for spec in audit.build_checklist()}
    assert "scheduled_net_de_to_zz" in checklist
    assert "physical_net_de_to_zz" in checklist


def test_checklist_picks_up_a_new_registry_entry_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(audit._RAW_AVAILABILITY, "synthetic_raw_col", Availability.RT_ACTUAL)
    monkeypatch.setitem(audit._BASE_FETCH_GROUP, "synthetic_raw_col", "day_ahead_price")
    checklist = {spec.column: spec for spec in audit.build_checklist()}
    assert checklist["synthetic_raw_col"].availability_class is Availability.RT_ACTUAL


def test_checklist_raises_a_clear_error_for_an_unmapped_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry entry with no _BASE_FETCH_GROUP mapping fails loud, not silently.

    _BASE_FETCH_GROUP is knowledge this audit invents (which client function
    fetches which column) that has no other source of truth in the codebase
    -- so it can't be derived automatically the way the neighbor expansion
    can, and a maintainer adding a raw column must add this mapping too.
    """
    monkeypatch.setitem(audit._RAW_AVAILABILITY, "unmapped_raw_col", Availability.RT_ACTUAL)
    with pytest.raises(KeyError, match="unmapped_raw_col"):
        audit.build_checklist()


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "interim" / "hourly.parquet").exists(),
    reason="data/interim/hourly.parquet not present -- local integrity check, not a CI gate (spec §5.1)",
)
def test_checklist_matches_hourly_parquet_columns() -> None:
    """Known pre-existing discrepancy, deliberately not special-cased away
    (owner decision, 2026-08-19): hourly.parquet -- copied, not rebuilt, per
    CLAUDE.md -- carries solar_actual/wind_onshore_actual/wind_offshore_actual,
    three raw columns absent from _RAW_AVAILABILITY and unreferenced anywhere
    in src/tests/scripts. Likely a leftover from an earlier pipeline stage
    that predates the current fetch_generation_by_type-based gen_solar/
    gen_wind_onshore/gen_wind_offshore columns, but that is a data-provenance
    question outside 6.2's scope, not something this test should paper over.
    """
    hourly = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "hourly.parquet")
    checklist_columns = {spec.column for spec in audit.build_checklist()}
    real_columns = set(hourly.columns)

    assert checklist_columns == real_columns


# ---------------------------------------------------------------------------
# Row building: coverage counting on synthetic frames
# ---------------------------------------------------------------------------


def test_full_coverage_is_ok() -> None:
    spec = audit.SeriesSpec("day_ahead_price", Availability.DA_FIXED, "day_ahead_price")
    target_date = dt.date(2026, 8, 20)
    df = _day_ahead_price_frame("2026-08-19T00:00:00+02:00", 24)  # exactly D-1, full day
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "critical", target_date, run_ts_utc, run_ts_local)

    assert row["status"] == "ok"
    assert row["coverage_ratio"] == 1.0
    assert row["expected_count"] == 24
    assert row["present_count"] == 24
    assert row["native_resolution_min"] == 60


def test_a_gap_is_partial_with_correct_coverage_ratio() -> None:
    spec = audit.SeriesSpec("day_ahead_price", Availability.DA_FIXED, "day_ahead_price")
    target_date = dt.date(2026, 8, 20)
    df = _day_ahead_price_frame("2026-08-19T00:00:00+02:00", 18)  # only 18 of 24 hours
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "critical", target_date, run_ts_utc, run_ts_local)

    assert row["status"] == "partial"
    assert row["expected_count"] == 24
    assert row["present_count"] == 18
    assert row["coverage_ratio"] == round(18 / 24, 4)


def test_empty_result_is_missing() -> None:
    spec = audit.SeriesSpec("day_ahead_price", Availability.DA_FIXED, "day_ahead_price")
    target_date = dt.date(2026, 8, 20)
    df = pd.DataFrame(columns=["day_ahead_price"])
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "critical", target_date, run_ts_utc, run_ts_local)

    assert row["status"] == "missing"
    assert row["present_count"] == 0


# ---------------------------------------------------------------------------
# DA_FORECAST fetch-range extension and latest_target_offset_days
# ---------------------------------------------------------------------------


def test_da_forecast_not_yet_published_is_missing_with_offset_minus_one() -> None:
    spec = audit.SeriesSpec("load_forecast_day_ahead", Availability.DA_FORECAST, "load")
    target_date = dt.date(2026, 8, 20)
    # Data reaches through D-1 23:45 (quarter-hourly) but nothing for D itself.
    df = _series("2026-08-19T00:00:00+02:00", 96, 15).rename("load_forecast_day_ahead").to_frame()
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "critical", target_date, run_ts_utc, run_ts_local)

    assert row["coverage_ratio"] == 0.0
    assert row["status"] == "missing"
    assert row["latest_target_offset_days"] == -1
    assert row["latest_target_local"] is not None


def test_da_forecast_published_for_target_day_has_offset_zero() -> None:
    spec = audit.SeriesSpec("load_forecast_day_ahead", Availability.DA_FORECAST, "load")
    target_date = dt.date(2026, 8, 20)
    # Data reaches through D itself (192 quarter-hours = D-1 and D).
    df = _series("2026-08-19T00:00:00+02:00", 192, 15).rename("load_forecast_day_ahead").to_frame()
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "critical", target_date, run_ts_utc, run_ts_local)

    assert row["coverage_ratio"] == 1.0
    assert row["status"] == "ok"
    assert row["latest_target_offset_days"] == 0


def test_latest_target_local_stays_empty_for_training_window() -> None:
    spec = audit.SeriesSpec("load_forecast_day_ahead", Availability.DA_FORECAST, "load")
    target_date = dt.date(2026, 8, 20)
    df = _series("2026-08-19T00:00:00+02:00", 192, 15).rename("load_forecast_day_ahead").to_frame()
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._availability_row(df, None, spec, "training", target_date, run_ts_utc, run_ts_local)

    assert row["latest_target_local"] is None
    assert row["latest_target_offset_days"] is None


# ---------------------------------------------------------------------------
# Client errors are data, not job failures
# ---------------------------------------------------------------------------


def test_client_error_produces_an_error_row_without_raising() -> None:
    spec = audit.SeriesSpec("day_ahead_price", Availability.DA_FIXED, "day_ahead_price")
    target_date = dt.date(2026, 8, 20)
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)
    error = RuntimeError("x" * 500)  # long message, must be truncated

    row = audit._availability_row(
        None, error, spec, "critical", target_date, run_ts_utc, run_ts_local
    )

    assert row["status"] == "error"
    assert row["error"] is not None
    assert len(row["error"]) <= 200
    assert "RuntimeError" in row["error"]


def test_run_availability_audit_does_not_raise_when_a_fetch_group_errors() -> None:
    def _boom(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise RuntimeError("ENTSO-E is down")

    # Every group must be mocked, not just the one under test: an unmocked
    # entry in _FETCH_FUNCTIONS is a real ENTSO-E/Yahoo client call, which
    # this "no network" test file must never make.
    functions = _synthetic_fetch_functions()
    functions["day_ahead_price"] = _boom

    with patch.dict(audit._FETCH_FUNCTIONS, functions):
        outcome = audit.run_availability_audit(
            pd.Timestamp("2026-08-19T09:00:00+00:00"),
            "workflow_dispatch",
            is_first_run_of_day=False,
        )

    price_rows = [r for r in outcome.availability_rows if r["series"] == "day_ahead_price"]
    assert len(price_rows) == 1
    assert price_rows[0]["status"] == "error"


# ---------------------------------------------------------------------------
# COMMODITY special rule
# ---------------------------------------------------------------------------


def test_commodity_weekend_gap_is_ok_not_partial() -> None:
    spec = audit.SeriesSpec("ttf_gas_eur_per_mwh", Availability.COMMODITY, "ttf_gas")
    # Friday 2026-08-14 is the last settlement; D is Monday 2026-08-17 --
    # weekend gap, staleness_days=3, still <= the readability threshold of 4.
    target_date = dt.date(2026, 8, 17)
    df = pd.DataFrame(
        {"ttf_gas_eur_per_mwh": [30.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-14T00:00:00+02:00")]).tz_convert("UTC"),
    )
    run_ts_utc = pd.Timestamp("2026-08-16T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._commodity_row(df, None, spec, target_date, run_ts_utc, run_ts_local)

    assert row["staleness_days"] == 3
    assert row["status"] == "ok"
    assert row["expected_count"] is None
    assert row["coverage_ratio"] is None


def test_commodity_stale_beyond_threshold_is_partial() -> None:
    spec = audit.SeriesSpec("ttf_gas_eur_per_mwh", Availability.COMMODITY, "ttf_gas")
    target_date = dt.date(2026, 8, 20)
    df = pd.DataFrame(
        {"ttf_gas_eur_per_mwh": [30.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-14T00:00:00+02:00")]).tz_convert("UTC"),
    )
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._commodity_row(df, None, spec, target_date, run_ts_utc, run_ts_local)

    assert row["staleness_days"] == 6
    assert row["status"] == "partial"


def test_commodity_empty_result_is_missing() -> None:
    spec = audit.SeriesSpec("ttf_gas_eur_per_mwh", Availability.COMMODITY, "ttf_gas")
    target_date = dt.date(2026, 8, 20)
    df = pd.DataFrame(columns=["ttf_gas_eur_per_mwh"])
    run_ts_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")
    run_ts_local = run_ts_utc.tz_convert(audit.LOCAL_TZ)

    row = audit._commodity_row(df, None, spec, target_date, run_ts_utc, run_ts_local)

    assert row["status"] == "missing"
    assert row["staleness_days"] is None


# ---------------------------------------------------------------------------
# Run frequency: training only on the first run of the local date
# ---------------------------------------------------------------------------


def test_is_first_run_of_local_date_true_when_no_file(tmp_path: Path) -> None:
    assert (
        audit.is_first_run_of_local_date(tmp_path / "audit_runs.csv", dt.date(2026, 8, 20)) is True
    )


def test_is_first_run_of_local_date_false_when_date_already_present(tmp_path: Path) -> None:
    path = tmp_path / "audit_runs.csv"
    pd.DataFrame({"local_date": ["2026-08-20"]}).to_csv(path, index=False)
    assert audit.is_first_run_of_local_date(path, dt.date(2026, 8, 20)) is False


def test_is_first_run_of_local_date_true_for_a_new_date(tmp_path: Path) -> None:
    path = tmp_path / "audit_runs.csv"
    pd.DataFrame({"local_date": ["2026-08-19"]}).to_csv(path, index=False)
    assert audit.is_first_run_of_local_date(path, dt.date(2026, 8, 20)) is True


def _synthetic_fetch_functions() -> dict[str, Any]:
    def make(columns: list[str], periods: int = 200) -> Any:
        def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            idx = pd.date_range(start, periods=periods, freq="1h")
            return pd.DataFrame({c: 1.0 for c in columns}, index=idx)

        return fetch

    return {
        "day_ahead_price": make(["day_ahead_price"]),
        "load": make(["load_actual", "load_forecast_day_ahead"]),
        "wind_solar_forecast": make(
            ["wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"]
        ),
        "generation": make(
            [
                "gen_nuclear",
                "gen_lignite",
                "gen_hard_coal",
                "gen_gas",
                "gen_oil",
                "gen_biomass",
                "gen_hydro",
                "gen_wind_onshore",
                "gen_wind_offshore",
                "gen_solar",
                "gen_other",
            ]
        ),
        "scheduled_exchanges": make([f"scheduled_net_de_to_{n.lower()}" for n in audit.NEIGHBORS]),
        "cross_border_flows": make([f"physical_net_de_to_{n.lower()}" for n in audit.NEIGHBORS]),
        "ttf_gas": make(["ttf_gas_eur_per_mwh"]),
        "eua_co2": make(["eua_co2_eur_per_t"]),
    }


def test_training_window_only_checked_on_first_run_of_day() -> None:
    with patch.dict(audit._FETCH_FUNCTIONS, _synthetic_fetch_functions()):
        not_first = audit.run_availability_audit(
            pd.Timestamp("2026-08-19T09:00:00+00:00"), "schedule", is_first_run_of_day=False
        )
        first = audit.run_availability_audit(
            pd.Timestamp("2026-08-19T09:00:00+00:00"), "schedule", is_first_run_of_day=True
        )

    assert not any(r["window_kind"] == "training" for r in not_first.availability_rows)
    assert any(r["window_kind"] == "training" for r in first.availability_rows)
    assert not_first.audit_run_row["windows_checked"] == "critical"
    assert first.audit_run_row["windows_checked"] == "critical+training"


# ---------------------------------------------------------------------------
# audit_runs.csv bookkeeping: n_client_calls and status counts
# ---------------------------------------------------------------------------


def test_n_client_calls_counts_actual_fetch_group_calls() -> None:
    call_counter = {"n": 0}
    functions = _synthetic_fetch_functions()

    def counting_wrapper(fn: Any) -> Any:
        def wrapped(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            call_counter["n"] += 1
            return fn(start, end)

        return wrapped

    wrapped_functions = {name: counting_wrapper(fn) for name, fn in functions.items()}

    with patch.dict(audit._FETCH_FUNCTIONS, wrapped_functions):
        outcome = audit.run_availability_audit(
            pd.Timestamp("2026-08-19T09:00:00+00:00"),
            "workflow_dispatch",
            is_first_run_of_day=False,
        )

    assert outcome.audit_run_row["n_client_calls"] == call_counter["n"]
    assert call_counter["n"] == len(
        audit._FETCH_FUNCTIONS
    )  # one call per fetch group, not per column


def test_status_counts_sum_to_availability_row_count() -> None:
    with patch.dict(audit._FETCH_FUNCTIONS, _synthetic_fetch_functions()):
        outcome = audit.run_availability_audit(
            pd.Timestamp("2026-08-19T09:00:00+00:00"), "workflow_dispatch", is_first_run_of_day=True
        )

    counted = (
        outcome.audit_run_row["n_ok"]
        + outcome.audit_run_row["n_partial"]
        + outcome.audit_run_row["n_missing"]
        + outcome.audit_run_row["n_error"]
    )
    assert counted == len(outcome.availability_rows)


# ---------------------------------------------------------------------------
# CSV append behavior (all three log files)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "columns",
    [audit.AVAILABILITY_COLUMNS, audit.CHALLENGE_CATALOG_COLUMNS, audit.AUDIT_RUNS_COLUMNS],
)
def test_append_csv_rows_writes_header_only_once(tmp_path: Path, columns: list[str]) -> None:
    path = tmp_path / "log.csv"
    row = dict.fromkeys(columns, "x")

    audit.append_csv_rows([row], path, columns)
    first_write = path.read_text(encoding="utf-8")
    audit.append_csv_rows([row], path, columns)
    second_write = path.read_text(encoding="utf-8")

    assert first_write.splitlines()[0] == ",".join(columns)
    assert second_write.startswith(first_write)  # existing bytes untouched, header not duplicated
    assert second_write.count(",".join(columns)) == 1


def test_append_csv_rows_keeps_stable_column_order_regardless_of_dict_order(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    columns = ["a", "b", "c"]

    audit.append_csv_rows([{"c": 3, "a": 1, "b": 2}], path, columns)

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "a,b,c"


# ---------------------------------------------------------------------------
# Challenge catalog snapshot -- Arena-only, no ENTSO-E/Yahoo touched
# ---------------------------------------------------------------------------


def test_build_challenge_catalog_row_computes_deadline_and_target_from_d() -> None:
    # now_utc such that D (tomorrow, Europe/Berlin) is 2026-08-20.
    now_utc = pd.Timestamp("2026-08-19T09:00:00+00:00")

    row = audit.build_challenge_catalog_row(now_utc)

    assert row["challenge_id"] == "2"
    assert row["resolution"] == 15
    assert row["timezone"] == "Europe/Berlin"
    assert row["target_start"] == "2026-08-20T00:00:00+02:00"
    assert row["target_end"] == "2026-08-21T00:00:00+02:00"
    assert row["deadline"] == "2026-08-19T12:00:00+02:00"  # 12:00 local on D-1
    assert row["expected_values"] == 96
    assert row["allow_multiple"] is True
    assert row["selection_policy"] == "latest_before_deadline"
    assert row["precision_decimals"] == 2
    assert row["allow_negative"] is True
    assert row["max_forecast_points"] is None
    assert len(row["spec_sha256"]) == 64  # sha256 hex digest


def test_build_challenge_catalog_row_deadline_is_dst_correct() -> None:
    # D=2026-10-26 -> D-1=2026-10-25, the fall-back transition day itself:
    # deadline must be 12:00 *wall-clock* local time (+01:00, post-transition
    # -- the transition happens at 03:00->02:00 local, well before noon), not
    # "12 hours after D-1's local midnight" (+02:00, pre-transition), which
    # is what Timedelta(hours=12) on local midnight used to produce (11:00
    # wall-clock -- the bug this test was written to catch).
    now_utc = pd.Timestamp("2026-10-25T00:30:00+00:00")  # still CEST, D-2 side of midnight

    row = audit.build_challenge_catalog_row(now_utc)

    assert row["target_start"] == "2026-10-26T00:00:00+01:00"
    assert row["deadline"] == "2026-10-25T12:00:00+01:00"
