"""Unit tests for scripts/oos_store_job.py (spec 7.1, sections 2.2/2.3) --
the backfill/daily-top-up orchestration only. compute_oos_rows_for_day
itself is mocked throughout (its own tests live in
tests/test_run_daily_submission.py); these tests verify which days/rows
get attempted and how the result is merged/written, not the model fit.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pandas as pd

from energy_price_forecast.ops.oos_store import OosStoreRow, rows_to_frame
from scripts.oos_store_job import (
    DAILY_TOPUP_LOOKBACK_DAYS,
    run_backfill,
    run_daily_topup,
)

_WEATHER_RUN = pd.Timestamp("2026-07-14T00:00:00", tz="UTC")


def _synthetic_row(target_day: dt.date, ladder_row: str) -> OosStoreRow:
    return OosStoreRow(
        target_day=target_day,
        timestamp_utc=pd.Timestamp(target_day, tz="UTC"),
        ladder_row=ladder_row,
        forecast=42.0,
        model_version="price_v1",
        weather_run_init_utc=_WEATHER_RUN,
        source="backfill",
        created_at_utc=pd.Timestamp("2026-07-15T06:00:00", tz="UTC"),
    )


def test_run_backfill_computes_only_missing_days_and_writes_the_merged_result() -> None:
    day1 = dt.date(2026, 7, 1)
    day2 = dt.date(2026, 7, 2)
    existing = rows_to_frame(
        [_synthetic_row(day, row_name) for day in (day1, day2) for row_name in ("core_gas",)]
    )

    def fake_compute(df, renewables_predictions, prices_qh, target_day, *, source, only_rows=None):
        assert source == "backfill"
        return [_synthetic_row(target_day, row_name) for row_name in only_rows]

    with (
        patch("scripts.oos_store_job.load_oos_store", return_value=existing),
        patch("scripts.oos_store_job.assemble_price_model_inputs", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_quarterhourly_prices", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.oos_store_job._renewables_predictions_for_range",
            return_value=pd.DataFrame(),
        ),
        patch(
            "scripts.oos_store_job.compute_oos_rows_for_day", side_effect=fake_compute
        ) as mock_compute,
        patch("scripts.oos_store_job.write_oos_store") as mock_write,
    ):
        report = run_backfill(start=day1, end=day2)

    # core_gas already stored for both days -- only the other two rows are
    # ever requested, for both days (spec section 2.3's per-row re-attempt).
    requested_rows = {call.kwargs["only_rows"] for call in mock_compute.call_args_list}
    assert requested_rows == {frozenset({"core_gas_loadpatch", "base"})}
    assert report.days_attempted == 2
    assert report.rows_added == 4  # 2 days x 2 missing rows
    assert report.fully_skipped_days == []
    mock_write.assert_called_once()


def test_run_backfill_does_nothing_when_every_day_is_already_fully_stored() -> None:
    day = dt.date(2026, 7, 1)
    existing = rows_to_frame(
        [_synthetic_row(day, row_name) for row_name in ("core_gas", "core_gas_loadpatch", "base")]
    )

    with (
        patch("scripts.oos_store_job.load_oos_store", return_value=existing),
        patch("scripts.oos_store_job.assemble_price_model_inputs") as mock_assemble,
        patch("scripts.oos_store_job.compute_oos_rows_for_day") as mock_compute,
        patch("scripts.oos_store_job.write_oos_store") as mock_write,
    ):
        report = run_backfill(start=day, end=day)

    assert report.days_attempted == 0
    assert report.rows_added == 0
    mock_assemble.assert_not_called()
    mock_compute.assert_not_called()
    mock_write.assert_not_called()


def test_run_backfill_records_a_fully_skipped_day_without_failing_the_run() -> None:
    day = dt.date(2026, 7, 1)
    empty_store = rows_to_frame([])

    with (
        patch("scripts.oos_store_job.load_oos_store", return_value=empty_store),
        patch("scripts.oos_store_job.assemble_price_model_inputs", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_quarterhourly_prices", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.oos_store_job._renewables_predictions_for_range",
            return_value=pd.DataFrame(),
        ),
        patch("scripts.oos_store_job.compute_oos_rows_for_day", return_value=[]),
        patch("scripts.oos_store_job.write_oos_store") as mock_write,
    ):
        report = run_backfill(start=day, end=day)

    assert report.days_attempted == 1
    assert report.rows_added == 0
    assert report.fully_skipped_days == [day]
    mock_write.assert_not_called()


def test_run_daily_topup_covers_the_lookback_window_ending_yesterday() -> None:
    as_of = pd.Timestamp("2026-07-15T10:00:00", tz="UTC")
    empty_store = rows_to_frame([])
    seen_days: set[dt.date] = set()

    def fake_compute(df, renewables_predictions, prices_qh, target_day, *, source, only_rows=None):
        assert source == "daily"
        seen_days.add(target_day)
        return [_synthetic_row(target_day, "base")]

    with (
        patch("scripts.oos_store_job.load_oos_store", return_value=empty_store),
        patch("scripts.oos_store_job.assemble_price_model_inputs", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_quarterhourly_prices", return_value=pd.DataFrame()),
        patch("scripts.oos_store_job.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.oos_store_job._renewables_predictions_for_range",
            return_value=pd.DataFrame(),
        ),
        patch("scripts.oos_store_job.compute_oos_rows_for_day", side_effect=fake_compute),
        patch("scripts.oos_store_job.write_oos_store"),
    ):
        report = run_daily_topup(as_of=as_of)

    yesterday = dt.date(2026, 7, 14)
    expected_start = yesterday - dt.timedelta(days=DAILY_TOPUP_LOOKBACK_DAYS - 1)
    assert seen_days == set(pd.date_range(expected_start, yesterday, freq="D").date)
    assert report.days_attempted == DAILY_TOPUP_LOOKBACK_DAYS
