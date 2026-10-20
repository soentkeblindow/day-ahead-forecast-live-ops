"""Unit tests for ops/oos_store.py -- the OOS-Store schema/merge/I-O layer
(spec 7.1, section 2). All tests work against synthetic frames only --
scripts/oos_store_job.py's own tests cover the real pipeline wiring.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.ops.oos_store import (
    LADDER_ROW_NAMES,
    OOS_STORE_COLUMNS,
    OosStoreRow,
    load_oos_store,
    merge_rows,
    missing_target_days,
    rows_to_frame,
    validate_oos_store_frame,
    write_oos_store,
)

_TS = pd.Timestamp("2026-07-15T00:00:00", tz="UTC")
_WEATHER_RUN = pd.Timestamp("2026-07-14T00:00:00", tz="UTC")


def _row(
    *,
    target_day: dt.date = dt.date(2026, 7, 15),
    timestamp_utc: pd.Timestamp = _TS,
    ladder_row: str = "core_gas",
    forecast: float = 42.0,
    weather_run_init_utc: pd.Timestamp | None = _WEATHER_RUN,
) -> OosStoreRow:
    return OosStoreRow(
        target_day=target_day,
        timestamp_utc=timestamp_utc,
        ladder_row=ladder_row,
        forecast=forecast,
        model_version="price_v1",
        weather_run_init_utc=weather_run_init_utc,
        source="backfill",
        created_at_utc=pd.Timestamp("2026-07-15T06:00:00", tz="UTC"),
    )


def test_load_oos_store_returns_an_empty_typed_frame_when_the_file_does_not_exist(
    tmp_path: Path,
) -> None:
    frame = load_oos_store(tmp_path / "does_not_exist.csv")
    assert frame.empty
    assert list(frame.columns) == list(OOS_STORE_COLUMNS)


def test_validate_oos_store_frame_accepts_a_well_formed_frame() -> None:
    frame = rows_to_frame([_row()])
    validate_oos_store_frame(frame)  # must not raise


def test_validate_oos_store_frame_rejects_a_missing_column() -> None:
    frame = rows_to_frame([_row()]).drop(columns=["model_version"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_oos_store_frame(frame)


def test_validate_oos_store_frame_rejects_a_duplicate_key() -> None:
    frame = rows_to_frame([_row(), _row()])
    with pytest.raises(ValueError, match="duplicate OOS-store key"):
        validate_oos_store_frame(frame)


def test_validate_oos_store_frame_rejects_an_unknown_ladder_row() -> None:
    frame = rows_to_frame([_row(ladder_row="not_a_real_row")])
    with pytest.raises(ValueError, match="unknown ladder_row"):
        validate_oos_store_frame(frame)


def test_validate_oos_store_frame_rejects_a_nan_forecast() -> None:
    frame = rows_to_frame([_row(forecast=float("nan"))])
    with pytest.raises(ValueError, match="NaN forecast"):
        validate_oos_store_frame(frame)


def test_validate_oos_store_frame_rejects_a_naive_timestamp() -> None:
    frame = rows_to_frame([_row()])
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        validate_oos_store_frame(frame)


def test_merge_rows_adds_a_genuinely_new_row() -> None:
    existing = rows_to_frame([_row(ladder_row="core_gas")])
    new_rows = rows_to_frame([_row(ladder_row="base", weather_run_init_utc=None)])

    merged, n_added = merge_rows(existing, new_rows)

    assert n_added == 1
    assert len(merged) == 2
    assert set(merged["ladder_row"]) == {"core_gas", "base"}


def test_merge_rows_is_idempotent_on_an_identical_resubmission() -> None:
    existing = rows_to_frame([_row()])
    same_row_again = rows_to_frame([_row()])

    merged, n_added = merge_rows(existing, same_row_again)

    assert n_added == 0
    assert len(merged) == 1
    pd.testing.assert_frame_equal(merged.reset_index(drop=True), existing.reset_index(drop=True))


def test_merge_rows_never_overwrites_an_already_stored_value() -> None:
    """spec section 2.3: an already-written row is never silently
    overwritten -- a genuine disagreement raises, it does not pick a
    winner."""
    existing = rows_to_frame([_row(forecast=42.0)])
    conflicting = rows_to_frame([_row(forecast=99.0)])

    with pytest.raises(ValueError, match="OOS-store conflict"):
        merge_rows(existing, conflicting)


def test_merge_rows_tolerates_float_formatting_noise_within_the_round_trip_tolerance() -> None:
    existing = rows_to_frame([_row(forecast=42.0)])
    nearly_identical = rows_to_frame([_row(forecast=42.0 + 1e-9)])

    merged, n_added = merge_rows(existing, nearly_identical)

    assert n_added == 0
    assert len(merged) == 1


def test_merge_rows_on_an_empty_existing_store_returns_the_new_rows_unchanged() -> None:
    existing = load_oos_store(Path("this/path/does/not/exist.csv"))
    new_rows = rows_to_frame([_row()])

    merged, n_added = merge_rows(existing, new_rows)

    assert n_added == 1
    assert len(merged) == 1


def test_write_oos_store_round_trips_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "oos_store.csv"
    base_row = _row(ladder_row="base", weather_run_init_utc=None)
    frame = rows_to_frame([_row(), base_row])

    write_oos_store(frame, path)
    read_back = load_oos_store(path)

    assert len(read_back) == 2
    base_read = read_back.loc[read_back["ladder_row"] == "base"].iloc[0]
    assert pd.isna(base_read["weather_run_init_utc"])
    core_gas_read = read_back.loc[read_back["ladder_row"] == "core_gas"].iloc[0]
    assert core_gas_read["weather_run_init_utc"] == pd.Timestamp("2026-07-14T00:00:00", tz="UTC")
    assert float(core_gas_read["forecast"]) == 42.0
    assert core_gas_read["target_day"] == dt.date(2026, 7, 15)
    assert core_gas_read["timestamp_utc"].tzinfo is not None


def test_write_oos_store_rejects_an_invalid_frame_before_touching_disk(tmp_path: Path) -> None:
    path = tmp_path / "oos_store.csv"
    invalid = rows_to_frame([_row(), _row()])  # duplicate key

    with pytest.raises(ValueError, match="duplicate OOS-store key"):
        write_oos_store(invalid, path)

    assert not path.exists()


def test_missing_target_days_reports_only_the_rows_not_yet_present() -> None:
    day = dt.date(2026, 7, 15)
    existing = rows_to_frame(
        [_row(target_day=day, ladder_row="core_gas"), _row(target_day=day, ladder_row="base")]
    )

    missing = missing_target_days(existing, [day])

    assert missing == {day: frozenset({"core_gas_loadpatch"})}


def test_missing_target_days_omits_a_fully_present_day() -> None:
    day = dt.date(2026, 7, 15)
    existing = rows_to_frame(
        [_row(target_day=day, ladder_row=row_name) for row_name in LADDER_ROW_NAMES]
    )

    missing = missing_target_days(existing, [day])

    assert missing == {}


def test_missing_target_days_reports_every_row_for_a_wholly_absent_day() -> None:
    missing = missing_target_days(load_oos_store(Path("no/such/file.csv")), [dt.date(2026, 7, 16)])

    assert missing == {dt.date(2026, 7, 16): LADDER_ROW_NAMES}
