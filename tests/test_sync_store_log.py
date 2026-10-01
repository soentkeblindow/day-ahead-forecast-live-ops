"""Test for scripts/sync_store.py's _write_log_row (real incident,
2026-09-11): the function only wrote a CSV header when the file did not
exist yet. Adding a new per-source field (SourceLogRow.hints, spec
6.7.1a) to the row dict left an already-existing file's header at its old,
narrower width while the very next real run appended a row with more
fields than that header declared -- unparseable, pd.read_csv raised
ParserError on the real committed file. Same "found a real bug, add a
regression test for exactly that" precedent as test_sync_store_entsoe.py /
test_sync_store_weather.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from scripts import sync_store


def _row(source_name: str, *, with_hints: bool) -> sync_store.RunLog:
    log = sync_store.RunLog()
    row = log.get(source_name)
    row.fetched = True
    row.rows_added = 1
    row.validation = "ok"
    if with_hints:
        row.hints = "some non-blocking finding"
    return log


def test_write_log_row_migrates_header_when_the_column_set_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "store_sync.csv"
    monkeypatch.setattr(sync_store, "STORE_SYNC_LOG", log_path)

    # Simulate an old row written before SourceLogRow.hints existed: build
    # the frame by hand, without the hints column, exactly like the
    # pre-6.7.1a code would have.
    old_row = {
        "run_timestamp_utc": pd.Timestamp("2026-09-01", tz="UTC").isoformat(),
        "run_id": "1",
        "code_sha": "abc",
        "store_bytes": 100,
        "warnings": "",
        "exit_status": "ok",
        "day_ahead_price_fetched": True,
        "day_ahead_price_rows_added": 5,
        "day_ahead_price_healed_cells": 0,
        "day_ahead_price_validation": "ok",
    }
    pd.DataFrame([old_row]).to_csv(log_path, index=False)

    # A real run today writes a row with the new hints column -- through
    # the real, unmodified _write_log_row.
    log = _row("day_ahead_price", with_hints=True)
    sync_store._write_log_row(
        pd.Timestamp("2026-09-11", tz="UTC"),
        "2",
        "def",
        log,
        200,
        "ok",
        ec_load_target_day=dt.date(2026, 9, 12),
        ec_load_complete_for_target_day=False,
        sync_mode="all",
    )

    # Must be readable without error, both rows present, old value
    # preserved, new hint captured.
    restored = pd.read_csv(log_path)
    assert list(restored["run_id"]) == [1, 2]
    assert restored.loc[0, "day_ahead_price_rows_added"] == 5
    assert restored.loc[1, "day_ahead_price_hints"] == "some non-blocking finding"
    assert pd.isna(restored.loc[0, "day_ahead_price_hints"])


def test_write_log_row_appends_normally_when_the_column_set_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "store_sync.csv"
    monkeypatch.setattr(sync_store, "STORE_SYNC_LOG", log_path)

    sync_store._write_log_row(
        pd.Timestamp("2026-09-01", tz="UTC"),
        "1",
        "abc",
        _row("day_ahead_price", with_hints=False),
        100,
        "ok",
        ec_load_target_day=dt.date(2026, 9, 2),
        ec_load_complete_for_target_day=False,
        sync_mode="all",
    )
    sync_store._write_log_row(
        pd.Timestamp("2026-09-02", tz="UTC"),
        "2",
        "def",
        _row("day_ahead_price", with_hints=False),
        100,
        "ok",
        ec_load_target_day=dt.date(2026, 9, 3),
        ec_load_complete_for_target_day=True,
        sync_mode="relevant-only",
    )

    restored = pd.read_csv(log_path)
    assert list(restored["run_id"]) == [1, 2]
    # No rewrite needed -- a plain append, same row count as an ordinary
    # two-line log (no duplicated/rewritten header).
    assert log_path.read_text(encoding="utf-8").count("run_timestamp_utc") == 1
