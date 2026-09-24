"""Unit tests for scripts/fetch_energy_charts_forecast_history.py
(docs/sprint6_auftrag_energy_charts_backup_2.md, Teil 1)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.data.energy_charts import EnergyChartsRateLimitedError
from scripts.fetch_energy_charts_forecast_history import (
    _fetch_with_retry,
    _year_chunks,
    build_grid_report,
    build_history,
    fetch_series_history,
    write_monthly_files,
)

# ---------------------------------------------------------------------------
# _year_chunks
# ---------------------------------------------------------------------------


def test_year_chunks_splits_at_calendar_year_boundaries() -> None:
    chunks = _year_chunks(dt.date(2024, 3, 14), dt.date(2026, 9, 15))
    assert chunks == [
        (dt.date(2024, 3, 14), dt.date(2024, 12, 31)),
        (dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
        (dt.date(2026, 1, 1), dt.date(2026, 9, 15)),
    ]


def test_year_chunks_single_year_is_one_chunk() -> None:
    chunks = _year_chunks(dt.date(2024, 3, 14), dt.date(2024, 8, 1))
    assert chunks == [(dt.date(2024, 3, 14), dt.date(2024, 8, 1))]


def test_year_chunks_rejects_start_after_end() -> None:
    with pytest.raises(ValueError, match="after end"):
        _year_chunks(dt.date(2024, 6, 1), dt.date(2024, 1, 1))


# ---------------------------------------------------------------------------
# _fetch_with_retry: 429 handling honours Retry-After, caps at the limit
# ---------------------------------------------------------------------------


def _series_for(start: dt.date, end: dt.date, name: str) -> pd.Series:
    day_start = pd.Timestamp(start, tz="Europe/Berlin").tz_convert("UTC")
    day_end = pd.Timestamp(end + dt.timedelta(days=1), tz="Europe/Berlin").tz_convert("UTC")
    index = pd.date_range(day_start, day_end, freq="15min", inclusive="left")
    return pd.Series(range(len(index)), index=index, name=name, dtype="float64")


def test_fetch_with_retry_retries_on_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_fetch(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        calls["n"] += 1
        if calls["n"] < 3:
            raise EnergyChartsRateLimitedError(5.0)
        return _series_for(start, end, production_type)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", fake_fetch
    )

    sleeps: list[float] = []
    series, n_requests = _fetch_with_retry(
        "load", dt.date(2024, 6, 1), dt.date(2024, 6, 1), sleep=sleeps.append
    )

    assert n_requests == 3
    assert sleeps == [5.0, 5.0]
    assert len(series) == 96


def test_fetch_with_retry_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_rate_limited(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        raise EnergyChartsRateLimitedError(1.0)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", always_rate_limited
    )

    with pytest.raises(EnergyChartsRateLimitedError):
        _fetch_with_retry("load", dt.date(2024, 6, 1), dt.date(2024, 6, 1), sleep=lambda _s: None)


# ---------------------------------------------------------------------------
# fetch_series_history: chunk concatenation, dedupe guard, archive-start guard
# ---------------------------------------------------------------------------


def test_fetch_series_history_concatenates_chunks_and_paces_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        return _series_for(start, end, production_type)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", fake_fetch
    )

    from scripts.fetch_energy_charts_forecast_history import _RequestCounter

    sleeps: list[float] = []
    counter = _RequestCounter()
    series = fetch_series_history(
        "load",
        archive_start=dt.date(2024, 12, 30),
        as_of_date=dt.date(2025, 1, 2),
        sleep=sleeps.append,
        counter=counter,
    )

    # Two year chunks (2024, 2025) -> one inter-request pause, not a pause
    # before the very first request.
    assert sleeps == [30.0]
    assert counter.count == 2
    assert series.index.is_monotonic_increasing
    assert not series.index.has_duplicates


def test_fetch_series_history_raises_on_duplicate_timestamps_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overlapping_fetch(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        # Deliberately returns a fixed range regardless of the requested
        # chunk, producing an overlap between the two year chunks.
        return _series_for(dt.date(2024, 12, 31), dt.date(2025, 1, 1), production_type)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", overlapping_fetch
    )
    from scripts.fetch_energy_charts_forecast_history import _RequestCounter

    with pytest.raises(ValueError, match="duplicate timestamp"):
        fetch_series_history(
            "load",
            archive_start=dt.date(2024, 12, 30),
            as_of_date=dt.date(2025, 1, 2),
            sleep=lambda _s: None,
            counter=_RequestCounter(),
        )


def test_fetch_series_history_raises_when_archive_does_not_reach_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def truncated_fetch(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        # Archive only reaches back to 2024-06-01, regardless of what was
        # requested -- simulates the documented Rueckfrage-1 condition.
        clamped_start = max(start, dt.date(2024, 6, 1))
        return _series_for(clamped_start, end, production_type)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", truncated_fetch
    )
    from scripts.fetch_energy_charts_forecast_history import _RequestCounter

    with pytest.raises(ValueError, match="does not reach back"):
        fetch_series_history(
            "load",
            archive_start=dt.date(2024, 3, 14),
            as_of_date=dt.date(2024, 8, 1),
            sleep=lambda _s: None,
            counter=_RequestCounter(),
        )


# ---------------------------------------------------------------------------
# build_grid_report: DST-aware expectation, zero-point days get a row
# ---------------------------------------------------------------------------


def test_build_grid_report_flags_dst_days_and_full_coverage() -> None:
    series = _series_for(dt.date(2026, 3, 28), dt.date(2026, 3, 30), "load")
    report = build_grid_report(
        series, start_date=dt.date(2026, 3, 28), end_date=dt.date(2026, 3, 30)
    )

    spring_forward = report.loc[report["calendar_date"] == "2026-03-29"].iloc[0]
    assert spring_forward["expected_points"] == 92
    assert spring_forward["n_points"] == 92
    assert spring_forward["deviation"] == 0
    assert bool(spring_forward["is_dst_transition_day"])

    normal_day = report.loc[report["calendar_date"] == "2026-03-28"].iloc[0]
    assert normal_day["expected_points"] == 96
    assert not bool(normal_day["is_dst_transition_day"])


def test_build_grid_report_reports_a_full_day_gap_as_deviation() -> None:
    # Series covers 06-01 and 06-03 only -- 06-02 is a total gap.
    part1 = _series_for(dt.date(2024, 6, 1), dt.date(2024, 6, 1), "solar")
    part2 = _series_for(dt.date(2024, 6, 3), dt.date(2024, 6, 3), "solar")
    series = pd.concat([part1, part2])

    report = build_grid_report(series, start_date=dt.date(2024, 6, 1), end_date=dt.date(2024, 6, 3))

    gap_day = report.loc[report["calendar_date"] == "2024-06-02"].iloc[0]
    assert gap_day["n_points"] == 0
    assert gap_day["deviation"] == -96


def test_build_grid_report_counts_null_values_separately_from_points() -> None:
    series = _series_for(dt.date(2024, 6, 1), dt.date(2024, 6, 1), "wind_onshore")
    series.iloc[0:3] = float("nan")

    report = build_grid_report(series, start_date=dt.date(2024, 6, 1), end_date=dt.date(2024, 6, 1))
    row = report.iloc[0]
    assert row["n_points"] == 96
    assert row["n_non_null"] == 93
    assert row["deviation"] == 0


# ---------------------------------------------------------------------------
# write_monthly_files: one Parquet file per UTC calendar month
# ---------------------------------------------------------------------------


def test_write_monthly_files_splits_at_utc_month_boundary(tmp_path: Path) -> None:
    series = _series_for(dt.date(2024, 3, 14), dt.date(2024, 4, 2), "load")

    written = write_monthly_files(series, output_dir=tmp_path)

    names = sorted(p.name for p in written)
    assert names == ["load_2024-03.parquet", "load_2024-04.parquet"]

    march = pd.read_parquet(tmp_path / "load_2024-03.parquet")
    april = pd.read_parquet(tmp_path / "load_2024-04.parquet")
    assert len(march) + len(april) == len(series)
    assert list(march.columns) == ["load"]


# ---------------------------------------------------------------------------
# build_history: end-to-end orchestration
# ---------------------------------------------------------------------------


def test_build_history_writes_report_and_files_for_all_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_fetch(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
        return _series_for(start, end, production_type)

    monkeypatch.setattr(
        "scripts.fetch_energy_charts_forecast_history.fetch_series_range", fake_fetch
    )

    output_dir = tmp_path / "raw"
    report_path = tmp_path / "report.csv"
    # Mid-month dates, deliberately away from a month boundary: local
    # midnight at the start/end of a calendar month can fall in the
    # *neighbouring* UTC month (June 1st 00:00 CEST is May 31st 22:00 UTC),
    # which would otherwise split output into two files per series here --
    # exactly the real, correct UTC-bucketing behaviour, just not what this
    # particular "one file per series" assertion wants to exercise.
    summary = build_history(
        archive_start=dt.date(2024, 6, 5),
        as_of_date=dt.date(2024, 6, 7),
        output_dir=output_dir,
        report_path=report_path,
        sleep=lambda _s: None,
    )

    assert summary["total_requests"] == 4  # one chunk per series, single calendar year
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert set(report["production_type"]) == {"load", "solar", "wind_onshore", "wind_offshore"}
    assert len(list(output_dir.glob("*.parquet"))) == 4
