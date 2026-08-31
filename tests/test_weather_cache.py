"""Unit tests for data/_weather_cache.py -- the per-run weather cache.

Pure filesystem tests, no network. All tests use pytest's tmp_path (passed
as the `root` argument to cache_path) rather than the real
data/cache/weather_single_runs/, so nothing here touches the working tree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.data._weather_cache import (
    cache_path,
    check_columns,
    read_cached_run,
    write_cached_run,
)
from energy_price_forecast.data.weather_grid import CACHE_KEY, expected_columns

_WINDOWS_FORBIDDEN = set('<>:"|?*')


def _synthetic_frame(run_init_utc: pd.Timestamp, n_hours: int = 3) -> pd.DataFrame:
    valid_times = pd.date_range(run_init_utc, periods=n_hours, freq="h", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([run_init_utc] * n_hours), valid_times],
        names=["run_init_utc", "valid_time_utc"],
    )
    columns = expected_columns()
    data = {col: [1.0 * i for i in range(n_hours)] for col in columns}
    return pd.DataFrame(data, index=index, columns=columns).astype("float32")


# ---------------------------------------------------------------------------
# cache_path()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_str",
    [
        "2024-03-14T00:00",  # archive start
        "2024-12-31T18:00",  # year rollover
        "2024-02-29T06:00",  # leap day
        "2026-08-31T12:00",
    ],
)
def test_no_windows_forbidden_characters_in_path(run_str: str, tmp_path: Path) -> None:
    run = pd.Timestamp(run_str, tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    # Only the segments cache_path() itself constructs -- tmp_path's own
    # prefix is a real filesystem path already and irrelevant to this check.
    for part in path.relative_to(tmp_path).parts:
        assert not (_WINDOWS_FORBIDDEN & set(part)), f"forbidden char in {part!r}"
        assert not part.endswith(".") or part.endswith(".parquet")
        assert not part.endswith(" ")


def test_filename_format(tmp_path: Path) -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    assert path.name == "2024-06-01T00Z.parquet"


def test_year_month_subfolders_match_run(tmp_path: Path) -> None:
    run = pd.Timestamp("2024-12-05T12:00", tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    assert path.parent.name == "12"
    assert path.parent.parent.name == "2024"


def test_path_contains_model_and_cache_key(tmp_path: Path) -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    assert "ecmwf_ifs" in path.parts
    assert CACHE_KEY in path.parts


# ---------------------------------------------------------------------------
# check_columns() / read_cached_run() schema handling
# ---------------------------------------------------------------------------


def test_check_columns_accepts_exact_match() -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    check_columns(_synthetic_frame(run))  # must not raise


def test_check_columns_raises_on_missing_and_extra() -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    df = _synthetic_frame(run)
    df = df.drop(columns=[expected_columns()[0]])
    df["unexpected_extra_column"] = 0.0

    with pytest.raises(ValueError, match="missing") as exc_info:
        check_columns(df)
    assert expected_columns()[0] in str(exc_info.value)
    assert "unexpected_extra_column" in str(exc_info.value)


def test_read_cached_run_raises_not_none_on_schema_mismatch(tmp_path: Path) -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    bad_df = _synthetic_frame(run).drop(columns=[expected_columns()[0]])
    write_cached_run(bad_df, path)

    # Must raise, NOT return None (a schema mismatch is a finding, not a
    # cache miss that would trigger a silent re-fetch, spec §5.5).
    with pytest.raises(ValueError, match="missing"):
        read_cached_run(path)


def test_read_cached_run_returns_none_when_file_absent(tmp_path: Path) -> None:
    path = cache_path(pd.Timestamp("2024-06-01T00:00", tz="UTC"), "ecmwf_ifs", root=tmp_path)
    assert read_cached_run(path) is None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_file_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    df = _synthetic_frame(run)
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)

    def raising_to_parquet(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raising_to_parquet)

    with pytest.raises(RuntimeError, match="simulated crash"):
        write_cached_run(df, path)

    assert not path.exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_round_trip(tmp_path: Path) -> None:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    df = _synthetic_frame(run)
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)

    write_cached_run(df, path)
    result = read_cached_run(path)

    assert result is not None
    pd.testing.assert_frame_equal(df, result)


# ---------------------------------------------------------------------------
# fetch_run(use_cache=True): a cache hit skips the HTTP call entirely
# ---------------------------------------------------------------------------


def test_existing_cache_entry_is_not_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import energy_price_forecast.data.weather_client as client_module

    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    path = cache_path(run, "ecmwf_ifs", root=tmp_path)
    write_cached_run(_synthetic_frame(run), path)

    monkeypatch.setattr(client_module, "cache_path", lambda run_init_utc, model: path)

    call_count = 0

    def fake_get(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        raise AssertionError("requests.get must not be called on a cache hit")

    monkeypatch.setattr(client_module.requests, "get", fake_get)

    result = client_module.fetch_run(run, forecast_days=1)

    assert call_count == 0
    pd.testing.assert_frame_equal(result, _synthetic_frame(run))
