"""Unit tests for arena/live_inputs.py::coalesce_price and
price_provenance_report (spec 6.9 section 5.3/5.7, Schritt 7).

Everything here is pure/disk-only against tmp_path fixtures -- no client,
no network, same discipline as tests/test_live_inputs.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_price_forecast.arena.live_inputs import coalesce_price, price_provenance_report
from energy_price_forecast.ops.store_sources import EntsoeSource

# ---------------------------------------------------------------------------
# coalesce_price
# ---------------------------------------------------------------------------


def test_coalesce_price_entsoe_wins_when_both_present_and_agree() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    entsoe = pd.Series([1.0, 2.0, 3.0], index=idx, name="day_ahead_price")
    ec = pd.Series([1.0, 2.0, 3.0], index=idx, name="day_ahead_price_ec")

    merged, provenance, n_conflicts = coalesce_price(entsoe, ec)

    assert merged.tolist() == [1.0, 2.0, 3.0]
    assert not provenance.any()
    assert n_conflicts == 0


def test_coalesce_price_ec_fills_a_nan_cell_within_an_existing_row() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    entsoe = pd.Series([1.0, float("nan"), 3.0], index=idx, name="day_ahead_price")
    ec = pd.Series([1.0, 99.0, 3.0], index=idx, name="day_ahead_price_ec")

    merged, provenance, n_conflicts = coalesce_price(entsoe, ec)

    assert merged.tolist() == [1.0, 99.0, 3.0]
    assert provenance.tolist() == [False, True, False]
    assert n_conflicts == 0


def test_coalesce_price_ec_fills_a_row_entsoe_is_missing_entirely() -> None:
    """A real gap-fill, not just a NaN cell -- ENTSO-E's own index doesn't
    have this timestamp at all."""
    entsoe = pd.Series(
        [1.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-01T00:00Z", "2024-01-01T02:00Z"]),
        name="day_ahead_price",
    )
    ec = pd.Series(
        [1.0, 99.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T02:00Z"]),
        name="day_ahead_price_ec",
    )

    merged, provenance, n_conflicts = coalesce_price(entsoe, ec)

    assert len(merged) == 3
    missing_hour = pd.Timestamp("2024-01-01T01:00Z")
    assert merged.loc[missing_hour] == 99.0
    assert provenance.loc[missing_hour]
    assert n_conflicts == 0


def test_coalesce_price_entsoe_wins_silently_on_a_real_conflict() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    entsoe = pd.Series([50.0], index=idx, name="day_ahead_price")
    ec = pd.Series([50.5], index=idx, name="day_ahead_price_ec")  # disagrees

    merged, provenance, n_conflicts = coalesce_price(entsoe, ec)

    assert merged.iloc[0] == 50.0  # ENTSO-E value, not EC's
    assert not provenance.any()  # ENTSO-E was present -> not EC-sourced
    assert n_conflicts == 1


def test_coalesce_price_empty_ec_changes_nothing() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    entsoe = pd.Series([1.0, 2.0, 3.0], index=idx, name="day_ahead_price")
    ec = pd.Series(dtype="float64", name="day_ahead_price_ec", index=pd.DatetimeIndex([], tz="UTC"))

    merged, provenance, n_conflicts = coalesce_price(entsoe, ec)

    assert merged.tolist() == [1.0, 2.0, 3.0]
    assert not provenance.any()
    assert n_conflicts == 0


# ---------------------------------------------------------------------------
# price_provenance_report
# ---------------------------------------------------------------------------


def _entsoe_price_source(cache_dir: Path) -> EntsoeSource:
    def _never_call(
        start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True
    ) -> pd.DataFrame:
        raise AssertionError("price_provenance_report must never call a fetch function")

    return EntsoeSource(name="day_ahead_price", fetch=_never_call, cache_dir=cache_dir)


def test_price_provenance_report_all_entsoe_when_ec_unused(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [1.0, 2.0, 3.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    ec_dir = tmp_path / "energy_charts"  # deliberately empty -- no EC file at all

    report, n_conflicts = price_provenance_report(
        entsoe_sources=(_entsoe_price_source(price_dir),),
        energy_charts_dir=ec_dir,
        as_of=pd.Timestamp("2024-01-01T03:00Z"),
    )

    assert report == {
        "training_labels": "entsoe",
        "price_lags": "entsoe",
        "shape_window": "entsoe",
        "persistence": "entsoe",
    }
    assert n_conflicts == 0


def test_price_provenance_report_flags_energy_charts_when_a_gap_is_filled(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [1.0, float("nan"), 3.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )

    ec_dir = tmp_path / "energy_charts"
    ec_dir.mkdir()
    pd.DataFrame({"day_ahead_price_ec": [1.0, 99.0, 3.0]}, index=idx).to_parquet(
        ec_dir / "day_ahead_price_ec.parquet"
    )

    as_of = pd.Timestamp("2024-01-01T03:00Z")
    report, n_conflicts = price_provenance_report(
        entsoe_sources=(_entsoe_price_source(price_dir),),
        energy_charts_dir=ec_dir,
        as_of=as_of,
    )

    assert report["training_labels"] == "energy_charts"
    assert report["price_lags"] == "energy_charts"
    assert report["shape_window"] == "energy_charts"  # the gap is inside the last 28 days
    assert report["persistence"] == "energy_charts"
    assert n_conflicts == 0


def test_price_provenance_report_shape_window_excludes_an_old_gap(tmp_path: Path) -> None:
    """A gap filled far outside the shape profile's own lookback window
    must not falsely mark shape_window as energy_charts, even though the
    other three (full-history) consumers correctly do."""
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()
    old_idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [1.0, float("nan"), 3.0]}, index=old_idx).to_parquet(
        price_dir / "DE_LU_2023-01.parquet"
    )

    ec_dir = tmp_path / "energy_charts"
    ec_dir.mkdir()
    pd.DataFrame({"day_ahead_price_ec": [1.0, 99.0, 3.0]}, index=old_idx).to_parquet(
        ec_dir / "day_ahead_price_ec.parquet"
    )

    as_of = pd.Timestamp("2024-06-01T00:00Z")  # far more than 28 days after the old gap
    report, n_conflicts = price_provenance_report(
        entsoe_sources=(_entsoe_price_source(price_dir),),
        energy_charts_dir=ec_dir,
        as_of=as_of,
    )

    assert report["training_labels"] == "energy_charts"  # full-history proxy still sees it
    assert report["shape_window"] == "entsoe"  # but the 28-day window does not
    assert n_conflicts == 0


def test_price_provenance_report_counts_a_real_conflict(tmp_path: Path) -> None:
    price_dir = tmp_path / "day_ahead_price"
    price_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": [50.0]}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    ec_dir = tmp_path / "energy_charts"
    ec_dir.mkdir()
    pd.DataFrame({"day_ahead_price_ec": [50.7]}, index=idx).to_parquet(
        ec_dir / "day_ahead_price_ec.parquet"
    )

    report, n_conflicts = price_provenance_report(
        entsoe_sources=(_entsoe_price_source(price_dir),),
        energy_charts_dir=ec_dir,
        as_of=pd.Timestamp("2024-01-01T01:00Z"),
    )

    assert n_conflicts == 1
    # ENTSO-E was present, so it wins -- not EC-sourced for any consumer.
    assert report == {
        "training_labels": "entsoe",
        "price_lags": "entsoe",
        "shape_window": "entsoe",
        "persistence": "entsoe",
    }
