"""Unit tests for scripts/backfill_day_ahead_price_gap.py -- the one-off,
owner-invoked Energy-Charts backfill for a specific day_ahead_price gap
ENTSO-E never published (docs/sprint6_step6_7_2_log.md, "Sitzung 9").
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest

from energy_price_forecast.ops.windows import local_day_bounds
from scripts.backfill_day_ahead_price_gap import backfill_gap, fetch_energy_charts_price

_GAP_DATE = dt.date(2026, 9, 13)


def _expected_index(gap_date: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(gap_date)
    return pd.DatetimeIndex(
        pd.date_range(
            start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left"
        )
    )


def _energy_charts_payload(gap_date: dt.date, *, unit: str = "EUR / MWh") -> dict[str, Any]:
    index = _expected_index(gap_date)
    return {
        "license_info": "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de",
        "unix_seconds": [int(ts.timestamp()) for ts in index],
        "price": [100.0 + i for i in range(len(index))],
        "unit": unit,
        "deprecated": False,
    }


class _FakeResponse:
    def __init__(self, body: Any) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._body


def _install_fake_get(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        calls.append(params)
        return response

    import scripts.backfill_day_ahead_price_gap as module

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# fetch_energy_charts_price
# ---------------------------------------------------------------------------


def test_fetch_energy_charts_price_returns_the_expected_local_day_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_get(monkeypatch, _FakeResponse(_energy_charts_payload(_GAP_DATE)))

    result = fetch_energy_charts_price(_GAP_DATE)

    assert calls == [{"bzn": "DE-LU", "start": "2026-09-13", "end": "2026-09-13"}]
    assert result.index.equals(_expected_index(_GAP_DATE))
    assert result.name == "day_ahead_price"
    assert result.notna().all()


def test_fetch_energy_charts_price_rejects_wrong_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_get(
        monkeypatch, _FakeResponse(_energy_charts_payload(_GAP_DATE, unit="EUR / kWh"))
    )

    with pytest.raises(ValueError, match="unexpected unit"):
        fetch_energy_charts_price(_GAP_DATE)


def test_fetch_energy_charts_price_rejects_a_null_value(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _energy_charts_payload(_GAP_DATE)
    payload["price"][5] = None
    _install_fake_get(monkeypatch, _FakeResponse(payload))

    with pytest.raises(ValueError, match="null price value"):
        fetch_energy_charts_price(_GAP_DATE)


def test_fetch_energy_charts_price_rejects_a_short_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _energy_charts_payload(_GAP_DATE)
    payload["unix_seconds"] = payload["unix_seconds"][:-1]
    payload["price"] = payload["price"][:-1]
    _install_fake_get(monkeypatch, _FakeResponse(payload))

    with pytest.raises(ValueError, match="does not cover exactly the expected local-day span"):
        fetch_energy_charts_price(_GAP_DATE)


# ---------------------------------------------------------------------------
# backfill_gap
# ---------------------------------------------------------------------------


def _existing_month_frame(gap_date: dt.date) -> pd.DataFrame:
    """A month's worth of real data on either side of gap_date's own local
    day, which is entirely absent (no rows at all -- the real shape of
    this incident, not NaN-filled rows)."""
    month_start = pd.Timestamp(gap_date.year, gap_date.month, 1, tz="UTC")
    next_month = (month_start + pd.DateOffset(months=1)).normalize()
    full_index = pd.date_range(month_start, next_month, freq="15min", inclusive="left")
    gap_index = _expected_index(gap_date)
    kept_index = full_index.difference(gap_index)
    return pd.DataFrame({"day_ahead_price": 50.0}, index=kept_index)


def test_backfill_gap_fills_only_the_missing_cells() -> None:
    existing = _existing_month_frame(_GAP_DATE)
    fresh = pd.Series(
        [100.0 + i for i in range(len(_expected_index(_GAP_DATE)))],
        index=_expected_index(_GAP_DATE),
        name="day_ahead_price",
    )

    merged, cells_filled = backfill_gap(existing, fresh)

    assert cells_filled == len(fresh)
    assert merged.loc[fresh.index, "day_ahead_price"].equals(fresh)
    # Everything outside the gap is untouched, bit-identical.
    assert merged.loc[existing.index, "day_ahead_price"].equals(existing["day_ahead_price"])


def test_backfill_gap_never_overwrites_an_existing_real_value() -> None:
    existing = _existing_month_frame(_GAP_DATE)
    gap_index = _expected_index(_GAP_DATE)
    # One hour of the "gap" already has a real (later-healed) ENTSO-E value.
    existing.loc[gap_index[0], "day_ahead_price"] = 999.0
    fresh = pd.Series(
        [100.0 + i for i in range(len(gap_index))], index=gap_index, name="day_ahead_price"
    )
    fresh.iloc[0] = 999.0  # agrees with the already-real value

    merged, cells_filled = backfill_gap(existing, fresh)

    assert cells_filled == len(fresh) - 1
    assert merged.loc[gap_index[0], "day_ahead_price"] == 999.0


def test_backfill_gap_raises_on_a_genuine_conflict() -> None:
    existing = _existing_month_frame(_GAP_DATE)
    gap_index = _expected_index(_GAP_DATE)
    existing.loc[gap_index[0], "day_ahead_price"] = 999.0
    fresh = pd.Series(
        [100.0 + i for i in range(len(gap_index))], index=gap_index, name="day_ahead_price"
    )
    # fresh.iloc[0] == 100.0, disagreeing with the real 999.0 already on disk.

    with pytest.raises(ValueError, match="disagrees with Energy-Charts"):
        backfill_gap(existing, fresh)


def test_backfill_gap_reports_zero_when_nothing_is_missing() -> None:
    existing = _existing_month_frame(_GAP_DATE)
    gap_index = _expected_index(_GAP_DATE)
    fresh = pd.Series(
        [100.0 + i for i in range(len(gap_index))], index=gap_index, name="day_ahead_price"
    )
    # Simulate the gap already having been healed by a prior run: every
    # gap-day timestamp now genuinely exists on disk, matching fresh exactly.
    existing = pd.concat([existing, fresh.to_frame()]).sort_index()

    _merged, cells_filled = backfill_gap(existing, fresh)

    assert cells_filled == 0
