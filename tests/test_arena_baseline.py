"""Unit tests for the Energy-Arena persistence baseline replica.

All hand-checked: expected values are derived by reasoning about wall-clock
alignment directly, not by re-deriving them through persistence_forecast
itself.
"""

import datetime as dt

import pandas as pd
import pytest

from energy_price_forecast.models.arena_baseline import _local_day_index, persistence_forecast

_TZ = "Europe/Berlin"


def _sequential_prices(date: dt.date) -> pd.Series:
    """D-1 prices 0.0, 1.0, 2.0, ... in wall-clock order -- each slot's value
    equals its position within the local day, making "same wall-clock slot"
    assertions readable as plain index arithmetic."""
    idx = _local_day_index(date, _TZ)
    return pd.Series(range(len(idx)), index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Ordinary day
# ---------------------------------------------------------------------------


def test_normal_day_matches_hand_computed_example() -> None:
    prev_date = dt.date(2026, 6, 14)
    target_date = dt.date(2026, 6, 15)
    prices = _sequential_prices(prev_date)

    result = persistence_forecast(prices, pd.Timestamp(target_date))

    assert len(result) == 96
    # Wall-clock 00:00 on D takes D-1's 00:00 value (position 0); 12:00 takes
    # D-1's 12:00 value (position 48 = 12h * 4 slots/h); 23:45 takes D-1's
    # last slot (position 95).
    assert result.iloc[0] == 0.0
    assert result.iloc[48] == 48.0
    assert result.iloc[-1] == 95.0
    assert result.to_numpy().tolist() == list(range(96))


# ---------------------------------------------------------------------------
# Spring forward: D has 92 quarter-hours, D-1 has 96
# ---------------------------------------------------------------------------


def test_spring_forward_day_has_92_slots_and_skips_the_missing_hour() -> None:
    prev_date = dt.date(2026, 3, 28)  # normal day before the 2026 DE/LU change
    target_date = dt.date(2026, 3, 29)  # clocks 02:00 -> 03:00
    prices = _sequential_prices(prev_date)

    result = persistence_forecast(prices, pd.Timestamp(target_date))

    assert len(result) == 92
    # D-1's 02:00-02:45 values (positions 8-11) are simply never used: D has
    # no local 02:00 slot that spring-forward day.
    assert 8.0 not in result.to_numpy()
    assert 9.0 not in result.to_numpy()
    assert 10.0 not in result.to_numpy()
    assert 11.0 not in result.to_numpy()
    # 01:45 (D-1 position 7) is the last slot before the jump; 03:00 (D-1
    # position 12) is the first slot after it -- adjacent in D's output.
    local = pd.DatetimeIndex(result.index).tz_convert(_TZ)
    idx_0145 = list(zip(local.hour, local.minute, strict=True)).index((1, 45))
    assert result.iloc[idx_0145] == 7.0
    assert result.iloc[idx_0145 + 1] == 12.0  # 03:00 immediately follows


# ---------------------------------------------------------------------------
# Fall back: D has 100 quarter-hours, D-1 has 96
# ---------------------------------------------------------------------------


def test_fall_back_day_has_100_slots_and_repeats_the_doubled_hour() -> None:
    prev_date = dt.date(2026, 10, 24)  # normal day before the 2026 DE/LU change
    target_date = dt.date(2026, 10, 25)  # clocks 03:00 -> 02:00 (hour repeats)
    prices = _sequential_prices(prev_date)

    result = persistence_forecast(prices, pd.Timestamp(target_date))

    assert len(result) == 100
    # Chronological layout of D: 8 slots (00:00-01:45), then 02:00-02:45
    # twice (4 slots each, CEST then CET), then 03:00-23:45 (84 slots).
    # 8 + 4 + 4 + 84 = 100.
    first_pass = result.iloc[8:12].to_numpy().tolist()
    second_pass = result.iloc[12:16].to_numpy().tolist()
    assert first_pass == [8.0, 9.0, 10.0, 11.0]
    assert second_pass == [8.0, 9.0, 10.0, 11.0]  # both passes take D-1's single value
    assert result.iloc[16] == 12.0  # 03:00 immediately follows the doubled hour
    assert result.iloc[-1] == 95.0  # 23:45


# ---------------------------------------------------------------------------
# Contract: no silent partial results, no leakage
# ---------------------------------------------------------------------------


def test_missing_d1_value_raises() -> None:
    prev_date = dt.date(2026, 6, 14)
    target_date = dt.date(2026, 6, 15)
    prices = _sequential_prices(prev_date)
    prices = prices.drop(prices.index[10])  # one gap in D-1

    with pytest.raises(ValueError, match=r"2026-06-14"):
        persistence_forecast(prices, pd.Timestamp(target_date))


def test_no_leakage_past_end_of_d_minus_1() -> None:
    """A sentinel filling every timestamp from D onward must never appear in
    the baseline for D -- the function must not reach past end of D-1."""
    prev_date = dt.date(2026, 6, 14)
    target_date = dt.date(2026, 6, 15)
    sentinel = -99999.0

    prev_index = _local_day_index(prev_date, _TZ)
    target_index = _local_day_index(target_date, _TZ)
    future_index = target_index.union(_local_day_index(dt.date(2026, 6, 16), _TZ))

    prices = pd.concat(
        [
            pd.Series(range(len(prev_index)), index=prev_index, dtype=float),
            pd.Series(sentinel, index=future_index, dtype=float),
        ]
    )

    result = persistence_forecast(prices, pd.Timestamp(target_date))

    assert sentinel not in result.to_numpy()
