import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.rearrangement import rearrange_quantiles

LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _grid(values: list[list[float]]) -> dict[float, pd.Series]:
    index = _hourly_utc("2021-01-01", len(values))
    arr = np.array(values)
    return {a: pd.Series(arr[:, i], index=index) for i, a in enumerate(LEVELS)}


def test_rearrange_fixes_a_crossing_example() -> None:
    preds = _grid(
        [
            [10.0, 5.0, 20.0, 30.0, 40.0, 50.0, 60.0],  # q05 > q10: crossing
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],  # already monotonic
        ]
    )

    out = rearrange_quantiles(preds)

    crossing_hour = pd.Series({a: out[a].iloc[0] for a in LEVELS})
    assert crossing_hour.is_monotonic_increasing
    assert sorted(crossing_hour.to_numpy()) == [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    clean_hour = pd.Series({a: out[a].iloc[1] for a in LEVELS})
    assert list(clean_hour.to_numpy()) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


def test_rearrange_is_idempotent() -> None:
    preds = _grid(
        [
            [10.0, 5.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ]
    )

    once = rearrange_quantiles(preds)
    twice = rearrange_quantiles(once)

    for a in LEVELS:
        pd.testing.assert_series_equal(once[a], twice[a])


def test_rearrange_preserves_value_multiset_per_hour() -> None:
    rng = np.random.default_rng(0)
    raw_values = rng.normal(size=(50, len(LEVELS)))
    preds = _grid(raw_values.tolist())

    out = rearrange_quantiles(preds)

    for hour in range(50):
        raw_row = sorted(preds[a].iloc[hour] for a in LEVELS)
        out_row = sorted(out[a].iloc[hour] for a in LEVELS)
        assert raw_row == out_row


def test_rearrange_is_monotonic_after() -> None:
    rng = np.random.default_rng(1)
    raw_values = rng.normal(size=(200, len(LEVELS)))
    preds = _grid(raw_values.tolist())

    out = rearrange_quantiles(preds)

    frame = pd.DataFrame({a: out[a] for a in LEVELS})
    monotonic = frame.apply(lambda row: row.is_monotonic_increasing, axis=1)
    assert monotonic.all()


def test_rearrange_raises_on_misaligned_index() -> None:
    preds = _grid([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]])
    preds[0.95] = preds[0.95].set_axis(_hourly_utc("2022-01-01", 1))

    with pytest.raises(ValueError, match="identical index"):
        rearrange_quantiles(preds)
