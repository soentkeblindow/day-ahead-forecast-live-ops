import math

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.breakdown import breakdown_point, breakdown_quantiles
from energy_price_forecast.evaluation.metrics import summarise, summarise_quantiles
from energy_price_forecast.evaluation.regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _bool_col(value: bool | list[bool], n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    return np.full(n, bool(arr)) if arr.ndim == 0 else arr


def _float_col(value: float | list[float], n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.full(n, float(arr)) if arr.ndim == 0 else arr


def _flags(
    index: pd.DatetimeIndex,
    *,
    renewable_scarcity: bool | list[bool] = False,
    high_wind: bool | list[bool] = False,
    negative_price: bool | list[bool] = False,
    price_spike: bool | list[bool] = False,
    macro: str | list[str] = "calm",
) -> pd.DataFrame:
    """A hand-built stand-in for tag_regimes()'s output shape."""
    n = len(index)
    rs = _bool_col(renewable_scarcity, n)
    hw = _bool_col(high_wind, n)
    neg = _bool_col(negative_price, n)
    spike = _bool_col(price_spike, n)
    normal = ~(rs | hw | neg | spike)

    macro_arr = np.asarray(macro, dtype=object)
    macro_col = np.full(n, macro, dtype=object) if macro_arr.ndim == 0 else macro_arr

    return pd.DataFrame(
        {
            "renewable_scarcity": rs,
            "high_wind": hw,
            "negative_price": neg,
            "price_spike": spike,
            "normal": normal,
            MACRO_REGIME_COLUMN: pd.Categorical(
                macro_col, categories=["calm", "crisis", "post_crisis"]
            ),
        },
        index=index,
    )


def _predictions(
    index: pd.DatetimeIndex,
    *,
    y_true: float | list[float] = 50.0,
    y_pred: float | list[float] = 50.0,
) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "y_true": _float_col(y_true, n),
            "y_pred": _float_col(y_pred, n),
            "delivery_day": index.tz_convert("Europe/Berlin").normalize(),
        },
        index=index,
    )


def _assert_dict_equal(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == set(expected)
    for k, v in expected.items():
        if isinstance(v, float) and math.isnan(v):
            assert math.isnan(actual[k]), f"{k}: expected NaN, got {actual[k]}"
        else:
            assert actual[k] == pytest.approx(v), f"{k}: expected {v}, got {actual[k]}"


# ---------------------------------------------------------------------------
# Anchor identity
# ---------------------------------------------------------------------------


def test_breakdown_point_overall_matches_summarise() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    preds = _predictions(idx, y_true=[50, 52, 48, 55, 45, 50], y_pred=[49, 53, 47, 50, 44, 51])
    flags = _flags(idx, macro=["calm"] * 2 + ["crisis"] * 2 + ["post_crisis"] * 2)

    out = breakdown_point(preds, flags)
    expected = summarise(preds)

    _assert_dict_equal({k: float(out[k].loc["overall"]) for k in expected}, expected)


def test_breakdown_quantiles_overall_matches_summarise_quantiles() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    y = pd.Series([50.0, 52.0, 48.0, 55.0, 45.0, 50.0], index=idx)
    preds = {
        0.05: pd.Series([45.0] * 6, index=idx),
        0.5: pd.Series([50.0] * 6, index=idx),
        0.95: pd.Series([56.0] * 6, index=idx),
    }
    flags = _flags(idx, macro=["calm"] * 2 + ["crisis"] * 2 + ["post_crisis"] * 2)

    out = breakdown_quantiles(y, preds, flags)
    expected = summarise_quantiles(y, preds)

    _assert_dict_equal({k: float(out[k].loc["overall"]) for k in expected}, expected)


# ---------------------------------------------------------------------------
# Macro partition invariant
# ---------------------------------------------------------------------------


def test_macro_partition_invariant() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    preds = _predictions(idx)
    flags = _flags(idx, macro=["calm", "calm", "crisis", "crisis", "post_crisis", "post_crisis"])

    out = breakdown_point(preds, flags)

    assert out.loc["overall", "n"] == 6
    assert out.loc["calm", "n"] == 2
    assert out.loc["crisis", "n"] == 2
    assert out.loc["post_crisis", "n"] == 2
    assert out.loc[["calm", "crisis", "post_crisis"], "n"].sum() == out.loc["overall", "n"]


def test_macro_row_present_even_when_a_phase_is_empty() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    preds = _predictions(idx)
    flags = _flags(idx, macro="calm")  # no crisis / post_crisis hours at all

    out = breakdown_point(preds, flags)

    assert "post_crisis" in out.index
    assert out.loc["post_crisis", "n"] == 0
    assert math.isnan(float(out["mae"].loc["post_crisis"]))


# ---------------------------------------------------------------------------
# normal-vs-special partition invariant, and the allowed flag overlap
# ---------------------------------------------------------------------------


def test_normal_vs_special_partition_invariant() -> None:
    idx = _hourly_utc("2021-01-01", 5)
    # rows: 0 normal, 1 renewable_scarcity, 2 negative_price, 3 both rs+spike, 4 normal
    flags = _flags(
        idx,
        renewable_scarcity=[False, True, False, True, False],
        negative_price=[False, False, True, False, False],
        price_spike=[False, False, False, True, False],
    )
    preds = _predictions(idx)
    out = breakdown_point(preds, flags)

    any_special = (
        flags["renewable_scarcity"]
        | flags["high_wind"]
        | flags["negative_price"]
        | flags["price_spike"]
    )
    expected_special_n = int(any_special.sum())  # rows 1, 2, 3 -> 3
    normal_n = int(out["n"].loc["normal"])
    overall_n = int(out["n"].loc["overall"])

    assert normal_n == overall_n - expected_special_n
    assert normal_n + expected_special_n == overall_n


def test_flag_overlap_oversums_total() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    # row 1 satisfies BOTH renewable_scarcity and price_spike at once.
    flags = _flags(
        idx,
        renewable_scarcity=[False, True, False],
        negative_price=[False, False, True],
        price_spike=[False, True, False],
    )
    preds = _predictions(idx)
    out = breakdown_point(preds, flags)

    flag_n_sum = out.loc[list(REGIME_FLAG_COLUMNS), "n"].sum()
    assert flag_n_sum > out.loc["overall", "n"]


# ---------------------------------------------------------------------------
# Value correctness on a subset
# ---------------------------------------------------------------------------


def test_point_value_correctness_on_subset() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    # rows 0,1 calm with |error|=2; rows 2,3 crisis with |error|=5
    preds = _predictions(idx, y_true=[50.0] * 4, y_pred=[48.0, 48.0, 45.0, 45.0])
    flags = _flags(idx, macro=["calm", "calm", "crisis", "crisis"])

    out = breakdown_point(preds, flags)

    assert out.loc["calm", "mae"] == pytest.approx(2.0)
    assert out.loc["crisis", "mae"] == pytest.approx(5.0)


def test_quantile_value_correctness_on_subset() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    y = pd.Series([5.0] * 4, index=idx)
    preds = {
        0.05: pd.Series([3.0] * 4, index=idx),
        0.5: pd.Series([5.0] * 4, index=idx),
        0.95: pd.Series([8.0] * 4, index=idx),
    }
    flags = _flags(idx, macro=["calm", "calm", "crisis", "crisis"])

    out = breakdown_quantiles(y, preds, flags)

    assert out.loc["crisis", "coverage_0.05"] == pytest.approx(0.0)
    assert out.loc["crisis", "interval_coverage_90"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# axis / n column contract
# ---------------------------------------------------------------------------


def test_axis_and_n_columns_contract() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    preds = _predictions(idx)
    flags = _flags(idx, macro="calm")

    out = breakdown_point(preds, flags)

    assert set(out["axis"].unique()) == {"overall", "macro", "flag"}
    assert (out["n"] >= 0).all()
    assert out["axis"].value_counts().to_dict() == {"flag": 5, "macro": 3, "overall": 1}


# ---------------------------------------------------------------------------
# Quantile index intersection
# ---------------------------------------------------------------------------


def test_breakdown_quantiles_uses_index_intersection() -> None:
    base = _hourly_utc("2021-01-01", 5)
    y = pd.Series([5.0] * 5, index=base)
    preds = {
        0.05: pd.Series([3.0] * 5, index=base),
        0.5: pd.Series([5.0] * 4, index=base[:4]),  # missing the last hour
        0.95: pd.Series([8.0] * 5, index=base),
    }
    flags = _flags(base)

    out = breakdown_quantiles(y, preds, flags)

    assert out.loc["overall", "n"] == 4


# ---------------------------------------------------------------------------
# Fail-fast contracts
# ---------------------------------------------------------------------------


def test_breakdown_point_missing_column_raises() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    preds = _predictions(idx).drop(columns=["y_true"])
    flags = _flags(idx)
    with pytest.raises(ValueError, match="y_true"):
        breakdown_point(preds, flags)


def test_breakdown_point_index_not_subset_raises() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    preds = _predictions(idx)
    flags = _flags(_hourly_utc("2021-01-01", 2))  # shorter than predictions index
    with pytest.raises(ValueError, match="subset"):
        breakdown_point(preds, flags)


def test_breakdown_quantiles_index_not_subset_raises() -> None:
    idx = _hourly_utc("2021-01-01", 3)
    y = pd.Series([5.0] * 3, index=idx)
    q = pd.Series([5.0] * 3, index=idx)
    flags = _flags(_hourly_utc("2021-01-01", 2))
    with pytest.raises(ValueError, match="subset"):
        breakdown_quantiles(y, {0.05: q, 0.5: q, 0.95: q}, flags)
