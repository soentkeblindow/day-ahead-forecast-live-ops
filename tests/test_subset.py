"""Tests for features/subset.py — drop_features fail-fast contract."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_price_forecast.features.subset import drop_features


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})


def test_drop_removes_named_columns() -> None:
    df = _make_df()
    result = drop_features(df, ["a", "c"])
    assert list(result.columns) == ["b"]
    assert list(result["b"]) == [3, 4]


def test_drop_leaves_index_unchanged() -> None:
    df = _make_df()
    result = drop_features(df, ["b"])
    assert result.index.tolist() == df.index.tolist()


def test_drop_returns_copy() -> None:
    df = _make_df()
    result = drop_features(df, ["a"])
    assert result is not df
    # mutating the copy must not affect the original
    result["b"] = 99
    assert df["b"].tolist() == [3, 4]


def test_strict_raises_on_unknown_column() -> None:
    df = _make_df()
    with pytest.raises(KeyError, match="typo_col"):
        drop_features(df, ["a", "typo_col"])


def test_strict_error_lists_all_missing() -> None:
    df = _make_df()
    with pytest.raises(KeyError, match="missing1") as exc:
        drop_features(df, ["missing1", "missing2"])
    assert "missing2" in str(exc.value)


def test_strict_false_ignores_unknown() -> None:
    df = _make_df()
    result = drop_features(df, ["a", "does_not_exist"], strict=False)
    assert list(result.columns) == ["b", "c"]


def test_strict_false_still_drops_known() -> None:
    df = _make_df()
    result = drop_features(df, ["b", "does_not_exist"], strict=False)
    assert "b" not in result.columns
    assert "a" in result.columns


def test_empty_columns_list_returns_full_copy() -> None:
    df = _make_df()
    result = drop_features(df, [])
    assert list(result.columns) == ["a", "b", "c"]
    assert result is not df
