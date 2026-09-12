"""Unit tests for arena/candidates.py -- the declarative candidate table
and its selection loop (spec 6.7.2, sections 2.10, 5.5, 7).
"""

from __future__ import annotations

import pandas as pd

from energy_price_forecast.arena.candidates import (
    Candidate,
    full_live_set_candidate_table,
    select_candidate,
)

_TARGET_DAY = pd.Timestamp("2026-09-13").date()


def _feature_row(**columns: float) -> pd.DataFrame:
    index = pd.date_range("2026-09-13", periods=1, freq="h", tz="UTC")
    return pd.DataFrame({k: [v] for k, v in columns.items()}, index=index)


def test_candidate_selected_when_feature_row_is_complete() -> None:
    features = _feature_row(price_lag_24h=50.0, renewable_share_forecast_nwp=0.3)
    table = (
        Candidate("full_live_set", frozenset({"price_lag_24h", "renewable_share_forecast_nwp"})),
    )

    selection = select_candidate(table, features, _TARGET_DAY)

    assert selection.candidate is not None
    assert selection.candidate.name == "full_live_set"
    assert selection.result.ok is True


def test_no_submission_when_a_required_column_is_missing_and_it_is_named() -> None:
    features = _feature_row(price_lag_24h=50.0)  # renewable_share_forecast_nwp missing
    table = (
        Candidate("full_live_set", frozenset({"price_lag_24h", "renewable_share_forecast_nwp"})),
    )

    selection = select_candidate(table, features, _TARGET_DAY)

    assert selection.candidate is None
    assert selection.result.ok is False
    assert selection.result.missing_features == ("renewable_share_forecast_nwp",)


def test_selection_is_a_loop_and_the_first_viable_row_wins() -> None:
    """Two artificial rows prove both the loop shape and that it carries
    for 6.9 (spec section 7): the first (stricter) candidate is
    unviable, the second (looser) one wins, and order matters."""
    features = _feature_row(price_lag_24h=50.0)  # only this one column present
    table = (
        Candidate("strict_full_set", frozenset({"price_lag_24h", "renewable_share_forecast_nwp"})),
        Candidate("loose_fallback", frozenset({"price_lag_24h"})),
    )

    selection = select_candidate(table, features, _TARGET_DAY)

    assert selection.candidate is not None
    assert selection.candidate.name == "loose_fallback"


def test_selection_prefers_the_first_viable_row_even_if_a_later_one_would_also_pass() -> None:
    features = _feature_row(price_lag_24h=50.0, renewable_share_forecast_nwp=0.3)
    table = (
        Candidate("first_viable", frozenset({"price_lag_24h"})),
        Candidate("also_viable", frozenset({"price_lag_24h", "renewable_share_forecast_nwp"})),
    )

    selection = select_candidate(table, features, _TARGET_DAY)

    assert selection.candidate is not None
    assert selection.candidate.name == "first_viable"


def test_no_viable_candidate_returns_none_with_a_reason() -> None:
    features = _feature_row(price_lag_24h=float("nan"))
    table = (Candidate("full_live_set", frozenset({"price_lag_24h"})),)

    selection = select_candidate(table, features, _TARGET_DAY)

    assert selection.candidate is None
    assert selection.result.ok is False
    assert selection.result.missing_features == ("price_lag_24h",)


def test_empty_candidate_table_returns_none_without_raising() -> None:
    features = _feature_row(price_lag_24h=50.0)

    selection = select_candidate((), features, _TARGET_DAY)

    assert selection.candidate is None
    assert selection.result.ok is False


def test_full_live_set_candidate_table_has_exactly_one_row_requiring_all_given_columns() -> None:
    columns = frozenset({"price_lag_24h", "month_sin", "residual_load_forecast_nwp"})
    table = full_live_set_candidate_table(columns)

    assert len(table) == 1
    assert table[0].required_features == columns
