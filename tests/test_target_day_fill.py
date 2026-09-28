"""Unit tests for arena/target_day_fill.py -- the target-day gap-handling
mechanics spec 6.9 section 2.8/5.6/7 describes ("Zieltag-Füllung"). Synthetic,
no network, no real feature builders -- isolates plan_target_day_fill/
apply_forward_fill/apply_partial_fill from the real per-group column
derivation (arena/candidates.py's own test suite, tests/test_candidates.py,
covers three_row_ladder() itself).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from energy_price_forecast.arena.candidates import (
    FFILL_MAX_HOURS,
    MAX_GAP_HOURS,
    Candidate,
    FeatureGroup,
    GapPolicy,
)
from energy_price_forecast.arena.preflight import PreflightResult
from energy_price_forecast.arena.target_day_fill import (
    GroupFill,
    apply_forward_fill,
    apply_partial_fill,
    plan_target_day_fill,
)


def _index(n: int, start: str = "2026-09-14T00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def _complete_rows(
    columns: Iterable[str], n: int = 24, start: str = "2026-09-14T00:00"
) -> pd.DataFrame:
    index = _index(n, start)
    return pd.DataFrame({c: [1.0] * n for c in columns}, index=index)


def _gap_last(df: pd.DataFrame, columns: Iterable[str], hours: int) -> pd.DataFrame:
    out = df.copy()
    if hours:
        out.loc[out.index[-hours:], list(columns)] = float("nan")
    return out


def _one_group_candidate(name: str, columns: frozenset[str], policy: GapPolicy) -> Candidate:
    return Candidate(
        name="test",
        rank=1,
        groups=(FeatureGroup(name, columns, policy),),
        requires_nwp=False,
        load_source=None,
        measured_as="test",
    )


# ---------------------------------------------------------------------------
# plan_target_day_fill -- load_forecast (FILL_THEN_FAIL)
# ---------------------------------------------------------------------------

_LOAD = frozenset({"load_forecast_day_ahead"})


def test_load_forecast_3h_gap_forward_fills() -> None:
    candidate = _one_group_candidate("load_forecast", _LOAD, GapPolicy.FILL_THEN_FAIL)
    rows = _gap_last(_complete_rows(_LOAD), _LOAD, FFILL_MAX_HOURS)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert len(plan) == 1
    assert plan[0] == GroupFill(
        group="load_forecast",
        action="forward_fill",
        columns=_LOAD,
        hours=tuple(rows.index[-FFILL_MAX_HOURS:]),
    )


def test_load_forecast_4h_gap_fails() -> None:
    candidate = _one_group_candidate("load_forecast", _LOAD, GapPolicy.FILL_THEN_FAIL)
    rows = _gap_last(_complete_rows(_LOAD), _LOAD, FFILL_MAX_HOURS + 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False
    assert "load_forecast_day_ahead" in plan.missing_features


# ---------------------------------------------------------------------------
# plan_target_day_fill -- nwp_residual (FILL_THEN_PARTIAL)
# ---------------------------------------------------------------------------

_NWP = frozenset({"residual_load_forecast_nwp"})


def test_nwp_residual_3h_gap_forward_fills() -> None:
    candidate = _one_group_candidate("nwp_residual", _NWP, GapPolicy.FILL_THEN_PARTIAL)
    rows = _gap_last(_complete_rows(_NWP), _NWP, FFILL_MAX_HOURS)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert plan[0].action == "forward_fill"


def test_nwp_residual_4h_gap_partial_fills() -> None:
    candidate = _one_group_candidate("nwp_residual", _NWP, GapPolicy.FILL_THEN_PARTIAL)
    rows = _gap_last(_complete_rows(_NWP), _NWP, FFILL_MAX_HOURS + 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert plan[0].action == "partial_fill"
    assert len(plan[0].hours) == FFILL_MAX_HOURS + 1


def test_nwp_residual_6h_gap_partial_fills() -> None:
    candidate = _one_group_candidate("nwp_residual", _NWP, GapPolicy.FILL_THEN_PARTIAL)
    rows = _gap_last(_complete_rows(_NWP), _NWP, MAX_GAP_HOURS)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert plan[0].action == "partial_fill"
    assert len(plan[0].hours) == MAX_GAP_HOURS


def test_nwp_residual_7h_gap_fails() -> None:
    candidate = _one_group_candidate("nwp_residual", _NWP, GapPolicy.FILL_THEN_PARTIAL)
    rows = _gap_last(_complete_rows(_NWP), _NWP, MAX_GAP_HOURS + 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False


# ---------------------------------------------------------------------------
# plan_target_day_fill -- price_lags (FILL_ONLY)
# ---------------------------------------------------------------------------

_PRICE_LAGS = frozenset({"price_lag_24h"})


def test_price_lags_6h_gap_forward_fills() -> None:
    candidate = _one_group_candidate("price_lags", _PRICE_LAGS, GapPolicy.FILL_ONLY)
    rows = _gap_last(_complete_rows(_PRICE_LAGS), _PRICE_LAGS, MAX_GAP_HOURS)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert plan[0].action == "forward_fill"
    assert len(plan[0].hours) == MAX_GAP_HOURS


def test_price_lags_7h_gap_fails() -> None:
    candidate = _one_group_candidate("price_lags", _PRICE_LAGS, GapPolicy.FILL_ONLY)
    rows = _gap_last(_complete_rows(_PRICE_LAGS), _PRICE_LAGS, MAX_GAP_HOURS + 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False


def test_three_scattered_hours_are_counted_the_same_as_a_contiguous_gap() -> None:
    """Spec section 6.3: "3 verstreute Einzelstunden -> funktional geprüft".
    plan_target_day_fill counts gap hours via a plain column-wise isna() mask
    with no contiguity assumption -- this proves it, rather than trusting
    that from the (exclusively trailing-gap) tests above alone.
    """
    candidate = _one_group_candidate("price_lags", _PRICE_LAGS, GapPolicy.FILL_ONLY)
    rows = _complete_rows(_PRICE_LAGS)
    scattered = rows.index[[2, 10, 20]]
    rows.loc[scattered, list(_PRICE_LAGS)] = float("nan")

    plan = plan_target_day_fill(rows, candidate)

    assert isinstance(plan, tuple)
    assert len(plan) == 1
    assert plan[0].action == "forward_fill"
    assert set(plan[0].hours) == set(scattered)


# ---------------------------------------------------------------------------
# plan_target_day_fill -- calendar / gas (STRICT)
# ---------------------------------------------------------------------------


def test_strict_group_fails_on_a_single_missing_hour() -> None:
    columns = frozenset({"hour_sin"})
    candidate = _one_group_candidate("calendar", columns, GapPolicy.STRICT)
    rows = _gap_last(_complete_rows(columns), columns, 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False
    assert plan.missing_features == ("hour_sin",)


def test_gas_group_fails_on_a_single_missing_hour() -> None:
    columns = frozenset({"ttf_gas_lag_48h"})
    candidate = _one_group_candidate("gas", columns, GapPolicy.STRICT)
    rows = _gap_last(_complete_rows(columns), columns, 1)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False


def test_no_gap_returns_empty_plan() -> None:
    columns = frozenset({"hour_sin"})
    candidate = _one_group_candidate("calendar", columns, GapPolicy.STRICT)
    rows = _complete_rows(columns)
    plan = plan_target_day_fill(rows, candidate)
    assert plan == ()


# ---------------------------------------------------------------------------
# Plan before fill: an 8-hour load gap must not silently forward-fill
# ---------------------------------------------------------------------------


def test_eight_hour_load_gap_fails_rather_than_forward_filling() -> None:
    candidate = _one_group_candidate("load_forecast", _LOAD, GapPolicy.FILL_THEN_FAIL)
    rows = _gap_last(_complete_rows(_LOAD), _LOAD, 8)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, PreflightResult)
    assert plan.ok is False


# ---------------------------------------------------------------------------
# 23-/25-hour DST days -- gap counting is index-based, not a fixed-length
# assumption, and apply_partial_fill's hour-to-quarter-hour mapping holds on
# a 92/100-value delivery day too.
# ---------------------------------------------------------------------------


def test_plan_works_on_a_23_hour_day() -> None:
    columns = frozenset({"price_lag_24h"})
    candidate = _one_group_candidate("price_lags", columns, GapPolicy.FILL_ONLY)
    rows = _gap_last(_complete_rows(columns, n=23), columns, 3)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert len(plan[0].hours) == 3


def test_plan_works_on_a_25_hour_day() -> None:
    columns = frozenset({"price_lag_24h"})
    candidate = _one_group_candidate("price_lags", columns, GapPolicy.FILL_ONLY)
    rows = _gap_last(_complete_rows(columns, n=25), columns, 3)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    assert len(plan[0].hours) == 3


def test_apply_partial_fill_on_a_92_value_dst_day() -> None:
    qh_index = pd.date_range("2026-03-29T00:00", periods=92, freq="15min", tz="UTC")
    payload = pd.Series(100.0, index=qh_index)
    persistence = pd.Series(50.0, index=qh_index)
    gap_hour = qh_index[0]
    fills = (GroupFill("nwp_residual", "partial_fill", _NWP, (gap_hour,)),)
    filled = apply_partial_fill(payload, persistence, fills)
    assert len(filled) == 92
    assert (filled.iloc[:4] == 50.0).all()
    assert (filled.iloc[4:] == 100.0).all()


# ---------------------------------------------------------------------------
# Same hour, two groups: forward_fill loses to partial_fill
# ---------------------------------------------------------------------------


def test_same_hour_forward_fill_loses_to_partial_fill() -> None:
    price_lags_cols = frozenset({"price_lag_24h"})
    nwp_cols = frozenset({"residual_load_forecast_nwp"})
    candidate = Candidate(
        name="test",
        rank=1,
        groups=(
            FeatureGroup("price_lags", price_lags_cols, GapPolicy.FILL_ONLY),
            FeatureGroup("nwp_residual", nwp_cols, GapPolicy.FILL_THEN_PARTIAL),
        ),
        requires_nwp=False,
        load_source=None,
        measured_as="test",
    )
    rows = _complete_rows(price_lags_cols | nwp_cols)
    # price_lags has a 2h gap (well within FILL_ONLY -- would forward_fill on
    # its own); nwp_residual has a 5h gap covering the same trailing hours
    # (falls into FILL_THEN_PARTIAL's partial bracket) -- the two trailing
    # hours they share must end up partial_fill, not forward_fill.
    rows = _gap_last(rows, price_lags_cols, 2)
    rows = _gap_last(rows, nwp_cols, 5)
    plan = plan_target_day_fill(rows, candidate)
    assert isinstance(plan, tuple)
    by_group = {fill.group: fill for fill in plan}
    assert by_group["nwp_residual"].action == "partial_fill"
    assert len(by_group["nwp_residual"].hours) == 5
    # price_lags' own forward_fill must exclude the 2 hours that overlap
    # nwp_residual's partial_fill hours (spec: "wird sie teil-gefüllt").
    assert "price_lags" not in by_group or len(by_group["price_lags"].hours) == 0


# ---------------------------------------------------------------------------
# apply_forward_fill
# ---------------------------------------------------------------------------


def test_apply_forward_fill_fills_only_the_declared_hours() -> None:
    matrix = _complete_rows(_LOAD, n=27, start="2026-09-13T21:00")  # D-1 tail + all of D
    target_rows = matrix.loc["2026-09-14"]
    matrix.loc[target_rows.index[:2], "load_forecast_day_ahead"] = float("nan")
    fills = (GroupFill("load_forecast", "forward_fill", _LOAD, tuple(target_rows.index[:2])),)
    filled = apply_forward_fill(matrix, fills)
    assert filled.loc[target_rows.index[:2], "load_forecast_day_ahead"].notna().all()
    # the value comes from the last known (D-1) row, not a re-derived constant.
    assert (filled.loc[target_rows.index[:2], "load_forecast_day_ahead"] == 1.0).all()


def test_apply_forward_fill_gap_at_start_of_day_pulls_from_d_minus_1() -> None:
    matrix = _complete_rows(_LOAD, n=25, start="2026-09-13T23:00")
    matrix.loc["2026-09-13T23:00", "load_forecast_day_ahead"] = 42.0
    target_start = pd.Timestamp("2026-09-14T00:00", tz="UTC")
    matrix.loc[target_start, "load_forecast_day_ahead"] = float("nan")
    fills = (GroupFill("load_forecast", "forward_fill", _LOAD, (target_start,)),)
    filled = apply_forward_fill(matrix, fills)
    assert filled.loc[target_start, "load_forecast_day_ahead"] == 42.0


def test_apply_forward_fill_never_touches_hours_outside_the_plan() -> None:
    matrix = _complete_rows(_LOAD, n=48)
    train_rows = matrix.iloc[:24].copy()
    target_rows = matrix.iloc[24:]
    fills = (GroupFill("load_forecast", "forward_fill", _LOAD, (target_rows.index[-1],)),)
    filled = apply_forward_fill(matrix, fills)
    pd.testing.assert_frame_equal(filled.iloc[:24], train_rows)


# ---------------------------------------------------------------------------
# apply_partial_fill
# ---------------------------------------------------------------------------


def test_apply_partial_fill_replaces_exactly_the_four_quarter_hour_slots() -> None:
    qh_index = pd.date_range("2026-09-14T00:00", periods=96, freq="15min", tz="UTC")
    payload = pd.Series(100.0, index=qh_index)
    persistence = pd.Series(50.0, index=qh_index)
    gap_hour = pd.Timestamp("2026-09-14T10:00", tz="UTC")
    fills = (GroupFill("nwp_residual", "partial_fill", _NWP, (gap_hour,)),)
    filled = apply_partial_fill(payload, persistence, fills)
    affected = pd.date_range(gap_hour, periods=4, freq="15min")
    assert (filled.loc[affected] == 50.0).all()
    unaffected = filled.index.difference(affected)
    assert (filled.loc[unaffected] == 100.0).all()


def test_apply_partial_fill_ignores_forward_fill_entries() -> None:
    qh_index = pd.date_range("2026-09-14T00:00", periods=96, freq="15min", tz="UTC")
    payload = pd.Series(100.0, index=qh_index)
    persistence = pd.Series(50.0, index=qh_index)
    fills = (GroupFill("price_lags", "forward_fill", _PRICE_LAGS, (qh_index[0],)),)
    filled = apply_partial_fill(payload, persistence, fills)
    pd.testing.assert_series_equal(filled, payload)
