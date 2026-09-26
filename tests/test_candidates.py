"""Unit tests for arena/candidates.py -- the declarative fallback-ladder
table (spec 6.9, sections 2.1, 5.2, 2.9).
"""

from __future__ import annotations

from energy_price_forecast.arena.candidates import (
    Candidate,
    FeatureGroup,
    GapPolicy,
    best_accepted_rank,
    may_submit,
    three_row_ladder,
)

_CALENDAR = frozenset({"hour_sin", "hour_cos"})
_NWP_RESIDUAL = frozenset({"residual_load_forecast_nwp", "renewable_share_forecast_nwp"})
_LOAD_FORECAST = frozenset({"load_forecast_day_ahead"})
_PRICE_LAGS = frozenset({"price_lag_24h"})
_GAS = frozenset({"ttf_gas_lag_48h"})


def _ladder() -> tuple[Candidate, ...]:
    return three_row_ladder(
        calendar_columns=_CALENDAR,
        nwp_residual_columns=_NWP_RESIDUAL,
        load_forecast_columns=_LOAD_FORECAST,
        price_lag_columns=_PRICE_LAGS,
        gas_columns=_GAS,
    )


# --- FeatureGroup / Candidate -----------------------------------------------


def test_feature_group_carries_name_columns_and_policy() -> None:
    group = FeatureGroup("gas", _GAS, GapPolicy.STRICT)
    assert group.name == "gas"
    assert group.columns == _GAS
    assert group.policy is GapPolicy.STRICT


def test_candidate_required_columns_is_the_union_of_its_groups() -> None:
    candidate = Candidate(
        name="core_gas",
        rank=1,
        groups=(
            FeatureGroup("calendar", _CALENDAR, GapPolicy.STRICT),
            FeatureGroup("gas", _GAS, GapPolicy.STRICT),
        ),
        requires_nwp=True,
        load_source="entsoe",
        measured_as="test",
    )
    assert candidate.required_columns() == _CALENDAR | _GAS


def test_candidate_required_columns_is_empty_for_no_groups() -> None:
    candidate = Candidate(
        name="empty", rank=9, groups=(), requires_nwp=False, load_source=None, measured_as="test"
    )
    assert candidate.required_columns() == frozenset()


# --- three_row_ladder --------------------------------------------------------


def test_three_row_ladder_has_exactly_three_rows_in_rank_order() -> None:
    ladder = _ladder()
    assert [c.rank for c in ladder] == [1, 2, 3]
    assert [c.name for c in ladder] == ["core_gas", "core_gas_loadpatch", "base"]


def test_rows_1_and_2_share_the_exact_same_groups() -> None:
    """spec 6.9 section 2.1: "Zeile 2 ist dasselbe Modell wie Zeile 1: gleiche
    Features, gleiches Training." -- only load_source/measured_as differ."""
    core_gas, core_gas_loadpatch, _base = _ladder()
    assert core_gas.groups == core_gas_loadpatch.groups
    assert core_gas.required_columns() == core_gas_loadpatch.required_columns()
    assert core_gas.requires_nwp is True
    assert core_gas_loadpatch.requires_nwp is True
    assert core_gas.load_source == "entsoe"
    assert core_gas_loadpatch.load_source == "similar_day"


def test_row_3_is_gas_free_and_has_no_nwp_or_load_forecast_group() -> None:
    """spec 6.9 section 2.1, Fassung 2 (Owner 2026-09-24): row 3 is
    gas-free -- no single point of failure from a dead TTF feed."""
    _core_gas, _core_gas_loadpatch, base = _ladder()
    assert base.requires_nwp is False
    assert base.load_source is None
    assert base.required_columns() == _CALENDAR | _PRICE_LAGS
    group_names = {g.name for g in base.groups}
    assert "gas" not in group_names
    assert "nwp_residual" not in group_names
    assert "load_forecast" not in group_names


def test_full_and_no_fundamentals_are_not_in_the_ladder() -> None:
    """spec 6.9 section 2.1: "full (bisher live) ... fliegen raus" -- no
    replacement row, the retired candidate is simply absent."""
    names = {c.name for c in _ladder()}
    assert "full" not in names
    assert "live" not in names
    assert "no_fundamentals" not in names


def test_ladder_group_policies_match_spec_section_5_2_table() -> None:
    core_gas, _core_gas_loadpatch, base = _ladder()
    policy_by_group = {g.name: g.policy for g in core_gas.groups}
    assert policy_by_group["calendar"] is GapPolicy.STRICT
    assert policy_by_group["nwp_residual"] is GapPolicy.FILL_THEN_PARTIAL
    assert policy_by_group["load_forecast"] is GapPolicy.FILL_THEN_FAIL
    assert policy_by_group["price_lags"] is GapPolicy.FILL_ONLY
    assert policy_by_group["gas"] is GapPolicy.STRICT

    base_policy_by_group = {g.name: g.policy for g in base.groups}
    assert base_policy_by_group["calendar"] is GapPolicy.STRICT
    assert base_policy_by_group["price_lags"] is GapPolicy.FILL_ONLY


def test_measured_as_names_a_real_measurement_for_every_row() -> None:
    for candidate in _ladder():
        assert candidate.measured_as  # non-empty
        assert "Messung" in candidate.measured_as


# --- best_accepted_rank / may_submit (spec 6.9 section 2.9) ------------------


def _record(
    *, target_day: str, submitted: bool, mode: str, candidate: str | None
) -> dict[str, object]:
    return {
        "target_day": target_day,
        "submitted": submitted,
        "submission_mode": mode,
        "candidate_selected": candidate,
    }


def test_best_accepted_rank_is_none_with_no_prior_submissions() -> None:
    assert best_accepted_rank([], "2026-10-02", _ladder()) is None


def test_best_accepted_rank_reads_the_accepted_live_rank() -> None:
    records = [_record(target_day="2026-10-02", submitted=True, mode="live", candidate="core_gas")]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) == 1


def test_best_accepted_rank_ignores_rejected_submissions() -> None:
    records = [_record(target_day="2026-10-02", submitted=False, mode="live", candidate="core_gas")]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) is None


def test_best_accepted_rank_ignores_smoke_mode() -> None:
    records = [_record(target_day="2026-10-02", submitted=True, mode="smoke", candidate="core_gas")]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) is None


def test_best_accepted_rank_ignores_a_different_target_day() -> None:
    records = [_record(target_day="2026-10-01", submitted=True, mode="live", candidate="core_gas")]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) is None


def test_best_accepted_rank_picks_the_lowest_rank_among_several() -> None:
    records = [
        _record(target_day="2026-10-02", submitted=True, mode="live", candidate="base"),
        _record(
            target_day="2026-10-02", submitted=True, mode="live", candidate="core_gas_loadpatch"
        ),
    ]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) == 2


def test_best_accepted_rank_ignores_an_unknown_candidate_name() -> None:
    """An old record naming a since-removed candidate (e.g. a retired 'live')
    must not crash a live run."""
    records = [_record(target_day="2026-10-02", submitted=True, mode="live", candidate="live")]
    assert best_accepted_rank(records, "2026-10-02", _ladder()) is None


def test_may_submit_true_when_nothing_accepted_yet() -> None:
    assert may_submit(3, None) is True


def test_may_submit_true_at_equal_rank() -> None:
    assert may_submit(2, 2) is True


def test_may_submit_true_at_better_rank() -> None:
    assert may_submit(1, 3) is True


def test_may_submit_false_at_worse_rank() -> None:
    assert may_submit(3, 1) is False
