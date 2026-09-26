"""Unit tests for arena/load_patch.py -- the Similar-Day-Patch for the
ENTSO-E day-ahead load forecast (spec 6.9, section 2.2 / 5.5).

Not yet wired into the live path (step 9 of 6.9) -- this file only proves
the two pure functions in isolation, per the spec's own test file split
(tests/test_load_patch.py, section 3.2).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_price_forecast.arena.load_patch import (
    MAX_LOAD_PATCH_WEEKS_BACK,
    LoadPatchReference,
    _local_hour_index,
    build_patched_load,
    choose_reference_day,
)


def _rule(
    *, holidays: frozenset[dt.date] = frozenset(), incomplete: frozenset[dt.date] = frozenset()
):
    return {
        "is_holiday": lambda day: day in holidays,
        "is_complete": lambda day: day not in incomplete,
    }


# --- choose_reference_day -------------------------------------------------


def test_tue_fri_normal_case_uses_yesterday() -> None:
    target = dt.date(2026, 9, 16)  # a Wednesday
    ref = choose_reference_day(target, **_rule())

    assert ref == LoadPatchReference(reference_day=dt.date(2026, 9, 15), weeks_back=0, skipped=())


def test_mon_sat_sun_normal_case_uses_last_week_same_weekday() -> None:
    target = dt.date(2026, 9, 19)  # a Saturday
    ref = choose_reference_day(target, **_rule())

    assert ref == LoadPatchReference(reference_day=dt.date(2026, 9, 12), weeks_back=0, skipped=())


def test_holiday_target_uses_last_sunday_before_it() -> None:
    target = dt.date(2026, 10, 3)  # Tag der Deutschen Einheit, a Saturday
    ref = choose_reference_day(target, **_rule(holidays=frozenset({target})))

    assert ref == LoadPatchReference(reference_day=dt.date(2026, 9, 27), weeks_back=0, skipped=())


def test_tue_fri_escalates_to_last_week_when_yesterday_is_a_holiday() -> None:
    target = dt.date(2026, 9, 16)  # Wednesday, yesterday = Tue 2026-09-15
    yesterday = dt.date(2026, 9, 15)
    ref = choose_reference_day(target, **_rule(holidays=frozenset({yesterday})))

    assert ref == LoadPatchReference(
        reference_day=dt.date(2026, 9, 9),
        weeks_back=1,
        skipped=((yesterday, "holiday"),),
    )


def test_tue_fri_escalates_a_second_time_on_incomplete_data() -> None:
    target = dt.date(2026, 9, 16)
    yesterday = dt.date(2026, 9, 15)
    last_week = dt.date(2026, 9, 9)
    ref = choose_reference_day(
        target,
        **_rule(holidays=frozenset({yesterday}), incomplete=frozenset({last_week})),
    )

    assert ref == LoadPatchReference(
        reference_day=dt.date(2026, 9, 2),
        weeks_back=2,
        skipped=((yesterday, "holiday"), (last_week, "incomplete")),
    )


def test_mon_sat_sun_escalates_to_two_weeks_back() -> None:
    target = dt.date(2026, 9, 19)  # Saturday
    first = dt.date(2026, 9, 12)
    ref = choose_reference_day(target, **_rule(incomplete=frozenset({first})))

    assert ref == LoadPatchReference(
        reference_day=dt.date(2026, 9, 5),
        weeks_back=1,
        skipped=((first, "incomplete"),),
    )


def test_holiday_target_escalates_to_the_sunday_before() -> None:
    target = dt.date(2026, 10, 3)
    first_sunday = dt.date(2026, 9, 27)
    ref = choose_reference_day(
        target, **_rule(holidays=frozenset({target}), incomplete=frozenset({first_sunday}))
    )

    assert ref == LoadPatchReference(
        reference_day=dt.date(2026, 9, 20),
        weeks_back=1,
        skipped=((first_sunday, "incomplete"),),
    )


def test_a_sunday_reference_is_usable_even_if_flagged_a_holiday() -> None:
    """Spec 2.2: 'ein Sonntag ist immer brauchbar, auch wenn er ein
    Feiertag ist' -- the holiday veto has an exception for Sundays,
    exercised here via the holiday-target chain, whose references always
    land on a Sunday."""
    target = dt.date(2026, 10, 3)
    first_sunday = dt.date(2026, 9, 27)
    ref = choose_reference_day(target, **_rule(holidays=frozenset({target, first_sunday})))

    assert ref == LoadPatchReference(reference_day=first_sunday, weeks_back=0, skipped=())


def test_returns_none_when_nothing_usable_within_the_weeks_back_limit() -> None:
    target = dt.date(2026, 9, 16)
    all_candidates = {
        dt.date(2026, 9, 15),
        dt.date(2026, 9, 9),
        dt.date(2026, 9, 2),
        dt.date(2026, 8, 26),
        dt.date(2026, 8, 19),
    }
    assert len(all_candidates) == MAX_LOAD_PATCH_WEEKS_BACK + 1

    ref = choose_reference_day(target, **_rule(incomplete=frozenset(all_candidates)))

    assert ref is None


def test_weeks_back_boundary_is_exactly_max_load_patch_weeks_back() -> None:
    target = dt.date(2026, 9, 16)
    all_but_last = {
        dt.date(2026, 9, 15),
        dt.date(2026, 9, 9),
        dt.date(2026, 9, 2),
        dt.date(2026, 8, 26),
    }
    last = dt.date(2026, 8, 19)

    ref = choose_reference_day(target, **_rule(incomplete=frozenset(all_but_last)))

    assert ref == LoadPatchReference(
        reference_day=last,
        weeks_back=MAX_LOAD_PATCH_WEEKS_BACK,
        skipped=tuple((d, "incomplete") for d in sorted(all_but_last, reverse=True)),
    )


# --- build_patched_load ----------------------------------------------------


def _hourly_series(day: dt.date, n_hours: int, *, start: float = 0.0) -> pd.Series:
    """A value-per-hour series (local wall-clock position, i.e. already the
    DST-correct length for `day`) reindexed onto real UTC timestamps."""
    index = _local_hour_index(day)
    assert len(index) == n_hours
    values = start + pd.RangeIndex(n_hours).to_numpy(dtype=float)
    return pd.Series(values, index=index.tz_convert("UTC"), name="load_forecast_day_ahead")


def test_normal_day_to_normal_day_is_a_plain_hour_of_day_copy() -> None:
    reference_day = dt.date(2026, 9, 15)
    target_day = dt.date(2026, 9, 16)
    reference_load = _hourly_series(reference_day, 24)

    patched = build_patched_load(reference_load, reference_day, target_day)

    assert len(patched) == 24
    assert list(patched.to_numpy()) == list(range(24))
    assert patched.name == "load_forecast_day_ahead"


def test_reference_is_a_23h_spring_forward_day_fills_hour_2_from_hour_1() -> None:
    reference_day = dt.date(2026, 3, 29)  # DE spring-forward day, 23 local hours
    target_day = dt.date(2026, 3, 30)
    reference_load = _hourly_series(reference_day, 23)

    patched = build_patched_load(reference_load, reference_day, target_day)

    values = patched.to_numpy()
    assert len(values) == 24
    assert values[1] == 1.0  # hour 1, unaffected
    assert values[2] == 1.0  # missing hour 2 borrows hour 1's value
    assert values[3] == 2.0  # hour 3 was position 2 in the 23-value source


def test_reference_is_a_25h_fall_back_day_uses_the_first_of_the_two_hour_2s() -> None:
    reference_day = dt.date(2026, 10, 25)  # DE fall-back day, 25 local hours
    target_day = dt.date(2026, 10, 26)
    reference_load = _hourly_series(reference_day, 25)

    patched = build_patched_load(reference_load, reference_day, target_day)

    values = patched.to_numpy()
    assert len(values) == 24
    assert values[1] == 1.0
    assert values[2] == 2.0  # first occurrence of hour 2 (position 2), not the second (position 3)
    assert values[3] == 4.0  # hour 3 was position 4 in the 25-value source


def test_target_is_a_23h_spring_forward_day_drops_hour_2() -> None:
    reference_day = dt.date(2026, 3, 22)
    target_day = dt.date(2026, 3, 29)  # DE spring-forward day, 23 local hours
    reference_load = _hourly_series(reference_day, 24)

    patched = build_patched_load(reference_load, reference_day, target_day)

    values = patched.to_numpy()
    assert len(values) == 23
    assert list(values) == [
        0.0,
        1.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
    ]


def test_target_is_a_25h_fall_back_day_duplicates_hour_2() -> None:
    reference_day = dt.date(2026, 10, 18)
    target_day = dt.date(2026, 10, 25)  # DE fall-back day, 25 local hours
    reference_load = _hourly_series(reference_day, 24)

    patched = build_patched_load(reference_load, reference_day, target_day)

    values = patched.to_numpy()
    assert len(values) == 25
    assert values[2] == 2.0
    assert values[3] == 2.0  # hour 2 used twice
    assert values[4] == 3.0


def test_output_index_is_utc_and_covers_the_target_days_own_local_hours() -> None:
    reference_day = dt.date(2026, 9, 15)
    target_day = dt.date(2026, 9, 16)
    reference_load = _hourly_series(reference_day, 24)

    patched = build_patched_load(reference_load, reference_day, target_day)

    index = pd.DatetimeIndex(patched.index)
    assert str(index.tz) == "UTC"
    local = index.tz_convert("Europe/Berlin")
    assert all(d == target_day for d in local.date)
