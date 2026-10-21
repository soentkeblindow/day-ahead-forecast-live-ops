"""Unit tests for the Arena live switch (spec 6.7.3, section 2.3/5.3) and
its Challenge-8 sibling (Sprint 7.1 Teil B -> Probephase).
"""

import pytest

from energy_price_forecast.arena.config import (
    ARENA_LIVE_ENV_VAR,
    ARENA_QUANTILE_LIVE_ENV_VAR,
    is_live_enabled,
    is_quantile_live_enabled,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        (None, False),  # not set
        ("", False),
        ("True", False),
        ("1", False),
        ("yes", False),
        ("false", False),
    ],
)
def test_is_live_enabled_only_for_the_exact_value_true(value: str | None, expected: bool) -> None:
    env = {} if value is None else {ARENA_LIVE_ENV_VAR: value}
    assert is_live_enabled(env) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        (None, False),  # not set -- the real state throughout the Probephase
        ("", False),
        ("True", False),
        ("false", False),
    ],
)
def test_is_quantile_live_enabled_only_for_the_exact_value_true(
    value: str | None, expected: bool
) -> None:
    env = {} if value is None else {ARENA_QUANTILE_LIVE_ENV_VAR: value}
    assert is_quantile_live_enabled(env) is expected


def test_quantile_live_switch_is_independent_of_the_point_switch() -> None:
    """Challenge 2 being live must never make Challenge 8 live too, and
    vice versa -- a real risk this test pins given both switches look
    similar at a glance."""
    env = {ARENA_LIVE_ENV_VAR: "true"}
    assert is_live_enabled(env) is True
    assert is_quantile_live_enabled(env) is False
