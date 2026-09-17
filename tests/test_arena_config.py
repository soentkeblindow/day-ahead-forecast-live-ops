"""Unit tests for the Arena live switch (spec 6.7.3, section 2.3/5.3)."""

import pytest

from energy_price_forecast.arena.config import ARENA_LIVE_ENV_VAR, is_live_enabled


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
