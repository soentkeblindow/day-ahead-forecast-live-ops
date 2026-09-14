"""Unit tests for the Arena payload builder and validator. No network access."""

import json
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.arena.payload import (
    PayloadValidationError,
    build_payload,
    expected_value_count,
    validate_payload,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Matches the real challenge 2 (Day-Ahead Prices | Germany-Luxembourg | Point
# Forecast) as parsed by arena/catalog.py, without going through the network.
CHALLENGE = ChallengeSpec(
    challenge_id="2",
    name="Day-Ahead Prices | Germany-Luxembourg | Point Forecast",
    timezone="Europe/Berlin",
    resolution_minutes=15,
    precision_decimals=2,
    allow_negative=True,
    max_forecast_points=None,
    raw={},
)


def _ts(iso: str) -> pd.Timestamp:
    return pd.Timestamp(iso)


# ---------------------------------------------------------------------------
# expected_value_count
# ---------------------------------------------------------------------------


def test_expected_value_count_normal_day() -> None:
    assert expected_value_count(CHALLENGE, _ts("2026-08-20T00:00:00+02:00")) == 96


def test_expected_value_count_spring_forward_dst() -> None:
    # 2026-03-29: clocks jump forward 02:00 -> 03:00 CEST, 23h day.
    assert expected_value_count(CHALLENGE, _ts("2026-03-29T00:00:00+01:00")) == 92


def test_expected_value_count_fall_back_dst() -> None:
    # 2026-10-25: clocks fall back 03:00 -> 02:00 CET, 25h day.
    assert expected_value_count(CHALLENGE, _ts("2026-10-25T00:00:00+02:00")) == 100


def test_expected_value_count_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        expected_value_count(CHALLENGE, pd.Timestamp("2026-08-20T00:00:00"))


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_uses_local_offset_not_utc_z_form() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), [1.0] * 96)
    assert payload["target_start"] == "2026-08-20T00:00:00+02:00"
    assert not payload["target_start"].endswith("Z")


def test_build_payload_rounds_to_precision_decimals() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), [1.23456, 2.0])
    assert payload["values"] == [1.23, 2.0]


def test_build_payload_keeps_negative_values() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), [-5.5, 10.0])
    assert payload["values"] == [-5.5, 10.0]


def test_build_payload_rejects_naive_target_start() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        build_payload(CHALLENGE, pd.Timestamp("2026-08-20T00:00:00"), [1.0])


# ---------------------------------------------------------------------------
# validate_payload
# ---------------------------------------------------------------------------


def _valid_payload(n: int = 96) -> dict:
    return build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), [1.0] * n)


def test_validate_payload_accepts_a_valid_payload() -> None:
    validate_payload(_valid_payload(), CHALLENGE)  # must not raise


@pytest.mark.parametrize(
    ("target_start", "n"),
    [
        ("2026-08-20T00:00:00+02:00", 96),  # normal day
        ("2026-03-29T00:00:00+01:00", 92),  # DE/LU DST start -- spring forward, 23h day
        ("2026-10-25T00:00:00+02:00", 100),  # DE/LU DST end -- fall back, 25h day
    ],
)
def test_validate_payload_accepts_the_real_count_for_each_day_type(
    target_start: str, n: int
) -> None:
    """Restarbeit 6.7.2, Teil C.1, level 3 -- a full build_payload +
    validate_payload accept-path for each day type, not just
    expected_value_count's own arithmetic (test_expected_value_count_
    spring_forward_dst/test_expected_value_count_fall_back_dst above,
    already at these exact dates) -- the count a genuinely 92/100-value day
    actually produces must be *accepted*, not just correctly computed."""
    payload = build_payload(CHALLENGE, _ts(target_start), [1.0] * n)
    validate_payload(payload, CHALLENGE)  # must not raise


def test_validate_payload_rejects_too_few_values() -> None:
    with pytest.raises(PayloadValidationError, match="95.*expected 96"):
        validate_payload(_valid_payload(95), CHALLENGE)


def test_validate_payload_rejects_too_many_values() -> None:
    with pytest.raises(PayloadValidationError, match="97.*expected 96"):
        validate_payload(_valid_payload(97), CHALLENGE)


def test_validate_payload_rejects_nan() -> None:
    payload = _valid_payload()
    payload["values"][0] = float("nan")
    with pytest.raises(PayloadValidationError, match="non-finite"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_inf() -> None:
    payload = _valid_payload()
    payload["values"][0] = float("inf")
    with pytest.raises(PayloadValidationError, match="non-finite"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_nested_values() -> None:
    payload = _valid_payload()
    payload["values"][0] = [1.0, 2.0]
    with pytest.raises(PayloadValidationError, match="flat list"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_utc_z_form_target_start() -> None:
    payload = _valid_payload()
    payload["target_start"] = "2026-08-19T22:00:00Z"
    with pytest.raises(PayloadValidationError, match="local offset"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_naive_target_start() -> None:
    payload = _valid_payload()
    payload["target_start"] = "2026-08-20T00:00:00"
    with pytest.raises(PayloadValidationError, match="offset"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_exceeding_max_forecast_points() -> None:
    capped_challenge = ChallengeSpec(
        challenge_id="2",
        name=CHALLENGE.name,
        timezone=CHALLENGE.timezone,
        resolution_minutes=CHALLENGE.resolution_minutes,
        precision_decimals=CHALLENGE.precision_decimals,
        allow_negative=CHALLENGE.allow_negative,
        max_forecast_points=10,
        raw={},
    )
    payload = build_payload(capped_challenge, _ts("2026-08-20T00:00:00+02:00"), [1.0] * 96)
    with pytest.raises(PayloadValidationError, match="max_forecast_points"):
        validate_payload(payload, capped_challenge)


# ---------------------------------------------------------------------------
# Golden fixture (b)
# ---------------------------------------------------------------------------


def test_fixture_passes_validation() -> None:
    """If our validator rejects the Arena's own example, our validator is wrong."""
    fixture = json.loads((FIXTURES_DIR / "arena_public_example.json").read_text(encoding="utf-8"))
    validate_payload(fixture, CHALLENGE)  # must not raise


def test_our_payload_matches_fixture_structure() -> None:
    """Reverse test: a payload we build has the same shape as the Arena's example.

    Not the same keys exactly (ours must be a superset of what's required),
    same types, same values-nesting depth -- not byte-identical (spec §6.4).
    """
    fixture = json.loads((FIXTURES_DIR / "arena_public_example.json").read_text(encoding="utf-8"))
    ours = build_payload(
        CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), [1.0] * len(fixture["values"])
    )

    assert set(ours.keys()) >= {"challenge_id", "target_start", "values"}
    assert type(ours["challenge_id"]) is type(fixture["challenge_id"])
    assert type(ours["target_start"]) is type(fixture["target_start"])
    assert isinstance(ours["values"], list)
    assert isinstance(fixture["values"], list)
    assert len(ours["values"]) == len(fixture["values"])
    assert not any(isinstance(v, list | tuple) for v in ours["values"])
    assert not any(isinstance(v, list | tuple) for v in fixture["values"])
