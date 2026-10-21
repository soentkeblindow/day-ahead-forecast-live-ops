"""Unit tests for the Arena quantile-challenge payload builder and
validator (Sprint 7.1 Teil B -> Probephase). No network access.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import QuantileChallengeSpec
from energy_price_forecast.arena.payload import PayloadValidationError
from energy_price_forecast.arena.quantile_payload import (
    build_payload,
    expected_value_count,
    validate_payload,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Matches the real challenge 8 (Day-Ahead Prices | Germany-Luxembourg |
# Quantile Forecast) as parsed by arena/catalog.py, without going through
# the network.
CHALLENGE = QuantileChallengeSpec(
    challenge_id="8",
    name="Day-Ahead Prices | Germany-Luxembourg | Quantile Forecast (5 quantiles)",
    timezone="Europe/Berlin",
    resolution_minutes=15,
    quantile_levels=(0.025, 0.25, 0.5, 0.75, 0.975),
    precision_decimals=2,
    allow_negative=True,
    max_forecast_points=None,
    raw={},
)

LEVELS = CHALLENGE.quantile_levels


def _ts(iso: str) -> pd.Timestamp:
    return pd.Timestamp(iso)


def _quantiles(n: int, index: pd.DatetimeIndex | None = None) -> dict[float, pd.Series]:
    idx = index if index is not None else pd.RangeIndex(n)
    # level a=0.5 -> value 50.0 etc, strictly ascending across levels so the
    # monotonicity check in validate_payload always passes by construction.
    return {a: pd.Series([50.0 + 10 * a] * n, index=idx) for a in LEVELS}


# ---------------------------------------------------------------------------
# expected_value_count
# ---------------------------------------------------------------------------


def test_expected_value_count_normal_day() -> None:
    assert expected_value_count(CHALLENGE, _ts("2026-08-20T00:00:00+02:00")) == 96


def test_expected_value_count_spring_forward_dst() -> None:
    assert expected_value_count(CHALLENGE, _ts("2026-03-29T00:00:00+01:00")) == 92


def test_expected_value_count_fall_back_dst() -> None:
    assert expected_value_count(CHALLENGE, _ts("2026-10-25T00:00:00+02:00")) == 100


def test_expected_value_count_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        expected_value_count(CHALLENGE, pd.Timestamp("2026-08-20T00:00:00"))


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_uses_local_offset_not_utc_z_form() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), _quantiles(96))
    assert payload["target_start"] == "2026-08-20T00:00:00+02:00"
    assert not payload["target_start"].endswith("Z")


def test_build_payload_shape_is_a_list_of_level_length_lists() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), _quantiles(3))
    assert len(payload["values"]) == 3
    for row in payload["values"]:
        assert len(row) == len(LEVELS)


def test_build_payload_row_order_matches_quantile_levels_ascending() -> None:
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), _quantiles(1))
    row = payload["values"][0]
    assert row == sorted(row)
    assert row == [round(50.0 + 10 * a, 2) for a in LEVELS]


def test_build_payload_rounds_to_precision_decimals() -> None:
    quantiles = {a: pd.Series([1.23456]) for a in LEVELS}
    payload = build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), quantiles)
    assert payload["values"][0] == [1.23] * len(LEVELS)


def test_build_payload_rejects_naive_target_start() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        build_payload(CHALLENGE, pd.Timestamp("2026-08-20T00:00:00"), _quantiles(1))


def test_build_payload_rejects_missing_level() -> None:
    quantiles = {a: pd.Series([1.0]) for a in LEVELS if a != 0.5}
    with pytest.raises(ValueError, match="0.5"):
        build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), quantiles)


def test_build_payload_rejects_mismatched_indices() -> None:
    quantiles = {a: pd.Series([1.0], index=[i]) for i, a in enumerate(LEVELS)}
    with pytest.raises(ValueError, match="identical index"):
        build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), quantiles)


# ---------------------------------------------------------------------------
# validate_payload
# ---------------------------------------------------------------------------


def _valid_payload(n: int = 96) -> dict:
    return build_payload(CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), _quantiles(n))


def test_validate_payload_accepts_a_valid_payload() -> None:
    validate_payload(_valid_payload(), CHALLENGE)  # must not raise


@pytest.mark.parametrize(
    ("target_start", "n"),
    [
        ("2026-08-20T00:00:00+02:00", 96),
        ("2026-03-29T00:00:00+01:00", 92),
        ("2026-10-25T00:00:00+02:00", 100),
    ],
)
def test_validate_payload_accepts_the_real_count_for_each_day_type(
    target_start: str, n: int
) -> None:
    payload = build_payload(CHALLENGE, _ts(target_start), _quantiles(n))
    validate_payload(payload, CHALLENGE)  # must not raise


def test_validate_payload_rejects_too_few_rows() -> None:
    with pytest.raises(PayloadValidationError, match="95.*expected 96"):
        validate_payload(_valid_payload(95), CHALLENGE)


def test_validate_payload_rejects_too_many_rows() -> None:
    with pytest.raises(PayloadValidationError, match="97.*expected 96"):
        validate_payload(_valid_payload(97), CHALLENGE)


def test_validate_payload_rejects_a_row_with_wrong_level_count() -> None:
    payload = _valid_payload()
    payload["values"][0] = [1.0, 2.0, 3.0]
    with pytest.raises(PayloadValidationError, match="expected 5"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_nan() -> None:
    payload = _valid_payload()
    payload["values"][0][2] = float("nan")
    with pytest.raises(PayloadValidationError, match="non-finite"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_inf() -> None:
    payload = _valid_payload()
    payload["values"][0][2] = float("inf")
    with pytest.raises(PayloadValidationError, match="non-finite"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_non_ascending_row() -> None:
    payload = _valid_payload()
    payload["values"][0] = [50.0, 40.0, 60.0, 70.0, 80.0]  # second value dips below the first
    with pytest.raises(PayloadValidationError, match="not ascending"):
        validate_payload(payload, CHALLENGE)


def test_validate_payload_rejects_flat_values() -> None:
    payload = _valid_payload()
    payload["values"] = [1.0] * 96
    with pytest.raises(PayloadValidationError, match="list of lists"):
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


# ---------------------------------------------------------------------------
# Golden fixture: the real Challenge-8 example, confirmed live 2026-10-21
# ---------------------------------------------------------------------------


def test_fixture_passes_validation() -> None:
    """If our validator rejects the Arena's own example, our validator is wrong."""
    fixture = json.loads(
        (FIXTURES_DIR / "arena_public_example_challenge_8.json").read_text(encoding="utf-8")
    )
    validate_payload(fixture, CHALLENGE)  # must not raise


def test_our_payload_matches_fixture_structure() -> None:
    """Reverse test: a payload we build has the same shape as the Arena's example."""
    fixture = json.loads(
        (FIXTURES_DIR / "arena_public_example_challenge_8.json").read_text(encoding="utf-8")
    )
    ours = build_payload(
        CHALLENGE, _ts("2026-08-20T00:00:00+02:00"), _quantiles(len(fixture["values"]))
    )

    assert set(ours.keys()) >= {"challenge_id", "target_start", "values"}
    assert type(ours["challenge_id"]) is type(fixture["challenge_id"])
    assert type(ours["target_start"]) is type(fixture["target_start"])
    assert isinstance(ours["values"], list)
    assert isinstance(fixture["values"], list)
    assert len(ours["values"]) == len(fixture["values"])
    assert all(isinstance(row, list) for row in ours["values"])
    assert all(isinstance(row, list) for row in fixture["values"])
    assert all(len(row) == len(LEVELS) for row in ours["values"])
    assert all(len(row) == len(LEVELS) for row in fixture["values"])
