"""Unit tests for the Arena catalog client.

All tests mock requests.get -- the boundary between our code and the Arena
API -- so retry/backoff logic runs against real implementation code. No
network access.
"""

import copy
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from energy_price_forecast.arena.catalog import (
    ArenaApiError,
    UnsupportedChallengeError,
    canonical_hash,
    get_challenge,
    list_open_challenges,
)

MODULE = "energy_price_forecast.arena.catalog"
SLEEP = f"{MODULE}.time.sleep"

# Structurally faithful to a real GET /api/v1/challenges/2 response (verified
# against the Arena's live OpenAPI schema, 2026-08-19), trimmed to the fields
# this client reads plus a couple of untouched passenger fields so raw/
# canonical_hash have something to prove they preserve.
CHALLENGE_DETAIL: dict[str, Any] = {
    "code": "2",
    "name": "Day-Ahead Prices | Germany-Luxembourg | Point Forecast",
    "reference_timezone": "Europe/Berlin",
    "areas": ["DE_LU"],
    "target_name": "Day-Ahead Prices",
    "accepted_forecast_format": "point",
    "target_period": {
        "type": "calendar_day",
        "timezone": "Europe/Berlin",
        "end_is_exclusive": True,
    },
    "constraints": {"unit": "EUR/MWh", "precision_decimals": 2, "allow_negative": True},
    "submission_window": {
        "open_offset_from_deadline": "-P3D",
        "allow_multiple": True,
        "selection_policy": "latest_before_deadline",
    },
    "example": {
        "payload": {
            "challenge_id": "2",
            "target_start": "2026-01-15T00:00:00+01:00",
            "values": [0.0] * 96,
        },
        "target_start": "2026-01-15T00:00:00+01:00",
    },
    "latest_public_example": {
        "payload": {
            "challenge_id": "2",
            "target_start": "2026-08-19T00:00:00+02:00",
            "values": [100.0 + i for i in range(96)],
        },
        "target_start": "2026-08-19T00:00:00+02:00",
    },
}

OPEN_CHALLENGES: dict[str, Any] = {
    "generated_at": "2026-08-19T12:30:41.467075Z",
    "active_challenges": [
        {
            "challenge_id": "2",
            "challenge_name": "Day-Ahead Prices | Germany-Luxembourg | Point Forecast",
            "reference_timezone": "Europe/Berlin",
            "next_submission_deadline": "2026-08-20T12:00:00+02:00",
            "next_target_start": "2026-08-21T00:00:00+02:00",
        }
    ],
}


def _mock_response(status_code: int, json_body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.side_effect = None
    return response


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_API_KEY", "test-key")
    monkeypatch.setenv("ARENA_API_BASE_URL", "https://arena.example.invalid")


@pytest.fixture
def get_mock() -> Generator[MagicMock, None, None]:
    with patch(f"{MODULE}.requests.get") as p:
        yield p


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_fixture_response_into_challenge_spec(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(200, CHALLENGE_DETAIL)

    spec = get_challenge("2")

    assert spec.challenge_id == "2"
    assert isinstance(spec.challenge_id, str)
    assert spec.name == CHALLENGE_DETAIL["name"]
    assert spec.timezone == "Europe/Berlin"
    assert spec.resolution_minutes == 15  # 96 values in latest_public_example
    assert spec.precision_decimals == 2
    assert spec.allow_negative is True
    assert spec.max_forecast_points is None


def test_resolution_inferred_from_example_when_no_public_example(get_mock: MagicMock) -> None:
    detail = copy.deepcopy(CHALLENGE_DETAIL)
    detail["latest_public_example"] = None
    get_mock.return_value = _mock_response(200, detail)

    spec = get_challenge("2")

    assert spec.resolution_minutes == 15  # falls back to `example` (also 96 values)


@pytest.mark.parametrize(
    ("n_values", "expected_minutes"),
    [
        (96, 15),
        (92, 15),  # spring-forward calendar day: 23h at quarter-hour resolution
        (100, 15),  # fall-back calendar day: 25h at quarter-hour resolution
        (24, 60),
        (23, 60),
        (48, 30),
    ],
)
def test_resolution_inference_handles_dst_shifted_counts(
    get_mock: MagicMock, n_values: int, expected_minutes: int
) -> None:
    detail = copy.deepcopy(CHALLENGE_DETAIL)
    detail["latest_public_example"]["payload"]["values"] = [1.0] * n_values
    get_mock.return_value = _mock_response(200, detail)

    spec = get_challenge("2")

    assert spec.resolution_minutes == expected_minutes


def test_non_calendar_day_target_period_is_rejected(get_mock: MagicMock) -> None:
    detail = copy.deepcopy(CHALLENGE_DETAIL)
    detail["target_period"]["type"] = "rolling_hour"
    get_mock.return_value = _mock_response(200, detail)

    with pytest.raises(UnsupportedChallengeError, match="calendar_day"):
        get_challenge("2")


def test_nested_values_are_rejected_as_unsupported(get_mock: MagicMock) -> None:
    detail = copy.deepcopy(CHALLENGE_DETAIL)
    detail["latest_public_example"]["payload"]["values"] = [[1.0, 2.0]] * 96
    get_mock.return_value = _mock_response(200, detail)

    with pytest.raises(UnsupportedChallengeError, match="quantile/ensemble"):
        get_challenge("2")


def test_list_open_challenges_parses_active_challenges(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(200, OPEN_CHALLENGES)

    challenges = list_open_challenges()

    assert len(challenges) == 1
    assert challenges[0].challenge_id == "2"
    assert challenges[0].next_target_start == "2026-08-21T00:00:00+02:00"
    assert challenges[0].next_submission_deadline == "2026-08-20T12:00:00+02:00"


# ---------------------------------------------------------------------------
# raw / canonical_hash
# ---------------------------------------------------------------------------


def test_raw_holds_the_unmodified_response(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(200, CHALLENGE_DETAIL)

    spec = get_challenge("2")

    assert spec.raw == CHALLENGE_DETAIL


def test_canonical_hash_stable_under_key_reorder() -> None:
    reordered = dict(reversed(list(CHALLENGE_DETAIL.items())))
    assert canonical_hash(CHALLENGE_DETAIL) == canonical_hash(reordered)


def test_canonical_hash_changes_when_a_value_changes() -> None:
    changed = copy.deepcopy(CHALLENGE_DETAIL)
    changed["constraints"]["precision_decimals"] = 3
    assert canonical_hash(CHALLENGE_DETAIL) != canonical_hash(changed)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_4xx_does_not_retry(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(404)

    with pytest.raises(requests.HTTPError):
        get_challenge("2")

    assert get_mock.call_count == 1


def test_5xx_retries_then_succeeds(get_mock: MagicMock) -> None:
    get_mock.side_effect = [
        _mock_response(500),
        _mock_response(500),
        _mock_response(200, CHALLENGE_DETAIL),
    ]

    with patch(SLEEP):
        spec = get_challenge("2")

    assert spec.challenge_id == "2"
    assert get_mock.call_count == 3


def test_5xx_exhausts_retries_and_raises(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(500)

    with patch(SLEEP), pytest.raises(ArenaApiError):
        get_challenge("2")

    assert get_mock.call_count == 4  # 1 initial + 3 retries
