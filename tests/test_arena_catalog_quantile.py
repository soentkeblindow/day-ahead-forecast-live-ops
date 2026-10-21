"""Unit tests for the Arena catalog client's quantile-challenge sibling
path (Sprint 7.1 Teil B -> Probephase). All tests mock requests.get -- no
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
    get_quantile_challenge,
)

MODULE = "energy_price_forecast.arena.catalog"
SLEEP = f"{MODULE}.time.sleep"

# Structurally faithful to the real GET /api/v1/challenges/8 response
# (verified against the live Arena API, 2026-10-21), trimmed to the fields
# this client reads.
CHALLENGE_8_DETAIL: dict[str, Any] = {
    "code": "8",
    "name": "Day-Ahead Prices | Germany-Luxembourg | Quantile Forecast (5 quantiles)",
    "reference_timezone": "Europe/Berlin",
    "areas": ["DE_LU"],
    "target_name": "Day-Ahead Prices",
    "accepted_forecast_format": "[q2.5, q25, q50, q75, q97.5]",
    "probabilistic_quantiles": [0.025, 0.25, 0.5, 0.75, 0.975],
    "target_period": {
        "type": "calendar_day",
        "timezone": "Europe/Berlin",
        "end_is_exclusive": True,
    },
    "constraints": {"unit": "EUR/MWh", "precision_decimals": 2, "allow_negative": True},
    "submission_window": {
        "open_offset_from_deadline": "-P7D",
        "allow_multiple": True,
        "selection_policy": "latest_before_deadline",
    },
    "example": {
        "payload": {
            "challenge_id": "8",
            "target_start": "2026-01-15T00:00:00+01:00",
            "values": [[0.0] * 5] * 96,
        },
        "target_start": "2026-01-15T00:00:00+01:00",
    },
    "latest_public_example": {
        "payload": {
            "challenge_id": "8",
            "target_start": "2026-10-22T00:00:00+02:00",
            "values": [[100.0 + i, 105.0 + i, 110.0 + i, 115.0 + i, 120.0 + i] for i in range(96)],
        },
        "target_start": "2026-10-22T00:00:00+02:00",
    },
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


def test_parses_fixture_response_into_quantile_challenge_spec(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(200, CHALLENGE_8_DETAIL)

    spec = get_quantile_challenge("8")

    assert spec.challenge_id == "8"
    assert spec.name == CHALLENGE_8_DETAIL["name"]
    assert spec.timezone == "Europe/Berlin"
    assert spec.resolution_minutes == 15  # 96 rows in latest_public_example
    assert spec.quantile_levels == (0.025, 0.25, 0.5, 0.75, 0.975)
    assert spec.precision_decimals == 2
    assert spec.allow_negative is True
    assert spec.max_forecast_points is None


@pytest.mark.parametrize(
    ("n_rows", "expected_minutes"),
    [
        (96, 15),
        (92, 15),  # spring-forward calendar day
        (100, 15),  # fall-back calendar day
    ],
)
def test_resolution_inference_handles_dst_shifted_row_counts(
    get_mock: MagicMock, n_rows: int, expected_minutes: int
) -> None:
    detail = copy.deepcopy(CHALLENGE_8_DETAIL)
    detail["latest_public_example"]["payload"]["values"] = [[1.0] * 5] * n_rows
    get_mock.return_value = _mock_response(200, detail)

    spec = get_quantile_challenge("8")

    assert spec.resolution_minutes == expected_minutes


def test_flat_values_are_rejected_not_a_quantile_shape(get_mock: MagicMock) -> None:
    detail = copy.deepcopy(CHALLENGE_8_DETAIL)
    detail["latest_public_example"]["payload"]["values"] = [1.0] * 96
    get_mock.return_value = _mock_response(200, detail)

    with pytest.raises(UnsupportedChallengeError, match="nested"):
        get_quantile_challenge("8")


def test_missing_probabilistic_quantiles_field_is_rejected(get_mock: MagicMock) -> None:
    detail = copy.deepcopy(CHALLENGE_8_DETAIL)
    del detail["probabilistic_quantiles"]
    get_mock.return_value = _mock_response(200, detail)

    with pytest.raises(UnsupportedChallengeError, match="probabilistic_quantiles"):
        get_quantile_challenge("8")


def test_4xx_does_not_retry(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(404)

    with pytest.raises(requests.HTTPError):
        get_quantile_challenge("8")

    assert get_mock.call_count == 1


def test_5xx_retries_then_succeeds(get_mock: MagicMock) -> None:
    get_mock.side_effect = [
        _mock_response(500),
        _mock_response(500),
        _mock_response(200, CHALLENGE_8_DETAIL),
    ]

    with patch(SLEEP):
        spec = get_quantile_challenge("8")

    assert spec.challenge_id == "8"
    assert get_mock.call_count == 3


def test_5xx_exhausts_retries_and_raises(get_mock: MagicMock) -> None:
    get_mock.return_value = _mock_response(500)

    with patch(SLEEP), pytest.raises(ArenaApiError):
        get_quantile_challenge("8")

    assert get_mock.call_count == 4  # 1 initial + 3 retries
