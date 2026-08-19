"""Unit tests for the Arena submit client. No network access.

The dry-run tests prove behavior, not just the return value: a fake
transport that raises on every call stands in for the real one, and the
assertion is that its call counter stays at zero -- not that
result.sent is False, which only checks what the code claims about itself.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.arena.payload import PayloadValidationError, build_payload
from energy_price_forecast.arena.submit import SubmissionResult, submit

MODULE = "energy_price_forecast.arena.submit"

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


def _valid_payload() -> dict[str, Any]:
    return build_payload(CHALLENGE, pd.Timestamp("2026-08-20T00:00:00+02:00"), [1.0] * 96)


class _ExplodingTransport:
    """Fake transport that raises and counts every call -- proof, not trust."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        raise AssertionError("transport must not be called in dry-run")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_API_KEY", "test-key")
    monkeypatch.setenv("ARENA_API_BASE_URL", "https://arena.example.invalid")


# ---------------------------------------------------------------------------
# Dry-run makes zero HTTP calls
# ---------------------------------------------------------------------------


def test_dry_run_never_calls_transport() -> None:
    transport = _ExplodingTransport()

    result = submit(CHALLENGE, _valid_payload(), transport=transport)

    assert transport.call_count == 0
    assert result == SubmissionResult(sent=False, challenge_id="2")


def test_dry_run_is_the_default_without_passing_live() -> None:
    transport = _ExplodingTransport()

    submit(CHALLENGE, _valid_payload(), transport=transport)  # live not passed

    assert transport.call_count == 0


# ---------------------------------------------------------------------------
# live=True calls the real transport exactly once
# ---------------------------------------------------------------------------


def test_live_calls_post_exactly_once_with_correct_path_header_and_body() -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = None
    response.json.return_value = {
        "submission_id": 123,
        "status": "accepted",
        "message": "ok",
        "submitted_at": "2026-08-20T11:00:00+02:00",
    }
    payload = _valid_payload()

    with patch(f"{MODULE}.requests.post", return_value=response) as post_mock:
        result = submit(CHALLENGE, payload, live=True)

    assert post_mock.call_count == 1
    args, kwargs = post_mock.call_args
    assert args[0] == "https://arena.example.invalid/api/v1/submissions"
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
    assert kwargs["json"] == payload

    assert result == SubmissionResult(
        sent=True,
        challenge_id="2",
        submission_id=123,
        status="accepted",
        message="ok",
        submitted_at="2026-08-20T11:00:00+02:00",
    )


def test_live_via_injected_fake_transport_is_called_once() -> None:
    calls: list[dict[str, Any]] = []

    def fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {"submission_id": 1, "status": "accepted", "message": "ok", "submitted_at": "x"}

    payload = _valid_payload()
    result = submit(CHALLENGE, payload, live=True, transport=fake_transport)

    assert calls == [payload]
    assert result.sent is True


# ---------------------------------------------------------------------------
# Invalid payload is rejected before any transport is touched
# ---------------------------------------------------------------------------


def test_invalid_payload_rejected_before_transport_dry_run() -> None:
    transport = _ExplodingTransport()
    bad_payload = _valid_payload()
    bad_payload["values"] = bad_payload["values"][:-1]  # wrong count

    with pytest.raises(PayloadValidationError):
        submit(CHALLENGE, bad_payload, transport=transport)

    assert transport.call_count == 0


def test_invalid_payload_rejected_before_transport_live() -> None:
    transport = _ExplodingTransport()
    bad_payload = _valid_payload()
    bad_payload["values"][0] = float("nan")

    with pytest.raises(PayloadValidationError):
        submit(CHALLENGE, bad_payload, live=True, transport=transport)

    assert transport.call_count == 0
