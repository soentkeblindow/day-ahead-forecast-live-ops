"""Unit tests for the Arena submit client. No network access.

The dry-run tests prove behavior, not just the return value: a fake
transport that raises on every call stands in for the real one, and the
assertion is that its call counter stays at zero -- not that
result.sent is False, which only checks what the code claims about itself.
"""

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

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
        result = submit(
            CHALLENGE,
            payload,
            live=True,
            clock=lambda: pd.Timestamp("2026-08-20T11:00:01+00:00"),
            confirm=lambda submission_id: True,
        )

    assert post_mock.call_count == 1
    args, kwargs = post_mock.call_args
    assert args[0] == "https://arena.example.invalid/api/v1/submissions"
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
    assert kwargs["json"] == payload

    assert result == SubmissionResult(
        sent=True,
        challenge_id="2",
        accepted=True,
        submission_id=123,
        status="accepted",
        message="ok",
        submitted_at="2026-08-20T11:00:00+02:00",
        http_status=200,
        response_received_utc="2026-08-20T11:00:01+00:00",
        confirmed_via_query=True,
    )


def test_live_via_injected_fake_transport_is_called_once() -> None:
    calls: list[dict[str, Any]] = []

    def fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {"submission_id": 1, "status": "accepted", "message": "ok", "submitted_at": "x"}

    payload = _valid_payload()
    result = submit(
        CHALLENGE, payload, live=True, transport=fake_transport, confirm=lambda submission_id: True
    )

    assert calls == [payload]
    assert result.sent is True
    assert result.accepted is True


# ---------------------------------------------------------------------------
# Response evaluation (spec 6.7.3 section 5.5)
# ---------------------------------------------------------------------------


def _http_error(status_code: int, json_body: dict[str, Any] | None = None) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = "" if json_body is None else json.dumps(json_body)
    error = requests.HTTPError(f"{status_code} error", response=response)
    return error


def test_a_422_rejection_is_reported_as_rejected_not_raised() -> None:
    detail_body = {
        "detail": [{"loc": ["body", "target_start"], "msg": "target_start is in the past"}]
    }

    def raising_transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise _http_error(422, detail_body)

    result = submit(
        CHALLENGE,
        _valid_payload(),
        live=True,
        transport=raising_transport,
        clock=lambda: pd.Timestamp("2026-08-20T11:00:01+00:00"),
    )

    assert result.sent is True
    assert result.accepted is False
    assert result.error_kind == "rejected"
    assert result.http_status == 422
    assert "target_start is in the past" in (result.message or "")
    assert result.response_received_utc == "2026-08-20T11:00:01+00:00"


def test_a_5xx_is_reported_as_transport_error_not_rejected() -> None:
    def raising_transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise _http_error(503)

    result = submit(CHALLENGE, _valid_payload(), live=True, transport=raising_transport)

    assert result.sent is True
    assert result.accepted is False
    assert result.error_kind == "transport_error"
    assert result.http_status == 503


def test_a_timeout_is_reported_as_transport_error() -> None:
    def raising_transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise requests.Timeout("connection timed out")

    result = submit(CHALLENGE, _valid_payload(), live=True, transport=raising_transport)

    assert result.sent is True
    assert result.accepted is False
    assert result.error_kind == "transport_error"
    assert result.http_status is None
    assert "connection timed out" in (result.message or "")


def _recording_confirm(calls: list[int]) -> Callable[[int], bool]:
    """A bool-returning confirm fake that also records its calls -- a plain
    named function instead of a `calls.append(x) or True` lambda one-liner,
    since list.append() returns None and mypy's func-returns-value check
    flags using that return value directly (docs/feedback_typechecking.md's
    guessed-# type: ignore pattern; not verifiable on this machine, so
    avoided structurally instead of suppressed)."""

    def confirm(submission_id: int) -> bool:
        calls.append(submission_id)
        return True

    return confirm


def test_a_transport_error_does_not_call_confirm() -> None:
    calls: list[int] = []

    def raising_transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise requests.Timeout("boom")

    submit(
        CHALLENGE,
        _valid_payload(),
        live=True,
        transport=raising_transport,
        confirm=_recording_confirm(calls),
    )

    assert calls == []


def test_a_response_without_a_submission_id_never_calls_confirm() -> None:
    calls: list[int] = []

    def fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "accepted", "message": "ok", "submitted_at": "x"}

    result = submit(
        CHALLENGE,
        _valid_payload(),
        live=True,
        transport=fake_transport,
        confirm=_recording_confirm(calls),
    )

    assert calls == []
    assert result.confirmed_via_query is None


def test_a_failed_confirmation_query_does_not_flip_accepted_to_false() -> None:
    def fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
        return {"submission_id": 7, "status": "accepted", "message": "ok", "submitted_at": "x"}

    result = submit(
        CHALLENGE, _valid_payload(), live=True, transport=fake_transport, confirm=lambda _id: False
    )

    assert result.accepted is True
    assert result.confirmed_via_query is False


# ---------------------------------------------------------------------------
# _confirm_via_query -- the real GET, mocked at the requests boundary
# ---------------------------------------------------------------------------


def test_confirm_via_query_uses_bearer_auth_and_the_submission_id_path() -> None:
    from energy_price_forecast.arena.submit import _confirm_via_query

    response = MagicMock()
    response.raise_for_status.side_effect = None

    with patch(f"{MODULE}.requests.get", return_value=response) as get_mock:
        confirmed = _confirm_via_query(123)

    assert confirmed is True
    args, kwargs = get_mock.call_args
    assert args[0] == "https://arena.example.invalid/api/v1/auth/me/submissions/123"
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}


def test_confirm_via_query_returns_false_and_does_not_raise_on_failure() -> None:
    from energy_price_forecast.arena.submit import _confirm_via_query

    with patch(f"{MODULE}.requests.get", side_effect=requests.ConnectionError("down")):
        confirmed = _confirm_via_query(123)

    assert confirmed is False


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
