"""Submit validated payloads to the Energy Arena -- dry-run by default.

live defaults to False: nothing not-doing is the safe state, and going live
must be an active choice by the caller (spec 6.7.2 section 6.3). The actual
switch lives in arena/config.py::is_live_enabled() (spec 6.7.3 section 2.3).

Unlike arena/catalog.py, submit() does not retry on 5xx/timeout -- retrying a
POST whose response was lost to a network error risks a duplicate
submission, and the API already de-duplicates on (challenge_id, target_start)
via submission_window.selection_policy server-side. A failed live submission
should surface immediately, not be silently retried (spec 6.7.3 section 5.5).

Response evaluation (spec 6.7.3 section 5.5): a live POST either succeeds
(2xx, ForecastCreateResponse), is rejected by the platform (4xx, e.g. the
documented 422 HTTPValidationError), or fails as a transport error (timeout,
connection error, or 5xx -- grouped together since none of them carry a
platform message and none get retried within this run; the next scheduled
slot is the retry).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.arena.payload import validate_payload

logger = logging.getLogger(__name__)

load_dotenv()

_TIMEOUT_S = 10.0

Transport = Callable[[dict[str, Any]], dict[str, Any]]
Clock = Callable[[], pd.Timestamp]
ConfirmQuery = Callable[[int], bool]


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


@dataclass(frozen=True)
class SubmissionResult:
    """Outcome of a submit() call (spec 6.7.3 section 5.5).

    sent=False (dry-run): only challenge_id is meaningful; the rest stay at
    their defaults since no request was made.

    sent=True: a live POST was actually issued.
    - accepted=True: the platform's ForecastCreateResponse fields are
      populated (submission_id, status, message, submitted_at).
    - accepted=False, error_kind="rejected": the platform rejected the
      payload (4xx, e.g. 422) -- ``message`` carries the platform's own
      error text, extracted best-effort from the response body.
    - accepted=False, error_kind="transport_error": timeout, connection
      error, or 5xx -- there is no platform message, ``message`` carries
      the exception text instead.

    ``confirmed_via_query`` is only ever non-None after an accepted
    submission whose response carried a submission_id; a failed
    confirmation query never flips ``accepted`` back to False (spec 5.5:
    a failed query is a warning, not a red run).
    """

    sent: bool
    challenge_id: str
    accepted: bool = False
    submission_id: int | None = None
    status: str | None = None
    message: str | None = None
    submitted_at: str | None = None
    http_status: int | None = None
    error_kind: str | None = None  # "rejected" | "transport_error" | None
    response_received_utc: str | None = None
    confirmed_via_query: bool | None = None


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.getenv("ARENA_API_BASE_URL")
    api_key = os.getenv("ARENA_API_KEY")
    if not base_url:
        raise RuntimeError(
            "ARENA_API_BASE_URL is not set. Copy .env.example to .env and add the Arena base URL."
        )
    if not api_key:
        raise RuntimeError("ARENA_API_KEY is not set. Copy .env.example to .env and add your key.")

    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/submissions",
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _error_message(response: requests.Response | None, fallback: str) -> str:
    """Best-effort extraction of the platform's own error text from a 4xx
    body (spec 5.5's documented shape: ``HTTPValidationError`` ->
    ``detail: [ValidationError]``). This is for a human reading the
    protocol log, not a strict parser -- any other shape falls back to the
    raw body or the exception text.
    """
    if response is None:
        return fallback
    try:
        body = response.json()
    except ValueError:
        return response.text or fallback
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, list) and detail:
            msgs = [str(item.get("msg", item)) for item in detail if isinstance(item, dict)]
            if msgs:
                return "; ".join(msgs)
        if isinstance(detail, str):
            return detail
        message = body.get("message")
        if isinstance(message, str):
            return message
    return str(body)


def _confirm_via_query(submission_id: int) -> bool:
    """Query the Arena's own-submissions endpoint once after an accepted
    POST (spec 5.5) -- GET /api/v1/auth/me/submissions/{submission_id}, the
    most specific of the three query endpoints found in section 3.1 (no
    list-filtering needed). Uses HTTPBearer auth, not the X-API-Key header
    _post uses -- two auth schemes on the same API, confirmed in section
    3.1. Owner decision 2026-09-17: try the same ARENA_API_KEY as the
    bearer token (the spec names no separate secret) -- correctness only
    provable once the smoke test (section 2.5) actually calls this live.

    Never raises -- a failed query is a warning, not a red run (spec 5.5).
    """
    base_url = os.getenv("ARENA_API_BASE_URL")
    api_key = os.getenv("ARENA_API_KEY")
    if not base_url or not api_key:
        logger.warning(
            "cannot confirm submission %s via query: ARENA_API_BASE_URL/ARENA_API_KEY not set",
            submission_id,
        )
        return False
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/v1/auth/me/submissions/{submission_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("confirmation query for submission %s failed: %s", submission_id, exc)
        return False
    return True


def submit(
    challenge: ChallengeSpec,
    payload: dict[str, Any],
    *,
    live: bool = False,
    transport: Transport = _post,
    clock: Clock = _utcnow,
    confirm: ConfirmQuery = _confirm_via_query,
) -> SubmissionResult:
    """Validate payload against challenge, then submit it -- or not.

    Validation always runs first and raises PayloadValidationError before
    transport is touched, live or not. With live=False (the default),
    transport is never called: this is a structural guarantee proven in
    tests/test_arena_submit.py by asserting a call counter on a fake
    transport, not by trusting the sent=False on the return value.

    ``clock``/``confirm`` are injectable for the same testability reason
    ``transport`` already is (no real time, no real network in tests).
    """
    validate_payload(payload, challenge)

    if not live:
        logger.info(
            "Dry-run: payload validated for challenge %s, not sent (pass live=True to submit)",
            challenge.challenge_id,
        )
        return SubmissionResult(sent=False, challenge_id=challenge.challenge_id)

    try:
        response = transport(payload)
    except requests.HTTPError as exc:
        received_at = clock().isoformat()
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code is not None and 400 <= status_code < 500:
            message = _error_message(exc.response, str(exc))
            logger.warning(
                "Arena rejected submission for challenge %s (HTTP %s): %s",
                challenge.challenge_id,
                status_code,
                message,
            )
            return SubmissionResult(
                sent=True,
                challenge_id=challenge.challenge_id,
                accepted=False,
                error_kind="rejected",
                http_status=status_code,
                message=message,
                response_received_utc=received_at,
            )
        logger.warning(
            "Arena submission transport error for challenge %s (HTTP %s): %s",
            challenge.challenge_id,
            status_code,
            exc,
        )
        return SubmissionResult(
            sent=True,
            challenge_id=challenge.challenge_id,
            accepted=False,
            error_kind="transport_error",
            http_status=status_code,
            message=str(exc),
            response_received_utc=received_at,
        )
    except requests.RequestException as exc:
        received_at = clock().isoformat()
        logger.warning(
            "Arena submission transport error for challenge %s: %s", challenge.challenge_id, exc
        )
        return SubmissionResult(
            sent=True,
            challenge_id=challenge.challenge_id,
            accepted=False,
            error_kind="transport_error",
            message=str(exc),
            response_received_utc=received_at,
        )

    received_at = clock().isoformat()
    submission_id = response.get("submission_id")
    confirmed_via_query: bool | None = None
    if submission_id is not None:
        confirmed_via_query = confirm(submission_id)

    return SubmissionResult(
        sent=True,
        challenge_id=challenge.challenge_id,
        accepted=True,
        submission_id=submission_id,
        status=response.get("status"),
        message=response.get("message"),
        submitted_at=response.get("submitted_at"),
        http_status=200,
        response_received_utc=received_at,
        confirmed_via_query=confirmed_via_query,
    )
