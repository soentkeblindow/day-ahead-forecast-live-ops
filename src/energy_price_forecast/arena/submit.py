"""Submit validated payloads to the Energy Arena -- dry-run by default.

live defaults to False: nothing not-doing is the safe state, and going live
must be an active choice by the caller (spec §6.3). No caller in this repo
passes live=True; that only happens once 6.5's submission script exists.

Unlike arena/catalog.py, submit() does not retry on 5xx/timeout -- retrying a
POST whose response was lost to a network error risks a duplicate
submission, and the API already de-duplicates on (challenge_id, target_start)
via submission_window.selection_policy server-side. A failed live submission
should surface immediately, not be silently retried.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.arena.payload import validate_payload

logger = logging.getLogger(__name__)

load_dotenv()

_TIMEOUT_S = 10.0

Transport = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class SubmissionResult:
    """Outcome of a submit() call.

    sent=False (dry-run): only challenge_id is meaningful; the rest stay None
    because no request was made. sent=True: the remaining fields come from
    the API's ForecastCreateResponse.
    """

    sent: bool
    challenge_id: str
    submission_id: int | None = None
    status: str | None = None
    message: str | None = None
    submitted_at: str | None = None


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


def submit(
    challenge: ChallengeSpec,
    payload: dict[str, Any],
    *,
    live: bool = False,
    transport: Transport = _post,
) -> SubmissionResult:
    """Validate payload against challenge, then submit it -- or not.

    Validation always runs first and raises PayloadValidationError before
    transport is touched, live or not. With live=False (the default),
    transport is never called: this is a structural guarantee proven in
    tests/test_arena_submit.py by asserting a call counter on a fake
    transport, not by trusting the sent=False on the return value.
    """
    validate_payload(payload, challenge)

    if not live:
        logger.info(
            "Dry-run: payload validated for challenge %s, not sent (pass live=True to submit)",
            challenge.challenge_id,
        )
        return SubmissionResult(sent=False, challenge_id=challenge.challenge_id)

    response = transport(payload)
    return SubmissionResult(
        sent=True,
        challenge_id=challenge.challenge_id,
        submission_id=response.get("submission_id"),
        status=response.get("status"),
        message=response.get("message"),
        submitted_at=response.get("submitted_at"),
    )
