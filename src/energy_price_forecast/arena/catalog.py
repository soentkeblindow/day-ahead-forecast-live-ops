"""Typed client for the Energy Arena's read-only challenge endpoints.

Reads timezone, precision, and format constraints live from the API instead
of hardcoding them -- the Arena is the source of truth for its own challenge
format (see docs/sprint6_step6_2_spec.md, §6.1).

The real API's schema (verified against its OpenAPI document, 2026-08-19)
has no explicit "resolution" or "max_forecast_points" field:
resolution_minutes is inferred from the value count of the challenge's
example payload (see _infer_resolution_minutes), and max_forecast_points
stays None until the API exposes an actual limit.

challenge_id / deadline / target_start are deliberately absent from
ChallengeSpec: challenge_id is a string (not int) per the API's own schema,
and deadline/target_start exist only on the *list* endpoint
(GET /challenges/open) -- not on the per-challenge detail endpoint this
client's get_challenge() uses. Computing "today's" target date is the
caller's responsibility (the availability audit already derives its target
date D independently; see ops/windows.py), not something a challenge's
static format description should carry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_TIMEOUT_S = 10.0
_MAX_RETRIES = 3
_SUPPORTED_TARGET_PERIOD_TYPE = "calendar_day"  # Point challenges only; see spec §12.


class ArenaApiError(Exception):
    """Raised when an Arena API call fails after exhausting retries."""


class UnsupportedChallengeError(Exception):
    """Raised for challenge shapes outside 6.2's scope (see spec §12)."""


def _api_key() -> str:
    key = os.getenv("ARENA_API_KEY")
    if not key:
        raise RuntimeError("ARENA_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def _base_url() -> str:
    url = os.getenv("ARENA_API_BASE_URL")
    if not url:
        raise RuntimeError(
            "ARENA_API_BASE_URL is not set. Copy .env.example to .env and add the Arena base URL."
        )
    return url.rstrip("/")


def _call_with_retry[T](fn: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
    """Retry fn() on 5xx responses and timeouts with exponential backoff (1s, 2s, 4s).

    4xx responses (including 429) propagate immediately -- retrying a bad
    request or a bad key does not make it correct (spec §6.1).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is None or status < 500:
                raise
            if attempt >= _MAX_RETRIES:
                raise ArenaApiError(
                    f"Arena API failed after {attempt + 1} attempts: {exc}"
                ) from exc
            wait = 2.0**attempt
            logger.warning(
                "Arena API 5xx -- retrying in %.0fs (retry %d/%d)", wait, attempt + 1, _MAX_RETRIES
            )
            attempt += 1
            sleep(wait)
        except requests.Timeout as exc:
            if attempt >= _MAX_RETRIES:
                raise ArenaApiError(
                    f"Arena API timed out after {attempt + 1} attempts: {exc}"
                ) from exc
            wait = 2.0**attempt
            logger.warning(
                "Arena API timeout -- retrying in %.0fs (retry %d/%d)",
                wait,
                attempt + 1,
                _MAX_RETRIES,
            )
            attempt += 1
            sleep(wait)


def _get(path: str) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    headers = {"X-API-Key": _api_key()}

    def do_request() -> dict[str, Any]:
        response = requests.get(url, headers=headers, timeout=_TIMEOUT_S)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    return _call_with_retry(do_request)


@dataclass(frozen=True)
class ChallengeSpec:
    """One Arena challenge's submission format and constraints.

    Deliberately excludes deadline/target_start -- see the module docstring.
    resolution_minutes is not returned by the API; it is inferred by
    _infer_resolution_minutes and only supports calendar_day target periods
    (Point challenges, spec §12).
    """

    challenge_id: str
    name: str
    timezone: str
    resolution_minutes: int
    precision_decimals: int
    allow_negative: bool
    max_forecast_points: int | None
    raw: dict[str, Any]


def _infer_resolution_minutes(detail: dict[str, Any]) -> int:
    """Infer the challenge's submission-grid resolution from its example payload.

    The API exposes no "resolution" field (verified against its OpenAPI
    schema). For a calendar_day target period, the value count of the latest
    public example (or the static example, if no submission exists yet)
    implies the resolution -- but DST days shift that count by one hour's
    worth of periods, so a plain 24h/count division is wrong twice a year.
    This instead checks each plausible resolution against both the normal and
    DST-shifted value counts it would produce for a calendar day.
    """
    target_period = detail["target_period"]
    if target_period["type"] != _SUPPORTED_TARGET_PERIOD_TYPE:
        raise UnsupportedChallengeError(
            f"target_period.type {target_period['type']!r} is not supported in 6.2 "
            "(only calendar_day / Point challenges; see spec §12)"
        )

    example = detail.get("latest_public_example") or detail["example"]
    values = example["payload"]["values"]
    if not values or isinstance(values[0], list):
        raise UnsupportedChallengeError(
            "quantile/ensemble challenges (nested values) are not supported in 6.2 (spec §12)"
        )
    n_values = len(values)

    for candidate_minutes in (15, 30, 60):
        normal_count = 24 * 60 // candidate_minutes
        dst_shift = 60 // candidate_minutes
        if n_values in (normal_count, normal_count - dst_shift, normal_count + dst_shift):
            return candidate_minutes

    raise UnsupportedChallengeError(
        f"could not infer resolution from {n_values} values in the example payload"
    )


def _parse_challenge(detail: dict[str, Any]) -> ChallengeSpec:
    constraints = detail["constraints"]
    return ChallengeSpec(
        challenge_id=detail["code"],
        name=detail["name"],
        timezone=detail["reference_timezone"],
        resolution_minutes=_infer_resolution_minutes(detail),
        precision_decimals=constraints["precision_decimals"],
        allow_negative=constraints["allow_negative"],
        max_forecast_points=None,  # not exposed by the API; see module docstring
        raw=detail,
    )


def get_challenge(challenge_id: str) -> ChallengeSpec:
    """GET /api/v1/challenges/{challenge_id}."""
    detail = _get(f"/api/v1/challenges/{challenge_id}")
    return _parse_challenge(detail)


@dataclass(frozen=True)
class OpenChallengeSummary:
    """One entry from GET /challenges/open.

    Deliberately not a ChallengeSpec: the list endpoint carries the *next*
    schedule instance (next_target_start / next_submission_deadline) but none
    of the format/constraints fields ChallengeSpec needs -- those live only on
    the per-challenge detail endpoint (see get_challenge).
    """

    challenge_id: str
    name: str
    timezone: str
    next_target_start: str
    next_submission_deadline: str
    raw: dict[str, Any]


def list_open_challenges() -> list[OpenChallengeSummary]:
    """GET /api/v1/challenges/open."""
    data = _get("/api/v1/challenges/open")
    return [
        OpenChallengeSummary(
            challenge_id=c["challenge_id"],
            name=c["challenge_name"],
            timezone=c["reference_timezone"],
            next_target_start=c["next_target_start"],
            next_submission_deadline=c["next_submission_deadline"],
            raw=c,
        )
        for c in data["active_challenges"]
    ]


def canonical_hash(raw: dict[str, Any]) -> str:
    """SHA256 of raw, canonicalized (sorted keys, no whitespace).

    Changes whenever the Arena changes any field of a challenge response,
    even fields we don't read -- an early-warning signal for unanticipated
    format changes (see docs/sprint6_step6_2_spec.md, §5.6).
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
