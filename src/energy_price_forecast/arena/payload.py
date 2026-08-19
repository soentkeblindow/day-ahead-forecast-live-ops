"""Build and validate Point-challenge submission payloads for the Energy Arena.

Only the "dense payload" shape from the API's ForecastCreate schema is built
here: {"challenge_id", "target_start", "values"}. target_start/target_end are
not read off ChallengeSpec (it doesn't carry them -- see arena/catalog.py's
module docstring); the caller supplies target_start explicitly, e.g. the
availability audit's own D, or -- once 6.5 exists -- a submission script's
notion of tomorrow.

The validator is this module's real value: it runs locally, before every
submission, and is cheaper than a rejected one (spec §6.2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from energy_price_forecast.arena.catalog import ChallengeSpec


class PayloadValidationError(Exception):
    """Raised by validate_payload for a payload that the Arena would reject."""


def expected_value_count(challenge: ChallengeSpec, target_start: pd.Timestamp) -> int:
    """Number of values a calendar-day submission for target_start needs.

    target_end is not a separate input: ChallengeSpec only supports
    calendar_day target periods (catalog.py rejects anything else), so
    target_end is always target_start plus one calendar day. Using
    DateOffset(days=1) on a tz-aware target_start -- rather than
    Timedelta(hours=24) -- keeps this DST-correct, but only once target_start
    carries the challenge's named zone rather than a fixed numeric offset:
    ISO-8601 strings (e.g. from validate_payload, which parses target_start
    out of a JSON payload) carry only a UTC offset like "+01:00", never an
    IANA zone name, so pd.Timestamp(...) parses them as a *fixed*-offset
    tzinfo. DateOffset(days=1) on a fixed offset does not re-resolve the
    offset for the next day -- it silently produces a 24h day even across a
    DST transition. tz_convert(challenge.timezone) below re-tags the same
    instant with the named zone so the offset re-resolves correctly.
    """
    if target_start.tzinfo is None:
        raise ValueError("target_start must be tz-aware (Europe/Berlin, not naive)")
    target_start_zoned = target_start.tz_convert(challenge.timezone)
    target_end = target_start_zoned + pd.DateOffset(days=1)
    elapsed = target_end - target_start_zoned
    return int(elapsed / pd.Timedelta(minutes=challenge.resolution_minutes))


def build_payload(
    challenge: ChallengeSpec, target_start: pd.Timestamp, values: Sequence[float]
) -> dict[str, Any]:
    """Build {"challenge_id", "target_start", "values"} for challenge.

    target_start must be tz-aware in the challenge's own timezone. Serializing
    via .isoformat() on that tz-aware Timestamp yields the local-offset form
    (e.g. "2026-08-20T00:00:00+02:00") the Arena expects -- converting to UTC
    first and serializing that is a known mistake (it silently changes which
    wall-clock day/hour is being submitted for) and must not happen here.

    Values are rounded to challenge.precision_decimals; negative values are
    kept as-is (the challenge's allow_negative constraint is a Yes/No fact
    about the target quantity, not something this builder enforces).
    """
    if target_start.tzinfo is None:
        raise ValueError("target_start must be tz-aware (Europe/Berlin, not naive)")
    return {
        "challenge_id": challenge.challenge_id,
        "target_start": target_start.isoformat(),
        "values": [round(v, challenge.precision_decimals) for v in values],
    }


def validate_payload(payload: dict[str, Any], challenge: ChallengeSpec) -> None:
    """Raise PayloadValidationError if the Arena would reject payload for challenge.

    Checks (spec §6.2): value count, non-finite values (NaN/inf), target_start
    format/offset, nested values (Point challenges need a flat list), and
    max_forecast_points if the challenge declares one. precision_decimals and
    allow_negative are not checked here -- build_payload already rounds, and
    the spec does not ask this validator to reject negative values.
    """
    values = payload.get("values")
    if not isinstance(values, list):
        raise PayloadValidationError(f"payload['values'] must be a list, got {type(values)!r}")
    if any(isinstance(v, list | tuple) for v in values):
        raise PayloadValidationError(
            "payload['values'] must be a flat list; Point challenges do not accept "
            "nested values (that shape is for quantile/ensemble challenges)"
        )
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
        raise PayloadValidationError("payload['values'] contains a non-finite value (NaN/inf)")

    target_start_raw = payload.get("target_start")
    if not isinstance(target_start_raw, str) or target_start_raw.endswith("Z"):
        raise PayloadValidationError(
            f"payload['target_start'] must be an ISO-8601 string with a local offset "
            f"(not UTC 'Z' form), got {target_start_raw!r}"
        )
    try:
        target_start = pd.Timestamp(target_start_raw)
    except ValueError as exc:
        raise PayloadValidationError(
            f"payload['target_start'] is not a valid timestamp: {target_start_raw!r}"
        ) from exc
    if target_start.tzinfo is None:
        raise PayloadValidationError(
            f"payload['target_start'] has no offset (naive timestamp): {target_start_raw!r}"
        )

    expected = expected_value_count(challenge, target_start)
    if len(values) != expected:
        raise PayloadValidationError(
            f"payload['values'] has {len(values)} value(s), expected {expected} "
            f"for a {challenge.resolution_minutes}-minute grid on {target_start.date()}"
        )

    if challenge.max_forecast_points is not None and len(values) > challenge.max_forecast_points:
        raise PayloadValidationError(
            f"payload['values'] has {len(values)} value(s), "
            f"exceeding max_forecast_points={challenge.max_forecast_points}"
        )
