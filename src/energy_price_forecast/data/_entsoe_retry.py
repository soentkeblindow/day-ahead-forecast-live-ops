"""Retry utilities for the ENTSO-E API client."""

import logging
import time
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

_MAX_TRANSIENT_RETRIES = 3  # 3 retries → 4 total calls; waits: 1 s, 2 s, 4 s
_MAX_RATE_LIMIT_RETRIES = 9  # 9 retries → 10 total calls; fixed 60 s wait each
_RATE_LIMIT_WAIT = 60.0


class EntsoeFetchError(Exception):
    """Raised when all transient retry attempts for an ENTSO-E API call are exhausted."""


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _is_rate_limit(exc: Exception) -> bool:
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code == 429
    )


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(), retrying transient errors and rate limits per ENTSO-E API policy.

    Transient errors (5xx, ConnectionError, Timeout): up to 3 retries with
    exponential backoff (1 s, 2 s, 4 s); raises EntsoeFetchError on exhaustion.
    Rate limits (HTTP 429): up to 9 retries with a fixed 60 s wait each.
    Non-retryable client errors (4xx except 429) propagate immediately.

    Args:
        fn: Zero-argument callable wrapping the API call.
        sleep: Injectable sleep function; pass a mock in tests to avoid real waits.
    """
    transient_retries = 0
    rate_limit_retries = 0

    while True:
        try:
            return fn()
        except Exception as exc:
            if _is_rate_limit(exc):
                if rate_limit_retries >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                logger.warning(
                    "HTTP 429 rate limit — waiting %.0fs (retry %d/%d)",
                    _RATE_LIMIT_WAIT,
                    rate_limit_retries + 1,
                    _MAX_RATE_LIMIT_RETRIES,
                )
                rate_limit_retries += 1
                sleep(_RATE_LIMIT_WAIT)
            elif _is_transient(exc):
                if transient_retries >= _MAX_TRANSIENT_RETRIES:
                    raise EntsoeFetchError(
                        f"ENTSO-E API failed after {transient_retries + 1} attempts: {exc}"
                    ) from exc
                wait = 2.0**transient_retries  # 1.0 s, 2.0 s, 4.0 s
                logger.warning(
                    "Transient error — retrying in %.0fs (retry %d/%d): %s",
                    wait,
                    transient_retries + 1,
                    _MAX_TRANSIENT_RETRIES,
                    exc,
                )
                transient_retries += 1
                sleep(wait)
            else:
                raise
