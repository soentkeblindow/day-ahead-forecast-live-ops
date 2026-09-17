"""Static Arena settings, in one place (spec 6.7.3, section 2.3).

The actual go-live switch is deliberately *not* a value in this file. It is
the GitHub repository variable ``ARENA_LIVE``, passed into the submission
job's environment by ``.github/workflows/submit.yml``. That split is the
point: the switch is also the kill switch, and it must be flippable from the
GitHub UI without a commit or a deploy. Set the repository variable
``ARENA_LIVE`` to the literal ``true`` to submit live; unset it, leave it
empty, or anything else keeps the job in dry-run mode -- fail-safe by design.

``ARENA_CHALLENGE_ID`` moved here from scripts/run_daily_submission.py
(unchanged value, "2" -- Day-Ahead Prices | Germany-Luxembourg | Point
Forecast, matches ops/availability_audit.py's own constant).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# "Day-Ahead Prices | Germany-Luxembourg | Point Forecast" -- matches
# ops/availability_audit.py's own ARENA_CHALLENGE_ID (spec section 5.4 there).
ARENA_CHALLENGE_ID: Final[str] = "2"

# Name of the GitHub repository variable that is the actual switch (spec
# section 2.3). Not the switch's value -- see is_live_enabled().
ARENA_LIVE_ENV_VAR: Final[str] = "ARENA_LIVE"


def is_live_enabled(env: Mapping[str, str]) -> bool:
    """True only for the exact value "true" (spec 2.3).

    The switch itself lives in the GitHub repository variables so it can be
    flipped without a commit (kill switch). Anything else -- unset, empty,
    "True", "1", "yes" -- keeps the job in dry-run mode. Fail-safe by design.
    """
    return env.get(ARENA_LIVE_ENV_VAR) == "true"
