"""Build and validate quantile-challenge submission payloads for the Energy
Arena (Sprint 7.1 Teil B -> Probephase).

Sibling of ``arena/payload.py`` for ``arena.catalog.QuantileChallengeSpec``
(nested values -- one list of ``challenge.quantile_levels`` floats per
timestep) -- added alongside ``payload.py``, not a generalisation of it:
``payload.py``/``ChallengeSpec``/``build_payload``/``validate_payload`` stay
unchanged (Point challenges only), same convention as
``evaluation.conformal.quarterhour_sigma``.

Real shape confirmed against the Arena API, not guessed (spec §3.1): for
challenge 8 (DE-LU quantile track), ``values`` is a list of 96 lists, each
holding exactly 5 floats in ascending order
(``[q0.025, q0.25, q0.5, q0.75, q0.975]``), matching the challenge detail's
own ``probabilistic_quantiles`` field --
``tests/fixtures/arena_public_example_challenge_8.json``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from energy_price_forecast.arena.catalog import QuantileChallengeSpec
from energy_price_forecast.arena.payload import PayloadValidationError


def expected_value_count(challenge: QuantileChallengeSpec, target_start: pd.Timestamp) -> int:
    """Number of rows (one per timestep) a calendar-day submission for
    target_start needs -- the quantile-shape sibling of
    ``payload.expected_value_count``. Same DST-correctness reasoning
    (``DateOffset(days=1)`` after re-tagging with the challenge's named
    timezone, not a fixed-offset ``Timedelta(hours=24)``) -- duplicated
    rather than shared with ``payload.py``'s version to keep that module's
    ``ChallengeSpec``-typed signature untouched.
    """
    if target_start.tzinfo is None:
        raise ValueError("target_start must be tz-aware (Europe/Berlin, not naive)")
    target_start_zoned = target_start.tz_convert(challenge.timezone)
    target_end = target_start_zoned + pd.DateOffset(days=1)
    elapsed = target_end - target_start_zoned
    return int(elapsed / pd.Timedelta(minutes=challenge.resolution_minutes))


def build_payload(
    challenge: QuantileChallengeSpec,
    target_start: pd.Timestamp,
    quantiles: dict[float, pd.Series],
) -> dict[str, Any]:
    """Build {"challenge_id", "target_start", "values"} for a quantile
    challenge. ``values`` is a list of ``len(challenge.quantile_levels)``-
    length lists, one per timestep, each row's values in
    ``challenge.quantile_levels`` order (ascending, confirmed against the
    real API response).

    ``quantiles`` maps each level to its Series (e.g.
    ``evaluation.fhs_quantiles.compute_fhs_quantiles_for_row``'s return
    value, sliced to one target day) -- all Series must share an
    identical, chronologically sorted index; that order becomes
    ``values``'s row order (the API has no explicit per-row timestamp,
    same convention as the Point-challenge flat list).
    """
    if target_start.tzinfo is None:
        raise ValueError("target_start must be tz-aware (Europe/Berlin, not naive)")
    levels = challenge.quantile_levels
    missing = [a for a in levels if a not in quantiles]
    if missing:
        raise ValueError(f"quantiles is missing level(s): {missing}")

    indices = [quantiles[a].index for a in levels]
    if not all(indices[0].equals(idx) for idx in indices[1:]):
        raise ValueError("all levels in quantiles must share an identical index")

    rounded = {
        a: [round(float(v), challenge.precision_decimals) for v in quantiles[a].to_numpy()]
        for a in levels
    }
    n = len(indices[0])
    values = [[rounded[a][i] for a in levels] for i in range(n)]
    return {
        "challenge_id": challenge.challenge_id,
        "target_start": target_start.isoformat(),
        "values": values,
    }


def validate_payload(payload: dict[str, Any], challenge: QuantileChallengeSpec) -> None:
    """Raise ``PayloadValidationError`` if the Arena would reject payload
    for challenge -- the quantile-shape sibling of
    ``payload.validate_payload``.

    Checks: ``values`` is a list of lists, each row has exactly
    ``len(challenge.quantile_levels)`` finite values in ASCENDING order
    (the Arena requires monotonic quantiles per timestep -- a last-line
    structural check, not a second enforcement mechanism;
    ``evaluation.rearrangement.rearrange_quantiles`` is what actually
    produces that ordering upstream), the row count matches the
    resolution grid, and ``target_start``'s format/offset.
    """
    values = payload.get("values")
    if not isinstance(values, list):
        raise PayloadValidationError(f"payload['values'] must be a list, got {type(values)!r}")
    if not all(isinstance(row, list) for row in values):
        raise PayloadValidationError(
            "payload['values'] must be a list of lists for a quantile challenge "
            "(flat values are for Point challenges)"
        )

    n_levels = len(challenge.quantile_levels)
    for i, row in enumerate(values):
        if len(row) != n_levels:
            raise PayloadValidationError(
                f"payload['values'][{i}] has {len(row)} value(s), expected {n_levels} "
                f"(challenge.quantile_levels={challenge.quantile_levels})"
            )
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in row):
            raise PayloadValidationError(
                f"payload['values'][{i}] contains a non-finite value (NaN/inf)"
            )
        if list(row) != sorted(row):
            raise PayloadValidationError(f"payload['values'][{i}] is not ascending: {row}")

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
            f"payload['values'] has {len(values)} row(s), expected {expected} "
            f"for a {challenge.resolution_minutes}-minute grid on {target_start.date()}"
        )
