"""Month-stratified, day-block bootstrap (Sprint 4.4b).

The generic significance machine for the coverage/magnitude test family
(Kupiec breach rate, Acerbi-Szekely Z1/Z2, the calibrated-vs-fhs variant
contrast). It is a RESAMPLING OBJECT, not a test statistic -- the same split
``residuals.py``/``risk.py`` drew in 4.4a, mirrored here for
``bootstrap.py``/``backtest.py``.

Two design choices, both argued in the spec (section 2.6):

- Block = delivery day. The 24 (or 23/25 on a DST edge) hours of one delivery
  day share pool, weather and regime -- resampling individual hours would tear
  that dependence apart and understate the true uncertainty. The block keeps
  a day's hours together.
- Stratum = calendar MONTH NUMBER (1-12, Europe/Berlin), pooling the SAME
  month-of-year across every year in the sample -- e.g. every June in a
  5-year backtest is one stratum, not five. Only the month-of-year
  composition is a fixed calendar effect worth holding constant; which
  specific YEAR's June contributes how many of a replicate's June days is
  deliberately left free to vary -- that is the intended randomness. Each
  replicate draws, WITHIN each month-of-year pool, exactly that pool's own
  day-count with replacement, so the total day-count per month-of-year is
  reproduced exactly every replicate, but the year mix is not.

This module deliberately does NOT import ``christoffersen_independence`` (see
``tests/test_bootstrap.py`` for the import-boundary test): the independence
family relies on the asymptotic chi-square distribution, and an i.i.d. day
resample would destroy the very serial dependence that statistic measures.
Two testing families, one resampling regime -- this module serves only one
of them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ

log = logging.getLogger(__name__)


def _day_positions(index: pd.DatetimeIndex, *, local_tz: str) -> pd.Series:
    """Delivery day (local midnight) -> integer row positions, ascending index order."""
    local_day = index.tz_convert(local_tz).normalize()
    frame = pd.DataFrame({"day": local_day, "pos": np.arange(len(index))})
    grouped = frame.groupby("day")["pos"].apply(lambda s: s.to_numpy())
    return grouped.sort_index()


def _draw_replicate(
    month_days: dict[int, list[np.ndarray]], rng: np.random.Generator, *, block_days: int
) -> np.ndarray:
    """Moving block bootstrap draw, within each month-of-year pool independently.

    For a pool of n_m days: n_starts = max(1, n_m - block_days + 1) overlapping
    candidate block starts, n_blocks = ceil(n_m / block_days) blocks drawn WITH
    replacement, concatenated and TRUNCATED to exactly n_m days -- so every
    replicate reproduces the pool's own day-count exactly (spec Nachtrag 2,
    2.2). At block_days=1, n_starts == n_blocks == n_m and this collapses to
    `rng.integers(0, n_m, size=n_m)`, the same RNG call as the pre-Nachtrag-2
    code (bit-exact backward compatibility, spec 2.3).
    """
    chosen: list[np.ndarray] = []
    for days_in_month in month_days.values():
        n_m = len(days_in_month)
        n_starts = max(1, n_m - block_days + 1)
        n_blocks = -(-n_m // block_days)  # ceil(n_m / block_days)
        starts = rng.integers(0, n_starts, size=n_blocks)
        flat_days: list[np.ndarray] = []
        for s in starts:
            flat_days.extend(days_in_month[s : s + block_days])
        chosen.extend(flat_days[:n_m])
    return np.concatenate(chosen)


def stratified_day_block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    local_tz: str = LOCAL_TZ,
    seed: int,
    ci: float = 0.95,
    min_bootstrap: int,
    max_bootstrap: int,
    check_every: int,
    mc_tol: float,
    n_stable: int,
    block_days: int = 1,
) -> dict[str, float | int | bool]:
    """Month-stratified, MULTI-DAY-block bootstrap CI with MC convergence monitoring.

    Stratum = calendar MONTH NUMBER (1-12) of the delivery day
    (Europe/Berlin, `local_tz`), pooling that same month across every year
    present (unchanged from the original design, spec 2.6). Block = a run of
    `block_days` CONSECUTIVE delivery days (all their valid hours stay
    together), drawn as a MOVING BLOCK BOOTSTRAP within each month-of-year
    pool: each replicate draws ceil(n_m / block_days) blocks with replacement
    from that pool's overlapping candidate starts, concatenates them, and
    TRUNCATES to exactly n_m days -- so every replicate still reproduces the
    pool's real day-count (the stratification contract, spec 2.6).
    Truncation costs block integrity on at most the LAST block per pool;
    keeping the day-count exact is the stronger guarantee and wins (spec
    Nachtrag 2, 2.2).

    `block_days=1` (default) is the historical one-day block and reproduces
    pre-Nachtrag-2 results BIT-EXACTLY: the draw collapses to
    `rng.integers(0, n_m, size=n_m)`, the same RNG call as before (spec
    Nachtrag 2, 2.3).

    Why `block_days > 1` exists (spec Nachtrag 2, 1.1): the one-day block
    assumes days are exchangeable within a month. The daily Christoffersen
    LR_ind measured in the Nachtrag 1 run (53-104 vs. a chi2_1 critical value
    of 3.84) FALSIFIES that assumption -- day-level breaches cluster
    (multi-day cold snaps / Dunkelflauten). So the one-day-block CIs are
    somewhat TOO NARROW; the direction of the error is known, the magnitude
    is not. A grid over `block_days` (BacktestConfig.block_days_grid)
    measures it, without replacing the reported `block_days=1` headline.

    Overlapping CANDIDATE blocks are NOT the Basel mistake (Nachtrag 1, part
    B): there we took a MAXIMUM over overlapping windows, i.e. we SELECTED
    the worst alignment. Here blocks are drawn at RANDOM, with no selection
    -- this is the standard moving block bootstrap (Kuensch 1989) and
    introduces no selection bias.

    NOTE: "consecutive" means consecutive in that month-of-year pool's
    sorted list of AVAILABLE delivery days, not necessarily calendar-adjacent
    -- the warm-up drop and the `min_valid_hours` drop can leave holes within
    a year, and the month-of-year pooling itself means the list jumps from
    one year's instance of the month to the next year's at pool boundaries.
    A deliberate approximation (spec Nachtrag 2, 2.2): restricting to
    calendar-contiguous blocks would thin the candidate set and distort the
    stratification. A block that happens to straddle such a jump behaves
    like two independent days rather than a true dependency run, which
    dilutes (never inflates) the measured `block_days` effect.

    A month-of-year pool with `n_m < block_days` has exactly one candidate
    start (the whole pool); after truncation that pool's replicate equals the
    original (no resampling variance from it) -- no crash, logged at DEBUG.

    Convergence, `low_support` orthogonality, no-fishing stopping rule:
    unchanged from Nachtrag 1. `low_support` (`cell_occupancy`) is
    INDEPENDENT of `block_days` by design (spec Nachtrag 2, 2.5) -- thin
    cells are a DATA problem, block length is a RESAMPLING problem. Larger
    blocks shrink the effective sample size and WIDEN the CI; that widening
    IS the signal, not a second low-support flag.

    Raises ValueError if `block_days < 1`. Returns a `block_days` key
    (echoing the input) in addition to the fields below, so a row carries its
    own resampling regime without needing to be reconstructed from a file
    name.

    Convergence (Nachtrag 1, part A): every `check_every` replications, this
    treats the replications drawn so far as `B // check_every` equal-length
    blocks (one block per check_every-sized chunk, in draw order) and
    estimates the Monte-Carlo standard error of BOTH reported percentiles via
    batch means -- the per-block percentile, then MCSE = sd(block
    percentiles) / sqrt(n_blocks). Stops once
    `MCSE(q_low) < mc_tol * (ci_high - ci_low)` AND
    `MCSE(q_high) < mc_tol * (ci_high - ci_low)` hold on `n_stable`
    CONSECUTIVE checkpoints -- never before `min_bootstrap`, never past
    `max_bootstrap`. Percentiles, not the mean, are monitored deliberately:
    tail-percentile variance scales roughly as `p(1-p) / (B * f(q)**2)`, and
    the density `f(q)` is small at the edges, so the reported quantities
    converge slower than the mean would.

    IMPORTANT -- "converged" does NOT mean "trustworthy". It bounds only the
    Monte-Carlo error from a finite B. It says nothing about sampling
    uncertainty from a thin stratum: a clumpy bootstrap distribution (few
    distinct days in some month-of-year pool) also converges cleanly -- to
    the percentile of a BAD approximation. `converged` NEVER replaces
    `low_support` (`cell_occupancy`); the two flags are orthogonal and both
    are meant to be reported alongside each other.

    The stopping rule reads the MC error ONLY -- it never looks at the test
    outcome (e.g. never "stop once 0 falls outside the CI"). That would be
    fishing (optional-stopping bias tied to significance).

    `statistic` maps a resampled frame to one scalar (e.g. a breach rate, a
    Z1, or a calibrated-minus-fhs difference). Returns {"point", "ci_low",
    "ci_high", "n_days", "n_bootstrap_used", "converged"}. `point` is
    `statistic(frame)` on the ORIGINAL (unresampled) data, not the bootstrap
    mean. Deterministic given `seed`, INCLUDING `n_bootstrap_used` (the draw
    sequence and the stopping rule are both deterministic functions of the
    seed). Does NOT serve the Christoffersen family (which uses chi2): an
    i.i.d. day resample would destroy the serial dependence LR_ind measures.
    """
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")

    index = pd.DatetimeIndex(frame.index)
    day_positions = _day_positions(index, local_tz=local_tz)

    months = pd.DatetimeIndex(day_positions.index).month
    month_days: dict[int, list[np.ndarray]] = {}
    for month, positions in zip(months, day_positions, strict=True):
        month_days.setdefault(int(month), []).append(positions)

    for month, days_in_month in month_days.items():
        if len(days_in_month) < block_days:
            log.debug(
                "stratified_day_block_bootstrap: month-of-year %d has only %d day(s), "
                "fewer than block_days=%d -- every replicate for this pool collapses to "
                "the original day list (no resampling variance from it).",
                month,
                len(days_in_month),
                block_days,
            )

    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    thetas: list[float] = []
    stable_streak = 0
    converged = False
    next_checkpoint = check_every

    while True:
        target = min(next_checkpoint, max_bootstrap)
        while len(thetas) < target:
            replicate_positions = _draw_replicate(month_days, rng, block_days=block_days)
            thetas.append(statistic(frame.iloc[replicate_positions]))

        b = len(thetas)
        n_blocks = b // check_every
        if b >= min_bootstrap and n_blocks >= 2:
            arr = np.asarray(thetas[: n_blocks * check_every], dtype=float)
            blocks = arr.reshape(n_blocks, check_every)
            block_qlo = np.nanpercentile(blocks, lo_q, axis=1)
            block_qhi = np.nanpercentile(blocks, hi_q, axis=1)
            ci_low_now = float(np.nanpercentile(arr, lo_q))
            ci_high_now = float(np.nanpercentile(arr, hi_q))
            width = ci_high_now - ci_low_now
            mcse_lo = float(np.std(block_qlo, ddof=1) / np.sqrt(n_blocks))
            mcse_hi = float(np.std(block_qhi, ddof=1) / np.sqrt(n_blocks))
            ok = (
                (mcse_lo < mc_tol * width) and (mcse_hi < mc_tol * width)
                if width > 0
                else (mcse_lo == 0.0 and mcse_hi == 0.0)
            )
            stable_streak = stable_streak + 1 if ok else 0
            if stable_streak >= n_stable:
                converged = True
                break

        if b >= max_bootstrap:
            log.warning(
                "stratified_day_block_bootstrap did not converge within "
                "max_bootstrap=%d replications (mc_tol=%.4g, n_stable=%d); "
                "returned CI's Monte-Carlo error is not bounded as tightly as requested.",
                max_bootstrap,
                mc_tol,
                n_stable,
            )
            break
        next_checkpoint += check_every

    n_bootstrap_used = len(thetas)
    arr = np.asarray(thetas, dtype=float)
    ci_low = float(np.nanpercentile(arr, lo_q))
    ci_high = float(np.nanpercentile(arr, hi_q))
    point = float(statistic(frame))

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_days": float(len(day_positions)),
        "block_days": block_days,
        "n_bootstrap_used": n_bootstrap_used,
        "converged": converged,
    }


def cell_occupancy(
    frame: pd.DataFrame, *, local_tz: str = LOCAL_TZ, min_cell_days: int
) -> dict[str, object]:
    """Per-month-of-year VALID-day counts for a (already-subset) frame.

    Counts the number of distinct delivery days present in `frame` per
    calendar MONTH NUMBER (1-12), pooled across every year present -- the
    same stratum definition used by `stratified_day_block_bootstrap`. This is
    the day-block diversity the bootstrap actually draws from for THIS
    (already subset- and valid-hour-restricted) cell, regardless of whether
    any given day breached.

    (Two earlier, rejected definitions: counting BREACH-days instead of valid
    days made `min_cell_days=30` unreachable at a well-calibrated ~5% rate;
    keying strata by (year, month) instead of month-of-year made every
    February -- and any warmup-truncated boundary month, e.g. the first
    partial month after a pool warmup period -- structurally thin in EVERY
    subset, since each such month only ever has one calendar instance.
    Pooling by month-of-year across years fixes both: a "February" pool spans
    every year's February, and a partial boundary month is diluted by the
    same month-of-year's full instances in other years.)

    Flags low_support when any month-of-year pool contributes fewer than
    `min_cell_days` valid days -- the signal that a percentile CI on this
    subset is not to be trusted (spec 2.6). Returns {"low_support": bool,
    "thin_months": [...], "min_count": ...}. `thin_months` entries are the
    month numbers (1-12, as strings) below the threshold.
    """
    index = pd.DatetimeIndex(frame.index)
    local_day = index.tz_convert(local_tz).normalize()

    unique_days = pd.DatetimeIndex(sorted(pd.unique(local_day)))
    if unique_days.empty:
        return {"low_support": True, "thin_months": [], "min_count": 0.0}

    months = unique_days.month
    counts = pd.Series(1, index=months).groupby(level=0).sum()

    thin = counts[counts < min_cell_days]
    return {
        "low_support": bool(len(thin) > 0),
        "thin_months": [str(m) for m in thin.index],
        "min_count": float(counts.min()),
    }
