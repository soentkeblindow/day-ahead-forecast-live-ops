import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation import bootstrap as bootstrap_module
from energy_price_forecast.evaluation.bootstrap import (
    cell_occupancy,
    stratified_day_block_bootstrap,
)


def _month_frame(days_by_start: dict[str, int], hours_per_day: int = 24) -> pd.DataFrame:
    """Build an hourly UTC frame with an exact, known number of days per (calendar-month) group.

    Keys are start dates (e.g. "2021-01-01"); values are day counts, kept
    within the real length of that calendar month so groups never overlap.
    Different keys MAY share the same month-of-year across different years
    -- that overlap is intentional (it is what the month-of-year pooling
    under test relies on).
    """
    rows: list[pd.Timestamp] = []
    for start, n_days in days_by_start.items():
        day0 = pd.Timestamp(start, tz="UTC")
        for d in range(n_days):
            day = day0 + pd.Timedelta(days=d)
            rows.extend(day + pd.Timedelta(hours=h) for h in range(hours_per_day))
    index = pd.DatetimeIndex(rows)
    return pd.DataFrame({"value": np.arange(len(index), dtype=float)}, index=index)


def _month_of_year(df: pd.DataFrame) -> np.ndarray:
    return pd.DatetimeIndex(df.index).month.to_numpy()


def _one_shot_bootstrap(
    frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], *, seed: int
) -> dict[str, float | int | bool]:
    """Forces exactly one replication (min == max == check_every == 1).

    Reproduces the old n_bootstrap=1 exact-value tests under the new API.
    """
    return stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=seed,
        min_bootstrap=1,
        max_bootstrap=1,
        check_every=1,
        mc_tol=0.01,
        n_stable=2,
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_stratified_bootstrap_is_deterministic_given_seed() -> None:
    frame = _month_frame({"2021-01-01": 10, "2021-02-01": 8})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    def _run() -> dict[str, float | int | bool]:
        return stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=42,
            min_bootstrap=200,
            max_bootstrap=2000,
            check_every=100,
            mc_tol=0.05,
            n_stable=2,
        )

    r1 = _run()
    r2 = _run()
    assert r1 == r2
    assert r1["n_bootstrap_used"] == r2["n_bootstrap_used"]


# ---------------------------------------------------------------------------
# Month-of-year composition preserved, pooled ACROSS YEARS (core deliverable)
# ---------------------------------------------------------------------------


def test_stratified_bootstrap_preserves_month_of_year_composition_across_years() -> None:
    # January pool spans TWO years (20 + 15 = 35 days); February is a
    # separate, single-year pool (10 days). The stratum key is the month
    # NUMBER, not (year, month): every replicate must draw exactly 35 days
    # from the pooled January set (mixing 2021 and 2022 freely), never 20
    # and 15 held fixed separately per year.
    frame = _month_frame({"2021-01-01": 20, "2022-01-01": 15, "2021-02-01": 10})

    def count_january_rows(df: pd.DataFrame) -> float:
        return float((_month_of_year(df) == 1).sum())

    original_january_rows = count_january_rows(frame)
    assert original_january_rows == (20 + 15) * 24

    result = _one_shot_bootstrap(frame, count_january_rows, seed=0)
    # A single forced replicate -> ci_low == ci_high == that replicate's statistic.
    # The TOTAL January day-count (across both years) is invariant every
    # replicate; only which specific year's January days get drawn varies.
    assert result["ci_low"] == pytest.approx(original_january_rows)
    assert result["ci_high"] == pytest.approx(original_january_rows)
    assert result["n_bootstrap_used"] == 1


# ---------------------------------------------------------------------------
# Block integrity: a drawn day's hours never mix with another day's
# ---------------------------------------------------------------------------


def test_stratified_bootstrap_keeps_day_blocks_intact() -> None:
    hours_per_day = 24
    frame = _month_frame({"2021-01-01": 20, "2021-02-01": 15}, hours_per_day=hours_per_day)
    frame["day_id"] = pd.DatetimeIndex(frame.index).tz_convert("UTC").normalize()

    def blocks_intact(df: pd.DataFrame) -> float:
        ids = df["day_id"].to_numpy()
        blocks = ids.reshape(-1, hours_per_day)
        all_constant = all((block == block[0]).all() for block in blocks)
        return 1.0 if all_constant else 0.0

    result = _one_shot_bootstrap(frame, blocks_intact, seed=7)
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


# ---------------------------------------------------------------------------
# CI sanity: coarse, tolerant Monte-Carlo check (single draw, not a power test)
# ---------------------------------------------------------------------------


def test_stratified_bootstrap_ci_covers_true_value_coarse_check() -> None:
    rng = np.random.default_rng(3)
    days_per_month = {"2021-01-01": 20, "2021-02-01": 20, "2021-03-01": 20}
    frame = _month_frame(days_per_month)
    # Overwrite "value" with iid noise per row, true population mean == 0.0.
    frame["value"] = rng.normal(loc=0.0, scale=1.0, size=len(frame))
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=1,
        min_bootstrap=500,
        max_bootstrap=500,
        check_every=500,
        mc_tol=0.01,
        n_stable=2,
    )
    assert result["ci_low"] < 0.0 < result["ci_high"]


# ---------------------------------------------------------------------------
# Convergence monitoring (Nachtrag 1, part A)
# ---------------------------------------------------------------------------


def test_convergence_reached_well_before_max_bootstrap() -> None:
    # A statistic with ZERO Monte-Carlo variance (ignores the resampled
    # frame entirely) must converge at the very first eligible checkpoint,
    # far short of max_bootstrap.
    frame = _month_frame({"2021-01-01": 20, "2021-02-01": 20, "2021-03-01": 20})
    statistic = lambda df: 5.0  # noqa: E731, ARG005

    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=11,
        min_bootstrap=1000,
        max_bootstrap=50_000,
        check_every=250,
        mc_tol=0.01,
        n_stable=2,
    )
    assert result["converged"] is True
    assert result["n_bootstrap_used"] < 50_000
    assert result["point"] == 5.0
    assert result["ci_low"] == 5.0
    assert result["ci_high"] == 5.0


def test_non_convergence_reported_without_crash_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Real (nonzero) Monte-Carlo noise plus an unreachably tight mc_tol
    # forces the ceiling to be hit without the criterion ever holding.
    rng = np.random.default_rng(5)
    frame = _month_frame({"2021-01-01": 15, "2021-02-01": 15})
    frame["value"] = rng.normal(loc=0.0, scale=1.0, size=len(frame))
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    with caplog.at_level(logging.WARNING, logger="energy_price_forecast.evaluation.bootstrap"):
        result = stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=13,
            min_bootstrap=200,
            max_bootstrap=400,
            check_every=100,
            mc_tol=1e-9,
            n_stable=2,
        )

    assert result["converged"] is False
    assert result["n_bootstrap_used"] == 400
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])
    assert any("did not converge" in r.message for r in caplog.records)


def test_never_stops_before_min_bootstrap() -> None:
    # Zero-variance statistic would satisfy the convergence criterion at the
    # very first check_every checkpoint if allowed to -- min_bootstrap must
    # still gate it, so the floor is the binding constraint here.
    frame = _month_frame({"2021-01-01": 20, "2021-02-01": 20, "2021-03-01": 20})
    statistic = lambda df: 5.0  # noqa: E731, ARG005

    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=17,
        min_bootstrap=1000,
        max_bootstrap=50_000,
        check_every=100,
        mc_tol=0.01,
        n_stable=2,
    )
    assert result["n_bootstrap_used"] >= 1000


def test_hysteresis_requires_two_consecutive_stable_checkpoints() -> None:
    # A single quiet checkpoint must not stop the run: n_stable=2 needs one
    # MORE checkpoint than n_stable=1 for a statistic that, once converged,
    # stays converged (MCSE only shrinks further as B grows).
    rng = np.random.default_rng(19)
    frame = _month_frame({"2021-01-01": 25, "2021-02-01": 25, "2021-03-01": 25})
    frame["value"] = rng.normal(loc=0.0, scale=1.0, size=len(frame))
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    check_every = 200

    def _run(n_stable: int) -> dict[str, float | int | bool]:
        return stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=23,
            min_bootstrap=500,
            max_bootstrap=20_000,
            check_every=check_every,
            mc_tol=0.2,
            n_stable=n_stable,
        )

    result_1 = _run(1)
    result_2 = _run(2)

    assert result_1["converged"] is True
    assert result_2["converged"] is True
    assert result_2["n_bootstrap_used"] == result_1["n_bootstrap_used"] + check_every


# ---------------------------------------------------------------------------
# Orthogonality: converged (MC error) vs. low_support (data thinness) measure
# different things -- a thin/clumpy cell can still converge cleanly.
# ---------------------------------------------------------------------------


def test_converged_and_low_support_can_both_be_true() -> None:
    # Only 3 distinct days in a single month-of-year pool: cell_occupancy
    # flags it thin (min_cell_days=30), but the bootstrap distribution over
    # only 3 days has few enough distinct outcomes to converge quickly.
    frame = _month_frame({"2021-01-01": 3})

    statistic = lambda df: float(df["value"].mean())  # noqa: E731
    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=29,
        min_bootstrap=500,
        max_bootstrap=20_000,
        check_every=100,
        mc_tol=0.05,
        n_stable=2,
    )
    occupancy = cell_occupancy(frame, local_tz="UTC", min_cell_days=30)

    assert result["converged"] is True
    assert occupancy["low_support"] is True


# ---------------------------------------------------------------------------
# cell_occupancy: month-of-year pools, across years
# ---------------------------------------------------------------------------


def test_cell_occupancy_flags_thin_month() -> None:
    # January pool (month=1): 20 + 15 = 35 days across two years (>= threshold).
    # February pool (month=2): 5 days, one year only (thin).
    frame = _month_frame({"2021-01-01": 20, "2022-01-01": 15, "2021-02-01": 5})

    result = cell_occupancy(frame, local_tz="UTC", min_cell_days=30)
    thin_months = result["thin_months"]
    assert result["low_support"] is True
    assert isinstance(thin_months, list)
    assert thin_months == ["2"]
    assert result["min_count"] == 5.0


def test_cell_occupancy_pools_partial_month_across_years_above_threshold() -> None:
    # February is thin in ANY single year (28 days, 2021) but a second,
    # partial February from another year (5 days, 2022) pools with it to
    # clear min_cell_days=30 -- exactly the real-world case this
    # stratification is meant to fix (a warmup-truncated boundary month
    # diluted by full instances of the same month-of-year elsewhere).
    frame = _month_frame({"2021-01-01": 31, "2021-02-01": 28, "2022-02-01": 5})

    result = cell_occupancy(frame, local_tz="UTC", min_cell_days=30)
    assert result["low_support"] is False
    assert result["thin_months"] == []
    assert result["min_count"] == 31.0  # January is the (still comfortably passing) minimum


# ---------------------------------------------------------------------------
# Architecture: independence family stays untouched by the bootstrap module
# ---------------------------------------------------------------------------


def test_bootstrap_module_does_not_import_christoffersen() -> None:
    assert not hasattr(bootstrap_module, "christoffersen_independence")


# ---------------------------------------------------------------------------
# Nachtrag 2, part A: block_days moving-block bootstrap
# ---------------------------------------------------------------------------


def test_block_days_default_reproduces_pre_nachtrag2_golden_values() -> None:
    # The most important test (spec Nachtrag 2, 5.1): block_days=1 (default)
    # must reproduce the EXACT pre-Nachtrag-2 RNG call sequence. Golden
    # values frozen from the committed pre-Nachtrag-2 code (commit 3690007)
    # for this exact frame/statistic/seed combination.
    frame = _month_frame({"2021-01-01": 20, "2022-01-01": 15, "2021-02-01": 10})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=42,
        min_bootstrap=200,
        max_bootstrap=2000,
        check_every=100,
        mc_tol=0.05,
        n_stable=2,
    )

    assert result["point"] == pytest.approx(539.5)
    assert result["ci_low"] == pytest.approx(472.55333333333334)
    assert result["ci_high"] == pytest.approx(598.4466666666667)
    assert result["n_days"] == 45.0
    assert result["n_bootstrap_used"] == 300
    assert result["converged"] is True
    assert result["block_days"] == 1

    # Explicit block_days=1 must match the (implicit-default) call bit-exactly.
    explicit = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=42,
        min_bootstrap=200,
        max_bootstrap=2000,
        check_every=100,
        mc_tol=0.05,
        n_stable=2,
        block_days=1,
    )
    assert explicit == result


def test_block_days_keeps_runs_of_consecutive_days_intact() -> None:
    # block_days=3, tested directly against the internal draw (spec Nachtrag
    # 2, 5.2): each "day" is a single-element position array holding its own
    # list-index (0..19), so the concatenated replicate directly reveals
    # which day-list positions were drawn, in draw order. Chunking the flat
    # output into consecutive groups of `block_days` must recover exactly the
    # blocks _draw_replicate concatenated: each chunk (the last one possibly
    # shorter, due to truncation) is a run of consecutive list positions.
    month_days = {1: [np.array([i]) for i in range(20)]}
    rng = np.random.default_rng(5)
    block_days = 3

    replicate = bootstrap_module._draw_replicate(month_days, rng, block_days=block_days)
    day_order = replicate.tolist()
    assert len(day_order) == 20  # month invariant: truncated back to n_m=20

    for start in range(0, len(day_order), block_days):
        block = day_order[start : start + block_days]
        assert block == list(range(block[0], block[0] + len(block))), (
            f"block starting at {start} is not a run of consecutive positions: {block}"
        )


def test_block_days_preserves_month_composition_invariant() -> None:
    frame = _month_frame({"2021-01-01": 20, "2022-01-01": 15, "2021-02-01": 10})

    def count_by_month(df: pd.DataFrame) -> dict[int, int]:
        months = _month_of_year(df)
        return {1: int((months == 1).sum()) // 24, 2: int((months == 2).sum()) // 24}

    for block_days in (1, 3, 5, 7):
        counts: list[dict[int, int]] = []

        def record(df: pd.DataFrame, counts: list[dict[int, int]] = counts) -> float:
            counts.append(count_by_month(df))
            return 0.0

        stratified_day_block_bootstrap(
            frame,
            record,
            local_tz="UTC",
            seed=1,
            min_bootstrap=20,
            max_bootstrap=20,
            check_every=20,
            mc_tol=0.01,
            n_stable=1,
            block_days=block_days,
        )
        for c in counts:
            assert c == {1: 35, 2: 10}, f"block_days={block_days} broke the month invariant"


def test_block_days_widens_ci_for_autocorrelated_breach_series() -> None:
    # Behavioural core (spec Nachtrag 2, 5.4): with strongly clustered
    # (multi-day-run) daily values, a multi-day block must capture more of
    # the true variance than the one-day block -- the whole motivation for
    # this Nachtrag.
    hours_per_day = 24
    n_days = 60
    frame = _month_frame({"2021-01-01": n_days}, hours_per_day=hours_per_day)
    # Clumped signal: 5-day runs alternating between 0 and 1.
    day_values = (np.arange(n_days) // 5) % 2
    frame["value"] = np.repeat(day_values.astype(float), hours_per_day)

    def day_mean(df: pd.DataFrame) -> float:
        return float(df["value"].to_numpy().reshape(-1, hours_per_day)[:, 0].mean())

    result_1 = stratified_day_block_bootstrap(
        frame,
        day_mean,
        local_tz="UTC",
        seed=2,
        min_bootstrap=2000,
        max_bootstrap=2000,
        check_every=2000,
        mc_tol=0.01,
        n_stable=1,
        block_days=1,
    )
    result_5 = stratified_day_block_bootstrap(
        frame,
        day_mean,
        local_tz="UTC",
        seed=2,
        min_bootstrap=2000,
        max_bootstrap=2000,
        check_every=2000,
        mc_tol=0.01,
        n_stable=1,
        block_days=5,
    )

    width_1 = result_1["ci_high"] - result_1["ci_low"]
    width_5 = result_5["ci_high"] - result_5["ci_low"]
    assert width_5 > 1.2 * width_1


def test_block_days_does_not_widen_ci_for_iid_series() -> None:
    # Counterpart (spec Nachtrag 2, 5.5): with i.i.d. daily values, a
    # multi-day block must NOT fabricate extra variance.
    rng = np.random.default_rng(9)
    hours_per_day = 24
    n_days = 60
    frame = _month_frame({"2021-01-01": n_days}, hours_per_day=hours_per_day)
    day_values = rng.binomial(1, 0.5, size=n_days).astype(float)
    frame["value"] = np.repeat(day_values, hours_per_day)

    def day_mean(df: pd.DataFrame) -> float:
        return float(df["value"].to_numpy().reshape(-1, hours_per_day)[:, 0].mean())

    result_1 = stratified_day_block_bootstrap(
        frame,
        day_mean,
        local_tz="UTC",
        seed=2,
        min_bootstrap=2000,
        max_bootstrap=2000,
        check_every=2000,
        mc_tol=0.01,
        n_stable=1,
        block_days=1,
    )
    result_5 = stratified_day_block_bootstrap(
        frame,
        day_mean,
        local_tz="UTC",
        seed=2,
        min_bootstrap=2000,
        max_bootstrap=2000,
        check_every=2000,
        mc_tol=0.01,
        n_stable=1,
        block_days=5,
    )

    width_1 = result_1["ci_high"] - result_1["ci_low"]
    width_5 = result_5["ci_high"] - result_5["ci_low"]
    assert width_5 == pytest.approx(width_1, rel=0.20)


def test_block_days_short_month_collapses_to_original(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # n_m=3 < block_days=7: exactly one candidate start (the whole pool);
    # after truncation the replicate for this pool equals the original.
    frame = _month_frame({"2021-01-01": 3})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731
    original_value = statistic(frame)

    with caplog.at_level(logging.DEBUG, logger="energy_price_forecast.evaluation.bootstrap"):
        result = stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=3,
            min_bootstrap=50,
            max_bootstrap=50,
            check_every=50,
            mc_tol=0.01,
            n_stable=1,
            block_days=7,
        )

    assert result["ci_low"] == pytest.approx(original_value)
    assert result["ci_high"] == pytest.approx(original_value)
    assert any("fewer than block_days" in r.message for r in caplog.records)


def test_block_days_is_deterministic() -> None:
    frame = _month_frame({"2021-01-01": 20, "2021-02-01": 15})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    def _run() -> dict[str, float | int | bool]:
        return stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=8,
            min_bootstrap=200,
            max_bootstrap=2000,
            check_every=100,
            mc_tol=0.05,
            n_stable=2,
            block_days=5,
        )

    r1 = _run()
    r2 = _run()
    assert r1 == r2
    assert r1["n_bootstrap_used"] == r2["n_bootstrap_used"]


@pytest.mark.parametrize("bad_block_days", [0, -1])
def test_block_days_validation_rejects_non_positive(bad_block_days: int) -> None:
    frame = _month_frame({"2021-01-01": 10})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    with pytest.raises(ValueError, match="block_days"):
        stratified_day_block_bootstrap(
            frame,
            statistic,
            local_tz="UTC",
            seed=1,
            min_bootstrap=10,
            max_bootstrap=10,
            check_every=10,
            mc_tol=0.01,
            n_stable=1,
            block_days=bad_block_days,
        )


def test_low_support_is_invariant_to_block_days() -> None:
    frame = _month_frame({"2021-01-01": 20, "2022-01-01": 15, "2021-02-01": 5})
    result = cell_occupancy(frame, local_tz="UTC", min_cell_days=30)
    # cell_occupancy takes no block_days parameter at all (spec Nachtrag 2,
    # 2.5) -- calling it again on the same frame must be identical regardless
    # of what block_days any bootstrap call elsewhere used.
    result_again = cell_occupancy(frame, local_tz="UTC", min_cell_days=30)
    assert result == result_again


def test_block_days_is_echoed_in_return_contract() -> None:
    frame = _month_frame({"2021-01-01": 10})
    statistic = lambda df: float(df["value"].mean())  # noqa: E731

    result = stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz="UTC",
        seed=1,
        min_bootstrap=10,
        max_bootstrap=10,
        check_every=10,
        mc_tol=0.01,
        n_stable=1,
        block_days=3,
    )
    assert result["block_days"] == 3
