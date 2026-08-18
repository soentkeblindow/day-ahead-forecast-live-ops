import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.residuals import tail_quantile
from energy_price_forecast.evaluation.risk import (
    QUANTITY_MWH,
    SDAC_PRICE_LIMITS,
    book_pnl,
    es_ratio_conditional,
    realised_es,
    risk_measures,
    sdac_limits_asof,
)


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _single_day_index(n_hours: int) -> pd.DatetimeIndex:
    """Hours entirely inside one Europe/Berlin local delivery day (June: no DST edge)."""
    return pd.date_range(
        "2021-06-01 00:00", periods=n_hours, freq="h", tz="Europe/Berlin"
    ).tz_convert("UTC")


_DAY = pd.Timestamp("2021-06-01", tz="Europe/Berlin").date()


# ---------------------------------------------------------------------------
# book_pnl: sign convention and scaling
# ---------------------------------------------------------------------------


def test_book_pnl_long_sign_and_scaling() -> None:
    price = pd.Series([90.0, 110.0])
    mark = pd.Series([100.0, 100.0])

    pnl = book_pnl(price, mark, side="long", quantity=10.0)

    # price below mark -> long loses; price above mark -> long gains.
    pd.testing.assert_series_equal(pnl, pd.Series([-100.0, 100.0]))


def test_book_pnl_short_sign_and_scaling() -> None:
    price = pd.Series([90.0, 110.0])
    mark = pd.Series([100.0, 100.0])

    pnl = book_pnl(price, mark, side="short", quantity=10.0)

    # price below mark -> short gains; price above mark -> short loses.
    pd.testing.assert_series_equal(pnl, pd.Series([100.0, -100.0]))


def test_book_pnl_scales_linearly_with_quantity() -> None:
    price = pd.Series([80.0])
    mark = pd.Series([100.0])

    pnl_10 = book_pnl(price, mark, side="long", quantity=10.0)
    pnl_20 = book_pnl(price, mark, side="long", quantity=20.0)

    assert pnl_20.iloc[0] == pytest.approx(2 * pnl_10.iloc[0])


def test_book_pnl_default_quantity_is_10_mwh() -> None:
    price = pd.Series([90.0])
    mark = pd.Series([100.0])
    pnl = book_pnl(price, mark, side="long")
    assert pnl.iloc[0] == pytest.approx(QUANTITY_MWH * (90.0 - 100.0))


# ---------------------------------------------------------------------------
# sdac_limits_asof: edge test at all three date boundaries
# ---------------------------------------------------------------------------


def test_sdac_limits_asof_before_first_edge_uses_first_limits() -> None:
    index = _hourly_utc("2019-01-01", 3)
    floor, cap = sdac_limits_asof(index)
    assert (floor == -500.0).all()
    assert (cap == 3000.0).all()


@pytest.mark.parametrize("edge_date,new_floor,new_cap", SDAC_PRICE_LIMITS[1:])
def test_sdac_limits_asof_day_before_and_on_transition(
    edge_date: str, new_floor: float, new_cap: float
) -> None:
    edge_local_midnight = pd.Timestamp(edge_date, tz="Europe/Berlin")
    edge_utc = edge_local_midnight.tz_convert("UTC")

    day_before = pd.DatetimeIndex([edge_utc - pd.Timedelta(hours=1)])
    on_edge = pd.DatetimeIndex([edge_utc])

    floor_before, cap_before = sdac_limits_asof(day_before)
    floor_on, cap_on = sdac_limits_asof(on_edge)

    assert floor_before.iloc[0] != new_floor or cap_before.iloc[0] != new_cap
    assert floor_on.iloc[0] == new_floor
    assert cap_on.iloc[0] == new_cap


def test_sdac_limits_asof_uses_europe_berlin_local_day() -> None:
    """A UTC hour that is still the PREVIOUS Berlin-local day must use the
    limits in force on that previous local day, not the UTC calendar day.
    """
    edge_local_midnight = pd.Timestamp("2022-05-10", tz="Europe/Berlin")
    edge_utc = edge_local_midnight.tz_convert("UTC")
    # one hour before Berlin midnight, but (in winter->summer terms here,
    # May is CEST = UTC+2) still the same UTC calendar day as the edge.
    just_before_local_midnight = pd.DatetimeIndex([edge_utc - pd.Timedelta(hours=1)])

    floor, cap = sdac_limits_asof(just_before_local_midnight)

    assert floor.iloc[0] == -500.0
    assert cap.iloc[0] == 3000.0


# ---------------------------------------------------------------------------
# realised_es: hand-computed, model-free ex-post tail
# ---------------------------------------------------------------------------


def test_realised_es_hand_computed() -> None:
    # 20 pnl outcomes; level=0.95 -> alpha=0.05 -> bottom 5% tail.
    pnl = pd.Series(np.arange(1, 21, dtype=float))  # 1..20
    result = realised_es(pnl, level=0.95)

    threshold = float(np.quantile(pnl.to_numpy(), 0.05))
    expected_tail = pnl.to_numpy()[pnl.to_numpy() <= threshold]

    assert result["n"] == 20.0
    assert result["n_tail"] == pytest.approx(float(expected_tail.size))
    assert result["var"] == pytest.approx(-threshold)
    assert result["es"] == pytest.approx(-float(expected_tail.mean()))


def test_realised_es_drops_nan() -> None:
    pnl = pd.Series([1.0, 2.0, float("nan"), 3.0, 4.0, 5.0])
    result = realised_es(pnl, level=0.95)
    assert result["n"] == 5.0


# ---------------------------------------------------------------------------
# risk_measures: variant-identity and mark-invariance, the two tests that
# hold the architecture (spec 10, step 7)
# ---------------------------------------------------------------------------


def test_variant_identity_fhs_matches_calibrated_with_matching_threshold() -> None:
    """Proves risk_measures is really ONE tail estimator with three
    thresholds: "fhs" must equal "calibrated" when calibrated's
    threshold_quantile is constructed to be exactly the fhs threshold.
    """
    index = _single_day_index(4)
    price = pd.Series([70.0, 90.0, 110.0, 130.0], index=index)
    mark = pd.Series([100.0] * 4, index=index)
    sigma = pd.Series([10.0] * 4, index=index)
    pool = {_DAY: np.array([-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0])}

    for side in ("long", "short"):
        fhs = risk_measures(
            price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side=side, level=0.95
        )
        alpha = 0.05 if side == "long" else 0.95
        matched_threshold = mark + sigma * tail_quantile(pool[_DAY], alpha)
        calibrated = risk_measures(
            price=price,
            mark=mark,
            sigma=sigma,
            pool=pool,
            variant="calibrated",
            side=side,
            level=0.95,
            threshold_quantile=matched_threshold,
        )
        pd.testing.assert_frame_equal(fhs, calibrated)


def test_mark_invariance_var_shifts_by_quantity_times_c_breach_unchanged() -> None:
    """spec 2.2: for variants with an independent threshold_quantile input,
    shifting mark by a forecast-time-known constant c (holding
    threshold_quantile fixed) leaves breach unchanged and shifts VAR by
    exactly Q * c -- an exact algebraic identity, since mark cancels out of
    the unclipped threshold_price (mark + sigma * (threshold_quantile -
    mark) / sigma == threshold_quantile, independent of mark). ES is NOT
    claimed to shift by Q * c here -- that property belongs to book_pnl +
    realised_es (see test_realised_es_hand_computed and spec section 6).
    """
    index = _single_day_index(3)
    price = pd.Series([60.0, 70.0, 80.0], index=index)
    mark = pd.Series([100.0] * 3, index=index)
    sigma = pd.Series([10.0] * 3, index=index)
    threshold_quantile = pd.Series([80.0] * 3, index=index)
    pool = {_DAY: np.array([-10.0, -5.0, -3.0, -1.0, 1.0, 5.0, 10.0])}

    base = risk_measures(
        price=price,
        mark=mark,
        sigma=sigma,
        pool=pool,
        variant="calibrated",
        side="long",
        level=0.95,
        threshold_quantile=threshold_quantile,
    )

    c = 5.0
    shifted = risk_measures(
        price=price,
        mark=mark + c,
        sigma=sigma,
        pool=pool,
        variant="calibrated",
        side="long",
        level=0.95,
        threshold_quantile=threshold_quantile,
    )

    pd.testing.assert_series_equal(base["breach"], shifted["breach"])
    np.testing.assert_allclose(shifted["var"].to_numpy(), base["var"].to_numpy() + QUANTITY_MWH * c)


# ---------------------------------------------------------------------------
# risk_measures: threshold_quantile fail-fast contract
# ---------------------------------------------------------------------------


def test_threshold_quantile_required_for_calibrated_and_raw() -> None:
    index = _single_day_index(2)
    price = pd.Series([90.0, 95.0], index=index)
    mark = pd.Series([100.0, 100.0], index=index)
    sigma = pd.Series([10.0, 10.0], index=index)
    pool = {_DAY: np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}

    for variant in ("calibrated", "raw"):
        with pytest.raises(ValueError, match="threshold_quantile"):
            risk_measures(
                price=price, mark=mark, sigma=sigma, pool=pool, variant=variant, side="long"
            )


def test_threshold_quantile_forbidden_for_fhs() -> None:
    index = _single_day_index(2)
    price = pd.Series([90.0, 95.0], index=index)
    mark = pd.Series([100.0, 100.0], index=index)
    sigma = pd.Series([10.0, 10.0], index=index)
    threshold_quantile = pd.Series([80.0, 80.0], index=index)
    pool = {_DAY: np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}

    with pytest.raises(ValueError, match="threshold_quantile"):
        risk_measures(
            price=price,
            mark=mark,
            sigma=sigma,
            pool=pool,
            variant="fhs",
            side="long",
            threshold_quantile=threshold_quantile,
        )


# ---------------------------------------------------------------------------
# risk_measures: coherence, symmetry, level guard, column contract, NaN safety
# ---------------------------------------------------------------------------


def test_coherence_es_at_least_var_on_random_pools() -> None:
    rng = np.random.default_rng(3)
    index = _single_day_index(1)
    mark = pd.Series([100.0], index=index)
    sigma = pd.Series([10.0], index=index)
    price = pd.Series([95.0], index=index)
    pool = {_DAY: np.sort(rng.normal(0, 1, 300))}

    for side in ("long", "short"):
        result = risk_measures(
            price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side=side, level=0.95
        )
        assert result["es"].iloc[0] >= result["var"].iloc[0] - 1e-9


def test_symmetric_pool_gives_equal_es_long_and_short_under_symmetric_clip() -> None:
    index = _single_day_index(1)
    mark = pd.Series([100.0], index=index)
    sigma = pd.Series([10.0], index=index)
    price = pd.Series([100.0], index=index)
    pool = {_DAY: np.array([-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0])}

    es_by_side = {}
    for side in ("long", "short"):
        result = risk_measures(
            price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side=side, level=0.95
        )
        # clip doesn't bind here (scenario prices stay well inside SDAC
        # limits) -- verified structurally, not assumed, so the symmetry
        # comparison below is meaningful.
        assert result["n_clipped"].iloc[0] == 0
        es_by_side[side] = result["es"].iloc[0]

    assert es_by_side["long"] == pytest.approx(es_by_side["short"])


def test_level_guard_raises_outside_0_90_to_0_95() -> None:
    index = _single_day_index(1)
    price = pd.Series([95.0], index=index)
    mark = pd.Series([100.0], index=index)
    sigma = pd.Series([10.0], index=index)
    pool = {_DAY: np.array([-1.0, 0.0, 1.0])}

    for level in (0.99, 0.80):
        with pytest.raises(ValueError, match="level"):
            risk_measures(
                price=price,
                mark=mark,
                sigma=sigma,
                pool=pool,
                variant="fhs",
                side="long",
                level=level,
            )


def test_risk_measures_column_contract() -> None:
    index = _single_day_index(2)
    price = pd.Series([90.0, 95.0], index=index)
    mark = pd.Series([100.0, 100.0], index=index)
    sigma = pd.Series([10.0, 10.0], index=index)
    pool = {_DAY: np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}

    result = risk_measures(
        price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side="long", level=0.95
    )

    assert list(result.columns) == [
        "u",
        "threshold_price",
        "var",
        "es",
        "es_unclipped",
        "pi",
        "n_pool",
        "n_tail",
        "n_clipped",
        "clip_impact_es",
        "breach",
    ]


# ---------------------------------------------------------------------------
# risk_measures: SDAC clip logic (spec 2.8, section 7's four clip tests)
# ---------------------------------------------------------------------------


def test_clip_invariance_when_scenarios_stay_within_limits() -> None:
    """No literal +-inf floor/cap is injectable (floor/cap come only from the
    real SDAC_PRICE_LIMITS table via sdac_limits_asof), so this constructs
    scenario prices that stay far inside the real limits: es must equal
    es_unclipped and n_clipped must be 0.
    """
    index = _single_day_index(1)
    price = pd.Series([95.0], index=index)
    mark = pd.Series([100.0], index=index)
    sigma = pd.Series([10.0], index=index)
    pool = {_DAY: np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}

    for side in ("long", "short"):
        result = risk_measures(
            price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side=side, level=0.95
        )
        assert result["n_clipped"].iloc[0] == 0
        assert result["es"].iloc[0] == pytest.approx(result["es_unclipped"].iloc[0])


def test_clip_binding_floor_above_whole_tail() -> None:
    """Floor strictly above every tail scenario price -> every tail scenario
    clips to the floor, so es_long == Q * (mark - floor) exactly and
    n_clipped == n_tail.
    """
    index = _single_day_index(1)  # 2021-06-01: floor=-500, cap=3000
    price = pd.Series([50.0], index=index)
    mark = pd.Series([100.0], index=index)
    sigma = pd.Series([300.0], index=index)
    threshold_quantile = pd.Series([-800.0], index=index)  # u = (-800-100)/300 = -3.0
    pool = {_DAY: np.array([-10.0, -8.0, -6.0, -4.0, -3.0, -1.0, 0.0, 2.0, 4.0])}

    result = risk_measures(
        price=price,
        mark=mark,
        sigma=sigma,
        pool=pool,
        variant="calibrated",
        side="long",
        level=0.95,
        threshold_quantile=threshold_quantile,
    )

    floor, _ = sdac_limits_asof(index)
    assert result["n_tail"].iloc[0] == 5  # {-10,-8,-6,-4,-3} <= u=-3.0
    assert result["n_clipped"].iloc[0] == 5
    assert result["es"].iloc[0] == pytest.approx(QUANTITY_MWH * (mark.iloc[0] - floor.iloc[0]))


def test_clip_monotonicity_tail_membership_unaffected_by_clip() -> None:
    """The tail is selected in r-space (r <= u for long) BEFORE any clipping
    of scenario prices -- n_tail (hence which pool elements are selected)
    must be identical regardless of which SDAC floor/cap regime applies,
    even though the resulting (clipped) es differs across regimes.
    """
    index_pre = pd.DatetimeIndex(["2021-06-01T10:00:00"], tz="UTC")  # floor=-500
    index_post = pd.DatetimeIndex(["2026-06-01T10:00:00"], tz="UTC")  # floor=-600
    pool_values = np.array([-10.0, -8.0, -6.0, -4.0, -3.0, -1.0, 0.0, 2.0, 4.0])

    def _run(index: pd.DatetimeIndex) -> pd.DataFrame:
        day = index.tz_convert("Europe/Berlin").normalize()[0].date()
        return risk_measures(
            price=pd.Series([50.0], index=index),
            mark=pd.Series([100.0], index=index),
            sigma=pd.Series([300.0], index=index),
            pool={day: pool_values},
            variant="calibrated",
            side="long",
            level=0.95,
            threshold_quantile=pd.Series([-800.0], index=index),
        )

    result_pre = _run(index_pre)
    result_post = _run(index_post)

    assert result_pre["n_tail"].iloc[0] == result_post["n_tail"].iloc[0] == 5
    # floor genuinely differs (-500 vs -600) -> the clipped es differs too,
    # proving the clip bound (not the tail selection) is what changed.
    assert result_pre["es"].iloc[0] != result_post["es"].iloc[0]


def test_cap_never_binds_at_realistic_short_side_magnitudes() -> None:
    """spec 2.8: with sigma <= ~145, mark <= ~950 and cap = 4000, the cap
    never binds on the short (upper-tail) side, even at the most extreme
    observed shock (r ~ 12.33, spec section 11).
    """
    index = pd.date_range("2023-06-01 00:00", periods=1, freq="h", tz="Europe/Berlin").tz_convert(
        "UTC"
    )  # floor=-500, cap=4000
    day = pd.Timestamp("2023-06-01", tz="Europe/Berlin").date()
    price = pd.Series([950.0], index=index)
    mark = pd.Series([950.0], index=index)
    sigma = pd.Series([145.0], index=index)
    pool = {day: np.array([-12.0, -5.0, -1.0, 0.0, 1.0, 5.0, 9.0, 10.0, 11.0, 12.33])}

    result = risk_measures(
        price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side="short", level=0.95
    )

    assert result["n_clipped"].iloc[0] == 0


# ---------------------------------------------------------------------------
# es_ratio_conditional: NOT mean_es / realised_es (spec 6)
# ---------------------------------------------------------------------------


def test_es_ratio_conditional_is_not_mean_es_over_realised_es() -> None:
    """spec 6: es_ratio_conditional is NOT mean_es / realised_es. On a sigma
    in {1, 10} frame where the realised loss on every breach hour is set to
    exactly equal that hour's ES ("perfectly calibrated"), es_ratio_conditional
    must be exactly 1, while mean_es / realised_es -- a pure sigma-mixture
    artefact -- is provably NOT 1.
    """
    index = pd.DatetimeIndex(["2021-06-01T10:00:00", "2021-06-01T11:00:00"], tz="UTC")
    day = pd.Timestamp("2021-06-01", tz="Europe/Berlin").date()
    mark = pd.Series([100.0, 100.0], index=index)
    sigma = pd.Series([1.0, 10.0], index=index)
    # u = (threshold_quantile - mark) / sigma == -2.0 for both hours.
    threshold_quantile = mark + sigma * (-2.0)
    pool = {day: np.array([-10.0, -2.0, 5.0])}  # tail (<= -2.0) = {-10.0, -2.0}, mean = -6.0

    # Realised price set so the realised loss exactly equals that hour's ES
    # (the mean tail scenario price): price = mark - sigma * 6.0.
    price = mark - sigma * 6.0

    risk = risk_measures(
        price=price,
        mark=mark,
        sigma=sigma,
        pool=pool,
        variant="calibrated",
        side="long",
        level=0.95,
        threshold_quantile=threshold_quantile,
    )
    assert (risk["breach"] == 1.0).all()  # sanity: both hours actually breach

    pnl = book_pnl(price, mark, side="long")

    assert es_ratio_conditional(pnl, risk) == pytest.approx(1.0)

    mean_es = float(risk["es"].mean())
    realised = realised_es(pnl, level=0.95)
    assert mean_es / realised["es"] != pytest.approx(1.0)


def test_risk_measures_empty_pool_or_missing_sigma_yields_nan_no_crash() -> None:
    index = _single_day_index(2)
    price = pd.Series([90.0, 95.0], index=index)
    mark = pd.Series([100.0, 100.0], index=index)
    sigma = pd.Series([10.0, float("nan")], index=index)
    pool: dict = {}  # no pool at all for this day

    result = risk_measures(
        price=price, mark=mark, sigma=sigma, pool=pool, variant="fhs", side="long", level=0.95
    )

    assert result["n_pool"].eq(0).all()
    assert result[["u", "threshold_price", "var", "es", "es_unclipped", "pi"]].isna().all().all()
