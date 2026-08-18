import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2

from energy_price_forecast.evaluation.backtest import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    day_breach_series,
    kupiec_pof,
)
from energy_price_forecast.evaluation.risk import es_ratio_conditional


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# kupiec_pof
# ---------------------------------------------------------------------------


def test_kupiec_pof_hand_computed_lr() -> None:
    # T=100, x=10, p=0.05: hand-computed LR_uc.
    breach = pd.Series([True] * 10 + [False] * 90)
    result = kupiec_pof(breach, level=0.95)

    p, p_hat, t, x = 0.05, 0.10, 100, 10
    expected_lr = 2.0 * (x * np.log(p_hat / p) + (t - x) * np.log((1.0 - p_hat) / (1.0 - p)))
    assert result["lr"] == pytest.approx(expected_lr)
    assert result["breach_rate"] == pytest.approx(0.10)
    assert result["n"] == 100.0
    assert result["n_breach"] == 10.0
    assert result["pvalue"] == pytest.approx(float(chi2.sf(expected_lr, df=1)))


def test_kupiec_pof_perfectly_calibrated_series_has_lr_near_zero() -> None:
    rng = np.random.default_rng(0)
    breach = pd.Series(rng.random(100_000) < 0.05)
    result = kupiec_pof(breach, level=0.95)
    assert result["lr"] < 5.0  # well under the chi2_1 95th percentile (~3.84), loosely


def test_kupiec_pof_severe_miscoverage_has_large_lr() -> None:
    breach = pd.Series([True] * 30 + [False] * 70)  # 30% vs nominal 5%
    result = kupiec_pof(breach, level=0.95)
    assert result["lr"] > 50.0
    assert result["pvalue"] < 0.001


def test_kupiec_pof_drops_nan_before_counting() -> None:
    breach = pd.Series([1.0, 0.0, np.nan, np.nan, 1.0])
    result = kupiec_pof(breach, level=0.95)
    assert result["n"] == 3.0
    assert result["n_breach"] == 2.0


# ---------------------------------------------------------------------------
# day_breach_series: k=2 aggregation, LOCAL_TZ edge, dropped short days
# ---------------------------------------------------------------------------


def test_day_breach_series_k2_single_hour_is_not_a_breach_day() -> None:
    index = _hourly_utc("2021-06-01 00:00", 24)  # single Berlin day (June, no DST edge)
    breach = pd.Series([False] * 24, index=index)
    breach.iloc[5] = True  # one breaching hour only

    result = day_breach_series(breach, k=2, local_tz="Europe/Berlin", min_valid_hours=1)
    assert result.sum() == 0


def test_day_breach_series_k2_two_hours_is_a_breach_day() -> None:
    index = _hourly_utc("2021-06-01 00:00", 24)
    breach = pd.Series([False] * 24, index=index)
    breach.iloc[5] = True
    breach.iloc[19] = True

    result = day_breach_series(breach, k=2, local_tz="Europe/Berlin", min_valid_hours=1)
    assert result.sum() == 1
    assert bool(result.iloc[0])


def test_day_breach_series_drops_days_with_too_few_valid_hours() -> None:
    index = _hourly_utc("2021-06-01 00:00", 48)
    breach = pd.Series([0.0] * 48, index=index)
    breach.iloc[0:10] = np.nan  # first local delivery day: most hours invalid
    breach.iloc[12] = 1.0
    breach.iloc[13] = 1.0  # first day WOULD be a breach day if kept

    result = day_breach_series(breach, k=2, local_tz="Europe/Berlin", min_valid_hours=20)
    # first day dropped entirely (too few valid hours), not counted as non-breach
    assert len(result) == 1


def test_day_breach_series_uses_local_tz_delivery_day_boundary() -> None:
    # 22:00-23:00 UTC on 2021-06-01 is already 2021-06-02 00:00/01:00 in Berlin (CEST, +2).
    index = _hourly_utc("2021-06-01 22:00", 4)
    breach = pd.Series([True, True, False, False], index=index)  # both breaches land 00/01 local

    result = day_breach_series(breach, k=2, local_tz="Europe/Berlin", min_valid_hours=1)
    assert len(result) == 1
    local_day = result.index[0]
    assert local_day == pd.Timestamp("2021-06-02", tz="Europe/Berlin")
    assert bool(result.iloc[0])


# ---------------------------------------------------------------------------
# christoffersen_independence
# ---------------------------------------------------------------------------


def test_christoffersen_independence_hand_checked_lr() -> None:
    # A mixed sequence with a known, non-trivial transition structure. Expected
    # counts are derived independently below (same recurrence as the
    # implementation) and cross-checked against the closed-form LR formula.
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 0, 1, 0, 1, 0] + [1, 1, 1, 1, 1])
    breach = pd.Series(b.astype(bool))
    result = christoffersen_independence(breach)

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    assert result["n00"] == n00
    assert result["n01"] == n01
    assert result["n10"] == n10
    assert result["n11"] == n11

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    expected_lr = 2.0 * (
        n01 * np.log(pi01 / pi)
        + n11 * np.log(pi11 / pi)
        + n00 * np.log((1 - pi01) / (1 - pi))
        + n10 * np.log((1 - pi11) / (1 - pi))
    )
    assert result["lr"] == pytest.approx(expected_lr)
    assert result["pvalue"] == pytest.approx(float(chi2.sf(expected_lr, df=1)))


def test_christoffersen_independence_iid_series_has_lr_near_zero() -> None:
    rng = np.random.default_rng(1)
    breach = pd.Series(rng.random(100_000) < 0.05)
    result = christoffersen_independence(breach)
    assert result["lr"] < 5.0


def test_christoffersen_independence_clumped_series_has_large_lr() -> None:
    # Breaches occur in long runs -> strong serial dependence.
    block = [True] * 20 + [False] * 20
    breach = pd.Series(block * 200)
    result = christoffersen_independence(breach)
    assert result["lr"] > 50.0
    assert result["pvalue"] < 0.001


def test_christoffersen_independence_no_state_one_predecessor_does_not_crash() -> None:
    # Exactly one breach, at the very end -> n10 + n11 == 0 (degenerate).
    breach = pd.Series([False] * 10 + [True])
    result = christoffersen_independence(breach)
    assert result["lr"] == 0.0
    assert result["pvalue"] == 1.0
    assert result["n11"] == 0.0


def test_christoffersen_independence_pvalue_uses_df_one() -> None:
    breach = pd.Series([True, False] * 500)  # perfectly alternating -> strong dependence
    result = christoffersen_independence(breach)
    assert result["pvalue"] == pytest.approx(float(chi2.sf(result["lr"], df=1)))


# ---------------------------------------------------------------------------
# acerbi_szekely_z1: identity with es_ratio_conditional (4.4a), hand calc
# ---------------------------------------------------------------------------


def test_acerbi_szekely_z1_identity_with_es_ratio_conditional() -> None:
    pnl = pd.Series([-30.0, -5.0, 20.0, -50.0, -10.0])
    es = pd.Series([25.0, 10.0, 10.0, 40.0, np.nan])
    breach = pd.Series([1.0, 0.0, 0.0, 1.0, 1.0])

    risk = pd.DataFrame({"breach": breach, "es": es})
    expected = es_ratio_conditional(pnl, risk) - 1.0

    actual = acerbi_szekely_z1(-pnl, es, breach)
    assert actual == pytest.approx(expected)


def test_acerbi_szekely_z1_hand_computed() -> None:
    loss = pd.Series([30.0, 50.0, 10.0])
    es = pd.Series([25.0, 40.0, 100.0])
    breach = pd.Series([1.0, 1.0, 0.0])

    result = acerbi_szekely_z1(loss, es, breach)
    expected = np.mean([30.0 / 25.0, 50.0 / 40.0]) - 1.0
    assert result == pytest.approx(expected)


def test_acerbi_szekely_z1_no_breaches_returns_nan() -> None:
    loss = pd.Series([1.0, 2.0])
    es = pd.Series([10.0, 10.0])
    breach = pd.Series([0.0, 0.0])
    assert np.isnan(acerbi_szekely_z1(loss, es, breach))


# ---------------------------------------------------------------------------
# acerbi_szekely_z2: sign convention, division by T*alpha
# ---------------------------------------------------------------------------


def test_acerbi_szekely_z2_underestimated_es_is_positive() -> None:
    # ES systematically too small relative to realised loss on breach hours.
    t = 100
    loss = pd.Series([0.0] * t)
    es = pd.Series([10.0] * t)
    breach = pd.Series([False] * t)
    loss.iloc[:5] = 50.0  # 5 breaches, loss >> es
    breach.iloc[:5] = True

    z2 = acerbi_szekely_z2(loss, es, breach, level=0.95)
    assert z2 > 0.0


def test_acerbi_szekely_z2_conservative_es_is_negative() -> None:
    t = 100
    loss = pd.Series([0.0] * t)
    es = pd.Series([100.0] * t)
    breach = pd.Series([False] * t)
    loss.iloc[:5] = 10.0  # loss << es
    breach.iloc[:5] = True

    z2 = acerbi_szekely_z2(loss, es, breach, level=0.95)
    assert z2 < 0.0


def test_acerbi_szekely_z2_hand_computed_denominator() -> None:
    t = 20
    alpha = 0.05
    loss = pd.Series([0.0] * t)
    es = pd.Series([10.0] * t)
    breach = pd.Series([False] * t)
    loss.iloc[0] = 20.0
    breach.iloc[0] = True

    z2 = acerbi_szekely_z2(loss, es, breach, level=1.0 - alpha)
    expected = (20.0 / 10.0) / (t * alpha) - 1.0
    assert z2 == pytest.approx(expected)


# ---------------------------------------------------------------------------
# basel_traffic_light (Nachtrag 1, part B: non-overlapping windows)
# ---------------------------------------------------------------------------


def _windows(result: dict[str, object]) -> list[dict[str, object]]:
    windows = result["windows"]
    assert isinstance(windows, list)
    return windows


def test_basel_traffic_light_known_breach_count_known_zone() -> None:
    window_days = 10
    index = _hourly_utc("2021-01-01 00:00", window_days * 24)
    breach = pd.Series([0.0] * len(index), index=index)
    breach.iloc[:2] = 1.0  # 2 breaches out of 240 hours, alpha=0.05 -> green

    result = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=True, local_tz="UTC"
    )
    assert result["n_windows"] == 1
    assert result["latest_zone"] == "green"
    assert result["illustrative"] is True
    window = _windows(result)[0]
    assert window["n_breach"] == 2
    assert window["n_hours"] == window_days * 24
    assert window["zone"] == "green"


def test_basel_traffic_light_high_breach_count_is_red() -> None:
    window_days = 10
    index = _hourly_utc("2021-01-01 00:00", window_days * 24)
    breach = pd.Series([0.0] * len(index), index=index)
    breach.iloc[:60] = 1.0  # 60/240 = 25% vs nominal 5% -> deep red

    result = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=True, local_tz="UTC"
    )
    assert result["latest_zone"] == "red"


def test_basel_traffic_light_windows_are_disjoint_and_gapless() -> None:
    window_days = 5
    n_windows = 4
    index = _hourly_utc("2021-01-01 00:00", n_windows * window_days * 24)
    breach = pd.Series([0.0] * len(index), index=index)

    result = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=True, local_tz="UTC"
    )
    assert result["n_windows"] == n_windows
    windows = _windows(result)
    starts = [pd.Timestamp(w["window_start"]) for w in windows]  # type: ignore[arg-type]
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + pd.Timedelta(days=window_days)
    assert all(w["n_hours"] == window_days * 24 for w in windows)


def test_basel_traffic_light_drops_partial_trailing_window_by_default() -> None:
    window_days = 10
    index = _hourly_utc("2021-01-01 00:00", 25 * 24)  # 2 full windows + a 5-day remainder
    breach = pd.Series([0.0] * len(index), index=index)

    dropped = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=True, local_tz="UTC"
    )
    assert dropped["n_windows"] == 2

    kept = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=False, local_tz="UTC"
    )
    assert kept["n_windows"] == 3
    assert _windows(kept)[-1]["n_hours"] == 5 * 24


def test_basel_traffic_light_latest_zone_is_last_window_not_worst() -> None:
    # The exact case the old worst-over-all-windows logic got wrong: an
    # early window is red, the LATEST is green -> latest_zone must be green.
    window_days = 10
    window_hours = window_days * 24
    index = _hourly_utc("2021-01-01 00:00", 2 * window_hours)
    breach = pd.Series([0.0] * len(index), index=index)
    breach.iloc[:60] = 1.0  # first window: 60/240 = 25% -> red; second window: all-zero -> green

    result = basel_traffic_light(
        breach, level=0.95, window_days=window_days, drop_partial=True, local_tz="UTC"
    )
    assert result["n_windows"] == 2
    windows = _windows(result)
    assert windows[0]["zone"] == "red"
    assert windows[1]["zone"] == "green"
    assert result["latest_zone"] == "green"
    assert result["n_red"] == 1
    assert result["n_yellow"] == 0


def test_basel_traffic_light_zone_boundaries_exact_at_binomial_cutoffs() -> None:
    window_days = 10
    window_hours = window_days * 24
    alpha = 0.05
    b_yellow = next(b for b in range(window_hours + 1) if binom.cdf(b, window_hours, alpha) >= 0.95)
    b_red = next(b for b in range(window_hours + 1) if binom.cdf(b, window_hours, alpha) >= 0.9999)

    def _zone_for(n_breach: int) -> object:
        index = _hourly_utc("2021-01-01 00:00", window_hours)
        breach = pd.Series([0.0] * window_hours, index=index)
        breach.iloc[:n_breach] = 1.0
        result = basel_traffic_light(
            breach, level=1.0 - alpha, window_days=window_days, drop_partial=True, local_tz="UTC"
        )
        return _windows(result)[0]["zone"]

    assert _zone_for(b_yellow - 1) == "green"
    assert _zone_for(b_yellow) == "yellow"
    assert _zone_for(b_red - 1) == "yellow"
    assert _zone_for(b_red) == "red"


def test_basel_traffic_light_return_dict_has_no_worst_zone_key() -> None:
    index = _hourly_utc("2021-01-01 00:00", 240)
    breach = pd.Series([0.0] * 240, index=index)
    result = basel_traffic_light(
        breach, level=0.95, window_days=10, drop_partial=True, local_tz="UTC"
    )
    assert "zone" not in result
    assert "worst_zone" not in result
    assert "window_end" not in result
    assert "breach_count" not in result
    assert "latest_zone" in result


def test_basel_traffic_light_insufficient_data_returns_no_windows() -> None:
    index = _hourly_utc("2021-01-01 00:00", 24 * 3)  # only 3 valid days, window_days=10
    breach = pd.Series([0.0] * len(index), index=index)
    result = basel_traffic_light(
        breach, level=0.95, window_days=10, drop_partial=True, local_tz="UTC"
    )
    assert result["latest_zone"] is None
    assert result["n_windows"] == 0
    assert result["windows"] == []
