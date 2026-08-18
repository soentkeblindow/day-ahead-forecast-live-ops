"""Backtest validation test statistics for the trading-book risk measures (Sprint 4.4b).

4.4a computed VaR/ES point estimates (``breach_rate``, ``es_ratio_conditional``)
with no null distribution. This module supplies the three test statistics that
turn those point estimates into a validation verdict: Kupiec POF (frequency,
unconditional coverage), Christoffersen ``LR_ind`` (serial independence /
clumping), and Acerbi-Szekely ``Z1``/``Z2`` (severity of the expected shortfall).
Every function here is a PURE statistic on a realised series -- no resampling,
no simulation. Significance for the coverage/magnitude family comes from the
day-block bootstrap in ``evaluation.bootstrap``; the independence family uses
its own asymptotic chi-square distribution and is never bootstrapped (see
``evaluation.bootstrap`` module docstring).

Reference formulas are fixed in the spec (4.4b spec section 5.0) to avoid the
"self-consistent wrong test" trap from 4.4a: several textbook variants of these
statistics exist with different sign/normalisation conventions, and this module
implements exactly one of them, documented inline. All p-values use
``scipy.stats.chi2.sf(stat, df=1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from ..market_time import LOCAL_TZ


def kupiec_pof(breach: pd.Series, *, level: float) -> dict[str, float]:
    """Kupiec proportion-of-failures test (unconditional coverage, hourly).

    breach: boolean (or {0.0, 1.0, NaN}) series, True/1.0 == VaR breach, one
    row per hour. NaN rows (e.g. missing sigma) are dropped before counting.
    Returns {"lr": ..., "pvalue": ..., "breach_rate": ..., "n": ..., "n_breach": ...}.
    lr ~ chi2_1 under H0 (true breach probability == 1 - level). The chi2 is
    ANTI-CONSERVATIVE here because intra-day breaches are correlated -- the
    honest interval on breach_rate comes from the stratified day-block
    bootstrap, not from this p-value. Report both; let the bootstrap CI carry
    the verdict.
    """
    valid = breach.dropna().astype(bool)
    t = int(valid.shape[0])
    x = int(valid.sum())
    p = 1.0 - level

    if t == 0:
        return {
            "lr": float("nan"),
            "pvalue": float("nan"),
            "breach_rate": float("nan"),
            "n": 0.0,
            "n_breach": 0.0,
        }

    p_hat = x / t
    term1 = x * np.log(p_hat / p) if x > 0 else 0.0
    term2 = (t - x) * np.log((1.0 - p_hat) / (1.0 - p)) if x < t else 0.0
    lr = 2.0 * (term1 + term2)
    return {
        "lr": float(lr),
        "pvalue": float(chi2.sf(lr, df=1)),
        "breach_rate": float(p_hat),
        "n": float(t),
        "n_breach": float(x),
    }


def day_breach_series(
    breach: pd.Series, *, k: int, local_tz: str = LOCAL_TZ, min_valid_hours: int = 1
) -> pd.Series:
    """Aggregate the hourly breach series to a per-delivery-day breach indicator.

    A delivery day (Europe/Berlin, `local_tz`) is a breach-day iff at least `k`
    of its VALID hours breached. Days with fewer than `min_valid_hours` valid
    hours (e.g. the ~15 uncalibrated start days of 4.4a) are DROPPED, not
    counted as non-breach. Index: one row per delivery day, boolean.
    """
    idx = pd.DatetimeIndex(breach.index)
    local_day = idx.tz_convert(local_tz).normalize()

    valid_mask = breach.notna()
    is_breach = breach == 1.0  # NaN compares False; bool True also == 1.0

    frame = pd.DataFrame({"valid": valid_mask.to_numpy(), "breach": is_breach.to_numpy()})
    frame.index = local_day

    n_valid = frame["valid"].groupby(level=0).sum()
    n_breach = frame["breach"].groupby(level=0).sum()

    keep = n_valid >= min_valid_hours
    result = (n_breach >= k) & keep
    result = result[keep].sort_index()
    result.index.name = None
    return result


def christoffersen_independence(breach_indicator: pd.Series) -> dict[str, float]:
    """Christoffersen independence test (LR_ind) on a breach-indicator series.

    Works on ANY level's indicator series: hourly (diagnostic -- rejection
    expected, a calendar artefact) or daily (the headline -- regime
    persistence). Builds the 2x2 first-order Markov transition counts (n00,
    n01, n10, n11) and returns LR_ind ~ chi2_1 under H0 (breaches are serially
    independent). Needs NO target rate (pi_hat is the series' own empirical
    rate). By convention 0 * ln(0) := 0 when a transition cell is empty.
    Degenerate (no breaches, all breaches, or no state-1 predecessor) ->
    lr := 0, pvalue := 1. Returns
    {"lr": ..., "pvalue": ..., "n00": ..., "n01": ..., "n10": ..., "n11": ...}.
    """
    b = breach_indicator.dropna().astype(bool).astype(int).to_numpy()

    n00 = n01 = n10 = n11 = 0
    if b.size >= 2:
        prev, curr = b[:-1], b[1:]
        n00 = int(np.sum((prev == 0) & (curr == 0)))
        n01 = int(np.sum((prev == 0) & (curr == 1)))
        n10 = int(np.sum((prev == 1) & (curr == 0)))
        n11 = int(np.sum((prev == 1) & (curr == 1)))

    total = n00 + n01 + n10 + n11
    n1_row = n10 + n11
    pi = (n01 + n11) / total if total > 0 else float("nan")

    if total == 0 or pi in (0.0, 1.0) or n1_row == 0:
        return {
            "lr": 0.0,
            "pvalue": 1.0,
            "n00": float(n00),
            "n01": float(n01),
            "n10": float(n10),
            "n11": float(n11),
        }

    n0_row = n00 + n01
    pi01 = n01 / n0_row if n0_row > 0 else float("nan")
    pi11 = n11 / n1_row

    def _term(n: int, num: float, den: float) -> float:
        return 0.0 if n == 0 else n * float(np.log(num / den))

    lr = 2.0 * (
        _term(n01, pi01, pi)
        + _term(n11, pi11, pi)
        + _term(n00, 1.0 - pi01, 1.0 - pi)
        + _term(n10, 1.0 - pi11, 1.0 - pi)
    )
    return {
        "lr": float(lr),
        "pvalue": float(chi2.sf(lr, df=1)),
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }


def acerbi_szekely_z1(loss: pd.Series, es: pd.Series, breach: pd.Series) -> float:
    """Acerbi-Szekely conditional statistic Z1 = mean(loss/es | breach) - 1.

    Identical to `es_ratio_conditional - 1` from 4.4a (evaluation.risk). Tests
    ES SEVERITY given the breaches. Z1 > 0: ES too small (underestimate);
    Z1 < 0: ES too conservative. Point estimate only -- significance comes
    from the day-block bootstrap. `breach` is compared with `== 1.0` (matches
    both boolean True and the float {0.0, 1.0, NaN} risk_hourly convention;
    NaN compares False, so missing-sigma rows never enter the tail mean).
    """
    is_breach = breach == 1.0
    valid = is_breach & es.notna() & (es != 0)
    if not bool(valid.any()):
        return float("nan")
    ratio = loss.loc[valid] / es.loc[valid]
    return float(ratio.mean() - 1.0)


def acerbi_szekely_z2(loss: pd.Series, es: pd.Series, breach: pd.Series, *, level: float) -> float:
    """Acerbi-Szekely unconditional statistic Z2 (frequency AND severity).

    Z2 = sum_t[ breach_t * loss_t / es_t ] / (T * (1 - level)) - 1. Divides by
    the EXPECTED breach count (T * alpha), not the observed one, so a wrong
    frequency and a wrong severity both push it away from 0. `T` is the full
    length of `breach` (every hour in the sample, not just breach hours).
    Breach hours whose `es` is NaN or zero contribute 0 (their tail estimate
    is unusable, not fabricated). Point estimate only -- see the bootstrap.
    """
    t = int(breach.shape[0])
    alpha = 1.0 - level
    is_breach = breach == 1.0
    usable = is_breach & es.notna() & (es != 0)

    contribution = pd.Series(0.0, index=loss.index)
    contribution.loc[usable] = loss.loc[usable] / es.loc[usable]

    denom = t * alpha
    if denom == 0:
        return float("nan")
    return float(contribution.sum() / denom - 1.0)


def basel_traffic_light(
    breach: pd.Series,
    *,
    level: float,
    window_days: int,
    drop_partial: bool,
    local_tz: str = LOCAL_TZ,
) -> dict[str, object]:
    """Basel-style traffic light over NON-OVERLAPPING windows (Nachtrag 1, part B).

    Cuts the delivery-day sequence (Europe/Berlin, `local_tz`; days with zero
    valid hours -- e.g. the pre-warmup start -- are dropped, not counted as
    empty windows) into consecutive, DISJOINT windows of `window_days`
    delivery days. An incomplete trailing window is dropped when
    `drop_partial` (matches the actual Basel convention: a shorter window
    would need different zone boundaries, since those depend on the window's
    own hour count). Zone per window from the cumulative binomial at the
    ACTUAL alpha (= 1 - level), evaluated against that window's own valid-hour
    count: green cdf < 0.95, yellow 0.95 <= cdf < 0.9999, red otherwise.

    NO max-over-windows: taking the worst of the ~n_hours-many overlapping
    windows a ROLLING scheme would produce is a selection-bias machine (it
    hunts for the alignment that best captures a random breach cluster) and
    is NOT the Basel convention, which reports the state over the LAST
    window. The headline is therefore `latest_zone` -- the zone of the most
    recent COMPLETE window; `n_yellow`/`n_red` and the per-window list carry
    the rest, for anyone who wants the stricter view without it being sold as
    THE verdict.

    Run this ONLY on the full hourly series, NEVER on a conditioning subset:
    the window is a CALENDAR construct, and slicing it to a subset destroys
    that meaning (250 days of evening-ramp hours are ~750 hours, not 6,000; a
    `forecast_renewable_surplus` "window" would span years of spring/summer
    days). Kupiec already answers the conditional frequency question
    properly, with a bootstrap CI.

    CAVEAT (state it, do not hide it): the binomial zone bounds assume
    INDEPENDENT hours. Ours are not -- intra-day breaches cluster (the hourly
    `chris_ind_lr` is large by construction, spec 2.4), so the breach count
    is OVERDISPERSED relative to binomial and the zone bounds are TOO TIGHT.
    A perfectly calibrated model will show yellow too often. Fixing this
    (day-based zone boundaries instead of hourly-binomial ones) is explicit
    future work (Nachtrag 1, section G), not done here. Illustrative, not a
    regulatory verdict; the statistically clean tools for the same question
    are Kupiec and Christoffersen.

    Returns {"latest_zone": str | None, "n_windows": int, "n_yellow": int,
    "n_red": int, "windows": list[dict], "illustrative": True}. Each entry of
    `windows` is {"window_index", "window_start", "window_end", "n_hours",
    "n_breach", "cdf", "zone"}.
    """
    alpha = 1.0 - level
    idx = pd.DatetimeIndex(breach.index)
    local_day = idx.tz_convert(local_tz).normalize()
    is_breach = breach == 1.0  # NaN compares False; bool True also == 1.0

    frame = pd.DataFrame(
        {"valid": breach.notna().to_numpy(), "breach": is_breach.to_numpy()}, index=local_day
    )
    n_valid = frame["valid"].groupby(level=0).sum()
    n_breach = frame["breach"].groupby(level=0).sum()
    days = n_valid[n_valid > 0].index.sort_values()

    n_days_total = len(days)
    n_full_windows = n_days_total // window_days
    day_chunks = [days[w * window_days : (w + 1) * window_days] for w in range(n_full_windows)]
    remainder = n_days_total - n_full_windows * window_days
    if remainder > 0 and not drop_partial:
        day_chunks.append(days[n_full_windows * window_days :])

    windows: list[dict[str, object]] = []
    for w, day_slice in enumerate(day_chunks):
        n_hours = int(n_valid.loc[day_slice].sum())
        n_breach_w = int(n_breach.loc[day_slice].sum())
        cdf = float(binom.cdf(n_breach_w, n_hours, alpha))
        zone = "green" if cdf < 0.95 else ("yellow" if cdf < 0.9999 else "red")
        windows.append(
            {
                "window_index": w,
                "window_start": pd.Timestamp(day_slice[0]),
                "window_end": pd.Timestamp(day_slice[-1]),
                "n_hours": n_hours,
                "n_breach": n_breach_w,
                "cdf": cdf,
                "zone": zone,
            }
        )

    n_yellow = sum(1 for w_ in windows if w_["zone"] == "yellow")
    n_red = sum(1 for w_ in windows if w_["zone"] == "red")
    latest_zone = windows[-1]["zone"] if windows else None

    return {
        "latest_zone": latest_zone,
        "n_windows": len(windows),
        "n_yellow": n_yellow,
        "n_red": n_red,
        "windows": windows,
        "illustrative": True,
    }
