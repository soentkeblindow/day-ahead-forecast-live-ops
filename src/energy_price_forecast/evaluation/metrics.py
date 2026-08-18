import numpy as np
import pandas as pd


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align on the index and drop rows where either side is NaN."""
    df = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    return df["t"], df["p"]


def _aligned3(
    y_true: pd.Series, lower: pd.Series, upper: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Align three series on the index and drop rows where any value is NaN."""
    df = pd.concat([y_true.rename("t"), lower.rename("lo"), upper.rename("hi")], axis=1).dropna()
    return df["t"], df["lo"], df["hi"]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error (EUR/MWh)."""
    t, p = _aligned(y_true, y_pred)
    return float((t - p).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error (EUR/MWh)."""
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(((t - p) ** 2).mean()))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y_true|.

    The aggregated denominator makes WAPE robust where individual prices are
    near zero — unlike MAPE, which divides per observation. |y_true| keeps
    negative prices from cancelling in the denominator. Returns NaN if the
    denominator is 0.
    """
    t, p = _aligned(y_true, y_pred)
    denom = t.abs().sum()
    return float((t - p).abs().sum() / denom) if denom > 0 else float("nan")


def pinball(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> float:
    """Mean pinball loss (quantile loss) at level alpha.

    For each pair the loss is alpha * (y - q) when y >= q, else
    (1 - alpha) * (q - y), where q is the alpha-quantile prediction. The
    minimiser of the expected pinball loss is the alpha-quantile, which is why
    it is the shared probabilistic score. Sanity anchor: at alpha=0.5 it equals
    0.5 * MAE.
    """
    t, p = _aligned(y_true, y_pred)
    err = t - p
    return float(np.maximum(alpha * err, (alpha - 1.0) * err).mean())


def quantile_coverage(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Empirical one-sided coverage P(y_true <= q_pred) for an alpha-quantile.

    For a well-calibrated alpha-quantile this should sit close to alpha.
    """
    t, p = _aligned(y_true, y_pred)
    return float((t <= p).mean())


def interval_coverage(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> float:
    """Empirical coverage of [lower, upper]: P(lower <= y_true <= upper).

    For a nominal 90% band (q0.05 / q0.95) this should sit close to 0.90.
    """
    t, lo, hi = _aligned3(y_true, lower, upper)
    return float(((t >= lo) & (t <= hi)).mean())


def interval_width(lower: pd.Series, upper: pd.Series) -> float:
    """Mean interval width (upper - lower): a sharpness measure.

    Lower is sharper, but only meaningful read together with coverage
    (reliability vs sharpness trade-off).
    """
    lo, hi = _aligned(lower, upper)
    return float((hi - lo).mean())


def quantile_crossing_rate(
    q_low: pd.Series, q_mid: pd.Series, q_high: pd.Series
) -> dict[str, float]:
    """Fraction of timestamps where the ordering q_low <= q_mid <= q_high breaks.

    A pure diagnostic (decision 3): crossings are reported, never corrected. The
    three quantile models share identical (untuned) hyperparameters but are fitted
    independently per alpha, so a small, stable crossing rate is expected. Returns
    the overall violation rate plus its two components, so the caller can see WHERE
    it crosses. Each sample where BOTH q_low > q_mid AND q_mid > q_high holds is
    counted ONCE in crossing_rate (OR, not double-counted).
    """
    lo, mid, hi = _aligned3(q_low, q_mid, q_high)
    low_above_mid = lo > mid
    mid_above_high = mid > hi
    any_violation = low_above_mid | mid_above_high
    return {
        "crossing_rate": float(any_violation.mean()),
        "crossing_low_above_mid": float(low_above_mid.mean()),
        "crossing_mid_above_high": float(mid_above_high.mean()),
    }


def summarise_quantiles(
    y_true: pd.Series,
    preds: dict[float, pd.Series],
    *,
    levels: tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> dict[str, float]:
    """Consolidate three quantile forecasts into the decision-6 KPIs.

    `preds` maps each alpha level to its prediction series (one per backtest run).
    Per the no-single-headline rule (decision 6) each metric is returned SEPARATELY
    per level / band, never averaged into a one-number score:
      - pinball_{a:.2f}      : shared probabilistic score at level a
      - coverage_{a:.2f}     : P(y_true <= q_a); well-calibrated ~ a
      - interval_coverage_90 : P(q_low <= y_true <= q_high); target ~ 0.90
      - interval_width_90    : mean(q_high - q_low); sharpness (lower = sharper)
      - crossing_rate (+ two components) : pure ordering diagnostic (decision 3)

    NaN handling: each sub-metric drops its own NaN pairs independently via
    _aligned / _aligned3, so sample sizes may differ slightly across metrics.
    """
    low, mid, high = levels
    out: dict[str, float] = {}
    for a in levels:
        out[f"pinball_{a:.2f}"] = pinball(y_true, preds[a], a)
        out[f"coverage_{a:.2f}"] = quantile_coverage(y_true, preds[a])
    out["interval_coverage_90"] = interval_coverage(y_true, preds[low], preds[high])
    out["interval_width_90"] = interval_width(preds[low], preds[high])
    out.update(quantile_crossing_rate(preds[low], preds[mid], preds[high]))
    return out


def summarise(predictions: pd.DataFrame) -> dict[str, float]:
    """Pooled MAE/RMSE/WAPE over all test rows, plus per-delivery-day MAE
    distribution (mean/std/p05/p50/p95) as a stability read. NaN pairs dropped.
    """
    t, p = predictions["y_true"], predictions["y_pred"]
    per_day = (
        predictions.dropna(subset=["y_true", "y_pred"])
        .assign(abs_err=lambda df: (df["y_true"] - df["y_pred"]).abs())
        .groupby("delivery_day")["abs_err"]
        .mean()
    )
    return {
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "wape": wape(t, p),
        "mae_per_day_mean": float(per_day.mean()),
        "mae_per_day_std": float(per_day.std()),
        "mae_per_day_p05": float(per_day.quantile(0.05)),
        "mae_per_day_p50": float(per_day.quantile(0.50)),
        "mae_per_day_p95": float(per_day.quantile(0.95)),
    }
