"""Reliability curve and nested-band metrics for the extended quantile grid (Sprint 4.3a).

DIAGNOSTIC ONLY. This module measures the RAW quantile forecasts (empirical
vs nominal coverage) and corrects nothing -- recalibration is step 4.3b. It
reuses the atomic metrics from ``.metrics`` and adds two things they do not
provide: level-spanning aggregation across the extended {0.05..0.95} grid,
and slicing by forecast-level bucket (as opposed to ``breakdown.py``, which
slices by realised-market regime).
"""

from __future__ import annotations

import pandas as pd

from .metrics import interval_coverage, interval_width, pinball, quantile_coverage

# Fixed, interpretable forecast-level buckets (EUR/MWh), keyed off the median
# forecast q_0.50 -- known ex ante, hence leakage-free. Edges are constants
# (not data-derived percentiles) to stay reproducible and look-ahead-free.
FORECAST_LEVEL_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("negative", float("-inf"), 0.0),
    ("0_50", 0.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_150", 100.0, 150.0),
    ("150_250", 150.0, 250.0),
    ("250_400", 250.0, 400.0),
    ("400_plus", 400.0, float("inf")),
)

# Central bands available from the extended grid: (name, lower level, upper level).
NESTED_BANDS: tuple[tuple[str, float, float], ...] = (
    ("90", 0.05, 0.95),
    ("80", 0.10, 0.90),
    ("50", 0.25, 0.75),
)


def reliability_curve(
    y_true: pd.Series,
    preds: dict[float, pd.Series],
    *,
    forecast_level: pd.Series | None = None,
    buckets: tuple[tuple[str, float, float], ...] | None = None,
) -> pd.DataFrame:
    """Empirical vs nominal one-sided coverage per quantile level (reliability data).

    DIAGNOSTIC ONLY -- measures the RAW quantile forecasts and corrects nothing
    (recalibration is step 4.3b). For each level ``a`` it reports the nominal
    level against the empirical coverage P(y_true <= q_a); a calibrated model
    sits on the diagonal. Reuses the atomic metrics ``quantile_coverage`` and
    ``pinball``; nothing here is a new metric.

    Parameters
    ----------
    y_true:
        Realised price series, hourly UTC index.
    preds:
        Maps each level ``a`` to its prediction series (one per backtest run).
        Must contain 0.5 when bucketing is requested and ``forecast_level`` is
        None. Indices should coincide; each metric aligns pairwise and drops NaN.
    forecast_level:
        Series used to assign forecast-level buckets. Defaults to ``preds[0.5]``
        (the median forecast) -- known ex ante, hence leakage-free.
    buckets:
        Optional (name, low, high) edges. When given, per-bucket sections are
        appended; a row is assigned to a bucket by ``low <= forecast_level < high``.

    Returns
    -------
    pd.DataFrame
        One row per (bucket, level): columns ``bucket`` ("overall" for the
        ungrouped rows), ``level`` (nominal ``a``), ``coverage`` (empirical),
        ``pinball``, and ``n`` (hours in the slice). ``n`` is the raw hour
        count of the slice; the atomic metrics drop their own NaN pairs
        independently, so the sample actually scored can be marginally
        smaller. The overall rows equal the atomic metrics on the full
        sample exactly.
    """
    levels = sorted(preds)

    def _rows(mask: pd.Series | None, bucket: str) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for a in levels:
            q = preds[a] if mask is None else preds[a].loc[mask]
            yt = y_true if mask is None else y_true.loc[mask]
            n = len(q)
            cov = float("nan") if n == 0 else quantile_coverage(yt, q)
            pb = float("nan") if n == 0 else pinball(yt, q, a)
            out.append({"bucket": bucket, "level": a, "coverage": cov, "pinball": pb, "n": n})
        return out

    rows = _rows(None, "overall")
    if buckets is not None:
        if forecast_level is None:
            if 0.5 not in preds:
                raise ValueError(
                    "median forecast (level 0.5) required for bucketing: "
                    "pass forecast_level explicitly or include 0.5 in preds"
                )
            fl = preds[0.5]
        else:
            fl = forecast_level
        for name, low, high in buckets:
            mask = (fl >= low) & (fl < high)
            rows += _rows(mask, name)

    return pd.DataFrame(rows)


def band_metrics(
    y_true: pd.Series,
    preds: dict[float, pd.Series],
    *,
    bands: tuple[tuple[str, float, float], ...] = NESTED_BANDS,
    forecast_level: pd.Series | None = None,
    buckets: tuple[tuple[str, float, float], ...] | None = None,
) -> pd.DataFrame:
    """Interval coverage and width per nested central band (reliability + sharpness).

    DIAGNOSTIC ONLY (see ``reliability_curve``). For each band the nominal
    coverage is ``upper_level - lower_level`` (e.g. 0.90 for the (0.05, 0.95)
    band); the empirical coverage is P(q_low <= y_true <= q_high) via
    ``interval_coverage``, and the width is mean(q_high - q_low) via
    ``interval_width`` (lower = sharper). Bands whose two levels are not both
    in ``preds`` are skipped (e.g. an ARIMAX reference that only carries
    0.05/0.95).

    Parameters
    ----------
    y_true:
        Realised price series, hourly UTC index.
    preds:
        Maps each level to its prediction series; see ``reliability_curve``.
    bands:
        (name, lower level, upper level) triples to evaluate.
    forecast_level:
        Series used to assign forecast-level buckets. Defaults to ``preds[0.5]``
        (the median forecast) -- known ex ante, hence leakage-free.
    buckets:
        Optional (name, low, high) edges. When given, per-bucket sections are
        appended; a row is assigned to a bucket by ``low <= forecast_level < high``.

    Returns
    -------
    pd.DataFrame
        One row per (bucket, band): columns ``bucket``, ``band``,
        ``nominal_coverage``, ``coverage`` (empirical), ``width``, ``n``. ``n``
        is the raw hour count of the slice; the atomic metrics drop their own
        NaN pairs independently, so the sample actually scored can be
        marginally smaller.
    """
    available_bands = [
        (name, low, high) for name, low, high in bands if low in preds and high in preds
    ]

    def _rows(mask: pd.Series | None, bucket: str) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for name, low, high in available_bands:
            lo = preds[low] if mask is None else preds[low].loc[mask]
            hi = preds[high] if mask is None else preds[high].loc[mask]
            yt = y_true if mask is None else y_true.loc[mask]
            n = len(yt)
            cov = float("nan") if n == 0 else interval_coverage(yt, lo, hi)
            width = float("nan") if n == 0 else interval_width(lo, hi)
            out.append(
                {
                    "bucket": bucket,
                    "band": name,
                    "nominal_coverage": high - low,
                    "coverage": cov,
                    "width": width,
                    "n": n,
                }
            )
        return out

    rows = _rows(None, "overall")
    if buckets is not None:
        if forecast_level is None:
            if 0.5 not in preds:
                raise ValueError(
                    "median forecast (level 0.5) required for bucketing: "
                    "pass forecast_level explicitly or include 0.5 in preds"
                )
            fl = preds[0.5]
        else:
            fl = forecast_level
        for name, low, high in buckets:
            mask = (fl >= low) & (fl < high)
            rows += _rows(mask, name)

    return pd.DataFrame(rows)
