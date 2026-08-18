"""Per-regime performance breakdown (Sprint 4.2).

DIAGNOSTIC ONLY. This module slices already-computed forecasts and metrics
by the regime flags from Sprint 4.1 (``evaluation.regimes``). It never
recomputes a forecast, never touches ``evaluation.metrics``, and never turns
a flag into a model input -- it only groups existing results.
"""

from __future__ import annotations

import pandas as pd

from .metrics import summarise, summarise_quantiles
from .regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS

_MACRO_LEVELS: tuple[str, ...] = ("calm", "crisis", "post_crisis")


def _nan_row(metric_keys: list[str]) -> dict[str, float]:
    return {k: float("nan") for k in metric_keys}


def _assemble(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def breakdown_point(
    predictions: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    """Slice one model's point predictions by regime and summarise each slice.

    DIAGNOSTIC ONLY. The regime flags are evaluation labels (Sprint 4.1),
    never model inputs; this function only groups already-computed forecasts.

    Parameters
    ----------
    predictions:
        Tidy point-prediction frame with columns ``y_true``, ``y_pred``,
        ``delivery_day``, indexed by hourly UTC timestamps (the shape that
        ``summarise`` expects). Its index MUST be a subset of ``flags.index``.
    flags:
        Output of ``tag_regimes`` on the FULL interim history: the boolean
        regime flags (``REGIME_FLAG_COLUMNS``) and the categorical
        ``MACRO_REGIME_COLUMN``, indexed by hourly UTC timestamps.

    Returns
    -------
    pd.DataFrame
        One row per regime label (index), columns ``axis`` (overall/macro/
        flag), ``n`` (raw hours in the slice -- the metric functions may drop
        a few more pairwise NaN rows internally, so the sample actually
        scored can be marginally smaller), and the ``summarise`` metric keys.
        The ``overall`` row equals ``summarise(predictions)`` exactly.
    """
    required = ["y_true", "y_pred", "delivery_day"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions frame is missing required columns: {missing}")
    if not predictions.index.isin(flags.index).all():
        raise ValueError(
            "predictions index is not a subset of flags index; "
            "the interim history does not cover the test window."
        )

    # Pure selection -- the per-regime spike threshold was already fixed on
    # the full history in step 4.1, so nothing is recomputed here.
    f = flags.reindex(predictions.index)

    overall_metrics = summarise(predictions)
    metric_keys = list(overall_metrics.keys())

    def _row(mask: pd.Series | None) -> dict[str, float]:
        subset = predictions if mask is None else predictions.loc[mask]
        n = len(subset)
        metrics = _nan_row(metric_keys) if n == 0 else summarise(subset)
        return {"n": n, **metrics}

    rows: dict[str, dict[str, object]] = {
        "overall": {"axis": "overall", **_row(None)},
    }
    for level in _MACRO_LEVELS:
        rows[level] = {"axis": "macro", **_row(f[MACRO_REGIME_COLUMN] == level)}
    for flag in REGIME_FLAG_COLUMNS:
        rows[flag] = {"axis": "flag", **_row(f[flag])}

    return _assemble(rows)


def breakdown_quantiles(
    y_true: pd.Series,
    preds: dict[float, pd.Series],
    flags: pd.DataFrame,
    *,
    levels: tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Slice one model's quantile forecasts by regime and summarise each slice.

    DIAGNOSTIC ONLY (see ``breakdown_point``). Mirrors the
    ``summarise_quantiles`` contract: ``y_true`` is passed SEPARATELY (the
    actual price series), and ``preds`` maps each alpha level to its
    prediction series (one per backtest run).

    The three quantile series are first aligned on their common index
    (intersection); ``y_true`` is aligned onto that index. The resulting
    index MUST be a subset of ``flags.index``.

    Returns
    -------
    pd.DataFrame
        One row per regime label (index), columns ``axis``, ``n`` (see
        ``breakdown_point`` for the raw-vs-scored sample size caveat), and
        the ``summarise_quantiles`` metric keys (pinball_/coverage_ per
        level, interval_coverage_90, interval_width_90, crossing_rate +
        components).
    """
    common: pd.Index | None = None
    for a in levels:
        common = preds[a].index if common is None else common.intersection(preds[a].index)
    assert common is not None  # levels is never empty
    y = y_true.reindex(common)
    p = {a: preds[a].reindex(common) for a in levels}

    if not common.isin(flags.index).all():
        raise ValueError(
            "quantile index is not a subset of flags index; "
            "the interim history does not cover the test window."
        )
    f = flags.reindex(common)

    overall_metrics = summarise_quantiles(y, p, levels=levels)
    metric_keys = list(overall_metrics.keys())

    def _row(mask: pd.Series | None) -> dict[str, float]:
        y_sub = y if mask is None else y.loc[mask]
        p_sub = p if mask is None else {a: s.loc[mask] for a, s in p.items()}
        n = len(y_sub)
        metrics = (
            _nan_row(metric_keys) if n == 0 else summarise_quantiles(y_sub, p_sub, levels=levels)
        )
        return {"n": n, **metrics}

    rows: dict[str, dict[str, object]] = {
        "overall": {"axis": "overall", **_row(None)},
    }
    for level in _MACRO_LEVELS:
        rows[level] = {"axis": "macro", **_row(f[MACRO_REGIME_COLUMN] == level)}
    for flag in REGIME_FLAG_COLUMNS:
        rows[flag] = {"axis": "flag", **_row(f[flag])}

    return _assemble(rows)
