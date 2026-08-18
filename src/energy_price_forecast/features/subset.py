"""Feature subsetting utility for in-memory ablation (Sprint 3.5)."""

from __future__ import annotations

import pandas as pd


def drop_features(
    features: pd.DataFrame, columns: list[str], *, strict: bool = True
) -> pd.DataFrame:
    """Return a copy of `features` without the given columns (ablation helper).

    Used by the step-3.5 feature ablation to remove the weak D6 forecast-error
    candidates in memory, so no separate `features_pruned.parquet` is created
    (abstract decision 5 / spec 3.5 decision 6).

    Fail-fast contract: with `strict=True` (default) a requested column that is
    absent raises KeyError listing the offending names. This prevents a typo in
    `--drop-features` from silently ablating nothing and reporting a misleading
    zero effect. With `strict=False` absent names are ignored.
    """
    missing = [c for c in columns if c not in features.columns]
    if missing and strict:
        raise KeyError(f"columns to drop not found in feature matrix: {missing}")
    to_drop = [c for c in columns if c in features.columns]
    return features.drop(columns=to_drop)
