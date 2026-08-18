"""Prediction snapshot for the report/dashboard artefact bundle (Sprint 5.1).

Builds the small, checked-in ``predictions_snapshot.parquet`` that is the
data contract for the later Streamlit dashboard (Sprint 5.3): hourly price,
LightGBM median forecast, raw and conformal-calibrated (rearranged) 90 %
bands, and the realised-market regime flags. Pure function -- no file I/O,
no MLflow. The thin I/O layer (``scripts/export_report_assets.py``) loads
the inputs and writes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..evaluation.regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS

SNAPSHOT_FILENAME = "predictions_snapshot.parquet"

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "price_actual",
    "forecast_median",
    "lo_raw",
    "hi_raw",
    "lo_calibrated",
    "hi_calibrated",
    *REGIME_FLAG_COLUMNS,
    MACRO_REGIME_COLUMN,
)

_REQUIRED = ("price_actual", "forecast_median")
_MAX_INDEX_MISMATCH_FRACTION = 0.01


@dataclass(frozen=True)
class SnapshotStats:
    """Diagnostics from a `build_snapshot` call, meant to be logged by the caller."""

    n_rows: int
    n_dropped_index_mismatch: int
    n_dropped_missing_required: int
    crossing_rate_raw: float
    crossing_rate_calibrated: float


def build_snapshot(
    price_actual: pd.Series,
    forecast_median: pd.Series,
    lo_raw: pd.Series,
    hi_raw: pd.Series,
    lo_calibrated: pd.Series,
    hi_calibrated: pd.Series,
    regime_flags: pd.DataFrame,
) -> tuple[pd.DataFrame, SnapshotStats]:
    """Join the six series plus the regime-flag frame into the snapshot contract.

    All inputs must carry a UTC `DatetimeIndex`. Rows are joined on the
    INTERSECTION of every input's index; rows present in only some inputs
    are dropped and counted (`SnapshotStats.n_dropped_index_mismatch`). If
    the dropped fraction exceeds 1% of the union, raises `ValueError`
    (spec §7: a large mismatch signals inconsistent upstream runs, not a
    normal edge case to silently absorb).

    Rows missing `price_actual` or `forecast_median` are dropped and
    counted separately (`n_dropped_missing_required`) -- these two columns
    are load-bearing for every downstream consumer.

    Quantile crossing (`lo_* <= forecast_median <= hi_*` violated) is a
    documented finding from Sprint 3/4, NOT repaired here -- only its rate
    is measured and returned, separately for the raw and calibrated bands.

    Raises `ValueError` if the resulting index is not strictly hourly,
    UTC, ascending and unique.
    """
    inputs = {
        "price_actual": price_actual,
        "forecast_median": forecast_median,
        "lo_raw": lo_raw,
        "hi_raw": hi_raw,
        "lo_calibrated": lo_calibrated,
        "hi_calibrated": hi_calibrated,
    }

    union_index = pd.DatetimeIndex([])
    common_index: pd.DatetimeIndex | None = None
    for name, series in inputs.items():
        idx = pd.DatetimeIndex(series.index)
        if idx.tz is None or str(idx.tz) != "UTC":
            raise ValueError(f"{name!r} index must be UTC tz-aware, got tz={idx.tz!r}.")
        union_index = union_index.union(idx)
        common_index = idx if common_index is None else common_index.intersection(idx)
    assert common_index is not None  # inputs is non-empty by construction

    regime_index = pd.DatetimeIndex(regime_flags.index)
    if regime_index.tz is None or str(regime_index.tz) != "UTC":
        raise ValueError(f"regime_flags index must be UTC tz-aware, got tz={regime_index.tz!r}.")
    union_index = union_index.union(regime_index)
    common_index = common_index.intersection(regime_index)

    common_index = common_index.sort_values()
    n_union = len(union_index)
    n_common = len(common_index)
    n_dropped_mismatch = n_union - n_common
    if n_union > 0 and (n_dropped_mismatch / n_union) > _MAX_INDEX_MISMATCH_FRACTION:
        raise ValueError(
            f"Index mismatch across snapshot inputs drops {n_dropped_mismatch}/{n_union} rows "
            f"({n_dropped_mismatch / n_union:.1%}), above the 1% fail-fast threshold -- this "
            "points at inconsistent upstream runs, not a normal edge case."
        )

    if len(common_index) > 1:
        diffs = common_index.to_series().diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != pd.Timedelta("1h"):
            raise ValueError(
                "Snapshot index is not a regular hourly grid after alignment; "
                f"found irregular spacing: {list(diffs)}."
            )
    if not common_index.is_unique:
        raise ValueError("Snapshot index is not unique after alignment.")

    frame = pd.DataFrame(
        {name: series.reindex(common_index) for name, series in inputs.items()},
        index=common_index,
    )
    frame = frame.join(regime_flags.reindex(common_index))

    n_before_dropna = len(frame)
    frame = frame.dropna(subset=list(_REQUIRED))
    n_dropped_missing = n_before_dropna - len(frame)

    crossing_raw = ~frame["lo_raw"].le(frame["forecast_median"]) | ~frame["forecast_median"].le(
        frame["hi_raw"]
    )
    crossing_calibrated = ~frame["lo_calibrated"].le(frame["forecast_median"]) | ~frame[
        "forecast_median"
    ].le(frame["hi_calibrated"])

    for col in REGIME_FLAG_COLUMNS:
        frame[col] = frame[col].astype(bool)
    frame[MACRO_REGIME_COLUMN] = frame[MACRO_REGIME_COLUMN].astype("category")

    stats = SnapshotStats(
        n_rows=len(frame),
        n_dropped_index_mismatch=n_dropped_mismatch,
        n_dropped_missing_required=n_dropped_missing,
        crossing_rate_raw=float(crossing_raw.mean()) if len(frame) else 0.0,
        crossing_rate_calibrated=float(crossing_calibrated.mean()) if len(frame) else 0.0,
    )
    return frame[list(SNAPSHOT_COLUMNS)], stats
