"""The OOS-Store: out-of-sample median price forecasts per target day and
fallback-ladder row (spec 7.1, section 2).

For every historical target day and every one of the three ladder rows
(``core_gas``/``core_gas_loadpatch``/``base``, ``arena/candidates.py``),
the OOS-Store holds the out-of-sample median quarter-hourly forecast the
live pipeline would have produced on the morning of D-1 -- computed with
exactly the live information state (``scripts/run_daily_submission.py``'s
own ``evaluate_ladder_row``/``fit_predict_expand``, never a second
implementation). ``scripts/oos_store_job.py`` is the only writer (backfill
and daily top-up alike); this module is the schema, merge, and I/O layer
underneath it.

No prices, residuals, or sigmas live here (spec section 2.1): those are
derived fresh at read time from this store plus the existing price store,
so a later ENTSO-E correction is picked up automatically and the one
place a price can disagree with itself is the price store, not a second
copy of it here.

Persistence is a plain CSV committed to git under ``logs/``, not a new
``ops/store.py`` source. ``ops/store.py``'s own module docstring is
explicit that only raw, externally-fetched data ever enters that store,
"never anything derived" -- the OOS-Store is exactly that derived
artifact, so it follows ``logs/submissions.jsonl``/``logs/store_sync.csv``'s
precedent (committed directly, no encryption, no Release Asset) instead.
CSV over Parquet for the same reason ``ops/protocol.py`` gives for its own
format choice: "flat and stable -> CSV, nested or growing -> JSON lines"
-- this schema is flat and, unlike ``logs/store_sync.csv``'s per-source
columns, never grows a new column for an existing row, so there is no
header-migration case to handle the way ``scripts/sync_store.py::
_write_log_row`` must.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import pandas as pd

from energy_price_forecast.config import PROJECT_ROOT

# Bumped by hand whenever the point model/features/training window/objective
# changes -- never a computed hash (this repo's convention for anything that
# gates behaviour is a named constant, e.g. PRICE_TRAIN_SPAN_DAYS/
# SHAPE_WINDOW_DAYS in scripts/run_daily_submission.py, not an opaque hash of
# params that could silently drift if an unrelated refactor reordered a
# dict). Teil B's own model_version filter (spec section 3.2 step 1) is what
# makes a forgotten bump loud -- old rows visibly keep the old tag -- rather
# than silently wrong.
OOS_MODEL_VERSION: Final[str] = "price_v1"

OOS_STORE_PATH: Final[Path] = PROJECT_ROOT / "logs" / "oos_store.csv"

LADDER_ROW_NAMES: Final[frozenset[str]] = frozenset({"core_gas", "core_gas_loadpatch", "base"})

OOS_STORE_COLUMNS: Final[tuple[str, ...]] = (
    "target_day",
    "timestamp_utc",
    "ladder_row",
    "forecast",
    "model_version",
    "weather_run_init_utc",
    "source",
    "created_at_utc",
)

_KEY_COLUMNS: Final[tuple[str, ...]] = ("target_day", "timestamp_utc", "ladder_row")

# A stored forecast's own float formatting (CSV round-trip) can differ from
# the in-memory value by less than this without counting as a genuine
# conflict in merge_rows -- strictly a float-precision guard, not a
# tolerance for a real second opinion (spec section 2.3's idempotency
# contract: a key already present is never overwritten).
_FORECAST_ATOL: Final[float] = 1e-6


@dataclass(frozen=True)
class OosStoreRow:
    """One (target_day, timestamp_utc, ladder_row) forecast (spec section
    2.1's field table). ``weather_run_init_utc`` is ``None`` for ``base``
    (no weather dependency at all, ``arena/candidates.py``'s
    ``requires_nwp=False``)."""

    target_day: dt.date
    timestamp_utc: pd.Timestamp
    ladder_row: str
    forecast: float
    model_version: str
    weather_run_init_utc: pd.Timestamp | None
    source: Literal["backfill", "daily"]
    created_at_utc: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "target_day": self.target_day.isoformat(),
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "ladder_row": self.ladder_row,
            "forecast": self.forecast,
            "model_version": self.model_version,
            "weather_run_init_utc": (
                self.weather_run_init_utc.isoformat()
                if self.weather_run_init_utc is not None
                else ""
            ),
            "source": self.source,
            "created_at_utc": self.created_at_utc.isoformat(),
        }


def rows_to_frame(rows: Iterable[OosStoreRow]) -> pd.DataFrame:
    """The in-memory frame shape ``merge_rows``/``validate_oos_store_frame``
    expect -- typed columns, not the CSV's string-only on-disk shape."""
    records = [
        {
            "target_day": row.target_day,
            "timestamp_utc": row.timestamp_utc,
            "ladder_row": row.ladder_row,
            "forecast": row.forecast,
            "model_version": row.model_version,
            "weather_run_init_utc": row.weather_run_init_utc,
            "source": row.source,
            "created_at_utc": row.created_at_utc,
        }
        for row in rows
    ]
    frame = pd.DataFrame.from_records(records, columns=list(OOS_STORE_COLUMNS))
    if frame.empty:
        return _empty_frame()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["weather_run_init_utc"] = pd.to_datetime(frame["weather_run_init_utc"], utc=True)
    frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], utc=True)
    return frame


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(OOS_STORE_COLUMNS))
    frame["target_day"] = frame["target_day"].astype(object)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["weather_run_init_utc"] = pd.to_datetime(frame["weather_run_init_utc"], utc=True)
    frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], utc=True)
    frame["forecast"] = frame["forecast"].astype(float)
    return frame


def load_oos_store(path: Path = OOS_STORE_PATH) -> pd.DataFrame:
    """Read the on-disk store, typed. Returns an empty, correctly-typed
    frame if ``path`` does not exist yet -- the same "no rows yet is a
    valid, not exceptional, state" convention as
    ``ops.protocol.read_submission_records``."""
    if not path.exists():
        return _empty_frame()
    raw = pd.read_csv(path, dtype={"ladder_row": str, "model_version": str, "source": str})
    raw["target_day"] = pd.to_datetime(raw["target_day"]).dt.date
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    raw["weather_run_init_utc"] = pd.to_datetime(
        raw["weather_run_init_utc"].replace("", pd.NA), utc=True
    )
    raw["created_at_utc"] = pd.to_datetime(raw["created_at_utc"], utc=True)
    raw["forecast"] = raw["forecast"].astype(float)
    return raw[list(OOS_STORE_COLUMNS)]


def validate_oos_store_frame(df: pd.DataFrame) -> None:
    """Raises ValueError, naming the first offending row, on: a missing
    required column, a duplicate (target_day, timestamp_utc, ladder_row)
    key, an unknown ``ladder_row``, a NaN ``forecast``, or a
    ``timestamp_utc`` that is not UTC-aware. Pure and side-effect-free --
    never mutates or coerces ``df``."""
    missing_columns = [c for c in OOS_STORE_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"OOS-store frame is missing required column(s): {missing_columns}")

    duplicated = df.duplicated(subset=list(_KEY_COLUMNS), keep=False)
    if duplicated.any():
        first = df.loc[duplicated, list(_KEY_COLUMNS)].iloc[0]
        raise ValueError(f"duplicate OOS-store key: {dict(first)}")

    unknown_rows = df.loc[~df["ladder_row"].isin(LADDER_ROW_NAMES), "ladder_row"]
    if not unknown_rows.empty:
        raise ValueError(f"unknown ladder_row: {unknown_rows.iloc[0]!r}")

    nan_forecast = df.loc[df["forecast"].isna(), list(_KEY_COLUMNS)]
    if not nan_forecast.empty:
        raise ValueError(f"NaN forecast for key: {dict(nan_forecast.iloc[0])}")

    if not df.empty:
        timestamps = pd.DatetimeIndex(df["timestamp_utc"])
        if timestamps.tz is None:
            raise ValueError("timestamp_utc must be tz-aware (UTC)")


def merge_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Idempotent merge, keyed on (target_day, timestamp_utc, ladder_row)
    (spec section 2.3: the daily top-up "ändert nichts" on a day already
    stored). A key already in ``existing`` is never overwritten; if
    ``new_rows`` disagrees with an already-stored key's own ``forecast``
    beyond float-precision noise, this raises rather than silently keeping
    either value -- the live pipeline is deterministic on identical inputs
    (``tests/test_run_daily_submission.py::
    test_run_submission_for_day_is_deterministic_on_identical_inputs``), so
    a genuine disagreement means the inputs themselves changed (a store
    correction, a code change without a model_version bump) and needs a
    human look, not a silent pick.

    Returns ``(merged, n_added)`` -- ``n_added`` counts only genuinely new
    keys, never a re-submitted-but-identical row.
    """
    validate_oos_store_frame(new_rows)
    if existing.empty:
        return new_rows.reset_index(drop=True), len(new_rows)

    merged_keys = existing.set_index(list(_KEY_COLUMNS)).index
    new_keys = new_rows.set_index(list(_KEY_COLUMNS)).index
    overlap = new_keys.isin(merged_keys)

    if overlap.any():
        existing_forecast_by_key: dict[object, float] = (
            existing.set_index(list(_KEY_COLUMNS))["forecast"].astype(float).to_dict()
        )
        new_forecasts = new_rows.loc[overlap, "forecast"].astype(float).to_numpy()
        for key, new_forecast in zip(new_keys[overlap], new_forecasts, strict=True):
            stored_forecast = existing_forecast_by_key[key]
            if abs(stored_forecast - float(new_forecast)) > _FORECAST_ATOL:
                raise ValueError(
                    f"OOS-store conflict for key {key}: stored forecast "
                    f"{stored_forecast!r} != new forecast {new_forecast!r} -- "
                    "an already-written row must never be silently overwritten"
                )

    genuinely_new = new_rows.loc[~overlap]
    if genuinely_new.empty:
        return existing.reset_index(drop=True), 0

    merged = pd.concat([existing, genuinely_new], ignore_index=True)
    merged = merged.sort_values(list(_KEY_COLUMNS)).reset_index(drop=True)
    return merged, len(genuinely_new)


def write_oos_store(df: pd.DataFrame, path: Path = OOS_STORE_PATH) -> None:
    """Write the full, already-merged frame to ``path`` as CSV (temp file +
    ``os.replace`` for crash-safety mid-write). Unlike ``scripts/sync_store.
    py::_write_log_row``'s header-migration dance, this schema is fixed
    (``OOS_STORE_COLUMNS`` never grows a new column for an existing row the
    way ``logs/store_sync.csv``'s per-source columns do) -- a full rewrite
    of the whole, already-merged frame is simpler and exactly as correct.
    """
    validate_oos_store_frame(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[list(OOS_STORE_COLUMNS)].copy()
    out["target_day"] = out["target_day"].astype(object).map(lambda d: d.isoformat())
    out["timestamp_utc"] = out["timestamp_utc"].map(lambda ts: ts.isoformat())
    out["weather_run_init_utc"] = out["weather_run_init_utc"].map(
        lambda ts: "" if pd.isna(ts) else ts.isoformat()
    )
    out["created_at_utc"] = out["created_at_utc"].map(lambda ts: ts.isoformat())

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        out.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def missing_target_days(
    existing: pd.DataFrame,
    target_days: Iterable[dt.date],
    ladder_rows: frozenset[str] = LADDER_ROW_NAMES,
) -> dict[dt.date, frozenset[str]]:
    """Per ``target_day``, which of ``ladder_rows`` are not yet present in
    ``existing`` -- a day already stored for 2 of 3 rows (row 3 genuinely
    failed its own viability check that day, spec section 3.4's "Zeile hat
    zu wenige eigene Residuen"-shaped outcome, one level up) reports only
    the missing third, never re-attempting the two already present.
    Days that are fully present (all of ``ladder_rows``) are simply absent
    from the returned dict, not mapped to an empty set.
    """
    if existing.empty:
        stored_by_day: dict[dt.date, frozenset[str]] = {}
    else:
        stored_by_day = {
            cast(dt.date, day): frozenset(group["ladder_row"])
            for day, group in existing.groupby("target_day")
        }

    out: dict[dt.date, frozenset[str]] = {}
    for day in target_days:
        stored = stored_by_day.get(day, frozenset())
        still_missing = ladder_rows - stored
        if still_missing:
            out[day] = still_missing
    return out
