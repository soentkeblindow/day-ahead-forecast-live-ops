"""Walk-forward evaluation loop for the three renewables capacity-factor
models (spec 6.5.2, sections 2.7 and 5.7).

Orchestrates, computes no metrics itself -- same division of responsibility
as `evaluation/arena_walkforward.py` in 6.4, and for the same reason:
`evaluation/walkforward.py::walk_forward_splits` is shared, validated code
and stays untouched, reused here for fold generation exactly as it already
is for the price model.

The feature matrix is built exactly once, for the whole span the folds
need, not once per fold or once per refit -- weather alignment and solar
position are the expensive part, and every fold's train/test window is
just a slice of that one matrix by (run_init_utc, valid_time_utc).

Each production type gets its own available-target subset (ENTSO-E's own
14.1.D series has occasional gaps, e.g. 74 missing hours in
wind_onshore_forecast) and its own persistence baseline; a day missing from
one target's coverage does not affect the other two.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import pytz

from energy_price_forecast.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
    installed_capacity_at,
)
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.evaluation.walkforward import walk_forward_splits
from energy_price_forecast.features.renewables_forecast import build_feature_matrix, columns_for
from energy_price_forecast.models.lgbm import Objective
from energy_price_forecast.models.renewables import (
    RenewablesModel,
    apply_solar_night_zero,
    capacity_factor_label,
    daylight_mask_for_training,
    to_mw,
)

logger = logging.getLogger(__name__)

_TZ = "Europe/Berlin"

TARGET_COLUMNS: dict[ProductionType, str] = {
    ProductionType.WIND_ONSHORE: "wind_onshore_forecast",
    ProductionType.WIND_OFFSHORE: "wind_offshore_forecast",
    ProductionType.SOLAR: "solar_forecast",
}

_OUTPUT_COLUMNS = [
    f"{target.value}_{suffix}"
    for target in ProductionType
    for suffix in ("cf_pred", "cf_actual", "mw_pred", "mw_actual", "capacity_mw")
] + ["is_daylight_hour"]


def persistence_baseline_cf(label_flat: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """D-1's same local wall-clock hour, in capacity factor (spec 5.7, E6).

    ``pd.DateOffset(days=1)`` (calendar-aware, not a fixed 24h duration) so
    the lookup lands on the correct local hour across a DST transition.
    Raises ``pytz.exceptions.InvalidTimeError`` when that local hour does
    not exist, or is ambiguous, on D-1 -- both DST changeover days a year
    (spec 5.7; same treatment as 6.4's arena baseline).
    """
    local_hours = test_index.tz_convert(_TZ)
    prev_hours_utc = (local_hours - pd.DateOffset(days=1)).tz_convert("UTC")
    values = label_flat.reindex(prev_hours_utc)
    if values.isna().any():
        raise ValueError("persistence baseline undefined: D-1 value missing")
    return pd.Series(values.to_numpy(), index=test_index)


def _build_target_table(
    target_hourly: pd.DataFrame,
    target: ProductionType,
    features: pd.DataFrame,
    valid_time: pd.DatetimeIndex,
    *,
    source: CapacitySource,
    method: CapacityExtrapolation,
) -> pd.DataFrame:
    """Feature columns for ``target`` plus its capacity-factor label, for
    exactly the rows where the raw ENTSO-E target value actually exists
    (gaps are dropped here, never interpolated -- capacity_factor_label
    itself guards against being handed an *unexpected* gap in an
    already-filtered batch, not routine sparse coverage in the raw series).
    """
    raw = target_hourly[TARGET_COLUMNS[target]].reindex(valid_time)
    available = raw.notna().to_numpy()

    table = features.loc[available, columns_for(target)].copy()
    label = capacity_factor_label(
        raw[available], valid_time[available], target, source=source, method=method
    )
    table[f"{target.value}_cf_actual"] = label.to_numpy()
    return table


def run_renewables_backtest(
    target_hourly: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    window: Literal["expanding", "rolling"] = "rolling",
    train_span_days: int | None = 365,
    min_history_days: int | None = None,
    refit_every: int = 7,
    objective: Objective = "l2",
    seed: int = 0,
    source: CapacitySource = CapacitySource.PUBLIC_REGISTRY,
    method: CapacityExtrapolation = CapacityExtrapolation.LAST_INCREMENT,
) -> pd.DataFrame:
    """Run the rolling (or expanding) walk-forward for all three targets.

    The evaluable window is derived from the data, never hard-coded: the
    first yielded fold from ``walk_forward_splits`` (which itself skips
    rolling folds without a full training window) is the first evaluable
    day (spec 5.7). ``expanding`` has no such built-in floor -- its very
    first candidate fold only needs one prior day -- so ``min_history_days``
    (365 for the E5 comparison run) additionally drops any fold less than
    that many days after the series start, so both variants start on the
    same first evaluable day even though only ``rolling`` needs the floor
    structurally.

    Returns one tidy DataFrame indexed by (run_init_utc, valid_time_utc)
    with columns {target}_cf_pred, {target}_cf_actual, {target}_mw_pred,
    {target}_mw_actual, {target}_capacity_mw per target, plus
    is_daylight_hour. Does not persist or compute metrics -- the caller
    script does both (evaluation/metrics.py stays unmodified, spec 3.3).
    """
    hourly_index = pd.DatetimeIndex(target_hourly.index)
    test_start = hourly_index.min().strftime("%Y-%m-%d")
    test_end = hourly_index.max().strftime("%Y-%m-%d")

    candidate_folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=test_start,
            test_end=test_end,
            window=window,
            train_span_days=train_span_days,
        )
    )
    if min_history_days is not None:
        floor_day = pd.Timestamp(hourly_index.min().tz_convert(_TZ).date(), tz=_TZ) + pd.DateOffset(
            days=min_history_days
        )
        candidate_folds = [f for f in candidate_folds if f.delivery_day >= floor_day]
    if not candidate_folds:
        raise ValueError("no evaluable folds derived from the data")

    logger.info(
        "renewables backtest: first evaluable day %s, last %s (%d candidate folds, window=%s, train_span_days=%s)",
        candidate_folds[0].delivery_day.date(),
        candidate_folds[-1].delivery_day.date(),
        len(candidate_folds),
        window,
        train_span_days,
    )

    first_needed = candidate_folds[0].train_index.min()
    last_needed = candidate_folds[-1].test_index.max()
    span_hours = hourly_index[(hourly_index >= first_needed) & (hourly_index <= last_needed)]
    needed_days = sorted(set(span_hours.tz_convert(_TZ).date))

    features, feature_excluded_days = build_feature_matrix(needed_days, weather)
    logger.info(
        "feature matrix: %d rows over %d days (%d days excluded: missing/incomplete weather run)",
        len(features),
        len(needed_days),
        len(feature_excluded_days),
    )
    if feature_excluded_days:
        logger.info(
            "weather-excluded days: %s", ", ".join(d.isoformat() for d in feature_excluded_days)
        )

    valid_time = pd.DatetimeIndex(features.index.get_level_values("valid_time_utc"))

    target_tables: dict[ProductionType, pd.DataFrame] = {}
    target_labels_flat: dict[ProductionType, pd.Series] = {}
    for target in ProductionType:
        table = _build_target_table(
            target_hourly, target, features, valid_time, source=source, method=method
        )
        target_tables[target] = table
        vt = pd.DatetimeIndex(table.index.get_level_values("valid_time_utc"))
        target_labels_flat[target] = pd.Series(
            table[f"{target.value}_cf_actual"].to_numpy(), index=vt
        )

    models: dict[ProductionType, RenewablesModel] = {
        target: RenewablesModel(target, objective=objective, seed=seed) for target in ProductionType
    }

    skip_counts: dict[str, int] = {t.value: 0 for t in ProductionType}
    n_folds = len(candidate_folds)
    records: list[pd.DataFrame] = []

    for i, fold in enumerate(candidate_folds):
        train_lo, train_hi = fold.train_index.min(), fold.train_index.max()
        test_lo, test_hi = fold.test_index.min(), fold.test_index.max()

        row_columns: dict[str, np.ndarray] = {}
        row_index: pd.DatetimeIndex | None = None

        for target in ProductionType:
            table = target_tables[target]
            vt = pd.DatetimeIndex(table.index.get_level_values("valid_time_utc"))
            test_mask = (vt >= test_lo) & (vt <= test_hi)

            if test_mask.sum() != len(fold.test_index):
                skip_counts[target.value] += 1
                logger.info(
                    "%s / %s: skipped (%d/%d target+weather hours available)",
                    fold.delivery_day.date(),
                    target.value,
                    int(test_mask.sum()),
                    len(fold.test_index),
                )
                continue

            try:
                persistence_baseline_cf(target_labels_flat[target], fold.test_index)
            except (pytz.exceptions.InvalidTimeError, ValueError) as exc:
                skip_counts[target.value] += 1
                logger.info(
                    "%s / %s: skipped, persistence baseline undefined (%s)",
                    fold.delivery_day.date(),
                    target.value,
                    exc,
                )
                continue

            if i % refit_every == 0:
                train_mask = (vt >= train_lo) & (vt <= train_hi)
                x_train = table.loc[train_mask, columns_for(target)]
                y_train = table.loc[train_mask, f"{target.value}_cf_actual"]
                if target == ProductionType.SOLAR:
                    daylight = daylight_mask_for_training(
                        pd.DatetimeIndex(x_train.index.get_level_values("valid_time_utc"))
                    ).to_numpy()
                    x_train = x_train.loc[daylight]
                    y_train = y_train.loc[daylight]
                models[target].fit(x_train, y_train)

            x_test = table.loc[test_mask, columns_for(target)]
            y_test_actual = table.loc[test_mask, f"{target.value}_cf_actual"]
            cf_pred = models[target].predict_capacity_factor(x_test.index, x_test)
            if target == ProductionType.SOLAR:
                cf_pred = apply_solar_night_zero(cf_pred, fold.test_index)

            mw_pred = to_mw(cf_pred, fold.test_index, target, source=source, method=method)
            mw_actual = to_mw(
                pd.Series(y_test_actual.to_numpy(), index=fold.test_index),
                fold.test_index,
                target,
                source=source,
                method=method,
            )
            capacity_mw_series = installed_capacity_at(
                target, fold.test_index, source=source, method=method
            ).to_numpy()

            row_columns[f"{target.value}_cf_pred"] = cf_pred.to_numpy()
            row_columns[f"{target.value}_cf_actual"] = y_test_actual.to_numpy()
            row_columns[f"{target.value}_mw_pred"] = mw_pred.to_numpy()
            row_columns[f"{target.value}_mw_actual"] = mw_actual.to_numpy()
            row_columns[f"{target.value}_capacity_mw"] = capacity_mw_series
            row_index = fold.test_index

        if row_index is None:
            continue  # every target skipped this day

        row_columns["is_daylight_hour"] = daylight_mask_for_training(row_index).to_numpy()
        run_init = run_init_for_target_day(fold.delivery_day.date())
        full_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(row_index), tz="UTC"), row_index],
            names=["run_init_utc", "valid_time_utc"],
        )
        records.append(pd.DataFrame(row_columns, index=full_index))

        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            logger.info("renewables backtest progress: %d/%d folds", i + 1, n_folds)

    logger.info("skipped fold counts by target: %s", skip_counts)

    if not records:
        raise ValueError("no fold produced a usable row for any target")

    result = pd.concat(records).sort_index()
    for col in _OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
    return result[_OUTPUT_COLUMNS]
