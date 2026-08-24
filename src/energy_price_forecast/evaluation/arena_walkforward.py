"""Quarter-hourly walk-forward harness for the Energy-Arena architecture gate.

Orchestrates, computes no metrics itself (spec 6.4 section 5.4). For each
evaluable delivery day D: refits the hourly model on a rolling
train_span_days window ending D-1 (reusing
evaluation.walkforward.walk_forward_splits for fold generation --
evaluation/walkforward.py itself is not modified, spec 6.4 section 2.5),
predicts 24 (23/24/25 across DST) hourly values for D, expands them to
quarter-hourly via both the shape profile (bridge_shape) and flat
repetition (bridge_flat), builds the Arena baseline replica, and attaches
the realised quarter-hourly prices as y_true.
"""

import logging

import pandas as pd

from energy_price_forecast.evaluation.walkforward import Forecaster, walk_forward_splits
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile

logger = logging.getLogger(__name__)

_TZ = "Europe/Berlin"

_OUTPUT_COLUMNS = [
    "y_true",
    "pred_bridge_shape",
    "pred_bridge_flat",
    "pred_baseline",
    "delivery_day",
    "n_slots_in_day",
]


def run_arena_backtest(
    y_hourly: pd.Series,
    x_hourly: pd.DataFrame,
    prices_qh: pd.Series,
    model: Forecaster,
    *,
    shape_window_days: int,
    train_span_days: int = 90,
    refit_every: int = 1,
    tz: str = _TZ,
) -> pd.DataFrame:
    """Run the bridge-vs-baseline walk-forward and return one tidy frame.

    The evaluable test window is derived from the data, not from a hard-coded
    date (spec 6.4 section 5.4): the first evaluable delivery day is the
    first one with ``shape_window_days`` complete prior days of
    quarter-hourly prices. The 90-day hourly training window is never the
    binding constraint (hourly history reaches back to 2020). Delivery days
    whose hourly test window or quarter-hourly target is incomplete (e.g. the
    most recent, still-in-progress day) are skipped and logged, not padded.

    Returns a DataFrame indexed by the canonical UTC quarter-hourly index
    with columns y_true, pred_bridge_shape, pred_bridge_flat, pred_baseline,
    delivery_day, n_slots_in_day. Does not persist -- the caller script
    writes data/processed/preds_arena.parquet, mirroring how
    evaluation.walkforward.run_backtest does not persist either.
    """
    qh_index = pd.DatetimeIndex(prices_qh.index)
    qh_local_days = pd.DatetimeIndex(qh_index.tz_convert(tz).normalize().unique()).sort_values()
    if len(qh_local_days) <= shape_window_days:
        raise ValueError(
            f"Not enough quarter-hourly history ({len(qh_local_days)} local days) for a "
            f"{shape_window_days}-day shape window."
        )
    start_day = qh_local_days[shape_window_days]
    end_day = qh_local_days[-1]
    logger.info(
        "arena backtest candidate window: %s -> %s (shape_window_days=%d)",
        start_day.date(),
        end_day.date(),
        shape_window_days,
    )

    hourly_index = pd.DatetimeIndex(y_hourly.index)
    candidate_folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=start_day.strftime("%Y-%m-%d"),
            test_end=end_day.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=train_span_days,
        )
    )

    evaluable_folds = []
    for fold in candidate_folds:
        day_start = pd.Timestamp(fold.delivery_day.date(), tz=tz)
        day_end = day_start + pd.DateOffset(days=1)
        expected_hours = round((day_end - day_start) / pd.Timedelta(hours=1))
        if len(fold.test_index) != expected_hours:
            logger.info(
                "skipping delivery day %s: incomplete hourly test window (%d/%d hours)",
                fold.delivery_day.date(),
                len(fold.test_index),
                expected_hours,
            )
            continue

        qh_mask = (qh_index >= day_start) & (qh_index < day_end)
        if qh_mask.sum() != expected_hours * 4:
            logger.info(
                "skipping delivery day %s: incomplete quarter-hourly target (%d/%d slots)",
                fold.delivery_day.date(),
                int(qh_mask.sum()),
                expected_hours * 4,
            )
            continue

        # The day immediately after a spring-forward transition has no D-1
        # wall-clock 02:00 slot to persist from (D-1 is the transition day
        # itself, which lacks that local hour entirely) -- the baseline is
        # structurally undefined for this one day a year. Owner decision
        # 2026-08-21: skip and log, consistent with the other skip branches
        # here, rather than aborting the whole run.
        try:
            persistence_forecast(prices_qh, fold.delivery_day, tz=tz)
        except ValueError as exc:
            logger.info(
                "skipping delivery day %s: baseline replica undefined (%s)",
                fold.delivery_day.date(),
                exc,
            )
            continue

        evaluable_folds.append(fold)

    n_folds = len(evaluable_folds)
    logger.info(
        "arena backtest: %d evaluable folds (of %d candidates) start=%s end=%s",
        n_folds,
        len(candidate_folds),
        evaluable_folds[0].delivery_day.date() if evaluable_folds else None,
        evaluable_folds[-1].delivery_day.date() if evaluable_folds else None,
    )

    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % refit_every == 0:
            y_train = y_hourly.loc[fold.train_index]
            x_train = x_hourly.loc[fold.train_index]
            model.fit(y_train, x_train)

        x_test = x_hourly.loc[fold.test_index]
        hourly_pred = model.predict(
            fold.test_index, history=y_hourly.loc[fold.train_index], x_test=x_test
        )

        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=shape_window_days, tz=tz
        )

        pred_shape = expand_to_quarterhour(
            hourly_pred, profile, target_day=fold.delivery_day, tz=tz
        )
        pred_flat = expand_to_quarterhour(hourly_pred, None, target_day=fold.delivery_day, tz=tz)
        pred_baseline = persistence_forecast(prices_qh, fold.delivery_day, tz=tz)

        day_start = pd.Timestamp(fold.delivery_day.date(), tz=tz)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    "pred_bridge_shape": pred_shape.reindex(y_true_day.index).to_numpy(),
                    "pred_bridge_flat": pred_flat.reindex(y_true_day.index).to_numpy(),
                    "pred_baseline": pred_baseline.reindex(y_true_day.index).to_numpy(),
                    "delivery_day": fold.delivery_day,
                    "n_slots_in_day": len(y_true_day),
                },
                index=y_true_day.index,
            )
        )

        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            logger.info("arena backtest progress: %d/%d folds", i + 1, n_folds)

    result = pd.concat(records).sort_index()
    return result[_OUTPUT_COLUMNS]
