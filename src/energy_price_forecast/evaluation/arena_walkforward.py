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

Extended in 6.6 (spec section 5.2) with run_live_gate_backtest: two
independently trained hourly models, one per feature set (live vs.
original), built per delivery day via features.build.build_feature_set_for_day
/ build_original_feature_set_for_day -- not from the precomputed
features.parquet, a different code path (spec 6.6 section 5.1).
"""

import datetime as dt
import logging
from typing import Literal

import pandas as pd

from energy_price_forecast.evaluation.walkforward import Forecaster, walk_forward_splits
from energy_price_forecast.features.build import (
    build_feature_set_for_day,
    build_original_feature_set_for_day,
)
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.models.arena_baseline import persistence_forecast
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.ops.windows import local_day_bounds

logger = logging.getLogger(__name__)

_TZ = "Europe/Berlin"
_PRICE_COL = "day_ahead_price"

_LIVE_GATE_OUTPUT_COLUMNS = [
    "y_true",
    "pred_original",
    "pred_live",
    "pred_baseline",
    "delivery_day",
    "n_slots_in_day",
]

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


def _local_days(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Unique local calendar days covered by a UTC index, sorted."""
    return pd.DatetimeIndex(
        pd.DatetimeIndex(index).tz_convert(tz).normalize().unique()
    ).sort_values()


def _hourly_persistence_forecast(
    y_hourly: pd.Series, target_day: pd.Timestamp, *, tz: str
) -> pd.Series:
    """Hourly wall-clock-aligned persistence baseline (spec 6.6 section 3.3)
    -- the same D-1-by-wall-clock rule as
    models.arena_baseline.persistence_forecast, at hourly instead of
    quarter-hourly resolution. There is no hourly Arena challenge; this is
    the secondary run's own baseline, not part of the platform spec. Not
    added to models/arena_baseline.py itself (spec 6.6 section 4.3: that
    module stays unchanged).
    """
    target_date = target_day.date()
    prev_date = target_date - dt.timedelta(days=1)

    target_start, target_end = local_day_bounds(target_date)
    prev_start, prev_end = local_day_bounds(prev_date)
    target_index = pd.date_range(target_start, target_end, freq="h", inclusive="left").tz_convert(
        "UTC"
    )
    prev_index = pd.date_range(prev_start, prev_end, freq="h", inclusive="left").tz_convert("UTC")

    prev_values = y_hourly.reindex(prev_index)
    if prev_values.isna().any():
        missing = prev_index[prev_values.isna()]
        raise ValueError(
            f"Missing realised hourly price(s) for {prev_date} -- cannot build the hourly "
            f"persistence baseline for {target_date}. First missing timestamp: {missing[0]}."
        )

    # D-1 is never itself a DST-transition day (same reasoning as the
    # quarter-hourly replica), so its wall-clock hours are always distinct.
    prev_local = prev_index.tz_convert(tz)
    lookup = dict(zip(prev_local.hour, prev_values.to_numpy(), strict=True))

    target_local = target_index.tz_convert(tz)
    keys = list(target_local.hour)
    missing_keys = [k for k in keys if k not in lookup]
    if missing_keys:
        raise ValueError(
            f"No matching D-1 wall-clock hour for target day {target_date}: "
            f"{missing_keys[0]:02d}:00."
        )
    values = [lookup[k] for k in keys]
    return pd.Series(values, index=target_index, name="baseline")


def _build_matrix_over_days(days: pd.DatetimeIndex, builder) -> tuple[pd.DataFrame, set]:
    """Call ``builder(day) -> pd.DataFrame`` once per local day and
    concatenate into one matrix spanning the whole range -- built once, not
    per fold, since a fixed day's feature row never changes across folds
    (spec 6.6 section 5.1).

    Days where the builder raises IncompleteReconstructionError are logged
    and excluded rather than filled (spec 6.5.3 section 3.3's whole-day-out
    policy, reused here). Returns (matrix, excluded_days).
    """
    frames = []
    excluded: set = set()
    for day in days:
        try:
            frames.append(builder(day.date()))
        except IncompleteReconstructionError as exc:
            logger.info("excluding %s from this feature set: %s", day.date(), exc)
            excluded.add(day.date())
    if not frames:
        raise ValueError("no day produced a usable feature row -- nothing to build a matrix from")
    return pd.concat(frames).sort_index(), excluded


def run_live_gate_backtest(
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    model_live: Forecaster,
    model_original: Forecaster,
    *,
    resolution: Literal["quarterhourly", "hourly"],
    prices_qh: pd.Series | None = None,
    shape_window_days: int | None = None,
    train_span_days: int = 90,
    refit_every: int = 1,
    tz: str = _TZ,
) -> pd.DataFrame:
    """Run the 6.6 live-gate walk-forward: two independently trained hourly
    models, one per feature set (spec 6.6 section 3.1), fit on identical
    delivery-day folds and compared against the Arena baseline replica.

    ``pred_original`` is a reference line only (the 6.4 upper bound, rebuilt
    on this step's own window) -- never evaluated against the go-live
    criterion (spec 6.6 sections 2.1/3.1). ``pred_live`` is the only
    candidate the gate in scripts/compare_arena.py may act on.

    resolution="quarterhourly" (the binding run, spec 6.6 section 2.2)
    expands both hourly forecasts via the shape profile (bridge_shape only
    -- bridge_flat was 6.4's own question, not re-asked here) and compares
    against the quarter-hourly Arena baseline. resolution="hourly" (the
    secondary, non-binding run, spec 6.6 section 3.3) compares the raw
    hourly forecasts directly against realised hourly prices and an hourly
    wall-clock persistence baseline -- not itself an Arena metric.

    Both candidates are trained and evaluated on exactly the same set of
    delivery days (spec 6.6 section 3.4): a day excluded from the live
    feature set (incomplete NWP reconstruction) is excluded for the
    original candidate too, so pred_original and pred_live always stand on
    identical folds.
    """
    if resolution == "quarterhourly" and (prices_qh is None or shape_window_days is None):
        raise ValueError("resolution='quarterhourly' requires prices_qh and shape_window_days")

    hourly_index = pd.DatetimeIndex(df.index)
    renewables_valid_time = pd.DatetimeIndex(
        renewables_predictions.index.get_level_values("valid_time_utc")
    )
    renewables_days = _local_days(renewables_valid_time, tz)

    # Lower bound on start_day: the first candidate fold's own train_span_days
    # training window must fully post-date the NWP reconstruction artefact's
    # start, or x_train_live reindexes to nothing but NaN rows and dropna()
    # leaves LightGBM an empty frame. The quarterhourly branch's own
    # qh_local_days[shape_window_days] term happened to already clear this bar
    # (quarter-hourly prices start much later than the NWP artefact), but that
    # was incidental, not a guarantee -- make it explicit for both branches.
    min_start_day = renewables_days[0] + pd.Timedelta(days=train_span_days)

    if resolution == "quarterhourly":
        assert prices_qh is not None and shape_window_days is not None
        qh_local_days = _local_days(pd.DatetimeIndex(prices_qh.index), tz)
        if len(qh_local_days) <= shape_window_days:
            raise ValueError(
                f"Not enough quarter-hourly history ({len(qh_local_days)} local days) for a "
                f"{shape_window_days}-day shape window."
            )
        start_day = max(qh_local_days[shape_window_days], min_start_day)
        end_day = min(qh_local_days[-1], renewables_days[-1])
    else:
        start_day = min_start_day
        end_day = min(_local_days(hourly_index, tz)[-1], renewables_days[-1])

    logger.info(
        "live-gate backtest candidate window (%s): %s -> %s",
        resolution,
        start_day.date(),
        end_day.date(),
    )

    candidate_folds = list(
        walk_forward_splits(
            hourly_index,
            test_start=start_day.strftime("%Y-%m-%d"),
            test_end=end_day.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=train_span_days,
        )
    )
    if not candidate_folds:
        raise ValueError("no candidate folds in the derived window -- check data coverage")

    first_train_day = candidate_folds[0].train_index.tz_convert(tz).normalize().min()
    all_days_needed = pd.date_range(first_train_day, end_day, freq="D", tz=tz)

    live_matrix, live_excluded = _build_matrix_over_days(
        all_days_needed, lambda d: build_feature_set_for_day(d, df, renewables_predictions)
    )
    original_matrix, _ = _build_matrix_over_days(
        all_days_needed, lambda d: build_original_feature_set_for_day(d, df)
    )
    logger.info(
        "feature matrices built: live=%d rows (%d days excluded), original=%d rows",
        len(live_matrix),
        len(live_excluded),
        len(original_matrix),
    )

    y_hourly = df[_PRICE_COL]

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
        if fold.delivery_day.date() in live_excluded:
            logger.info(
                "skipping delivery day %s: incomplete NWP reconstruction -- "
                "excluded for both candidates (spec 6.6 section 3.4)",
                fold.delivery_day.date(),
            )
            continue

        if resolution == "quarterhourly":
            assert prices_qh is not None
            qh_index = pd.DatetimeIndex(prices_qh.index)
            qh_mask = (qh_index >= day_start) & (qh_index < day_end)
            if qh_mask.sum() != expected_hours * 4:
                logger.info(
                    "skipping delivery day %s: incomplete quarter-hourly target (%d/%d slots)",
                    fold.delivery_day.date(),
                    int(qh_mask.sum()),
                    expected_hours * 4,
                )
                continue
            try:
                persistence_forecast(prices_qh, fold.delivery_day, tz=tz)
            except ValueError as exc:
                logger.info(
                    "skipping delivery day %s: baseline replica undefined (%s)",
                    fold.delivery_day.date(),
                    exc,
                )
                continue
        else:
            try:
                _hourly_persistence_forecast(y_hourly, fold.delivery_day, tz=tz)
            except ValueError as exc:
                logger.info(
                    "skipping delivery day %s: hourly baseline undefined (%s)",
                    fold.delivery_day.date(),
                    exc,
                )
                continue

        evaluable_folds.append(fold)

    n_folds = len(evaluable_folds)
    if n_folds == 0:
        raise ValueError("no evaluable folds after applying skip rules -- check data coverage")
    logger.info(
        "live-gate backtest: %d evaluable folds (of %d candidates) start=%s end=%s",
        n_folds,
        len(candidate_folds),
        evaluable_folds[0].delivery_day.date(),
        evaluable_folds[-1].delivery_day.date(),
    )

    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        day_start = pd.Timestamp(fold.delivery_day.date(), tz=tz)
        day_end = day_start + pd.DateOffset(days=1)

        if i % refit_every == 0:
            y_train = y_hourly.reindex(fold.train_index)
            # Whole-day-out policy for training rows too (not just delivery
            # days): a day excluded from live_matrix reindexes to an
            # all-NaN row here, which dropna() removes -- consistent with
            # never silently filling incomplete NWP reconstruction (spec
            # 6.5.3 section 3.3), not a workaround for LightGBM's (separate,
            # legitimate) tolerance for missing individual feature values.
            x_train_live = live_matrix.reindex(fold.train_index).dropna()
            x_train_original = original_matrix.reindex(fold.train_index).dropna()
            model_live.fit(y_train.loc[x_train_live.index], x_train_live)
            model_original.fit(y_train.loc[x_train_original.index], x_train_original)

        x_test_live = live_matrix.loc[fold.test_index]
        x_test_original = original_matrix.loc[fold.test_index]
        history = y_hourly.loc[fold.train_index]
        pred_live_hourly = model_live.predict(fold.test_index, history=history, x_test=x_test_live)
        pred_original_hourly = model_original.predict(
            fold.test_index, history=history, x_test=x_test_original
        )

        if resolution == "quarterhourly":
            assert prices_qh is not None and shape_window_days is not None
            end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
            profile = fit_shape_profile(
                prices_qh, end_day=end_day_minus_1, n_days=shape_window_days, tz=tz
            )
            pred_live = expand_to_quarterhour(
                pred_live_hourly, profile, target_day=fold.delivery_day, tz=tz
            )
            pred_original = expand_to_quarterhour(
                pred_original_hourly, profile, target_day=fold.delivery_day, tz=tz
            )
            pred_baseline = persistence_forecast(prices_qh, fold.delivery_day, tz=tz)
            qh_index = pd.DatetimeIndex(prices_qh.index)
            y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()
        else:
            pred_live = pred_live_hourly
            pred_original = pred_original_hourly
            pred_baseline = _hourly_persistence_forecast(y_hourly, fold.delivery_day, tz=tz)
            y_true_day = y_hourly.loc[fold.test_index].sort_index()

        records.append(
            pd.DataFrame(
                {
                    "y_true": y_true_day.to_numpy(),
                    "pred_original": pred_original.reindex(y_true_day.index).to_numpy(),
                    "pred_live": pred_live.reindex(y_true_day.index).to_numpy(),
                    "pred_baseline": pred_baseline.reindex(y_true_day.index).to_numpy(),
                    "delivery_day": fold.delivery_day,
                    "n_slots_in_day": len(y_true_day),
                },
                index=y_true_day.index,
            )
        )

        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            logger.info("live-gate backtest progress: %d/%d folds", i + 1, n_folds)

    result = pd.concat(records).sort_index()
    return result[_LIVE_GATE_OUTPUT_COLUMNS]
