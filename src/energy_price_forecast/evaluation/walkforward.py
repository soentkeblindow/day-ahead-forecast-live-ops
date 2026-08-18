import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from ..market_time import GATE_CLOSURE_LOCAL_HOUR, LOCAL_TZ

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward step: forecast a single delivery day D.

    delivery_day : local date of D (tz-aware Europe/Berlin midnight).
    train_index  : UTC timestamps of delivery days < D the model may learn from.
                   Expanding: all days from series start to D-1.
                   Rolling: the trailing train_span_days local days before D.
    test_index   : UTC timestamps of delivery day D (23/24/25 hours across DST).
    gate_closure : UTC instant of 12:00 Europe/Berlin on D-1. Documents the
                   information contract; no model input may postdate it. The price
                   split is enforced by the delivery-day boundary; the feature
                   contract is checked against this instant in step 2.3.
    """

    delivery_day: pd.Timestamp
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    gate_closure: pd.Timestamp


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    window: Literal["expanding", "rolling"] = "expanding",
    train_span_days: int | None = None,
) -> Iterator[Fold]:
    """Yield one Fold per delivery day in [test_start, test_end].

    Delivery days are Europe/Berlin local calendar days. The UTC index is grouped
    by local date so each test_index is exactly one traded day-ahead block
    (23/24/25 hours across DST). For each delivery day D: train_index covers
    delivery days < D, test_index is day D, gate_closure is 12:00 local on D-1.

    Raises ValueError if the index is not UTC, or if window == "rolling" and
    train_span_days is None. Rolling folds without a full trailing window (early
    in the series) are skipped, so every yielded fold has a complete training window.
    """
    if str(index.tz) != "UTC":
        raise ValueError("index must be a UTC tz-aware DatetimeIndex")
    if window == "rolling" and train_span_days is None:
        raise ValueError("train_span_days is required when window == 'rolling'")

    local_date = index.tz_convert(LOCAL_TZ).normalize()  # local midnight per row
    all_days = pd.DatetimeIndex(local_date.unique()).sort_values()

    start = pd.Timestamp(test_start, tz=LOCAL_TZ).normalize()
    end = (
        pd.Timestamp(test_end, tz=LOCAL_TZ).normalize() if test_end is not None else all_days.max()
    )
    test_days = all_days[(all_days >= start) & (all_days <= end)]

    for d in test_days:
        prev_days = all_days[all_days < d]
        if len(prev_days) == 0:
            continue  # no prior day; cannot build training set or gate_closure
        prev_day = prev_days[-1]

        if window == "expanding":
            train_mask = local_date < d
        else:  # rolling
            # pd.DateOffset respects DST so lower lands at local midnight
            assert train_span_days is not None  # guarded by ValueError above
            lower = d - pd.DateOffset(days=train_span_days)
            if lower < all_days.min():
                continue  # full rolling window not yet available; skip
            train_mask = (local_date >= lower) & (local_date < d)

        test_index = index[local_date == d]
        train_index = index[train_mask]
        if len(train_index) == 0 or len(test_index) == 0:
            continue

        # 12:00 Europe/Berlin on D-1 is never in the ambiguous 02:00–03:00 DST
        # window, so tz_convert always yields an unambiguous UTC instant.
        gate_closure = (prev_day + pd.Timedelta(hours=GATE_CLOSURE_LOCAL_HOUR)).tz_convert("UTC")
        yield Fold(d, train_index, test_index, gate_closure)


class Forecaster(Protocol):
    """Minimal contract every model must satisfy to run in the harness.

    The runner is the single gatekeeper of information: it only ever passes
    data available at the fold's gate closure. history is the price series on
    delivery days < D; X_train / X_test are leakage-correct feature frames
    (None in Sprint 2, populated from step 2.3 onward). A model uses whatever
    it needs and ignores the rest.
    """

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None: ...

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series: ...


def run_backtest(
    y: pd.Series,
    model: Forecaster,
    folds: Iterable[Fold],
    *,
    refit_every: int = 1,
    x: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drive a model through the walk-forward folds and collect predictions.

    On every refit_every-th fold the model is refit on that fold's train slice
    (freshest history < gate closure); on the intervening folds the previously
    fitted model is reused, but each day is still predicted from its own current
    inputs. Fold 0 always triggers a fit. Returns a tidy frame indexed by the
    UTC test timestamps with columns [y_true, y_pred, delivery_day].
    """
    fold_list = list(folds)
    n_folds = len(fold_list)
    logger.info("starting backtest: %d folds", n_folds)
    _milestones = {max(0, int(n_folds * p) - 1) for p in (0.25, 0.5, 0.75)} | {n_folds - 1}
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(fold_list):
        n_train_days = len(fold.train_index) // 24
        if i % refit_every == 0:
            logger.debug(
                "fold %d/%d %s: fitting on %d train days",
                i + 1,
                n_folds,
                fold.delivery_day.date(),
                n_train_days,
            )
        else:
            logger.debug(
                "fold %d/%d %s: predicting (%d train days, no refit)",
                i + 1,
                n_folds,
                fold.delivery_day.date(),
                n_train_days,
            )
        if i in _milestones:
            logger.info(
                "backtest progress: %d/%d folds (%.0f%%)",
                i + 1,
                n_folds,
                100 * (i + 1) / n_folds,
            )
        history = y.loc[fold.train_index]
        if i % refit_every == 0:
            x_train = x.loc[fold.train_index] if x is not None else None
            model.fit(history, x_train)
        x_test = x.loc[fold.test_index] if x is not None else None
        y_pred = model.predict(fold.test_index, history=history, x_test=x_test)
        records.append(
            pd.DataFrame(
                {
                    "y_true": y.loc[fold.test_index].to_numpy(),
                    "y_pred": np.asarray(y_pred),
                    "delivery_day": fold.delivery_day,
                },
                index=fold.test_index,
            )
        )
    return pd.concat(records).sort_index()
