"""Unit tests for scripts/run_daily_submission.py (spec 6.7.2, section 5.1).
run_renewables_backtest itself is mocked throughout -- a real call needs 365+
real days of weather/capacity data and takes real wall-clock minutes; these
tests verify the window arithmetic and wiring around it, not the renewables
model itself (that is evaluation/renewables_walkforward.py's own test
suite's job).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.arena.catalog import ChallengeSpec
from energy_price_forecast.arena.live_inputs import (
    DEFAULT_WEATHER_MODEL,
    assemble_price_model_inputs,
    read_quarterhourly_prices,
    read_weather_runs,
)
from energy_price_forecast.arena.payload import PayloadValidationError
from energy_price_forecast.arena.preflight import PreflightResult
from energy_price_forecast.arena.submit import SubmissionResult
from energy_price_forecast.data._weather_cache import cache_path, write_cached_run
from energy_price_forecast.data.capacity import CapacitySource
from energy_price_forecast.data.weather_client import run_init_for_target_day
from energy_price_forecast.data.weather_grid import expected_columns
from energy_price_forecast.evaluation.walkforward import Fold
from energy_price_forecast.features.build import build_feature_set_for_day
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.ops import protocol, store
from energy_price_forecast.ops.store_sources import EntsoeSource
from energy_price_forecast.ops.windows import LOCAL_TZ, local_day_bounds
from scripts.run_daily_submission import (
    PRICE_TRAIN_SPAN_DAYS,
    RENEWABLES_TRAIN_SPAN_DAYS,
    SubmissionOutcome,
    _silence_turns_run_red,
    accepted_earlier_today,
    build_arena_payload,
    build_price_feature_matrix,
    build_submission_record,
    check_a_inputs,
    fit_predict_expand,
    holiday_calendar_covers_target,
    is_past_gate_closure,
    renewables_window,
    run_daily_submission,
    run_daily_submission_smoke,
    run_renewables_step,
    run_smoke_submission_for_day,
    run_submission_for_day,
    smoke_target_day,
    source_ages_from_manifest,
    write_payload_to_repo,
)

CHALLENGE = ChallengeSpec(
    challenge_id="2",
    name="Day-Ahead Prices | Germany-Luxembourg | Point Forecast",
    timezone="Europe/Berlin",
    resolution_minutes=15,
    precision_decimals=2,
    allow_negative=True,
    max_forecast_points=None,
    raw={
        "submission_window": {"allow_multiple": True, "selection_policy": "latest_before_deadline"},
    },
)


def test_renewables_window_spans_both_training_lookbacks() -> None:
    target_day = dt.date(2026, 9, 20)

    start, end = renewables_window(target_day)

    # end is target_day's own LAST UTC hour (2026-09-20 is CEST, UTC+2), not UTC
    # midnight of the same calendar date -- the real bug found live during the
    # wiring probe (see renewables_window's own docstring).
    assert end == pd.Timestamp("2026-09-20T21:00:00", tz="UTC")
    expected_start = end - pd.Timedelta(days=PRICE_TRAIN_SPAN_DAYS + RENEWABLES_TRAIN_SPAN_DAYS)
    assert start == expected_start


def test_renewables_window_end_covers_the_full_local_target_day() -> None:
    """The window's end must be late enough that target_day's own local
    24-hour block is fully inside [start, end] -- the exact property whose
    absence caused target_day itself to be wrongly excluded on the real
    store (see renewables_window's docstring)."""
    target_day = dt.date(2026, 9, 20)
    _, end = renewables_window(target_day)

    day_start_local, day_end_local = local_day_bounds(target_day)
    last_hour_of_target_day = day_end_local.tz_convert("UTC") - pd.Timedelta(hours=1)

    assert end >= last_hour_of_target_day
    assert day_start_local.tz_convert("UTC") <= end


def test_run_renewables_step_slices_to_the_window_and_measures_runtime() -> None:
    target_day = dt.date(2026, 9, 20)
    start, end = renewables_window(target_day)

    # df/weather deliberately span far outside the needed window -- proves
    # the slice actually happens rather than passing the whole frame through.
    wide_index = pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq="h")
    df = pd.DataFrame(
        {
            "wind_onshore_forecast": 1.0,
            "wind_offshore_forecast": 2.0,
            "solar_forecast": 3.0,
            "day_ahead_price": 50.0,
        },
        index=wide_index,
    )
    run_inits = pd.date_range(start - pd.Timedelta(days=5), end + pd.Timedelta(days=5), freq="D")
    weather_index = pd.MultiIndex.from_arrays(
        [run_inits, run_inits], names=["run_init_utc", "valid_time_utc"]
    )
    weather = pd.DataFrame({"dummy_col": 0.0}, index=weather_index)

    fake_predictions = pd.DataFrame({"solar_cf_pred": [0.5]})
    with patch(
        "scripts.run_daily_submission.run_renewables_backtest", return_value=fake_predictions
    ) as mock_backtest:
        predictions, runtime_seconds = run_renewables_step(df, weather, target_day)

    assert predictions is fake_predictions
    assert runtime_seconds >= 0.0
    mock_backtest.assert_called_once()
    call_kwargs = mock_backtest.call_args.kwargs
    assert call_kwargs["window"] == "rolling"
    assert call_kwargs["train_span_days"] == RENEWABLES_TRAIN_SPAN_DAYS
    assert call_kwargs["refit_every"] == 1
    assert call_kwargs["objective"] == "l2"

    target_hourly_arg, weather_arg = mock_backtest.call_args.args
    assert list(target_hourly_arg.columns) == [
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
    ]
    assert target_hourly_arg.index.min() >= start
    assert target_hourly_arg.index.max() <= end
    passed_run_inits = pd.DatetimeIndex(weather_arg.index.get_level_values("run_init_utc"))
    assert passed_run_inits.min() >= start


# ---------------------------------------------------------------------------
# build_price_feature_matrix
# ---------------------------------------------------------------------------

# A DST-free week so every local day has exactly 24 hours -- keeps the
# expected row-count arithmetic in these tests simple; DST-day row counts are
# evaluation/walkforward.py's and models/bridge.py's own test suites' job.
_TARGET_DAY = dt.date(2026, 7, 15)
_AS_OF = pd.Timestamp("2026-07-14T10:00", tz="UTC")
# _AS_OF is exactly _TARGET_DAY's own gate closure moment (see _GATE_CLOSURE_UTC
# below -- same value). spec 6.7.3 section 2.4's second, pre-POST gate check reads a
# SEPARATE, later wall-clock moment (``now``) -- tests that need run_submission_for_day
# to actually reach submit() must inject a ``now`` comfortably before closure, since the
# real default (real wall-clock "now") would otherwise always see _TARGET_DAY as long
# past its (fixed, 2026) deadline.
_BEFORE_GATE_CLOSURE = _AS_OF - pd.Timedelta(hours=1)


def _fake_feature_row(day: dt.date, df: pd.DataFrame, renewables: pd.DataFrame) -> pd.DataFrame:
    index = pd.date_range(
        pd.Timestamp(day, tz=LOCAL_TZ), periods=24, freq="h", tz=LOCAL_TZ
    ).tz_convert("UTC")
    return pd.DataFrame({"feat_a": 1.0}, index=index)


def _hourly_price_df(start: str, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), end_exclusive, freq="h", inclusive="left")
    return pd.DataFrame({"day_ahead_price": 50.0}, index=idx)


def test_build_price_feature_matrix_covers_training_window_and_target_day() -> None:
    df = _hourly_price_df("2026-01-01", pd.Timestamp(_TARGET_DAY, tz="UTC") + pd.Timedelta(days=1))
    renewables_predictions = pd.DataFrame()

    with patch(
        "scripts.run_daily_submission.build_feature_set_for_day", side_effect=_fake_feature_row
    ) as mock_builder:
        matrix, fold, excluded = build_price_feature_matrix(df, renewables_predictions, _TARGET_DAY)

    assert fold.delivery_day.date() == _TARGET_DAY
    assert excluded == set()
    first_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().min().date()
    expected_n_days = (_TARGET_DAY - first_train_day).days + 1
    assert mock_builder.call_count == expected_n_days
    assert mock_builder.call_args_list[-1].args[0] == _TARGET_DAY
    assert len(matrix) == expected_n_days * 24
    assert fold.test_index.isin(matrix.index).all()


def test_build_price_feature_matrix_excludes_a_day_with_incomplete_reconstruction() -> None:
    df = _hourly_price_df("2026-01-01", pd.Timestamp(_TARGET_DAY, tz="UTC") + pd.Timedelta(days=1))
    renewables_predictions = pd.DataFrame()
    bad_day = _TARGET_DAY - dt.timedelta(days=5)

    def builder_with_one_bad_day(
        day: dt.date, df_arg: pd.DataFrame, renewables_arg: pd.DataFrame
    ) -> pd.DataFrame:
        if day == bad_day:
            raise IncompleteReconstructionError("synthetic gap for this test")
        return _fake_feature_row(day, df_arg, renewables_arg)

    with patch(
        "scripts.run_daily_submission.build_feature_set_for_day",
        side_effect=builder_with_one_bad_day,
    ):
        matrix, fold, excluded = build_price_feature_matrix(df, renewables_predictions, _TARGET_DAY)

    assert excluded == {bad_day}
    first_train_day = fold.train_index.tz_convert(LOCAL_TZ).normalize().min().date()
    expected_n_days = (_TARGET_DAY - first_train_day).days + 1
    # One fewer day's worth of rows than the full window -- the excluded
    # day's 24 hours never entered the matrix at all (not filled with NaN).
    assert len(matrix) == (expected_n_days - 1) * 24


# ---------------------------------------------------------------------------
# fit_predict_expand
# ---------------------------------------------------------------------------


def test_fit_predict_expand_returns_a_full_quarterhourly_day() -> None:
    df = _hourly_price_df("2026-01-01", pd.Timestamp(_TARGET_DAY, tz="UTC") + pd.Timedelta(days=1))
    renewables_predictions = pd.DataFrame()

    with patch(
        "scripts.run_daily_submission.build_feature_set_for_day", side_effect=_fake_feature_row
    ):
        matrix, fold, _excluded = build_price_feature_matrix(
            df, renewables_predictions, _TARGET_DAY
        )

    # 30 days of quarter-hourly history ending the day before target_day --
    # comfortably more than SHAPE_WINDOW_DAYS=28 needs, with a mild intra-day
    # pattern so the shape profile isn't degenerate.
    qh_start = pd.Timestamp(_TARGET_DAY, tz="UTC") - pd.Timedelta(days=30)
    qh_end = pd.Timestamp(_TARGET_DAY, tz="UTC")
    qh_index = pd.date_range(qh_start, qh_end, freq="15min", inclusive="left")
    prices_qh = pd.DataFrame(
        {"day_ahead_price": 50.0 + (qh_index.hour % 4).astype(float)}, index=qh_index
    )

    result, n_training_rows, n_training_labels = fit_predict_expand(df, matrix, fold, prices_qh)

    assert len(result) == 96  # DST-free day
    assert result.notna().all()
    assert pd.DatetimeIndex(result.index).tz is not None
    # Restarbeit Teil A.5: complete data -> both counters equal.
    assert n_training_rows == n_training_labels
    assert n_training_rows == len(matrix.reindex(fold.train_index).dropna())


def test_fit_predict_expand_drops_a_training_day_with_missing_price() -> None:
    """docs/sprint6_fix_partial_today.md section 3.1: assemble_price_model_inputs
    no longer drops a row missing only its price (a genuine real gap, or a
    genuine future target day), so a training-window day without a real
    price can now reach fit_predict_expand -- must not crash or fabricate a
    value. Confirmed (2026-09-13) that models/lgbm.py::LGBMForecaster.fit
    already masks out any NaN-target row before calling LightGBM, so no
    change to fit_predict_expand itself was needed; this test locks in that
    existing protection now that a NaN-price row can actually reach it."""
    df = _hourly_price_df("2026-01-01", pd.Timestamp(_TARGET_DAY, tz="UTC") + pd.Timedelta(days=1))
    gap_day = _TARGET_DAY - dt.timedelta(days=10)
    gap_start = pd.Timestamp(gap_day, tz=LOCAL_TZ).tz_convert("UTC")
    gap_end = gap_start + pd.Timedelta(hours=24)
    df.loc[(df.index >= gap_start) & (df.index < gap_end), "day_ahead_price"] = float("nan")
    renewables_predictions = pd.DataFrame()

    with patch(
        "scripts.run_daily_submission.build_feature_set_for_day", side_effect=_fake_feature_row
    ):
        matrix, fold, _excluded = build_price_feature_matrix(
            df, renewables_predictions, _TARGET_DAY
        )

    qh_start = pd.Timestamp(_TARGET_DAY, tz="UTC") - pd.Timedelta(days=30)
    qh_end = pd.Timestamp(_TARGET_DAY, tz="UTC")
    qh_index = pd.date_range(qh_start, qh_end, freq="15min", inclusive="left")
    prices_qh = pd.DataFrame(
        {"day_ahead_price": 50.0 + (qh_index.hour % 4).astype(float)}, index=qh_index
    )

    # Must not raise (LightGBM rejects a NaN target) and must still produce
    # a complete day -- the gap day is silently excluded from training, not
    # fabricated and not allowed to crash the fit.
    result, n_training_rows, n_training_labels = fit_predict_expand(df, matrix, fold, prices_qh)

    assert len(result) == 96
    assert result.notna().all()
    # Restarbeit Teil A.5: the gap day's 24 hours stay in the feature matrix
    # (features unaffected by the price gap) but lose their label -- checked
    # on both sides, not just the difference, so the counter can't be
    # accidentally len(matrix) computed twice.
    assert n_training_rows == len(matrix.reindex(fold.train_index).dropna())
    assert n_training_labels == n_training_rows - 24


# ---------------------------------------------------------------------------
# build_arena_payload
# ---------------------------------------------------------------------------


def _quarterhourly_series(target_day: dt.date, value: float = 50.0) -> pd.Series:
    day_start = pd.Timestamp(target_day, tz=LOCAL_TZ)
    index = pd.date_range(day_start, periods=96, freq="15min", tz=LOCAL_TZ).tz_convert("UTC")
    return pd.Series(value, index=index)


def test_build_arena_payload_fetches_challenge_and_builds_a_valid_payload() -> None:
    forecast = _quarterhourly_series(_TARGET_DAY)

    with patch(
        "scripts.run_daily_submission.get_challenge", return_value=CHALLENGE
    ) as mock_get_challenge:
        payload, challenge = build_arena_payload(forecast, _TARGET_DAY)

    mock_get_challenge.assert_called_once_with("2")
    assert challenge is CHALLENGE
    assert payload["challenge_id"] == "2"
    assert len(payload["values"]) == 96
    assert all(v == 50.0 for v in payload["values"])
    assert payload["target_start"] == pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat()


def test_build_arena_payload_raises_on_wrong_value_count() -> None:
    # 48 values for a 96-value day -- must be caught by validate_payload, not
    # silently accepted.
    forecast = _quarterhourly_series(_TARGET_DAY).iloc[:48]

    with (
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
        pytest.raises(PayloadValidationError),
    ):
        build_arena_payload(forecast, _TARGET_DAY)


# ---------------------------------------------------------------------------
# holiday_calendar_covers_target
# ---------------------------------------------------------------------------


def test_holiday_calendar_covers_a_real_near_future_year() -> None:
    assert holiday_calendar_covers_target(_TARGET_DAY) is True


def test_holiday_calendar_does_not_cover_an_implausible_year() -> None:
    # Long before the modern German public-holiday calendar existed --
    # the holidays package returns an empty set rather than raising.
    assert holiday_calendar_covers_target(dt.date(1500, 1, 1)) is False


# ---------------------------------------------------------------------------
# check_a_inputs
# ---------------------------------------------------------------------------


def test_check_a_inputs_assembles_reads_and_delegates_to_check_reconstruction_inputs() -> None:
    fake_weather_run = pd.DataFrame({"col": [1.0]})
    fake_anchor_valid_until = pd.Timestamp("2099-01-01", tz="UTC")

    with (
        patch(
            "scripts.run_daily_submission.read_cached_run", return_value=fake_weather_run
        ) as mock_read,
        patch(
            "scripts.run_daily_submission.anchor_table_valid_until",
            return_value=fake_anchor_valid_until,
        ) as mock_anchor,
        patch(
            "scripts.run_daily_submission.check_reconstruction_inputs",
            return_value=PreflightResult(ok=True),
        ) as mock_check,
    ):
        result = check_a_inputs(_TARGET_DAY)

    assert result.ok
    mock_read.assert_called_once()
    mock_anchor.assert_called_once_with(CapacitySource.PUBLIC_REGISTRY)
    mock_check.assert_called_once_with(
        weather_run=fake_weather_run,
        target_day=_TARGET_DAY,
        anchor_valid_until=fake_anchor_valid_until,
        holiday_calendar_covers_target=True,
    )


# ---------------------------------------------------------------------------
# run_submission_for_day
# ---------------------------------------------------------------------------


def _fake_fold(target_day: dt.date, train_days: int = PRICE_TRAIN_SPAN_DAYS) -> Fold:
    delivery_day = pd.Timestamp(target_day, tz=LOCAL_TZ)
    test_index = pd.date_range(delivery_day, periods=24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    train_start = delivery_day - pd.Timedelta(days=train_days)
    train_index = pd.date_range(
        train_start, periods=train_days * 24, freq="h", tz=LOCAL_TZ
    ).tz_convert("UTC")
    return Fold(
        delivery_day=delivery_day,
        train_index=train_index,
        test_index=test_index,
        gate_closure=delivery_day - pd.Timedelta(hours=12),
    )


def test_run_submission_for_day_stops_at_check_a_without_running_renewables() -> None:
    with (
        patch(
            "scripts.run_daily_submission.check_a_inputs",
            return_value=PreflightResult(ok=False, reasons=("no weather run for D-1",)),
        ),
        patch("scripts.run_daily_submission.run_renewables_step") as mock_renewables,
    ):
        outcome = run_submission_for_day(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert outcome.candidate_selected is None
    assert outcome.skip_reason is not None
    assert "Check A" in outcome.skip_reason
    assert "no weather run for D-1" in outcome.skip_reason
    mock_renewables.assert_not_called()


def test_run_submission_for_day_stops_at_training_extent() -> None:
    fold = _fake_fold(_TARGET_DAY)
    # Drop the first 10 training days entirely -- simulates a frozen source
    # silently shortening the window (spec section 2.4), never filled with NaN.
    kept_train_index = fold.train_index[10 * 24 :]
    matrix_index = kept_train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch("scripts.run_daily_submission.fit_predict_expand") as mock_fit,
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert outcome.candidate_selected is None
    assert outcome.skip_reason is not None
    assert "training extent" in outcome.skip_reason
    assert outcome.renewables_runtime_seconds == 1.5
    mock_fit.assert_not_called()


def test_run_submission_for_day_tolerates_a_known_defect_gap_in_training_window() -> None:
    """The real 2026-09-12 finding: a training window missing exactly the
    hours of a documented, permanent weather defect (2026-06-24, its own
    D-1 run_init 2026-06-23 is a KNOWN_WEATHER_DEFECTS entry) plus its
    real knock-on (2026-06-25, no D-1 persistence-lag value) must NOT be
    treated as a frozen source -- unlike
    test_run_submission_for_day_stops_at_training_extent's unexplained
    10-day gap, which must still block.
    """
    fold = _fake_fold(_TARGET_DAY)
    gap_days = {dt.date(2026, 6, 24), dt.date(2026, 6, 25)}
    local_days = fold.train_index.tz_convert(LOCAL_TZ).normalize()
    keep_mask = [day not in gap_days for day in local_days.date]
    kept_train_index = fold.train_index[keep_mask]
    matrix_index = kept_train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    forecast = _quarterhourly_series(_TARGET_DAY)
    payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        "values": [50.0 + i * 0.01 for i in range(96)],
    }

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, gap_days),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(sent=False, challenge_id="2"),
        ),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
        )

    assert outcome.skip_reason is None
    assert outcome.candidate_selected == "full_live_set"
    assert outcome.excluded_training_days == frozenset(gap_days)


def test_run_submission_for_day_stops_at_check_b_with_named_missing_feature() -> None:
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    # Blank out feat_a for target_day's own hours only -- Check B must catch
    # this even though every training-day value is present.
    matrix.loc[fold.test_index, "feat_a"] = float("nan")
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert outcome.candidate_selected is None
    assert outcome.skip_reason is not None
    assert "Check B" in outcome.skip_reason
    assert "feat_a" in outcome.missing_features


def test_run_submission_for_day_stops_at_payload_plausibility() -> None:
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    absurd_forecast = _quarterhourly_series(_TARGET_DAY, value=5000.0)
    absurd_payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        "values": [5000.0] * 96,
    }

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand",
            return_value=(absurd_forecast, 2160, 2160),
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload",
            return_value=(absurd_payload, CHALLENGE),
        ),
        patch("scripts.run_daily_submission.submit") as mock_submit,
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert outcome.candidate_selected is not None  # candidate was selected...
    assert outcome.skip_reason is not None
    assert "payload plausibility" in outcome.skip_reason  # ...but the payload itself is rejected
    mock_submit.assert_not_called()


def test_run_submission_for_day_happy_path_submits_dry_run() -> None:
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    forecast = _quarterhourly_series(_TARGET_DAY)
    payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        # Non-degenerate values -- an all-identical payload would (correctly)
        # fail check_payload_plausibility's own wiring-fault guard.
        "values": [50.0 + i * 0.01 for i in range(96)],
    }
    fake_submission_result = SubmissionResult(sent=False, challenge_id="2")

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, {dt.date(2026, 4, 20)}),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit", return_value=fake_submission_result
        ) as mock_submit,
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
        )

    assert outcome.candidate_selected == "full_live_set"
    assert outcome.skip_reason is None
    assert outcome.payload == payload
    assert outcome.submission_result is fake_submission_result
    assert outcome.renewables_runtime_seconds == 1.5
    assert outcome.excluded_training_days == frozenset({dt.date(2026, 4, 20)})
    mock_submit.assert_called_once()
    assert mock_submit.call_args.kwargs["live"] is False


def test_run_submission_for_day_carries_commodity_staleness_warnings_through() -> None:
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    # ttf_gas_eur_per_mwh's last real value is 5 days before as_of -- inside the 7-day ffill
    # limit (so it still ffills into every feature row, Check B sees no NaN), but at/beyond
    # COMMODITY_STALENESS_WARN_DAYS=4 -- must surface as a warning, not silently disappear.
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    df["ttf_gas_eur_per_mwh"] = float("nan")
    df.loc[
        matrix_index[matrix_index <= _AS_OF - pd.Timedelta(days=5)][-1], "ttf_gas_eur_per_mwh"
    ] = 30.0

    forecast = _quarterhourly_series(_TARGET_DAY)
    payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        "values": [50.0 + i * 0.01 for i in range(96)],
    }

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(sent=False, challenge_id="2"),
        ),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
        )

    assert "ttf_gas_eur_per_mwh" in outcome.commodity_staleness_warnings
    assert outcome.commodity_staleness_warnings["ttf_gas_eur_per_mwh"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Restarbeit Teil B.2 (docs/sprint6_fix_partial_today.md section 4.1): the
# real morning state -- "today" (the training window's own last day) has a
# complete day_ahead_price/14.1.D forecast, but its own actuals/physical-flow
# columns only reach ~11:00 local. A full run_submission_for_day over this
# must produce a payload with today inside the training window. Unlike the
# other run_submission_for_day tests in this file, build_price_feature_matrix
# and fit_predict_expand run for REAL here (not mocked) -- that is the whole
# point: only the real features/build.py + check_training_extent + LightGBM
# fit path can prove today's row survives.
#
# Deliberately does NOT re-derive df via the real assemble_price_model_inputs
# (df is built directly, already in the shape a correctly-fixed
# assemble_price_model_inputs produces) -- that function's own row-keeping
# fix already has its own direct test
# (tests/test_live_inputs.py::test_assemble_price_model_inputs_keeps_a_row_missing_only_price).
# run_renewables_step stays mocked, returning today's renewables prediction
# directly, per this file's own stated convention -- and confirmed while
# building this test (evaluation/renewables_walkforward.py::_build_target_table
# already judges completeness against `raw = target_hourly[TARGET_COLUMNS[target]]`,
# its own single target column, never the whole row -- section 3.2's "falls
# P2 zeigt, dass die Strenge in der Zieltabelle sitzt" branch was not taken;
# the actual bug lived entirely in assemble_price_model_inputs).
# ---------------------------------------------------------------------------

# Small, not the real 90/365 -- comfortably more than FeatureConfig's own
# 191h/~8-day max lookback (section 4.1: "wenige Tage Historie").
_SMALL_PRICE_TRAIN_SPAN_DAYS = 15
_HISTORY_DAYS = 40  # >= SHAPE_WINDOW_DAYS (28) + _SMALL_PRICE_TRAIN_SPAN_DAYS margin


def _build_partial_today_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dt.date]:
    """The section 4.1 table, literally: today's day_ahead_price/14.1.D-style
    forecast columns are complete; today's actuals/physical-flow columns are
    NaN from 11:00 local onward. target_day's own day_ahead_price is NaN
    (the real future-price state), everything else about target_day is
    normal. Returns (df, renewables_predictions, today)."""
    target_day = _TARGET_DAY
    today = target_day - dt.timedelta(days=1)
    history_start = target_day - dt.timedelta(days=_HISTORY_DAYS - 1)
    idx = pd.date_range(
        pd.Timestamp(history_start, tz=LOCAL_TZ),
        pd.Timestamp(target_day + dt.timedelta(days=1), tz=LOCAL_TZ),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")

    # Same raw-column shape as tests/test_arena_walkforward.py's own
    # _build_full_hourly_df -- every column build_feature_set_for_day needs,
    # constant-filled.
    df = pd.DataFrame(
        {
            # A real hour-of-day pattern, not a flat constant -- a constant
            # target gives LightGBM nothing to learn from, which trips
            # check_payload_plausibility's own degenerate-payload guard
            # ("all values identical") for the wrong reason (no signal to
            # fit, not a wiring fault). Structural realism only, per this
            # test's own point (section 4.1: "prueft die Struktur, nicht die
            # Zahlenqualitaet").
            "day_ahead_price": 50.0 + 10.0 * np.sin(2 * np.pi * idx.hour / 24),
            "load_actual": np.full(len(idx), 40000.0),
            "load_forecast_day_ahead": np.full(len(idx), 40000.0),
            "gen_wind_onshore": np.full(len(idx), 8000.0),
            "wind_onshore_forecast": np.full(len(idx), 8000.0),
            "gen_wind_offshore": np.full(len(idx), 2000.0),
            "wind_offshore_forecast": np.full(len(idx), 2000.0),
            "gen_solar": np.full(len(idx), 5000.0),
            "solar_forecast": np.full(len(idx), 5000.0),
            "scheduled_net_de_to_AT": np.full(len(idx), 1000.0),
            "scheduled_net_de_to_BE": np.full(len(idx), 500.0),
            "physical_net_de_to_AT": np.full(len(idx), 1200.0),
            "physical_net_de_to_BE": np.full(len(idx), 600.0),
            "ttf_gas_eur_per_mwh": np.full(len(idx), 30.0),
            "eua_co2_eur_per_t": np.full(len(idx), 70.0),
        },
        index=idx,
    )

    # target_day's own price genuinely doesn't exist yet (that's the whole
    # point of forecasting it) -- never a feature, only ever a label for
    # some future training window, so NaN-ing it out cannot itself affect
    # target_day's own feature row.
    target_day_mask = pd.DatetimeIndex(df.index).tz_convert(LOCAL_TZ).normalize() == pd.Timestamp(
        target_day, tz=LOCAL_TZ
    )
    df.loc[target_day_mask, "day_ahead_price"] = float("nan")

    # today's actuals/physical-flow columns: real only through ~11:00 local
    # (section 4.1's table row 5) -- day_ahead_price/14.1.D-style forecast
    # columns above are deliberately left untouched for today (rows 1-2 of
    # the same table: both are already fully published by now).
    today_local = pd.DatetimeIndex(df.index).tz_convert(LOCAL_TZ)
    today_after_11 = (today_local.normalize() == pd.Timestamp(today, tz=LOCAL_TZ)) & (
        today_local.hour >= 11
    )
    actuals_columns = [
        "load_actual",
        "gen_wind_onshore",
        "gen_wind_offshore",
        "gen_solar",
        "physical_net_de_to_AT",
        "physical_net_de_to_BE",
    ]
    df.loc[today_after_11, actuals_columns] = float("nan")

    # renewables_predictions covers the whole history through target_day,
    # INCLUDING today -- section 3.2's fix (the renewables target table
    # judges completeness by its own 3 target columns, not the whole row)
    # is what makes this real in production; here it is the mocked
    # run_renewables_step's return value, since that fix's own correctness
    # is evaluation/renewables_walkforward.py's test suite's job, not this
    # file's (see this file's own module docstring).
    frames = []
    for day_offset in range(_HISTORY_DAYS + 1):
        day = history_start + dt.timedelta(days=day_offset)
        day_start = pd.Timestamp(day, tz=LOCAL_TZ)
        day_end = pd.Timestamp(day + dt.timedelta(days=1), tz=LOCAL_TZ)
        target_index = pd.date_range(day_start, day_end, freq="h", inclusive="left").tz_convert(
            "UTC"
        )
        run_init = run_init_for_target_day(day)
        pred_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(target_index), tz="UTC"), target_index],
            names=["run_init_utc", "valid_time_utc"],
        )
        frames.append(
            pd.DataFrame(
                {
                    "wind_onshore_mw_pred": np.full(len(target_index), 8000.0),
                    "wind_offshore_mw_pred": np.full(len(target_index), 2000.0),
                    "solar_mw_pred": np.full(len(target_index), 5000.0),
                },
                index=pred_index,
            )
        )
    renewables_predictions = pd.concat(frames)

    return df, renewables_predictions, today


def test_run_submission_for_day_includes_the_real_morning_state_of_today() -> None:
    """docs/sprint6_fix_partial_today.md section 4.1 -- the fixture that
    would have caught this a third time. as_of pinned to a real submission
    slot (section 3.4 / this project's own convention against reading the
    wall clock in any probe or test)."""
    df, renewables_predictions, today = _build_partial_today_fixture()
    as_of = pd.Timestamp(_TARGET_DAY - dt.timedelta(days=1), tz=LOCAL_TZ).replace(
        hour=10, minute=40
    )

    prices_qh = df["day_ahead_price"].resample("15min").ffill().to_frame()

    with (
        patch("scripts.run_daily_submission.PRICE_TRAIN_SPAN_DAYS", _SMALL_PRICE_TRAIN_SPAN_DAYS),
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(renewables_predictions, 1.5),
        ),
        # get_challenge is a real, read-only Arena API call in production
        # (build_arena_payload's own docstring) -- mocked here the same way
        # test_build_arena_payload_fetches_challenge_and_builds_a_valid_payload
        # already does, so this test doesn't depend on network access or
        # ARENA_API_BASE_URL being set (a real regression: this call was left
        # unmocked in the first version of this test, which passed locally
        # only because this machine happens to have that env var set, then
        # failed in CI where it doesn't -- build_payload/validate_payload
        # below still run for real).
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            prices_qh,
            _TARGET_DAY,
            as_of=as_of.tz_convert("UTC"),
            now=lambda: as_of.tz_convert("UTC"),
        )

    assert outcome.skip_reason is None, outcome.skip_reason
    assert outcome.candidate_selected == "full_live_set"
    assert outcome.payload is not None
    assert len(outcome.payload["values"]) == 96
    # The actual point: today is NOT excluded from the training window,
    # despite its own actuals/flow columns being NaN from 11:00 local
    # onward.
    assert today not in outcome.excluded_training_days
    # And today's label really was fitted, not silently masked out --
    # cross-checks Restarbeit Teil A's own counters against this fixture.
    assert outcome.n_training_rows is not None
    assert outcome.n_training_labels == outcome.n_training_rows


def test_todays_feature_row_is_unaffected_by_todays_own_actuals_being_empty() -> None:
    """docs/sprint6_fix_partial_today.md section 4.3, point 4 -- the
    empirical 48-hour-lag proof, not just an argument. Every lag on a
    lagging-actuals series (load_actual, gen_*, physical_net_*) is >=48h
    (features/config.py::FeatureConfig.max_lookback_hours's own inputs),
    reaching back past today entirely -- so emptying ALL 24 of today's own
    hours in those columns (not just from 11:00 onward, unlike the section
    4.1 fixture above) must leave today's own feature row bit-identical.
    A future feature rebuild that introduces a shorter lag on one of these
    series would make this test fail, by design."""
    df, renewables_predictions, today = _build_partial_today_fixture()
    baseline_row = build_feature_set_for_day(today, df, renewables_predictions)

    actuals_columns = [
        "load_actual",
        "gen_wind_onshore",
        "gen_wind_offshore",
        "gen_solar",
        "physical_net_de_to_AT",
        "physical_net_de_to_BE",
    ]
    df_emptied = df.copy()
    today_local = pd.DatetimeIndex(df_emptied.index).tz_convert(LOCAL_TZ)
    today_all_hours = today_local.normalize() == pd.Timestamp(today, tz=LOCAL_TZ)
    df_emptied.loc[today_all_hours, actuals_columns] = float("nan")

    emptied_row = build_feature_set_for_day(today, df_emptied, renewables_predictions)

    pd.testing.assert_frame_equal(baseline_row, emptied_row)


def test_run_submission_for_day_is_deterministic_on_identical_inputs() -> None:
    """Restarbeit Teil C.2 -- spec section 4 rule 2: two runs over the exact
    same store state and the exact same as_of produce a bit-identical
    payload. This is what makes the three-submission-slot schedule safe (a
    later slot only ever overwrites an earlier one's payload with something
    different if the underlying data actually changed, per
    write_payload_to_repo's own docstring) -- LGBMForecaster's own
    deterministic=True/force_col_wise=True/fixed random_state (models/lgbm.py,
    unchanged) is what this test actually exercises, not just asserts."""
    df, renewables_predictions, _today = _build_partial_today_fixture()
    as_of = pd.Timestamp(_TARGET_DAY - dt.timedelta(days=1), tz=LOCAL_TZ).replace(
        hour=10, minute=40
    )
    prices_qh = df["day_ahead_price"].resample("15min").ffill().to_frame()

    with (
        patch("scripts.run_daily_submission.PRICE_TRAIN_SPAN_DAYS", _SMALL_PRICE_TRAIN_SPAN_DAYS),
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(renewables_predictions, 1.5),
        ),
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
    ):
        outcome_1 = run_submission_for_day(
            df,
            pd.DataFrame(),
            prices_qh,
            _TARGET_DAY,
            as_of=as_of.tz_convert("UTC"),
            now=lambda: as_of.tz_convert("UTC"),
        )
        outcome_2 = run_submission_for_day(
            df,
            pd.DataFrame(),
            prices_qh,
            _TARGET_DAY,
            as_of=as_of.tz_convert("UTC"),
            now=lambda: as_of.tz_convert("UTC"),
        )

    assert outcome_1.skip_reason is None
    assert outcome_1.payload == outcome_2.payload
    assert outcome_1.n_training_rows == outcome_2.n_training_rows
    assert outcome_1.n_training_labels == outcome_2.n_training_labels


# ---------------------------------------------------------------------------
# is_past_gate_closure
# ---------------------------------------------------------------------------

# Gate closure for _TARGET_DAY (2026-07-15, local Europe/Berlin) is 12:00 CEST
# on 2026-07-14 == 10:00 UTC (July is DST, UTC+2).
_GATE_CLOSURE_UTC = pd.Timestamp("2026-07-14T10:00:00", tz="UTC")


def test_is_past_gate_closure_true_at_and_after_deadline() -> None:
    assert is_past_gate_closure(_TARGET_DAY, _GATE_CLOSURE_UTC) is True
    assert is_past_gate_closure(_TARGET_DAY, _GATE_CLOSURE_UTC + pd.Timedelta(seconds=1)) is True


def test_is_past_gate_closure_false_before_deadline() -> None:
    assert is_past_gate_closure(_TARGET_DAY, _GATE_CLOSURE_UTC - pd.Timedelta(seconds=1)) is False


# ---------------------------------------------------------------------------
# accepted_earlier_today (spec 6.7.3 section 2.4 -- day-coloring only, never a lock)
# ---------------------------------------------------------------------------


def _sample_record(
    *,
    target_day: dt.date,
    candidate_selected: str | None,
    skip_reason: str | None = None,
    submitted: bool = False,
) -> protocol.SubmissionRecord:
    return protocol.SubmissionRecord(
        run_timestamp_utc="2026-07-14T09:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc123",
        target_day=target_day.isoformat(),
        nominal_slot="10:40",
        gate_closure_ok=True,
        candidate_selected=candidate_selected,
        skip_reason=skip_reason,
        submitted=submitted,
    )


def test_accepted_earlier_today_true_after_an_accepted_submission(tmp_path: Path) -> None:
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected="full_live_set", submitted=True),
    )
    assert accepted_earlier_today(log_path, _TARGET_DAY) is True


def test_accepted_earlier_today_false_after_a_candidate_selected_but_not_submitted(
    tmp_path: Path,
) -> None:
    """A dry-run (or a rejected/error live attempt) selected a candidate but
    never got accepted -- must not count (spec section 2.4's own wording is
    "keine Einreichung angenommen", not "kein Kandidat gewaehlt")."""
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected="full_live_set", submitted=False),
    )
    assert accepted_earlier_today(log_path, _TARGET_DAY) is False


def test_accepted_earlier_today_false_after_only_a_skip(tmp_path: Path) -> None:
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected=None, skip_reason="Check A: x"),
    )
    assert accepted_earlier_today(log_path, _TARGET_DAY) is False


def test_accepted_earlier_today_false_when_log_does_not_exist(tmp_path: Path) -> None:
    assert accepted_earlier_today(tmp_path / "does_not_exist.jsonl", _TARGET_DAY) is False


# ---------------------------------------------------------------------------
# source_ages_from_manifest
# ---------------------------------------------------------------------------


def _manifest(sources: dict[str, store.SourceManifestEntry]) -> store.Manifest:
    return store.Manifest(
        store_format_version=1,
        created_at_utc="2026-07-14T09:00:00+00:00",
        run_id="1",
        run_url="",
        code_sha="abc123",
        sources=sources,
    )


def test_source_ages_from_manifest_computes_hours_since_covered_end() -> None:
    manifest = _manifest(
        {
            "day_ahead_price": store.SourceManifestEntry(
                covered_start_utc="2026-01-01T00:00:00+00:00",
                covered_end_utc="2026-07-14T08:00:00+00:00",
                count=100,
                last_success_utc="2026-07-14T09:00:00+00:00",
                last_attempt_utc="2026-07-14T09:00:00+00:00",
            ),
            "generation": store.SourceManifestEntry(
                covered_start_utc=None,
                covered_end_utc=None,
                count=0,
                last_success_utc=None,
                last_attempt_utc="2026-07-14T09:00:00+00:00",
            ),
        }
    )
    as_of = pd.Timestamp("2026-07-14T10:00:00", tz="UTC")

    ages = source_ages_from_manifest(manifest, as_of)

    assert ages["day_ahead_price"] == pytest.approx(2.0)
    assert ages["generation"] is None


# ---------------------------------------------------------------------------
# build_submission_record
# ---------------------------------------------------------------------------


def test_build_submission_record_reads_the_actual_payload_and_merges_staleness() -> None:
    manifest = _manifest({})
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0, 2.0, 3.0]}
    outcome = SubmissionOutcome(
        candidate_selected="full_live_set",
        skip_reason=None,
        payload=payload,
        submission_result=SubmissionResult(sent=False, challenge_id="2"),
        commodity_staleness_warnings={"ttf_gas_eur_per_mwh": 5.0},
    )
    as_of = pd.Timestamp("2026-07-14T10:00:00", tz="UTC")

    record = build_submission_record(
        outcome,
        manifest,
        target_day=_TARGET_DAY,
        nominal_slot="10:40",
        gate_closure_ok=True,
        as_of=as_of,
        runtime_seconds=12.3,
    )

    assert record.n_values == 3
    assert record.payload_min == 1.0
    assert record.payload_mean == 2.0
    assert record.payload_max == 3.0
    assert record.submitted is False
    assert record.api_status == "dry_run"
    assert record.source_ages["ttf_gas_eur_per_mwh"] == pytest.approx(5.0 * 24)
    assert record.candidate_selected == "full_live_set"
    assert record.runtime_seconds == 12.3


def test_build_submission_record_for_a_skip_has_no_payload_stats() -> None:
    manifest = _manifest({})
    outcome = SubmissionOutcome(candidate_selected=None, skip_reason="Check A: no weather run")
    as_of = pd.Timestamp("2026-07-14T10:00:00", tz="UTC")

    record = build_submission_record(
        outcome,
        manifest,
        target_day=_TARGET_DAY,
        nominal_slot="10:40",
        gate_closure_ok=True,
        as_of=as_of,
        runtime_seconds=1.0,
    )

    assert record.n_values is None
    assert record.payload_min is None
    assert record.submitted is False
    assert record.api_status is None
    assert record.skip_reason == "Check A: no weather run"


# ---------------------------------------------------------------------------
# write_payload_to_repo
# ---------------------------------------------------------------------------


def test_write_payload_to_repo_writes_json(tmp_path: Path) -> None:
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0, 2.0]}

    with patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path):
        path = write_payload_to_repo(payload, _TARGET_DAY)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload


# ---------------------------------------------------------------------------
# run_daily_submission (spec section 5.1, the full sequence)
# ---------------------------------------------------------------------------


def test_run_daily_submission_exits_cleanly_past_gate_closure() -> None:
    with patch("scripts.run_daily_submission.store") as mock_store:
        exit_code = run_daily_submission(
            _GATE_CLOSURE_UTC, nominal_slot="11:45", is_last_slot_of_day=True
        )

    assert exit_code == 0
    mock_store.load_store.assert_not_called()


def test_run_daily_submission_has_no_idempotency_lock_two_runs_both_load_the_store(
    tmp_path: Path,
) -> None:
    """spec 6.7.3 section 2.4: the idempotency lock is gone -- an earlier
    accepted submission for target_day must not make a second run for the
    same day exit early. Proven the same way the 6.7.2 "Kein Netz" tests
    prove their own guarantees: a call-count assertion on the thing that
    would have been skipped, not just the exit code."""
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected="full_live_set", submitted=True),
    )
    fake_outcome = SubmissionOutcome(
        candidate_selected="full_live_set",
        skip_reason=None,
        payload={"challenge_id": "2", "target_start": "x", "values": [1.0, 2.0]},
        submission_result=SubmissionResult(sent=False, challenge_id="2"),
    )
    fake_manifest = _manifest({})

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch(
            "scripts.run_daily_submission.run_submission_for_day", return_value=fake_outcome
        ) as mock_run,
    ):
        mock_store.load_store.return_value.manifest = fake_manifest
        exit_code = run_daily_submission(as_of, nominal_slot="11:15", is_last_slot_of_day=False)

    assert exit_code == 0
    mock_run.assert_called_once()
    mock_store.load_store.assert_called_once()


def test_run_daily_submission_happy_path_appends_protocol_and_writes_payload(
    tmp_path: Path,
) -> None:
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    log_path = tmp_path / "submissions.jsonl"
    payloads_dir = tmp_path / "payloads"
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0, 2.0]}
    fake_outcome = SubmissionOutcome(
        candidate_selected="full_live_set", skip_reason=None, payload=payload
    )
    fake_manifest = _manifest({})

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", payloads_dir),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch(
            "scripts.run_daily_submission.run_submission_for_day", return_value=fake_outcome
        ) as mock_run,
    ):
        mock_store.load_store.return_value.manifest = fake_manifest
        exit_code = run_daily_submission(as_of, nominal_slot="10:40", is_last_slot_of_day=True)

    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["as_of"] == as_of
    records = protocol.read_submission_records(log_path)
    assert len(records) == 1
    assert records[0]["candidate_selected"] == "full_live_set"
    written_payload_path = payloads_dir / f"{records[0]['target_day']}.json"
    assert written_payload_path.exists()
    assert json.loads(written_payload_path.read_text(encoding="utf-8")) == payload


def test_run_daily_submission_silent_run_is_red_only_on_the_last_slot(tmp_path: Path) -> None:
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    log_path = tmp_path / "submissions.jsonl"
    fake_outcome = SubmissionOutcome(candidate_selected=None, skip_reason="Check A: no weather run")
    fake_manifest = _manifest({})

    def _run_once(is_last_slot: bool) -> int:
        with (
            patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
            patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
            patch("scripts.run_daily_submission.store") as mock_store,
            patch(
                "scripts.run_daily_submission.assemble_price_model_inputs",
                return_value=pd.DataFrame(),
            ),
            patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
            patch(
                "scripts.run_daily_submission.read_quarterhourly_prices",
                return_value=pd.DataFrame(),
            ),
            patch("scripts.run_daily_submission.run_submission_for_day", return_value=fake_outcome),
        ):
            mock_store.load_store.return_value.manifest = fake_manifest
            return run_daily_submission(
                as_of, nominal_slot="10:40", is_last_slot_of_day=is_last_slot
            )

    assert _run_once(is_last_slot=False) == 0
    assert _run_once(is_last_slot=True) == 1


def test_run_daily_submission_last_slot_silence_stays_green_if_earlier_run_was_accepted(
    tmp_path: Path,
) -> None:
    """spec 6.7.3 section 2.4: the last slot reads the protocol -- never a
    lock -- for its own day-coloring decision. An earlier accepted
    submission this day means a silent final run must NOT turn the day
    red, unlike test_run_daily_submission_silent_run_is_red_only_on_the_last_slot's
    own no-earlier-acceptance case."""
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected="full_live_set", submitted=True),
    )
    fake_outcome = SubmissionOutcome(candidate_selected=None, skip_reason="Check A: no weather run")
    fake_manifest = _manifest({})

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.run_submission_for_day", return_value=fake_outcome),
    ):
        mock_store.load_store.return_value.manifest = fake_manifest
        exit_code = run_daily_submission(as_of, nominal_slot="11:40", is_last_slot_of_day=True)

    assert exit_code == 0


# ---------------------------------------------------------------------------
# Response evaluation at the run_daily_submission level (spec 6.7.3 section
# 5.5): accepted -> green with a SUBMITTED line; rejected/transport error ->
# red immediately, on every slot, distinct from "SILENT".
# ---------------------------------------------------------------------------


def _run_with_fake_outcome(
    tmp_path: Path, outcome: SubmissionOutcome, *, is_last_slot_of_day: bool = False
) -> int:
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", tmp_path / "submissions.jsonl"),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.run_submission_for_day", return_value=outcome),
    ):
        mock_store.load_store.return_value.manifest = _manifest({})
        return run_daily_submission(
            as_of, nominal_slot="10:40", is_last_slot_of_day=is_last_slot_of_day
        )


def test_run_daily_submission_accepted_live_submission_is_green_with_a_submitted_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0] * 96}
    outcome = SubmissionOutcome(
        candidate_selected="full_live_set",
        skip_reason=None,
        payload=payload,
        submission_result=SubmissionResult(
            sent=True,
            challenge_id="2",
            accepted=True,
            submission_id=42,
            status="accepted",
            response_received_utc="2026-07-14T09:10:00+00:00",
            confirmed_via_query=True,
        ),
    )

    exit_code = _run_with_fake_outcome(tmp_path, outcome)

    assert exit_code == 0
    assert "SUBMITTED" in capsys.readouterr().out

    records = protocol.read_submission_records(tmp_path / "submissions.jsonl")
    assert records[0]["submitted"] is True
    assert records[0]["submission_mode"] == "live"
    assert records[0]["submission_id"] == 42
    assert records[0]["api_response_received_utc"] == "2026-07-14T09:10:00+00:00"
    assert records[0]["confirmed_via_query"] is True


def test_run_daily_submission_rejected_live_submission_is_red(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0] * 96}
    outcome = SubmissionOutcome(
        candidate_selected="full_live_set",
        skip_reason=None,
        payload=payload,
        submission_result=SubmissionResult(
            sent=True,
            challenge_id="2",
            accepted=False,
            error_kind="rejected",
            http_status=422,
            message="target_start is in the past",
        ),
    )

    exit_code = _run_with_fake_outcome(tmp_path, outcome)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "REJECTED" in out
    assert "target_start is in the past" in out


def test_run_daily_submission_transport_error_is_red_on_every_slot_not_just_the_last(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """spec 6.7.3 section 5.5: distinct from 'SILENT', a genuine failed
    attempt turns THIS run red immediately -- unlike the day-coloring
    logic that only reddens a silent day on its last slot."""
    payload = {"challenge_id": "2", "target_start": "x", "values": [1.0] * 96}
    outcome = SubmissionOutcome(
        candidate_selected="full_live_set",
        skip_reason=None,
        payload=payload,
        submission_result=SubmissionResult(
            sent=True,
            challenge_id="2",
            accepted=False,
            error_kind="transport_error",
            message="timed out",
        ),
    )

    exit_code = _run_with_fake_outcome(tmp_path, outcome, is_last_slot_of_day=False)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "TRANSPORT_ERROR" in out


# ---------------------------------------------------------------------------
# Restarbeit Teil C.3: SILENCE_STREAK_THRESHOLD must genuinely decide the
# outcome, not just document one -- proven by patching it to a different
# value and observing the behavior actually change.
# ---------------------------------------------------------------------------


def test_silence_turns_run_red_default_threshold_matches_is_last_slot_of_day() -> None:
    assert _silence_turns_run_red(is_last_slot_of_day=False) is False
    assert _silence_turns_run_red(is_last_slot_of_day=True) is True


def test_silence_streak_threshold_of_two_turns_every_silent_run_red(tmp_path: Path) -> None:
    """The actual proof Teil C.3 asks for: with SILENCE_STREAK_THRESHOLD=2,
    even the maximum 1 remaining attempt this schedule can ever report is
    < 2, so *every* silent run -- including a non-last one, which the
    default threshold=1 explicitly tolerates -- turns red. If this constant
    were decorative (read by nothing), patching it could not change this
    test's outcome."""
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    log_path = tmp_path / "submissions.jsonl"
    fake_outcome = SubmissionOutcome(candidate_selected=None, skip_reason="Check A: no weather run")
    fake_manifest = _manifest({})

    with (
        patch("scripts.run_daily_submission.SILENCE_STREAK_THRESHOLD", 2),
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.run_submission_for_day", return_value=fake_outcome),
    ):
        mock_store.load_store.return_value.manifest = fake_manifest
        exit_code = run_daily_submission(as_of, nominal_slot="10:40", is_last_slot_of_day=False)

    assert exit_code == 1


# ---------------------------------------------------------------------------
# Restarbeit Teil C.4: an unexpected exception (spec section 2.7's second
# red condition) must print a summary line distinguishable from
# "SILENT -- ..." without reading the full traceback, and must still
# propagate (never silently swallowed).
# ---------------------------------------------------------------------------


def test_run_daily_submission_prints_a_distinguishable_line_on_an_unexpected_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", tmp_path / "submissions.jsonl"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.assemble_price_model_inputs", return_value=pd.DataFrame()
        ),
        patch("scripts.run_daily_submission.read_weather_runs", return_value=pd.DataFrame()),
        patch(
            "scripts.run_daily_submission.read_quarterhourly_prices", return_value=pd.DataFrame()
        ),
        patch(
            "scripts.run_daily_submission.run_submission_for_day",
            side_effect=RuntimeError("synthetic failure for this test"),
        ),
        pytest.raises(RuntimeError, match="synthetic failure"),
    ):
        mock_store.load_store.return_value.manifest = _manifest({})
        run_daily_submission(as_of, nominal_slot="10:40", is_last_slot_of_day=False)

    captured = capsys.readouterr()
    assert "EXCEPTION — RuntimeError: synthetic failure for this test" in captured.out
    assert "SILENT" not in captured.out


# ---------------------------------------------------------------------------
# "Kein Netz" acceptance tests (spec section 7 / section 4 rule 4). The design
# note settled on two narrower, separately-tested guarantees instead of one
# blanket zero-HTTP test -- neither needs a real _post/requests mock
# (arena/submit.py's own `if not live: return` already makes the transport
# structurally unreachable, per tests/test_arena_submit.py). Both use a guard
# that raises loudly if the guarantee is ever broken, not just an assertion
# on the outcome.
# ---------------------------------------------------------------------------


def _raising_entsoe_source(name: str, cache_dir: Path) -> EntsoeSource:
    def _fetch(start: pd.Timestamp, end: pd.Timestamp, *, use_cache: bool = True) -> pd.DataFrame:
        raise AssertionError(
            f"{name}: the daily submission job must never call an ENTSO-E fetch function"
        )

    return EntsoeSource(name=name, fetch=_fetch, cache_dir=cache_dir)


def _raising_commodity_fetch(name: str) -> Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
    def _fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise AssertionError(
            f"{name}: the daily submission job must never call a commodity fetch function"
        )

    return _fetch


def _synthetic_weather_frame(run_init_utc: pd.Timestamp) -> pd.DataFrame:
    valid_times = pd.date_range(run_init_utc, periods=2, freq="h", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([run_init_utc] * 2), valid_times],
        names=["run_init_utc", "valid_time_utc"],
    )
    columns = expected_columns()
    return pd.DataFrame({col: [1.0, 2.0] for col in columns}, index=index).astype("float32")


def test_run_submission_for_day_completes_without_calling_any_fetch_client(
    tmp_path: Path,
) -> None:
    """Spec section 4 rule 4 / section 7's first 'Kein Netz' guarantee: the
    disk-read boundary (arena/live_inputs.py) must never reach an ENTSO-E or
    weather fetch client, even on a run that completes end to end. Builds
    df/weather/prices_qh via the REAL assemble_price_model_inputs/
    read_weather_runs/read_quarterhourly_prices (not mocked -- that would
    defeat the point), wired to stand-in fetch functions that raise if
    invoked, then drives run_submission_for_day through a full dry-run
    submit with the heavy per-day computation mocked (as in the happy-path
    test above; that computation is pure and was never a network risk)."""
    price_dir = tmp_path / "day_ahead_price"
    load_dir = tmp_path / "load"
    price_dir.mkdir()
    load_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    pd.DataFrame({"day_ahead_price": 50.0}, index=idx).to_parquet(
        price_dir / "DE_LU_2024-01.parquet"
    )
    pd.DataFrame({"load_actual": 1.0, "load_forecast_day_ahead": 1.0}, index=idx).to_parquet(
        load_dir / "DE_LU_2024-01.parquet"
    )
    entsoe_sources = (
        _raising_entsoe_source("day_ahead_price", price_dir),
        _raising_entsoe_source("load", load_dir),
    )
    commodities_dir = tmp_path / "commodities"
    commodities_dir.mkdir()

    weather_root = tmp_path / "weather_single_runs"
    run_init = run_init_for_target_day(_TARGET_DAY)
    write_cached_run(
        _synthetic_weather_frame(run_init),
        cache_path(run_init, DEFAULT_WEATHER_MODEL, root=weather_root),
    )

    with patch(
        "energy_price_forecast.data.weather_client.fetch_run",
        side_effect=AssertionError(
            "the daily submission job must never call the weather fetch client"
        ),
    ) as mock_fetch_run:
        df = assemble_price_model_inputs(
            entsoe_sources=entsoe_sources,
            commodity_sources=(
                ("ttf_gas", _raising_commodity_fetch("ttf_gas"), "ttf_gas_eur_per_mwh"),
                ("eua_co2", _raising_commodity_fetch("eua_co2"), "eua_co2_eur_per_t"),
            ),
            commodities_dir=commodities_dir,
        )
        weather = read_weather_runs([_TARGET_DAY], root=weather_root)
        prices_qh = read_quarterhourly_prices(entsoe_sources=entsoe_sources)

        fold = _fake_fold(_TARGET_DAY)
        matrix = pd.DataFrame({"feat_a": 1.0}, index=fold.train_index.union(fold.test_index))
        forecast = _quarterhourly_series(_TARGET_DAY)
        payload = {
            "challenge_id": "2",
            "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
            "values": [50.0 + i * 0.01 for i in range(96)],
        }

        with (
            patch(
                "scripts.run_daily_submission.check_a_inputs",
                return_value=PreflightResult(ok=True),
            ),
            patch(
                "scripts.run_daily_submission.run_renewables_step",
                return_value=(pd.DataFrame(), 1.5),
            ),
            patch(
                "scripts.run_daily_submission.build_price_feature_matrix",
                return_value=(matrix, fold, set()),
            ),
            patch(
                "scripts.run_daily_submission.fit_predict_expand",
                return_value=(forecast, 2160, 2160),
            ),
            patch(
                "scripts.run_daily_submission.build_arena_payload",
                return_value=(payload, CHALLENGE),
            ),
            patch(
                "scripts.run_daily_submission.submit",
                return_value=SubmissionResult(sent=False, challenge_id="2"),
            ),
        ):
            outcome = run_submission_for_day(
                df,
                weather,
                prices_qh,
                _TARGET_DAY,
                as_of=_AS_OF,
                now=lambda: _BEFORE_GATE_CLOSURE,
            )

        mock_fetch_run.assert_not_called()

    assert outcome.skip_reason is None
    assert outcome.candidate_selected == "full_live_set"


# ---------------------------------------------------------------------------
# live wiring (spec 6.7.3 section 2.4/5.4): live defaults to False, but is
# threaded straight through to submit() when the caller (run_daily_submission,
# ultimately main()'s is_live_enabled(os.environ)) passes it. The 6.7.2-era
# guarantee that live is NEVER True is gone by design -- these two tests
# together prove the opposite: the default stays safe, AND live=True
# genuinely reaches submit() when asked for.
# ---------------------------------------------------------------------------


def _fixture_for_live_wiring_tests() -> tuple[pd.DataFrame, pd.DataFrame, Fold, dict[str, Any]]:
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        "values": [50.0 + i * 0.01 for i in range(96)],
    }
    return df, matrix, fold, payload


def test_run_submission_for_day_defaults_to_dry_run_without_explicit_live() -> None:
    df, matrix, fold, payload = _fixture_for_live_wiring_tests()
    forecast = _quarterhourly_series(_TARGET_DAY)

    def _guarded_submit(
        challenge: ChallengeSpec, payload: dict, *, live: bool = False
    ) -> SubmissionResult:
        if live:
            raise AssertionError("must not default to live=True")
        return SubmissionResult(sent=False, challenge_id=challenge.challenge_id)

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch("scripts.run_daily_submission.submit", side_effect=_guarded_submit),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
            # live not passed -- exercises the default
        )

    assert outcome.skip_reason is None
    assert outcome.submission_result is not None
    assert outcome.submission_result.sent is False


def test_run_submission_for_day_passes_live_true_through_to_submit_when_requested() -> None:
    df, matrix, fold, payload = _fixture_for_live_wiring_tests()
    forecast = _quarterhourly_series(_TARGET_DAY)

    def _guarded_submit(
        challenge: ChallengeSpec, payload: dict, *, live: bool = False
    ) -> SubmissionResult:
        if not live:
            raise AssertionError("live=True must reach submit() when the caller asked for it")
        return SubmissionResult(sent=True, challenge_id=challenge.challenge_id, accepted=True)

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch("scripts.run_daily_submission.submit", side_effect=_guarded_submit),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
            live=True,
        )

    assert outcome.skip_reason is None
    assert outcome.submission_result is not None
    assert outcome.submission_result.sent is True
    assert outcome.submission_result.accepted is True


def test_run_submission_for_day_live_with_missing_api_key_raises_never_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spec 6.7.3 section 2.3: 'Ist der Schalter an, aber ARENA_API_KEY
    fehlt, wird der Lauf rot -- nicht still zum Dry-Run.' Uses the REAL
    submit()/_post() (no submit mock) so the actual missing-key guard in
    arena/submit.py::_post runs; requests.post itself is patched only to
    prove it is never reached."""
    monkeypatch.delenv("ARENA_API_KEY", raising=False)
    monkeypatch.setenv("ARENA_API_BASE_URL", "https://arena.example.invalid")
    df, matrix, fold, payload = _fixture_for_live_wiring_tests()
    forecast = _quarterhourly_series(_TARGET_DAY)

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "energy_price_forecast.arena.submit.requests.post",
            side_effect=AssertionError("must never POST without an API key"),
        ),
        pytest.raises(RuntimeError, match="ARENA_API_KEY"),
    ):
        run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_AS_OF,
            now=lambda: _BEFORE_GATE_CLOSURE,
            live=True,
        )


# ---------------------------------------------------------------------------
# The second, pre-POST gate-closure check (spec 6.7.3 section 2.4)
# ---------------------------------------------------------------------------


def test_run_submission_for_day_skips_post_if_gate_closed_between_start_and_post() -> None:
    """A run whose fit/predict/build pass took long enough to cross the
    deadline must not send late -- proven the same way this file's other
    'Kein Netz' guarantees are: an exploding fake stands in for submit()."""
    df, matrix, fold, payload = _fixture_for_live_wiring_tests()
    forecast = _quarterhourly_series(_TARGET_DAY)

    def _exploding_submit(*args: Any, **kwargs: Any) -> SubmissionResult:
        raise AssertionError("submit() must not be called once the gate has closed")

    with (
        patch("scripts.run_daily_submission.check_a_inputs", return_value=PreflightResult(ok=True)),
        patch(
            "scripts.run_daily_submission.run_renewables_step",
            return_value=(pd.DataFrame(), 1.5),
        ),
        patch(
            "scripts.run_daily_submission.build_price_feature_matrix",
            return_value=(matrix, fold, set()),
        ),
        patch(
            "scripts.run_daily_submission.fit_predict_expand", return_value=(forecast, 2160, 2160)
        ),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch("scripts.run_daily_submission.submit", side_effect=_exploding_submit),
    ):
        outcome = run_submission_for_day(
            df,
            pd.DataFrame(),
            pd.DataFrame(),
            _TARGET_DAY,
            as_of=_BEFORE_GATE_CLOSURE,
            now=lambda: _AS_OF,  # _AS_OF == gate closure moment exactly (>= closes it)
        )

    assert outcome.candidate_selected == "full_live_set"
    assert outcome.skip_reason == "gate closure passed before POST"
    # spec section 5.4 step 11: the payload archive still runs even though nothing sent.
    assert outcome.payload == payload
    assert outcome.submission_result is None


# ---------------------------------------------------------------------------
# Smoke mode (spec 6.7.3 sections 2.5/5.4)
# ---------------------------------------------------------------------------


def test_smoke_target_day_is_d_plus_2() -> None:
    assert smoke_target_day(_AS_OF) == _TARGET_DAY + dt.timedelta(days=1)


def test_run_smoke_submission_for_day_builds_baseline_payload_and_sends_unconditionally() -> None:
    target_day = _TARGET_DAY
    source_day = target_day - dt.timedelta(days=1)
    prices_qh = _quarterhourly_series(source_day, value=42.0).to_frame(name="day_ahead_price")

    with (
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
        patch("scripts.run_daily_submission.read_quarterhourly_prices", return_value=prices_qh),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(
                sent=True, challenge_id="2", accepted=True, submission_id=7, status="accepted"
            ),
        ) as mock_submit,
    ):
        outcome = run_smoke_submission_for_day(target_day)

    assert outcome.is_smoke is True
    assert outcome.smoke_baseline_source_day == source_day
    assert outcome.candidate_selected is None
    assert outcome.skip_reason is None
    assert outcome.payload is not None
    assert len(outcome.payload["values"]) == 96
    assert all(v == 42.0 for v in outcome.payload["values"])
    assert outcome.submission_result is not None
    assert outcome.submission_result.accepted is True
    mock_submit.assert_called_once()
    assert mock_submit.call_args.kwargs["live"] is True


def test_run_smoke_submission_for_day_raises_if_source_day_price_is_incomplete() -> None:
    """The baseline is never silently partial (arena_baseline.py's own
    contract) -- a genuinely missing D-1 price must propagate, not be
    swallowed into a degraded smoke payload."""
    target_day = _TARGET_DAY
    incomplete = _quarterhourly_series(target_day - dt.timedelta(days=1)).iloc[:48]
    prices_qh = incomplete.to_frame(name="day_ahead_price")

    with (
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
        patch("scripts.run_daily_submission.read_quarterhourly_prices", return_value=prices_qh),
        pytest.raises(ValueError, match="Missing realised price"),
    ):
        run_smoke_submission_for_day(target_day)


def test_run_smoke_submission_for_day_never_triggers_a_model_fit() -> None:
    """spec 6.7.3 section 5.4: steps 4-10 (renewables walk-forward, feature
    build, price fit/predict) do not apply in smoke mode -- proven with
    exploding fakes, not just by the absence of a call in the source."""
    target_day = _TARGET_DAY
    source_day = target_day - dt.timedelta(days=1)
    prices_qh = _quarterhourly_series(source_day, value=42.0).to_frame(name="day_ahead_price")

    def _exploding(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("smoke mode must never trigger a model fit")

    with (
        patch("scripts.run_daily_submission.get_challenge", return_value=CHALLENGE),
        patch("scripts.run_daily_submission.read_quarterhourly_prices", return_value=prices_qh),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(sent=True, challenge_id="2", accepted=True),
        ),
        patch("scripts.run_daily_submission.run_renewables_step", side_effect=_exploding),
        patch("scripts.run_daily_submission.build_price_feature_matrix", side_effect=_exploding),
        patch("scripts.run_daily_submission.fit_predict_expand", side_effect=_exploding),
    ):
        outcome = run_smoke_submission_for_day(target_day)

    assert outcome.submission_result is not None
    assert outcome.submission_result.accepted is True


_SMOKE_TARGET_DAY = _TARGET_DAY + dt.timedelta(days=1)  # smoke_target_day(_AS_OF)


def _fake_smoke_outcome(*, submission_result: SubmissionResult) -> SubmissionOutcome:
    return SubmissionOutcome(
        candidate_selected=None,
        skip_reason=None,
        payload={"challenge_id": "2", "target_start": "x", "values": [1.0] * 96},
        submission_result=submission_result,
        is_smoke=True,
        smoke_baseline_source_day=_SMOKE_TARGET_DAY - dt.timedelta(days=1),
    )


def test_run_daily_submission_smoke_exits_cleanly_past_its_own_gate_closure() -> None:
    """By construction, smoke_target_day(as_of) is always 2 local calendar
    days after as_of, and its own gate closure sits only 1 day after --
    always in as_of's future, for any real wall-clock as_of. This branch
    exists as the same structural safety net the regular flow has (spec
    section 5.4 step 1, "unveraendert"), not something a realistic as_of
    can trigger -- proven here via is_past_gate_closure itself, not a
    contrived as_of."""
    with (
        patch("scripts.run_daily_submission.is_past_gate_closure", return_value=True),
        patch("scripts.run_daily_submission.store") as mock_store,
    ):
        exit_code = run_daily_submission_smoke(_AS_OF)

    assert exit_code == 0
    mock_store.load_store.assert_not_called()


def test_run_daily_submission_smoke_accepted_appends_protocol_with_smoke_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "submissions.jsonl"
    outcome = _fake_smoke_outcome(
        submission_result=SubmissionResult(
            sent=True, challenge_id="2", accepted=True, submission_id=99, status="accepted"
        )
    )

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", log_path),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.run_smoke_submission_for_day", return_value=outcome
        ) as mock_run,
    ):
        mock_store.load_store.return_value.manifest = _manifest({})
        exit_code = run_daily_submission_smoke(_AS_OF)

    assert exit_code == 0
    assert "SUBMITTED" in capsys.readouterr().out
    mock_run.assert_called_once_with(_SMOKE_TARGET_DAY)

    records = protocol.read_submission_records(log_path)
    assert len(records) == 1
    assert records[0]["submission_mode"] == "smoke"
    assert records[0]["submitted"] is True
    assert records[0]["submission_id"] == 99
    assert records[0]["smoke_baseline_source_day"] == _TARGET_DAY.isoformat()
    written_path = tmp_path / "payloads" / f"{records[0]['target_day']}.json"
    assert written_path.exists()


def test_run_daily_submission_smoke_rejected_is_red(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outcome = _fake_smoke_outcome(
        submission_result=SubmissionResult(
            sent=True,
            challenge_id="2",
            accepted=False,
            error_kind="rejected",
            http_status=422,
            message="too late",
        )
    )

    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", tmp_path / "submissions.jsonl"),
        patch("scripts.run_daily_submission.PAYLOADS_DIR", tmp_path / "payloads"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch("scripts.run_daily_submission.run_smoke_submission_for_day", return_value=outcome),
    ):
        mock_store.load_store.return_value.manifest = _manifest({})
        exit_code = run_daily_submission_smoke(_AS_OF)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "REJECTED" in out


def test_run_daily_submission_smoke_prints_a_distinguishable_line_on_an_unexpected_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("scripts.run_daily_submission.SUBMISSIONS_LOG", tmp_path / "submissions.jsonl"),
        patch("scripts.run_daily_submission.store") as mock_store,
        patch(
            "scripts.run_daily_submission.run_smoke_submission_for_day",
            side_effect=RuntimeError("synthetic smoke failure"),
        ),
        pytest.raises(RuntimeError, match="synthetic smoke failure"),
    ):
        mock_store.load_store.return_value.manifest = _manifest({})
        run_daily_submission_smoke(_AS_OF)

    assert "EXCEPTION — RuntimeError: synthetic smoke failure" in capsys.readouterr().out
