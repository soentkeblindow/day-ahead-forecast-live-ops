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
from unittest.mock import patch

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
from energy_price_forecast.features.nwp_fundamentals import IncompleteReconstructionError
from energy_price_forecast.ops import protocol, store
from energy_price_forecast.ops.store_sources import EntsoeSource
from energy_price_forecast.ops.windows import LOCAL_TZ, local_day_bounds
from scripts.run_daily_submission import (
    PRICE_TRAIN_SPAN_DAYS,
    RENEWABLES_TRAIN_SPAN_DAYS,
    SubmissionOutcome,
    already_submitted,
    build_arena_payload,
    build_price_feature_matrix,
    build_submission_record,
    check_a_inputs,
    fit_predict_expand,
    holiday_calendar_covers_target,
    is_past_gate_closure,
    renewables_window,
    run_daily_submission,
    run_renewables_step,
    run_submission_for_day,
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

    result = fit_predict_expand(df, matrix, fold, prices_qh)

    assert len(result) == 96  # DST-free day
    assert result.notna().all()
    assert pd.DatetimeIndex(result.index).tz is not None


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
    result = fit_predict_expand(df, matrix, fold, prices_qh)

    assert len(result) == 96
    assert result.notna().all()


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
        patch("scripts.run_daily_submission.fit_predict_expand", return_value=forecast),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(sent=False, challenge_id="2"),
        ),
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
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
        patch("scripts.run_daily_submission.fit_predict_expand", return_value=absurd_forecast),
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
        patch("scripts.run_daily_submission.fit_predict_expand", return_value=forecast),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit", return_value=fake_submission_result
        ) as mock_submit,
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
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
        patch("scripts.run_daily_submission.fit_predict_expand", return_value=forecast),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch(
            "scripts.run_daily_submission.submit",
            return_value=SubmissionResult(sent=False, challenge_id="2"),
        ),
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert "ttf_gas_eur_per_mwh" in outcome.commodity_staleness_warnings
    assert outcome.commodity_staleness_warnings["ttf_gas_eur_per_mwh"] == pytest.approx(5.0)


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
# already_submitted
# ---------------------------------------------------------------------------


def _sample_record(
    *, target_day: dt.date, candidate_selected: str | None, skip_reason: str | None = None
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
    )


def test_already_submitted_true_after_a_completed_run(tmp_path: Path) -> None:
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path, _sample_record(target_day=_TARGET_DAY, candidate_selected="full_live_set")
    )
    assert already_submitted(log_path, _TARGET_DAY) is True


def test_already_submitted_false_after_only_a_skip(tmp_path: Path) -> None:
    log_path = tmp_path / "submissions.jsonl"
    protocol.append_submission_record(
        log_path,
        _sample_record(target_day=_TARGET_DAY, candidate_selected=None, skip_reason="Check A: x"),
    )
    assert already_submitted(log_path, _TARGET_DAY) is False


def test_already_submitted_false_when_log_does_not_exist(tmp_path: Path) -> None:
    assert already_submitted(tmp_path / "does_not_exist.jsonl", _TARGET_DAY) is False


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


def test_run_daily_submission_exits_cleanly_when_already_submitted() -> None:
    as_of = _GATE_CLOSURE_UTC - pd.Timedelta(hours=1)
    with (
        patch("scripts.run_daily_submission.already_submitted", return_value=True),
        patch("scripts.run_daily_submission.store") as mock_store,
    ):
        exit_code = run_daily_submission(as_of, nominal_slot="10:40", is_last_slot_of_day=False)

    assert exit_code == 0
    mock_store.load_store.assert_not_called()


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
            patch("scripts.run_daily_submission.fit_predict_expand", return_value=forecast),
            patch(
                "scripts.run_daily_submission.build_arena_payload",
                return_value=(payload, CHALLENGE),
            ),
            patch(
                "scripts.run_daily_submission.submit",
                return_value=SubmissionResult(sent=False, challenge_id="2"),
            ),
        ):
            outcome = run_submission_for_day(df, weather, prices_qh, _TARGET_DAY, as_of=_AS_OF)

        mock_fetch_run.assert_not_called()

    assert outcome.skip_reason is None
    assert outcome.candidate_selected == "full_live_set"


def test_run_submission_for_day_never_passes_live_true_to_submit() -> None:
    """Spec section 7's second 'Kein Netz' guarantee: run_submission_for_day
    must never pass live=True to submit(). arena/submit.py's own
    `if not live: return` already makes the transport unreachable when
    live=False (tests/test_arena_submit.py), so this needs no _post mock --
    it is a guard on the call site itself, so an accidental live=True would
    fail loudly here even before it ever reached submit()."""
    fold = _fake_fold(_TARGET_DAY)
    matrix_index = fold.train_index.union(fold.test_index)
    matrix = pd.DataFrame({"feat_a": 1.0}, index=matrix_index)
    df = pd.DataFrame({"day_ahead_price": 50.0}, index=matrix_index)
    forecast = _quarterhourly_series(_TARGET_DAY)
    payload = {
        "challenge_id": "2",
        "target_start": pd.Timestamp(_TARGET_DAY, tz="Europe/Berlin").isoformat(),
        "values": [50.0 + i * 0.01 for i in range(96)],
    }

    def _guarded_submit(
        challenge: ChallengeSpec, payload: dict, *, live: bool = False
    ) -> SubmissionResult:
        if live:
            raise AssertionError("run_submission_for_day must never pass live=True to submit()")
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
        patch("scripts.run_daily_submission.fit_predict_expand", return_value=forecast),
        patch(
            "scripts.run_daily_submission.build_arena_payload", return_value=(payload, CHALLENGE)
        ),
        patch("scripts.run_daily_submission.submit", side_effect=_guarded_submit),
    ):
        outcome = run_submission_for_day(
            df, pd.DataFrame(), pd.DataFrame(), _TARGET_DAY, as_of=_AS_OF
        )

    assert outcome.skip_reason is None
    assert outcome.submission_result is not None
    assert outcome.submission_result.sent is False
