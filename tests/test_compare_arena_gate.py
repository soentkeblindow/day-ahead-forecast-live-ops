"""Guardrail for the one calculation in compare_arena.py that decides
whether the live gate passes (spec 6.6 section 5.3's exception to the
untested-glue-script convention, spec section 7). Synthetic data only, no
real backtest needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compare_arena import (
    _CANDIDATE_SETS,
    _log_gate_verdict,
    _run_comparison,
    _upsert_by_resolution,
    gate_verdict,
)

_TZ = "Europe/Berlin"


def test_gate_passes_when_live_beats_baseline_and_dm_test_significant() -> None:
    assert gate_verdict(rmse_live=10.0, rmse_baseline=20.0, p_value=0.01) == "PASS"


def test_gate_fails_when_live_does_not_beat_baseline_rmse() -> None:
    """RMSE(live) >= RMSE(baseline) fails regardless of the p-value -- even
    an implausibly significant one."""
    assert gate_verdict(rmse_live=20.0, rmse_baseline=20.0, p_value=0.0001) == "FAIL"
    assert gate_verdict(rmse_live=25.0, rmse_baseline=20.0, p_value=0.0001) == "FAIL"


def test_gate_fails_when_rmse_better_but_not_significant() -> None:
    """spec 6.6 Entscheidung 24 is a conjunction, not just the RMSE
    comparison alone: a better RMSE with p >= 0.10 still fails."""
    assert gate_verdict(rmse_live=10.0, rmse_baseline=20.0, p_value=0.10) == "FAIL"
    assert gate_verdict(rmse_live=10.0, rmse_baseline=20.0, p_value=0.50) == "FAIL"


def _build_predictions(
    n_days: int, *, original_bias: float, live_bias: float, baseline_bias: float, seed: int = 0
) -> pd.DataFrame:
    """One quarter-hourly row per hour (4 identical slots, to keep the
    synthetic construction simple) over ``n_days`` ordinary 96-slot days.
    ``*_bias`` controls how far each candidate's prediction sits from
    y_true -- smaller bias means a better (lower-RMSE) candidate."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz=_TZ)
    frames = []
    for i in range(n_days):
        day = start + pd.DateOffset(days=i)
        idx = pd.date_range(day, day + pd.DateOffset(days=1), freq="15min", inclusive="left")
        y_true = 50 + rng.normal(0, 5, len(idx))
        noise = rng.normal(0, 1, len(idx))
        frames.append(
            pd.DataFrame(
                {
                    "y_true": y_true,
                    "pred_original": y_true + original_bias + noise,
                    "pred_live": y_true + live_bias + noise,
                    "pred_baseline": y_true + baseline_bias + noise,
                    "delivery_day": day,
                    "n_slots_in_day": len(idx),
                },
                index=idx.tz_convert("UTC"),
            )
        )
    return pd.concat(frames)


def _extract_gate_inputs(table: pd.DataFrame, dm_table: pd.DataFrame) -> tuple[float, float, float]:
    rmse_live = float(
        table.loc[
            (table["period"] == "full")
            & (table["day_type"] == "overall")
            & (table["candidate"] == "live"),
            "rmse",
        ].iloc[0]
    )
    rmse_baseline = float(
        table.loc[
            (table["period"] == "full")
            & (table["day_type"] == "overall")
            & (table["candidate"] == "baseline"),
            "rmse",
        ].iloc[0]
    )
    p_value = float(
        dm_table.loc[
            (dm_table["comparison"] == "live_vs_baseline")
            & (dm_table["period"] == "full")
            & (dm_table["variant"] == "quarter_hourly"),
            "p_value",
        ].iloc[0]
    )
    return rmse_live, rmse_baseline, p_value


def test_pred_original_does_not_influence_the_gate_outcome() -> None:
    """spec 6.6 section 7: a synthetic series where pred_original is better
    than both pred_live and pred_baseline must not change the result --
    'original' is a reference line, never a candidate for the gate."""
    candidates = _CANDIDATE_SETS["live-gate"]
    changeover_start = pd.Timestamp("2099-01-01", tz=_TZ)  # keep 'full' == everything
    comparisons = (
        ("live_vs_baseline", "live", "baseline"),
        ("live_vs_original", "live", "original"),
    )

    predictions_bad_original = _build_predictions(
        40, original_bias=5.0, live_bias=0.1, baseline_bias=3.0
    )
    predictions_great_original = _build_predictions(
        40, original_bias=0.0, live_bias=0.1, baseline_bias=3.0
    )

    table_bad, dm_bad = _run_comparison(
        predictions_bad_original,
        changeover_start,
        candidates,
        comparisons,
        "quarter_hourly",
        192,
        96,
    )
    table_great, dm_great = _run_comparison(
        predictions_great_original,
        changeover_start,
        candidates,
        comparisons,
        "quarter_hourly",
        192,
        96,
    )

    inputs_bad = _extract_gate_inputs(table_bad, dm_bad)
    inputs_great = _extract_gate_inputs(table_great, dm_great)

    assert inputs_bad == inputs_great
    assert gate_verdict(*inputs_bad) == gate_verdict(*inputs_great) == "PASS"


def test_log_gate_verdict_reads_its_own_resolutions_row_after_upsert(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression test (found 2026-09-07): _log_gate_verdict's row filters
    were missing a resolution condition. Once an hourly run's rows are on
    disk, _upsert_by_resolution's merged table carries both resolutions'
    period=full/candidate=live rows -- with the hourly one sorting first
    (existing rows are concatenated before the fresh ones). .iloc[0] then
    silently picked the hourly row even when this invocation's resolution
    was quarterhourly, so the logged PASS/FAIL used stale numbers from
    whichever resolution ran first. Deliberately gives the two resolutions
    very different RMSE gaps so a wrong pick is unambiguous."""
    candidates = _CANDIDATE_SETS["live-gate"]
    changeover_start = pd.Timestamp("2099-01-01", tz=_TZ)  # keep 'full' == everything
    comparisons = (
        ("live_vs_baseline", "live", "baseline"),
        ("live_vs_original", "live", "original"),
    )

    hourly_preds = _build_predictions(40, original_bias=0.0, live_bias=4.0, baseline_bias=4.5)
    hourly_table, hourly_dm = _run_comparison(
        hourly_preds, changeover_start, candidates, comparisons, "hourly", 24, 24
    )
    hourly_table = hourly_table.assign(resolution="hourly")
    hourly_dm = hourly_dm.assign(resolution="hourly")

    out_path = tmp_path / "arena_live_gate.csv"
    dm_out_path = tmp_path / "dm_test_live_gate.csv"
    hourly_table.to_csv(out_path, index=False)
    hourly_dm.to_csv(dm_out_path, index=False)

    qh_preds = _build_predictions(40, original_bias=0.0, live_bias=0.1, baseline_bias=5.0, seed=1)
    qh_table, qh_dm = _run_comparison(
        qh_preds, changeover_start, candidates, comparisons, "quarter_hourly", 192, 96
    )
    qh_table = qh_table.assign(resolution="quarterhourly")
    qh_dm = qh_dm.assign(resolution="quarterhourly")

    combined_table = _upsert_by_resolution(out_path, qh_table, "quarterhourly")
    combined_dm = _upsert_by_resolution(dm_out_path, qh_dm, "quarterhourly")

    # sanity: both resolutions' period=full/candidate=live rows are really
    # both present, with hourly sorting first -- otherwise this test would
    # not exercise the bug at all.
    live_rows = combined_table.loc[
        (combined_table["period"] == "full")
        & (combined_table["day_type"] == "overall")
        & (combined_table["candidate"] == "live")
    ]
    assert list(live_rows["resolution"]) == ["hourly", "quarterhourly"]

    expected_live_rmse = float(
        qh_table.loc[
            (qh_table["period"] == "full")
            & (qh_table["day_type"] == "overall")
            & (qh_table["candidate"] == "live"),
            "rmse",
        ].iloc[0]
    )
    expected_baseline_rmse = float(
        qh_table.loc[
            (qh_table["period"] == "full")
            & (qh_table["day_type"] == "overall")
            & (qh_table["candidate"] == "baseline"),
            "rmse",
        ].iloc[0]
    )
    assert expected_live_rmse != pytest.approx(
        float(
            hourly_table.loc[
                (hourly_table["period"] == "full")
                & (hourly_table["day_type"] == "overall")
                & (hourly_table["candidate"] == "live"),
                "rmse",
            ].iloc[0]
        )
    )

    caplog.set_level(logging.INFO, logger="scripts.compare_arena")
    _log_gate_verdict(combined_table, combined_dm, "quarterhourly")

    gate_records = [r.message for r in caplog.records if "GATE" in r.message]
    assert len(gate_records) == 1
    assert f"RMSE(live)={expected_live_rmse:.4f}" in gate_records[0]
    assert f"RMSE(baseline)={expected_baseline_rmse:.4f}" in gate_records[0]
