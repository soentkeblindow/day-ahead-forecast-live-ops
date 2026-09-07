"""Sprint 6 / Step 6.4 + 6.6: candidate comparison and significance tests.

Thin I/O and glue layer (spec 6.4 section 5.5, spec 6.6 section 5.3) -- all
metric logic lives in the existing, UNCHANGED evaluation.metrics.summarise
and evaluation.dm_test modules. Loads a persisted preds_arena*.parquet
(written by scripts/backtest_arena.py).

--candidates bridge (default, unchanged from 6.4): three candidates
(bridge_shape, bridge_flat, baseline), writes
outputs/results/arena_bridge_backtest.csv / dm_test_bridge.csv. UPPER BOUND
run -- full feature set, not a live-capability claim.

--candidates live-gate (spec 6.6): three candidates (original, live,
baseline). ``original`` is a reference line only (never evaluated against
the go-live criterion, spec 6.6 section 2.1/3.1); the two DM comparisons
are live-vs-baseline (the gate itself) and live-vs-original (what the NWP
reconstruction costs). Writes outputs/results/arena_live_gate.csv /
dm_test_live_gate.csv, one row set per (resolution, period, day_type,
candidate) -- each invocation processes one resolution's predictions and
upserts its rows into the shared CSVs (spec 6.6 section 6: quarterhourly
and hourly runs land in the same files, in their own ``resolution``
column, not two separate files). The bindende Lauf (spec 6.6 section 2.2:
resolution=quarterhourly, period=full, live vs baseline) is additionally
logged at INFO with an explicit PASS/FAIL against the section 2.1
criterion.

Daily-block loss is aggregated by the ``delivery_day`` column
backtest_arena.py already carries (the correct Europe/Berlin local calendar
day for that fold) rather than dm_test.daily_mean_loss's UTC-midnight
normalize, which would misalign daily boundaries against a UTC-indexed
quarter-hourly series -- a deliberate, contained deviation from the exact
compare_objectives.py helper call, not from the module itself (dm_test.py
is not modified).

No model runs here, no MLflow queries -- reads only the persisted parquet.
Fails fast on missing files or malformed columns.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from energy_price_forecast.evaluation.dm_test import dm_test
from energy_price_forecast.evaluation.metrics import summarise

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS_BASE = ("y_true", "pred_baseline", "delivery_day", "n_slots_in_day")
_CANDIDATE_SETS = {
    "bridge": ("bridge_shape", "bridge_flat", "baseline"),
    "live-gate": ("original", "live", "baseline"),
}
_DAY_TYPE_LABELS = {96: "normal", 92: "spring_dst", 100: "fall_dst"}
_METRIC_KEYS = (
    "mae",
    "rmse",
    "wape",
    "mae_per_day_mean",
    "mae_per_day_std",
    "mae_per_day_p05",
    "mae_per_day_p50",
    "mae_per_day_p95",
)

# Daily-block: same lag/horizon as 6.3's daily variant (compare_objectives.py).
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
# Native quarter-hourly resolution robustness check: 2-day lag / 1-day
# horizon at 15-minute granularity, the direct analogue of 6.3's hourly
# robustness check (48-hour lag / 24-hour horizon at hourly granularity).
_QH_HAC_LAG = 192
_QH_HORIZON = 96
# Hourly resolution (spec 6.6 section 3.3, secondary/non-binding run):
# 2-day lag / 1-day horizon at hourly granularity.
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

_UPPER_BOUND_COMMENT = (
    "# UPPER BOUND: run with the full feature set, including DA_FORECAST series that are "
    "NOT available at gate closure (see sprint 6 abstract, section 9). Not a live-capability "
    "claim.\n"
)
_LIVE_GATE_COMMENT = (
    "# 'live' uses only features available at gate closure (NWP reconstruction). 'original' "
    "is an UPPER BOUND using TSO DA_FORECAST series that are NOT available at gate closure "
    "-- reference line, never a live candidate.\n"
)
_DM_SIGN_COMMENT = (
    "# mean_loss_diff_sq_eur2_mwh2 = loss(candidate) - loss(reference); "
    "a NEGATIVE value means the candidate is better.\n"
)

# spec 6.6 section 2.1, the literal wording from Entscheidung 24.
_GATE_RMSE_MARGIN = 0.0  # RMSE(live) strictly under RMSE(baseline)
_GATE_P_THRESHOLD = 0.10


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Candidate comparison and DM tests: the 6.4 bridge-vs-baseline "
        "architecture gate, or the 6.6 live-gate (live vs. original vs. baseline)."
    )
    p.add_argument("--candidates", choices=("bridge", "live-gate"), default="bridge")
    p.add_argument(
        "--resolution",
        choices=("quarterhourly", "hourly"),
        default="quarterhourly",
        help="--candidates live-gate only: tags the resolution column and selects which "
        "row set in the shared output CSVs this invocation upserts.",
    )
    p.add_argument("--preds-path", default=None, type=Path)
    p.add_argument(
        "--changeover-start",
        default="2026-01-01",
        help="First delivery day whose entire training window post-dates the switch to the "
        "quarter-hourly auction product (spec 6.4 section 11); start of the post_changeover "
        "period.",
    )
    p.add_argument("--tz", default="Europe/Berlin")
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--dm-out", default=None, type=Path)
    args = p.parse_args()
    if args.preds_path is None:
        if args.candidates == "bridge":
            args.preds_path = Path("data/processed/preds_arena.parquet")
        else:
            args.preds_path = Path(
                "data/processed/preds_arena_live.parquet"
                if args.resolution == "quarterhourly"
                else "data/processed/preds_arena_live_hourly.parquet"
            )
    if args.out is None:
        args.out = Path(
            "outputs/results/arena_bridge_backtest.csv"
            if args.candidates == "bridge"
            else "outputs/results/arena_live_gate.csv"
        )
    if args.dm_out is None:
        args.dm_out = Path(
            "outputs/results/dm_test_bridge.csv"
            if args.candidates == "bridge"
            else "outputs/results/dm_test_live_gate.csv"
        )
    return args


def _load_predictions(path: Path, candidates: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/backtest_arena.py first "
            "(see spec 6.4 section 6 / spec 6.6 section 6 for the exact invocation)"
        )
    frame = pd.read_parquet(path)
    required = (*_REQUIRED_COLUMNS_BASE, *(f"pred_{c}" for c in candidates))
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing} (got {list(frame.columns)})"
        )
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError(f"{path}: index must be a UTC tz-aware DatetimeIndex, got {index!r}")
    return frame


def _day_type_label(n_slots: int) -> str:
    return _DAY_TYPE_LABELS.get(n_slots, f"n{n_slots}")


def _metrics_row(
    frame: pd.DataFrame, candidate: str, period: str, day_type: str
) -> dict[str, object]:
    cand_frame = frame[["y_true", "delivery_day"]].assign(y_pred=frame[f"pred_{candidate}"])
    metrics = summarise(cand_frame)
    return {
        "period": period,
        "day_type": day_type,
        "candidate": candidate,
        "n": len(frame),
        **{k: metrics[k] for k in _METRIC_KEYS},
    }


def _period_frames(
    predictions: pd.DataFrame, changeover_start: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    full = predictions
    post = predictions.loc[predictions["delivery_day"] >= changeover_start]
    return {"full": full, "post_changeover": post}


def _comparison_table(
    predictions: pd.DataFrame, changeover_start: pd.Timestamp, candidates: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, period_frame in _period_frames(predictions, changeover_start).items():
        if period_frame.empty:
            log.warning(
                "period=%s has no rows (changeover_start=%s) -- skipping",
                period,
                changeover_start,
            )
            continue
        for candidate in candidates:
            rows.append(_metrics_row(period_frame, candidate, period, "overall"))
        for n_slots in sorted(period_frame["n_slots_in_day"].unique().tolist()):
            day_type_frame = period_frame.loc[period_frame["n_slots_in_day"] == n_slots]
            label = _day_type_label(int(n_slots))
            for candidate in candidates:
                rows.append(_metrics_row(day_type_frame, candidate, period, label))
    return pd.DataFrame(rows)


def _daily_loss(frame: pd.DataFrame, candidate: str) -> pd.Series:
    loss = (frame[f"pred_{candidate}"] - frame["y_true"]) ** 2
    return loss.groupby(frame["delivery_day"]).mean()


def _dm_rows_for_period(
    period_frame: pd.DataFrame,
    period: str,
    comparisons: tuple[tuple[str, str, str], ...],
    native_variant: str,
    native_hac_lag: int,
    native_horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison, candidate, reference in comparisons:
        loss_candidate_native = (period_frame[f"pred_{candidate}"] - period_frame["y_true"]) ** 2
        loss_reference_native = (period_frame[f"pred_{reference}"] - period_frame["y_true"]) ** 2
        daily_candidate = _daily_loss(period_frame, candidate)
        daily_reference = _daily_loss(period_frame, reference)

        for variant, loss_a, loss_b, hac_lag, horizon in (
            (
                native_variant,
                loss_candidate_native,
                loss_reference_native,
                native_hac_lag,
                native_horizon,
            ),
            ("daily", daily_candidate, daily_reference, _DAILY_HAC_LAG, _DAILY_HORIZON),
        ):
            try:
                result = dm_test(loss_a, loss_b, hac_lag=hac_lag, horizon=horizon)
            except ValueError as exc:
                log.warning("skipping DM test %s/%s/%s: %s", comparison, period, variant, exc)
                continue
            rows.append(
                {"comparison": comparison, "period": period, "variant": variant, **asdict(result)}
            )
    return rows


def _log_overall_rows(table: pd.DataFrame, period: str) -> None:
    overall = table.loc[(table["period"] == period) & (table["day_type"] == "overall")]
    for _, row in overall.iterrows():
        metrics_str = ", ".join(f"{m}={row[m]:.4f}" for m in _METRIC_KEYS)
        log.info("period=%s candidate=%s overall: %s", period, row["candidate"], metrics_str)


def _upsert_by_resolution(path: Path, new_rows: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Merge ``new_rows`` (all tagged with this ``resolution``) into the
    existing CSV at ``path``, replacing only that resolution's previous
    rows -- so a quarterhourly run followed by an hourly run (or vice
    versa) accumulates into one shared file (spec 6.6 section 6) instead
    of each overwriting the other."""
    if path.exists():
        existing = pd.read_csv(path, comment="#")
        existing = existing.loc[existing["resolution"] != resolution]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    return combined


def _run_comparison(
    predictions: pd.DataFrame,
    changeover_start: pd.Timestamp,
    candidates: tuple[str, ...],
    comparisons: tuple[tuple[str, str, str], ...],
    native_variant: str,
    native_hac_lag: int,
    native_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = _comparison_table(predictions, changeover_start, candidates)
    for period in table["period"].unique():
        _log_overall_rows(table, period)

    dm_rows: list[dict[str, object]] = []
    for period, period_frame in _period_frames(predictions, changeover_start).items():
        if period_frame.empty:
            continue
        dm_rows += _dm_rows_for_period(
            period_frame, period, comparisons, native_variant, native_hac_lag, native_horizon
        )
    dm_table = pd.DataFrame(dm_rows).rename(
        columns={"mean_loss_diff": "mean_loss_diff_sq_eur2_mwh2"}
    )
    dm_columns = [
        "comparison",
        "period",
        "variant",
        "n_obs",
        "hac_lag",
        "horizon",
        "mean_loss_diff_sq_eur2_mwh2",
        "dm_stat",
        "p_value",
    ]
    dm_table = dm_table[dm_columns] if not dm_table.empty else pd.DataFrame(columns=dm_columns)
    return table, dm_table


def gate_verdict(rmse_live: float, rmse_baseline: float, p_value: float) -> str:
    """The literal criterion from spec 6.6 Entscheidung 24, and nothing
    else: 'PASS' iff RMSE(live) is strictly under RMSE(baseline) AND a
    one-sided DM test gives p < 0.10; 'FAIL' otherwise. Pulled out as its
    own pure function (spec 6.6 section 5.3's guardrail-test exception to
    the usual untested-glue-script convention) so tests/test_compare_arena_
    gate.py can exercise the decision directly, without a full backtest."""
    passed = rmse_live < rmse_baseline - _GATE_RMSE_MARGIN and p_value < _GATE_P_THRESHOLD
    return "PASS" if passed else "FAIL"


def _log_gate_verdict(table: pd.DataFrame, dm_table: pd.DataFrame, resolution: str) -> None:
    """spec 6.6 section 5.3 step 8: log the bindende Lauf (resolution
    quarterhourly, period full, live vs baseline) with an explicit
    PASS/FAIL against the section 2.1 criterion. Logged only when this
    invocation's resolution is quarterhourly -- the criterion is never
    evaluated on the hourly run."""
    if resolution != "quarterhourly":
        return
    # table has been through _upsert_by_resolution by this point, so it holds
    # both resolutions' rows (this run's fresh ones plus the other
    # resolution's untouched existing ones) -- filtering on resolution too is
    # required, or .iloc[0] below can silently pick the other resolution's
    # row when it happens to sort first in the concatenation.
    rmse_row = table.loc[
        (table["resolution"] == resolution)
        & (table["period"] == "full")
        & (table["day_type"] == "overall")
        & (table["candidate"] == "live")
    ]
    baseline_row = table.loc[
        (table["resolution"] == resolution)
        & (table["period"] == "full")
        & (table["day_type"] == "overall")
        & (table["candidate"] == "baseline")
    ]
    dm_row = dm_table.loc[
        (dm_table["comparison"] == "live_vs_baseline")
        & (dm_table["period"] == "full")
        & (dm_table["variant"] == "quarter_hourly")
    ]
    if rmse_row.empty or baseline_row.empty or dm_row.empty:
        log.warning(
            "cannot evaluate the go-live criterion (spec 6.6 section 2.1): "
            "missing period=full/candidate=live|baseline row or the live_vs_baseline "
            "quarter_hourly DM test -- run the quarterhourly resolution first."
        )
        return
    rmse_live = float(rmse_row["rmse"].iloc[0])
    rmse_baseline = float(baseline_row["rmse"].iloc[0])
    p_value = float(dm_row["p_value"].iloc[0])
    verdict = gate_verdict(rmse_live, rmse_baseline, p_value)
    log.info(
        "GATE %s (spec 6.6 section 2.1): RMSE(live)=%.4f RMSE(baseline)=%.4f "
        "DM p=%.4f (threshold %.2f)",
        verdict,
        rmse_live,
        rmse_baseline,
        p_value,
        _GATE_P_THRESHOLD,
    )


def _run(args: argparse.Namespace) -> None:
    candidates = _CANDIDATE_SETS[args.candidates]
    predictions = _load_predictions(args.preds_path, candidates)
    changeover_start = pd.Timestamp(args.changeover_start, tz=args.tz)

    if args.candidates == "bridge":
        comparisons = (
            ("bridge_shape_vs_baseline", "bridge_shape", "baseline"),
            ("bridge_shape_vs_bridge_flat", "bridge_shape", "bridge_flat"),
        )
        native_variant, native_hac_lag, native_horizon = "quarter_hourly", _QH_HAC_LAG, _QH_HORIZON
    else:
        comparisons = (
            ("live_vs_baseline", "live", "baseline"),
            ("live_vs_original", "live", "original"),
        )
        native_variant, native_hac_lag, native_horizon = (
            ("quarter_hourly", _QH_HAC_LAG, _QH_HORIZON)
            if args.resolution == "quarterhourly"
            else ("hourly", _HOURLY_HAC_LAG, _HOURLY_HORIZON)
        )

    table, dm_table = _run_comparison(
        predictions,
        changeover_start,
        candidates,
        comparisons,
        native_variant,
        native_hac_lag,
        native_horizon,
    )

    if args.candidates == "live-gate":
        table = table.assign(resolution=args.resolution)
        dm_table = dm_table.assign(resolution=args.resolution)
        table = _upsert_by_resolution(args.out, table, args.resolution)
        dm_table = _upsert_by_resolution(args.dm_out, dm_table, args.resolution)
        comment = _LIVE_GATE_COMMENT
    else:
        comment = _UPPER_BOUND_COMMENT

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="\n") as f:
        f.write(comment)
        table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.out, len(table))

    args.dm_out.parent.mkdir(parents=True, exist_ok=True)
    with args.dm_out.open("w", newline="\n") as f:
        f.write(comment)
        f.write(_DM_SIGN_COMMENT)
        dm_table.to_csv(f, index=False, lineterminator="\n")
    log.info("written %s (%d rows)", args.dm_out, len(dm_table))

    if args.candidates == "live-gate":
        _log_gate_verdict(table, dm_table, args.resolution)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run(_parse_args())


if __name__ == "__main__":
    main()
