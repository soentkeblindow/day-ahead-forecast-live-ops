"""Sprint 6.8 follow-up, Messung E (not in the original spec -- an owner
request 2026-09-24, triggered by the same day's real ENTSO-E load_forecast_
day_ahead delay, see docs/data_sources_for_live_model_use.md section 1.2's
2026-09-24 update): if ENTSO-E's day-ahead load forecast for the target day
is not there yet, how well does a simple weekday-shape heuristic reconstruct
it, compared to the real (settled) value Messung A was measured against?

Reuses, unchanged: scripts.measurement_a_candidate_intake._prepare (window,
folds, matrices), features.nwp_fundamentals.build_nwp_forecast_fundamentals/
build_residual_load_nwp/build_renewable_share_nwp, features.build._build_day_
matrix (the exact Bausteinliste `live` uses), scripts.measurement_d_
ec_load._renamed (same rename-after-combine helper the EC-load patch already
needed), and (step 10 addition) scripts.compare_arena.gate_verdict (the
literal Entscheidung-24 criterion, spec 6.6/6.9) for the FINAL variant's
official go-live check.

Four reconstruction variants (owner-specified 2026-09-24), one lookback
table per target-day weekday type. "Forecast" reads load_forecast_day_ahead
from the source day, "Actual" reads load_actual:

  Feiertag (nationwide, holidays.country_holidays("DE", ...) -- the same
            check features/calendar.py's own is_holiday uses, NOT including
            the two regional holidays is_regional_holiday separately tracks):
            source day = the most recent Sunday strictly before target_day.
  Sa/So/Mo: source day = target_day - 7 days (same weekday last week).
  Di-Fr:    variant A/B -> source day = target_day - 1 day (yesterday)
            variant C/D -> source day = target_day - 7 days (last week)

  Variant A: Forecast, Di-Fr source = yesterday
  Variant B: Actual,   Di-Fr source = yesterday
  Variant C: Forecast, Di-Fr source = last week
  Variant D: Actual,   Di-Fr source = last week

A and C (B and D) are literally the same reconstruction outside the Di-Fr
branch -- there is only one lookback rule for Feiertag/Sa/So/Mo, the
variant split only matters on Tue-Fri.

Only three candidates are affected at all: `live`, `core_no_commodities`,
`core_gas_only` -- the only three of Messung A's six that call
build_nwp_forecast_fundamentals (hence read load_forecast_day_ahead,
residual_load_forecast_nwp, renewable_share_forecast_nwp). `floor_core`/
`floor_core_gas` have no fundamentals block at all; `no_fundamentals` only
reads load_forecast_error_lag_48h (a settled, already-past lag value, never
the raw target-day forecast) -- confirmed by source read
(scripts/ablation_no_fundamentals.py:136), not assumed. Re-measuring those
three would reproduce Messung A's own numbers bit-for-bit, no need to.

Training is untouched (same principle as Messung C, not Messung B): only
the target day's own row gets its load forecast replaced; every training
day's load_forecast_day_ahead is real and already settled by the time it's
used for training, live or in this backtest.

DST edge case (rare, ~2 folds/year): if the source day and target day have
a different number of local hours (a DST-transition source or target day),
values are aligned by hour-of-day POSITION from local midnight, not by
matching UTC timestamp -- the shorter side is used as-is, the longer side's
trailing hour(s) repeat the last available source value. A heuristic
reconstruction has no exact answer for this edge case either way; this is a
documented, deliberate simplification, not a silent gap.

Sprint 6.9 step 10 addition (spec section 2.2/10): re-measures `core_gas_only`
with the "Endregel" -- the real arena.load_patch package functions
(choose_reference_day/build_patched_load), which add the holiday/
availability escalation chain and the exact 2-o'clock DST table that this
module's own Variant A never had (a plain D-1 lookback with no fallback, and
a crude "repeat the last value" pad for a length mismatch). Only
`core_gas_only` is re-measured (the row this feeds, `core_gas_loadpatch`,
per the 6.9 spec) -- `live`/`core_no_commodities` stay unmeasured against the
final rule, same as the original four variants. The FINAL variant is
computed inline in the existing per-fold loop (same fit models as the A-D
variants, since row 2 is explicitly the same model as row 1, spec 2.1), not
a separate run.

Usage:
  python -m scripts.measurement_e_load_forecast_reconstruction --sanity-check
  python -m scripts.measurement_e_load_forecast_reconstruction --probe-folds 10
  python -m scripts.measurement_e_load_forecast_reconstruction
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import holidays
import mlflow
import numpy as np
import pandas as pd

from energy_price_forecast.arena.load_patch import (
    LoadPatchReference,
    build_patched_load,
    choose_reference_day,
)
from energy_price_forecast.data.loaders import load_interim_hourly, load_renewables_predictions
from energy_price_forecast.evaluation.dm_test import DMResult, dm_test
from energy_price_forecast.evaluation.metrics import mae, rmse
from energy_price_forecast.features.availability import Feature, build_matrix
from energy_price_forecast.features.build import _build_day_matrix
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig
from energy_price_forecast.features.fundamentals import build_commodity_features
from energy_price_forecast.features.lags import build_price_lags
from energy_price_forecast.features.nwp_fundamentals import (
    build_nwp_forecast_fundamentals,
    build_renewable_share_nwp,
    build_residual_load_nwp,
)
from energy_price_forecast.market_time import gate_closure_for_index
from energy_price_forecast.models.bridge import expand_to_quarterhour, fit_shape_profile
from energy_price_forecast.models.lgbm import LGBMForecaster
from scripts.compare_arena import gate_verdict
from scripts.measurement_a_candidate_intake import (
    _DAILY_HAC_LAG,
    _DAILY_HORIZON,
    _QH_HAC_LAG,
    _QH_HORIZON,
    _RANDOM_STATE,
    _REFIT_EVERY,
    _SHAPE_WINDOW_DAYS,
    _TZ,
    _hourly_index_for_local_day,
    _prepare,
)
from scripts.measurement_d_ec_load import _renamed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("measurement_e")

_MLFLOW_EXPERIMENT = "live_robustness_6_8"
_OUT_PATH = Path("data/processed/measurement_e_load_forecast_reconstruction_predictions.parquet")
_SUMMARY_PATH = Path("outputs/results/measurement_e_load_forecast_reconstruction_summary.csv")
_DIVERGENCE_PATH = Path("outputs/results/measurement_e_final_rule_reference_divergence.csv")

# spec 6.9 section 2.1/10: the provisional RMSE from this module's original Variant A
# (no holiday/availability escalation, no DST table) -- Rückfrage-Anlass 1 if the
# Endregel re-measurement deviates from it by more than the threshold below.
_PROVISIONAL_VARIANT_A_RMSE = 29.70
_PROVISIONAL_DEVIATION_RUECKFRAGE_THRESHOLD = 0.5

_CANDIDATES = ("live", "core_no_commodities", "core_gas_only")
_VARIANTS = ("A", "B", "C", "D")
_FORECAST_COL = "load_forecast_day_ahead"
_ACTUAL_COL = "load_actual"

_CAVEAT = (
    "# CAVEAT: a weekday-shape heuristic reconstruction, not a model -- these numbers describe "
    "how costly the simplest possible stand-in is, not a lower bound on what's achievable. "
    "See scripts/measurement_e_load_forecast_reconstruction.py's own module docstring for the "
    "exact per-weekday lookback table."
)


def _is_holiday(day: dt.date) -> bool:
    """Nationwide DE holidays only -- the same check features/calendar.py's
    own is_holiday feature uses (holidays.country_holidays("DE", ...)), not
    the two additional regional holidays is_regional_holiday tracks
    separately (Fronleichnam/Allerheiligen)."""
    return day in holidays.country_holidays("DE", years=[day.year])


def _last_sunday_before(day: dt.date) -> dt.date:
    """Most recent Sunday strictly before `day` (never `day` itself, even
    if `day` happens to be a Sunday -- "letzter Sonntag" reads as past
    tense)."""
    days_back = (day.weekday() - 6) % 7
    if days_back == 0:
        days_back = 7
    return day - dt.timedelta(days=days_back)


def source_day_for_variant(target_day: dt.date, variant: str) -> dt.date:
    """The single source day this variant reads load from, per the
    owner-specified lookback table (module docstring)."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {_VARIANTS}")

    if _is_holiday(target_day):
        return _last_sunday_before(target_day)

    weekday = target_day.weekday()  # Mon=0 .. Sun=6
    if weekday in (5, 6, 0):  # Sat, Sun, Mon
        return target_day - dt.timedelta(days=7)

    # Tue-Fri: A/B look back 1 day, C/D look back 7 days.
    if variant in ("A", "B"):
        return target_day - dt.timedelta(days=1)
    return target_day - dt.timedelta(days=7)


def reconstruct_load_forecast(target_day: dt.date, df: pd.DataFrame, variant: str) -> pd.Series:
    """The reconstructed load_forecast_day_ahead series for target_day's own
    hourly index, per `variant`'s lookback rule -- Forecast source
    (variant A/C) reads _FORECAST_COL, Actual source (variant B/D) reads
    _ACTUAL_COL, both from source_day_for_variant's source day."""
    column = _FORECAST_COL if variant in ("A", "C") else _ACTUAL_COL
    source_day = source_day_for_variant(target_day, variant)

    source_index = _hourly_index_for_local_day(source_day)
    target_index = _hourly_index_for_local_day(target_day)
    source_values = df[column].reindex(source_index).to_numpy()

    n_target = len(target_index)
    n_source = len(source_values)
    if n_source < n_target:
        # Rare DST edge case (module docstring): pad by repeating the last
        # available source value for the target day's extra hour(s).
        pad = np.full(n_target - n_source, source_values[-1] if n_source else float("nan"))
        source_values = np.concatenate([source_values, pad])
    elif n_source > n_target:
        source_values = source_values[:n_target]

    return pd.Series(source_values, index=target_index, name=_FORECAST_COL)


_FINAL_RULE_CANDIDATE = "core_gas_only"  # spec 6.9 section 2.1: the only row this re-measures


def _entsoe_load_forecast_complete(day: dt.date, df: pd.DataFrame) -> bool:
    """Spec 2.2 point 2: a reference day is unusable if its ENTSO-E day-ahead
    load forecast isn't present for every one of its own local hours."""
    index = _hourly_index_for_local_day(day)
    return bool(df[_FORECAST_COL].reindex(index).notna().all())


def reconstruct_load_forecast_final_rule(
    target_day: dt.date, df: pd.DataFrame
) -> tuple[pd.Series, LoadPatchReference]:
    """The "Endregel" reconstruction (spec 2.2/5.5): the real
    arena.load_patch package functions, not a copy -- holiday/availability
    escalation chain plus the exact 2-o'clock DST table, unlike this
    module's own Variant A (module docstring)."""
    reference = choose_reference_day(
        target_day,
        is_holiday=_is_holiday,
        is_complete=lambda day: _entsoe_load_forecast_complete(day, df),
    )
    if reference is None:
        raise ValueError(
            f"no usable reference day for target_day={target_day} within "
            f"MAX_LOAD_PATCH_WEEKS_BACK -- unexpected for real historical data, "
            f"see spec 2.2's own Rückfrage-Anlässe"
        )
    patched = build_patched_load(df[_FORECAST_COL], reference.reference_day, target_day)
    return patched, reference


def _patch_fundamentals(
    fundamentals: list[Feature], reconstructed_load: pd.Series
) -> list[Feature]:
    """Replaces fundamentals[0] (load_forecast_day_ahead) with the
    reconstructed series and recomputes the two arithmetically derived
    columns from it -- won/woff/solar (indices 1-3) are untouched, since
    they never depend on ENTSO-E load. Mirrors measurement_d_ec_load.py's
    build_patched_target_row exactly, generalised over the load source."""
    load, won, woff, solar = fundamentals[0], fundamentals[1], fundamentals[2], fundamentals[3]
    target_index = pd.DatetimeIndex(load.values.index)

    patched_load = Feature(
        _FORECAST_COL,
        pd.Series(
            reconstructed_load.reindex(target_index).to_numpy(),
            index=target_index,
            name=_FORECAST_COL,
        ),
        pd.Series(gate_closure_for_index(target_index), index=target_index),
    )
    residual = _renamed(
        build_residual_load_nwp(patched_load, won, woff, solar), "residual_load_forecast_nwp"
    )
    share = _renamed(
        build_renewable_share_nwp(patched_load, won, woff, solar), "renewable_share_forecast_nwp"
    )
    return [patched_load, won, woff, solar, residual, share]


def build_reconstructed_row(
    candidate: str,
    target_day: dt.date,
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    cfg: FeatureConfig,
    reconstructed_load: pd.Series,
) -> pd.DataFrame:
    """One candidate's target-day feature row, with load_forecast_day_ahead
    (and its two derived columns) replaced by `reconstructed_load` -- same
    Bausteinliste each candidate's own real builder uses, only the
    fundamentals list differs (patched vs. real)."""
    target_index = _hourly_index_for_local_day(target_day)
    fundamentals = build_nwp_forecast_fundamentals(df, renewables, target_index)
    patched = _patch_fundamentals(fundamentals, reconstructed_load)

    if candidate == "live":
        return _build_day_matrix(target_index, patched, df, cfg)
    if candidate == "core_no_commodities":
        price = build_price_lags(df, target_index, cfg)
        features = [*build_calendar_features(target_index, cfg), *patched, *price]
        return build_matrix(features)
    if candidate == "core_gas_only":
        price = build_price_lags(df, target_index, cfg)
        commodity_feats = build_commodity_features(df, target_index, cfg)[:1]  # ttf_gas only
        features = [*build_calendar_features(target_index, cfg), *patched, *commodity_feats, *price]
        return build_matrix(features)
    raise ValueError(f"unknown candidate {candidate!r}, expected one of {_CANDIDATES}")


def run_sanity_check(
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    cfg: FeatureConfig,
    target_day: dt.date,
) -> None:
    """Feeds the REAL load_forecast_day_ahead of target_day itself through
    build_reconstructed_row (bypassing reconstruct_load_forecast entirely)
    and asserts the result is bit-identical to the real matrix row from
    Messung A's own _prepare() -- proves the patch mechanism is correct,
    independent of the weekday heuristic (same discipline as measurement_d_
    ec_load.py's D0 sanity check)."""
    target_index = _hourly_index_for_local_day(target_day)
    real_load = df[_FORECAST_COL].reindex(target_index)

    for candidate in _CANDIDATES:
        patched_row = build_reconstructed_row(candidate, target_day, df, renewables, cfg, real_load)
        real_row = matrices[candidate].loc[target_index]
        patched_aligned = patched_row.reindex(columns=real_row.columns)
        max_abs_diff = (patched_aligned - real_row).abs().to_numpy().max()
        log.info("sanity check %s: max abs diff = %s", candidate, max_abs_diff)
        if max_abs_diff != 0:
            raise AssertionError(
                f"{candidate}: patch mechanism does not reproduce the real row "
                f"(max abs diff {max_abs_diff}), even with the real load fed through it"
            )
    log.info("sanity check passed for all %d candidates", len(_CANDIDATES))


def _predict_qh(
    model: LGBMForecaster, x_test: pd.DataFrame, fold, history: pd.Series, profile
) -> pd.Series:
    hourly_pred = model.predict(fold.test_index, history=history, x_test=x_test)
    return expand_to_quarterhour(hourly_pred, profile, target_day=fold.delivery_day, tz=_TZ)


def run_measurement_e(
    df: pd.DataFrame,
    renewables: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    y_hourly: pd.Series,
    prices_qh: pd.Series,
    evaluable_folds: list,
    cfg: FeatureConfig,
    *,
    refit_every: int,
    divergences: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """`divergences`, if given, is appended to (spec 2.2: "Im Log stehen
    außerdem die Tage im Fenster, an denen die Endregel eine andere Referenz
    wählt als die alte Variante A") -- one row per evaluated target_day for
    `_FINAL_RULE_CANDIDATE`, whether or not the reference actually differs."""
    restricted = {name: matrices[name] for name in _CANDIDATES}
    qh_index = pd.DatetimeIndex(prices_qh.index)
    models = {
        name: LGBMForecaster(objective="quantile", alpha=0.5, random_state=_RANDOM_STATE, n_jobs=1)
        for name in _CANDIDATES
    }

    n_folds = len(evaluable_folds)
    records: list[pd.DataFrame] = []
    for i, fold in enumerate(evaluable_folds):
        if i % refit_every == 0:
            y_train = y_hourly.reindex(fold.train_index)
            for name in _CANDIDATES:
                x_train = restricted[name].reindex(fold.train_index).dropna()
                models[name].fit(y_train.loc[x_train.index], x_train)

        history = y_hourly.loc[fold.train_index]
        end_day_minus_1 = fold.delivery_day - pd.DateOffset(days=1)
        profile = fit_shape_profile(
            prices_qh, end_day=end_day_minus_1, n_days=_SHAPE_WINDOW_DAYS, tz=_TZ
        )

        target_day = fold.delivery_day.date()
        day_start = pd.Timestamp(target_day, tz=_TZ)
        day_end = day_start + pd.DateOffset(days=1)
        y_true_day = prices_qh.loc[(qh_index >= day_start) & (qh_index < day_end)].sort_index()

        row: dict[str, object] = {
            "y_true": y_true_day.to_numpy(),
            "delivery_day": fold.delivery_day,
        }

        # Real (unpatched) reference prediction per candidate -- what Messung A itself measured.
        for name in _CANDIDATES:
            x_test_real = restricted[name].loc[fold.test_index]
            pred_qh = _predict_qh(models[name], x_test_real, fold, history, profile)
            row[f"pred_{name}_real"] = pred_qh.reindex(y_true_day.index).to_numpy()

        # Four reconstruction variants per candidate -- same models, only x_test differs.
        for variant in _VARIANTS:
            reconstructed = reconstruct_load_forecast(target_day, df, variant)
            for name in _CANDIDATES:
                x_test = build_reconstructed_row(
                    name, target_day, df, renewables, cfg, reconstructed
                )
                x_test = x_test[restricted[name].columns]
                pred_qh = _predict_qh(models[name], x_test, fold, history, profile)
                row[f"pred_{name}_{variant}"] = pred_qh.reindex(y_true_day.index).to_numpy()

        # Endregel (spec 6.9 section 2.2/5.5), the real package functions -- only for the one
        # row this re-measures, same fitted model as the A-D variants above (no extra fit).
        name = _FINAL_RULE_CANDIDATE
        reconstructed_final, patch_reference = reconstruct_load_forecast_final_rule(target_day, df)
        x_test_final = build_reconstructed_row(
            name, target_day, df, renewables, cfg, reconstructed_final
        )
        x_test_final = x_test_final[restricted[name].columns]
        pred_qh_final = _predict_qh(models[name], x_test_final, fold, history, profile)
        row[f"pred_{name}_FINAL"] = pred_qh_final.reindex(y_true_day.index).to_numpy()

        if divergences is not None:
            old_source_day = source_day_for_variant(target_day, "A")
            divergences.append(
                {
                    "target_day": target_day,
                    "old_variant_a_source_day": old_source_day,
                    "final_rule_reference_day": patch_reference.reference_day,
                    "final_rule_weeks_back": patch_reference.weeks_back,
                    "final_rule_skipped": patch_reference.skipped,
                    "differs": old_source_day != patch_reference.reference_day,
                }
            )

        records.append(pd.DataFrame(row, index=y_true_day.index))
        if (i + 1) % max(1, n_folds // 10) == 0 or i == n_folds - 1:
            log.info("progress: %d/%d folds", i + 1, n_folds)

    return pd.concat(records).sort_index()


def _run_dm(
    loss_a: pd.Series, loss_b: pd.Series, delivery_day: pd.Series
) -> tuple[DMResult, DMResult]:
    native = dm_test(loss_a, loss_b, hac_lag=_QH_HAC_LAG, horizon=_QH_HORIZON)
    daily_a = loss_a.groupby(delivery_day).mean()
    daily_b = loss_b.groupby(delivery_day).mean()
    daily = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    return native, daily


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-folds", type=int, default=None)
    p.add_argument("--sanity-check", action="store_true")
    args = p.parse_args()

    cfg = FeatureConfig()
    y_hourly, matrices, prices_qh, evaluable_folds, end_day = _prepare(cfg)
    df = load_interim_hourly()
    renewables = load_renewables_predictions()

    if args.sanity_check:
        target_day = evaluable_folds[len(evaluable_folds) // 2].delivery_day.date()
        log.info("running sanity check against target_day=%s", target_day)
        run_sanity_check(df, renewables, matrices, cfg, target_day)
        return

    folds_to_run = evaluable_folds[: args.probe_folds] if args.probe_folds else evaluable_folds

    t0 = time.monotonic()
    divergences: list[dict[str, object]] = []
    out = run_measurement_e(
        df,
        renewables,
        matrices,
        y_hourly,
        prices_qh,
        folds_to_run,
        cfg,
        refit_every=_REFIT_EVERY,
        divergences=divergences,
    )
    elapsed = time.monotonic() - t0

    if args.probe_folds:
        per_fold = elapsed / len(folds_to_run)
        estimated_total = per_fold * len(evaluable_folds)
        log.info(
            "PROBE: %d folds took %.1fs (%.2fs/fold) -- extrapolated total for %d folds: "
            "%.0fs (%.1f min)",
            len(folds_to_run),
            elapsed,
            per_fold,
            len(evaluable_folds),
            estimated_total,
            estimated_total / 60,
        )
        return

    reference = pd.read_parquet(
        Path("data/processed/measurement_a_candidate_intake_predictions.parquet")
    )
    ref = reference.loc[out.index]
    out["pred_baseline"] = ref["pred_baseline"].to_numpy()

    for name in _CANDIDATES:
        check = (out[f"pred_{name}_real"] - ref[f"pred_{name}"]).abs().max()
        log.info(
            "sanity: %s_real vs Messung A's own %s predictions, max abs diff=%.6g",
            name,
            name,
            check,
        )

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT_PATH)

    print(f"\nn_folds={len(folds_to_run)}  refit_every={_REFIT_EVERY}  window_end={end_day}\n")

    metrics: dict[str, dict[str, float]] = {}
    for name in _CANDIDATES:
        suffixes = (
            ("real", *_VARIANTS, "FINAL") if name == _FINAL_RULE_CANDIDATE else ("real", *_VARIANTS)
        )
        for suffix in suffixes:
            col = f"pred_{name}_{suffix}"
            key = f"{name}_{suffix}"
            metrics[key] = {
                "mae": mae(out["y_true"], out[col]),
                "rmse": rmse(out["y_true"], out[col]),
            }
            print(f"{key:28s}  MAE={metrics[key]['mae']:.4f}  RMSE={metrics[key]['rmse']:.4f}")
    metrics["baseline"] = {
        "mae": mae(out["y_true"], out["pred_baseline"]),
        "rmse": rmse(out["y_true"], out["pred_baseline"]),
    }
    print(
        f"{'baseline':28s}  MAE={metrics['baseline']['mae']:.4f}  RMSE={metrics['baseline']['rmse']:.4f}"
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    summary_rows: list[dict[str, object]] = []
    log_metrics: dict[str, float] = {}
    final_gate_native_p: float | None = None
    with mlflow.start_run(run_name="measurement_e_load_forecast_reconstruction"):
        mlflow.log_params({"n_folds": len(folds_to_run), "refit_every": _REFIT_EVERY})
        mlflow.set_tags({"spec": "sprint6_step6_8_followup", "measurement": "E"})

        print("\n--- DM per variant vs 'real' (own candidate reference) and vs baseline ---\n")
        for name in _CANDIDATES:
            variants = (*_VARIANTS, "FINAL") if name == _FINAL_RULE_CANDIDATE else _VARIANTS
            for variant in variants:
                arm_key = f"{name}_{variant}"
                print(f"{arm_key}:")
                for reference_name in (f"{name}_real", "baseline"):
                    loss_a = (out[f"pred_{arm_key}"] - out["y_true"]) ** 2
                    loss_b = (out[f"pred_{reference_name}"] - out["y_true"]) ** 2
                    native, daily = _run_dm(loss_a, loss_b, out["delivery_day"])
                    better = "better" if native.mean_loss_diff < 0 else "worse"
                    # Entscheidung 24's literal criterion (spec 6.6/6.9), not just an
                    # RMSE-only comparison -- reused via scripts.compare_arena.gate_verdict.
                    gate_pass = (
                        gate_verdict(
                            metrics[arm_key]["rmse"], metrics["baseline"]["rmse"], native.p_value
                        )
                        == "PASS"
                        if reference_name == "baseline"
                        else None
                    )
                    if (
                        name == _FINAL_RULE_CANDIDATE
                        and variant == "FINAL"
                        and reference_name == "baseline"
                    ):
                        final_gate_native_p = native.p_value
                    print(
                        f"  vs {reference_name:18s} loss_diff={native.mean_loss_diff:+.4f} "
                        f"p={native.p_value:.4f} ({better})"
                        + (
                            f"  gate={'PASS' if gate_pass else 'FAIL'}"
                            if gate_pass is not None
                            else ""
                        )
                    )
                    summary_rows.append(
                        {
                            "candidate": name,
                            "variant": variant,
                            "reference": reference_name,
                            "rmse_arm": metrics[arm_key]["rmse"],
                            "rmse_reference": metrics[reference_name]["rmse"],
                            "native_loss_diff": native.mean_loss_diff,
                            "native_p": native.p_value,
                            "daily_loss_diff": daily.mean_loss_diff,
                            "daily_p": daily.p_value,
                            "gate_pass": gate_pass,
                        }
                    )
                    log_metrics[f"{arm_key}_vs_{reference_name}_native_p"] = native.p_value
                    log_metrics[f"{arm_key}_vs_{reference_name}_native_loss_diff"] = (
                        native.mean_loss_diff
                    )
                print()
        for key, m in metrics.items():
            log_metrics[f"{key}_rmse"] = m["rmse"]
            log_metrics[f"{key}_mae"] = m["mae"]

        mlflow.log_metrics(log_metrics)
        summary = pd.DataFrame(summary_rows)
        _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SUMMARY_PATH.open("w", newline="\n") as f:
            f.write(_CAVEAT + "\n")
            summary.to_csv(f, index=False, lineterminator="\n")
        mlflow.log_artifact(str(_SUMMARY_PATH))
        mlflow.log_artifact(str(_OUT_PATH))

        # Step 10 addition (spec 2.2): the official go-live check for row 2, and the
        # provisional-value sanity band from the Rückfrage-Anlässe (spec section 10).
        assert final_gate_native_p is not None  # set inside the DM loop above, always reached
        final_key = f"{_FINAL_RULE_CANDIDATE}_FINAL"
        final_rmse = metrics[final_key]["rmse"]
        final_native_p = final_gate_native_p
        final_verdict = gate_verdict(final_rmse, metrics["baseline"]["rmse"], final_native_p)
        deviation_from_provisional = final_rmse - _PROVISIONAL_VARIANT_A_RMSE
        print(
            f"\n=== OFFICIAL GATE for row 2 ({final_key}), spec 6.9 Entscheidung 24: "
            f"{final_verdict} ===\n"
            f"RMSE(final)={final_rmse:.4f}  RMSE(baseline)={metrics['baseline']['rmse']:.4f}  "
            f"DM p={final_native_p:.4f}\n"
            f"deviation from provisional 29.70 (Messung E Variante A, ohne Endregel): "
            f"{deviation_from_provisional:+.4f}"
            + (
                "  -- exceeds the 0.5 Rückfrage-Anlass, flag to owner"
                if abs(deviation_from_provisional) > _PROVISIONAL_DEVIATION_RUECKFRAGE_THRESHOLD
                else ""
            )
        )
        mlflow.log_metrics(
            {
                f"{final_key}_gate_pass": float(final_verdict == "PASS"),
                f"{final_key}_deviation_from_provisional": deviation_from_provisional,
            }
        )

        divergence_days = pd.DataFrame(divergences)
        n_differs = int(divergence_days["differs"].sum())
        print(
            f"\nEndregel weicht an {n_differs}/{len(divergence_days)} Tagen von der alten "
            f"Variante A ab (vollständige Liste: {_DIVERGENCE_PATH})."
        )
        _DIVERGENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        divergence_days.to_csv(_DIVERGENCE_PATH, index=False)
        mlflow.log_artifact(str(_DIVERGENCE_PATH))
        mlflow.log_metric(f"{final_key}_n_reference_divergence_days", n_differs)

    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
