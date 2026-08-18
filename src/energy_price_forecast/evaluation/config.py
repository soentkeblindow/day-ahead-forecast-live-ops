from dataclasses import dataclass, field

import pandas as pd

from ..market_time import LOCAL_TZ

EXPERIMENT_NAME = "baselines"
SPRINT3_EXPERIMENT_NAME = "sprint3_models"
CALIBRATION_EXPERIMENT_NAME = "calibration_and_risk"


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds and boundaries for realised-market regime tagging.

    All flags derived from this config are DIAGNOSTIC labels computed from
    realised (actual) market outcomes. They must never enter the model as
    features (that would be look-ahead leakage). See the
    ``evaluation.regimes`` module docstring for the full contract.
    """

    # Renewable scarcity: residual_load / load above this share.
    # residual_load = load - (wind_onshore + wind_offshore + solar).
    # Equivalent to renewable share < (1 - this value). A self-normalising
    # ratio, so it is robust to installed-capacity growth over time.
    renewable_scarcity_residual_share: float = 0.90

    # High wind: combined onshore + offshore generation above this level.
    # Units are MW (interim generation columns are in MW): 20_000 MW = 20 GW.
    high_wind_generation_mw: float = 20_000.0

    # Price spike: day-ahead price above this per-macro-regime quantile,
    # computed over the full interim history. Regime-relative by design.
    price_spike_quantile: float = 0.95

    # Negative price uses a fixed 0.0 boundary; no configurable field needed.

    # Macro-regime date boundaries, canonical from the EDA. Europe/Berlin
    # local midnight -- identical to FeatureConfig.crisis_start /
    # post_crisis_start (features/config.py) -- so this diagnostic axis
    # matches the is_crisis / is_post_crisis model feature exactly.
    crisis_start: pd.Timestamp = pd.Timestamp("2021-09-01", tz=LOCAL_TZ)
    post_crisis_start: pd.Timestamp = pd.Timestamp("2023-04-01", tz=LOCAL_TZ)


# Extended quantile grid from step 4.3a (leftmost = lowest level).
QUANTILE_GRID: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class ConformalConfig:
    """Constants for the scaled conformal recalibration (step 4.3b).

    One place for every tunable number in the calibration layer. The rolling
    calibration-window length is additionally exposed on the CLI
    (``--calibration-window``); the CLI value overrides ``window_days`` here.
    All other knobs are config-only, on purpose, to stay reproducible.
    """

    # Rolling calibration-window length in days (CLI-overridable default).
    window_days: int = 90

    # Embargo (gap) in days between the calibration window and the test day.
    # Guarantees the calibration realizations are known at gate closure of the
    # test day's forecast. 1 day already covers the day-ahead convention.
    embargo_days: int = 1

    # k for the kNN-in-forecast-level local-scale estimator (sigma).
    n_neighbors: int = 200

    # Floor for sigma, as a fraction of the global calibration-window std of y.
    # Prevents division by a near-zero local scale in flat neighbourhoods.
    sigma_floor_fraction: float = 0.10

    # Recompute sigma + the seven Q_alpha once per this many days.
    recompute_cadence_days: int = 1

    # Minimum calibration hours required to calibrate a day. Below this the
    # day's raw predictions are passed through UNCHANGED and flagged as
    # "uncalibrated" in the diagnostics (honest: no correction from too little
    # data). Early test days without a full window fall here.
    min_calibration_hours: int = 24 * 14


# Local hour-of-day phases (Europe/Berlin), from the std(r) table in 4.4a spec 2.5.
_NIGHT: tuple[int, ...] = (2, 3, 4, 5)
_MORNING_RAMP: tuple[int, ...] = (7, 8)
_MIDDAY: tuple[int, ...] = (10, 11, 12, 13)
_EVENING_RAMP: tuple[int, ...] = (18, 19, 20)


@dataclass(frozen=True)
class BacktestConfig:
    """Single source for all Sprint 4.4b backtest constants (spec section 3.2)."""

    level: float = 0.95  # VaR/ES level, MUST match the 4.4a run
    day_breach_k: int = 2  # >=k breach hours make a breach-day (spec 2.5)
    n_bootstrap: int = (
        10_000  # legacy fixed-B default; superseded by min/max_bootstrap below (Nachtrag 1)
    )
    bootstrap_seed: int = 20260710  # fixed for reproducibility
    basel_window_days: int = (
        250  # non-overlapping Basel window, in delivery days (Nachtrag 1, part B)
    )
    # Below this, CI flagged low_support (spec 2.6). cell_occupancy counts
    # VALID days per (subset, MONTH-OF-YEAR pooled across every year in the
    # sample) -- e.g. "February" spans every February in a 5-year backtest,
    # not one calendar instance -- so 30 is comfortably reachable for any
    # well-populated subset while still catching genuinely thin cells.
    min_cell_days: int = 30

    # --- Part A: bootstrap convergence monitoring (Nachtrag 1, section A) ---
    min_bootstrap: int = 2_000  # never stop before this many replications
    max_bootstrap: int = 50_000  # hard ceiling
    check_every: int = 500  # convergence check interval (also the batch-means block size)
    mc_tol: float = 0.01  # MCSE < mc_tol * CI width, required on BOTH bounds
    n_stable: int = 2  # consecutive checkpoints the rule must hold (hysteresis)

    # --- Part B: Basel traffic light (Nachtrag 1, section B) ---
    # basel_window_days above is reused; windows are now NON-OVERLAPPING and
    # the light runs ONLY on the full hourly series, never on a conditioning
    # subset (spec Nachtrag 1, B.2c).
    basel_drop_partial_window: bool = True  # discard an incomplete trailing window

    # Ex-ante-known hour-of-day conditioning phases (LOCAL time).
    hour_phases: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "night": _NIGHT,
            "morning_ramp": _MORNING_RAMP,
            "midday": _MIDDAY,
            "evening_ramp": _EVENING_RAMP,
            "ramp": _MORNING_RAMP + _EVENING_RAMP,  # headline subset
        }
    )

    # Ex-ante-known FORECAST-based regime subsets (spec 2.7). Both are the
    # forecast twins of realised, ex-post RegimeConfig flags, so they are valid
    # conditioning sets. The dunkelflaute threshold REUSES the realised scarcity
    # threshold from RegimeConfig (import it; do NOT re-hardcode 0.90) so the
    # ex-ante and ex-post twins share exactly one number. The surplus threshold
    # is 0 by definition (residual_load_forecast < 0 == renewable prod > load),
    # not a tunable magic number.
    dunkelflaute_forecast_uses_regime_threshold: bool = (
        True  # residual_share_fc > RegimeConfig scarcity thr.
    )
    surplus_forecast_residual_load_threshold: float = 0.0  # residual_load_forecast < 0

    # --- Nachtrag 2: block bootstrap sensitivity ---------------------------
    # Block length in DELIVERY DAYS for the stratified block bootstrap.
    # 1 == the historical one-day block (Nachtrag 1 and earlier) and is the
    # REPORTED headline. Larger values capture day-to-day dependence (multi-day
    # cold snaps / Dunkelflauten) at the cost of effective sample size. The
    # daily Christoffersen LR_ind of 53-104 in the Nachtrag 1 run is the direct
    # evidence that day-to-day dependence EXISTS, i.e. that the one-day block
    # yields CIs that are somewhat TOO NARROW. We do not "fix" this by picking a
    # different single value -- we measure how far the conclusions move across a
    # grid (spec Nachtrag 2, 2.4).
    block_days: int = 1
    # Grid for the sensitivity run. Upper bound 7 == the plausible persistence of
    # the relevant regimes (cold snaps, Dunkelflauten); beyond that we only lose
    # effective sample size without capturing more structure.
    block_days_grid: tuple[int, ...] = (1, 3, 5, 7)
