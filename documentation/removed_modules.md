# Removed modules

This repo was forked from `energy-price-forecast-risk-validation` (Sprints 1-5 there) to
build and run a live, unattended day-ahead price forecast for the Energy
Arena. Before publication, modules that are (a) not reachable from the
Arena live/backtest/result-generating paths and (b) byte-identical to their
counterpart in the source repo were removed here — the authoritative
version of each lives there.

Everything listed below is still recoverable from this repo's own history at
the tag `pre-publication-cleanup`, e.g.
`git show pre-publication-cleanup:src/energy_price_forecast/evaluation/risk.py`.

## Removed — risk/reporting track (byte-identical to `energy-price-forecast-risk-validation`, see there)

Market-risk backtesting (VaR / Expected Shortfall) and the dashboard
reporting pipeline are not part of the Arena forecasting problem; for that
side of the work, see `energy-price-forecast-risk-validation`.

- `src/energy_price_forecast/evaluation/backtest.py`
- `src/energy_price_forecast/evaluation/bootstrap.py`
- `src/energy_price_forecast/evaluation/residuals.py`
- `src/energy_price_forecast/evaluation/risk.py`
- `src/energy_price_forecast/reporting/__init__.py`
- `src/energy_price_forecast/reporting/assets.py`
- `src/energy_price_forecast/reporting/snapshot.py`
- `src/energy_price_forecast/reporting/tables.py`
- `scripts/backtest_block_sensitivity.py`
- `scripts/backtest_risk.py`
- `scripts/compute_risk.py`

## Removed — superseded batch feature pipeline

Replaced by the live, per-day feature build (`features/build.py`'s
`build_feature_set_for_day` / `build_original_feature_set_for_day`), which
calls the same shared building blocks. No feature logic was lost.

- `scripts/build_features.py`
- `scripts/build_interim.py`
- `scripts/extend_interim.py`

## Removed — superseded one-off comparison tooling

- `scripts/run_dm_test.py` — superseded by direct `evaluation.dm_test` calls
  in the current comparison scripts (`compare_objectives.py`,
  `compare_arena.py`, ...).
- `scripts/smoke_check_commodities.py`, `scripts/smoke_check_loader.py` —
  ad hoc development checks, never referenced by any workflow or runbook.

## Moved — one-off operational repair scripts

Kept locally only, in a gitignored `scripts_local/` directory (not part of
this public repo): `backfill_day_ahead_price_gap.py`,
`backfill_energy_charts_store.py`, `backfill_entsoe_actuals_gap.py`,
`ablation_refit_every_sensitivity.py`. Each was a one-time fix for a
specific real incident or a local sensitivity study, not something the live
system calls.
