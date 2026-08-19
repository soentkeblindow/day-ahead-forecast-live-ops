# sbl-energy-forecast

Day-ahead electricity price forecasting for Germany (DE-LU) with LightGBM, quantile regression, and walk-forward backtesting. This repository is the submission codebase for the Energy Arena, built on top of the validated model from [`energy-price-forecast`](https://github.com/soentkeblindow/energy-price-forecast).

This README is intentionally short for now; the full version (setup, usage, model details) is written later in Sprint 6. What follows is the one section that is load-bearing from day one.

## Provenance

**Source:** [`energy-price-forecast`](https://github.com/soentkeblindow/energy-price-forecast) @ commit [`e6047806aaa4673cd0f5698e9e9e3151c202b379`](https://github.com/soentkeblindow/energy-price-forecast/commit/e6047806aaa4673cd0f5698e9e9e3151c202b379) (2026-07-20), working tree clean at copy time.

**What was omitted, and why:**

- `src/energy_price_forecast/dashboard/` and its tests — the Streamlit app belongs to the previous project's packaging and has no function in the Arena repo.
- `scripts/export_report_assets.py` and its test — a Sprint-5 packaging script (README asset export) with no Arena use.
- `tests/test_smoke.py` — its `test_subpackages_importable` asserts that `dashboard` is importable; since `dashboard/` is not copied, the whole file is left out rather than editing test code.
- `notebooks/`, `outputs/`, `mlruns/`, `docs/`, `.venv/`, `README.md`, `LICENSE` — exploratory or packaging artifacts of the predecessor project, not needed here.

Everything else in `src/`, `tests/`, and `scripts/` was copied verbatim, byte-for-byte, with the importable package name (`energy_price_forecast`) kept identical to the source repo on purpose (see Beleg A below).

**Data artifacts were copied as files, not rebuilt.** ENTSO-E revises historical time series, so a fresh pull today could silently differ from the data the Sprints 1-5 validation was run on. Copying the exact `.parquet` files removes that variable; rebuilding is deliberately out of scope for this step.

**Relationship to `energy-arena-participate`:** a separate clone, kept as a reference during development and a one-time fixture generator for Sprint 6.2 (its recorded Arena payload gets frozen into `tests/fixtures/` once). There is no runtime dependency on it from this repo.

### The four provenance belege

| Beleg | What it checks | Result |
|---|---|---|
| **A** — file-hash diff of `src/` and `tests/` against the source repo | Every line, not a sample | Byte-identical (42/42 files in `src/`, 26/26 in `tests/`) except the documented omissions above |
| **B** — copied test suite (`uv run pytest`) | Behavior on the source repo's own tests | 447 passed, 1 deselected (integration marker), no test code changed. `mlflow` was pinned to `3.11.1` (matching the source repo's resolved version) after an initial run surfaced a hard failure caused by newer mlflow's `file://`-store behavior change |
| **C** — golden fixture (`tests/test_provenance.py`) | Default `LGBMForecaster()` behavior on a fixed, in-code synthetic dataset | Bit-identical match against the frozen fixture; regression anchor for all later changes starting with 6.3 |
| **D** — production backtest (`scripts/backtest.py --model lgbm --alpha 0.5 --window rolling --train-span-days 90 --refit-every 1 --test-start 2021-01-01 --test-end 2025-12-31`) | End-to-end reproduction of the documented headline metric | MAE = 15.3988 EUR/MWh vs. the documented reference MAE ≈ 15.40 EUR/MWh (`model_validation_report.md` in the source repo). RMSE 26.5926 vs. 26.59; WAPE 0.1285 vs. 0.129. MLflow run under experiment `arena_models`, tag `study=provenance`. Reviewed and accepted by the owner (2026-08-18) |

**Dependency pinning:** `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `pyarrow` are pinned to the exact versions resolved in the source repo's `uv.lock` (`4.6.0`, `2.3.5`, `2.3.3`, `1.8.0`, `23.0.1`). `mlflow` was pinned as a sixth package (`3.11.1`) after Beleg B surfaced the version-drift failure described above.

## Daily availability audit

`.github/workflows/audit.yml` runs four times a day (`09:00`, `09:45`, `10:00`, `10:45` UTC) and measures whether the raw input series the model would need for tomorrow's forecast are actually published yet on ENTSO-E and Yahoo Finance, using the same data clients the model pipeline uses. **It never submits anything to the Energy Arena** — no `POST` happens anywhere in this repo outside a live-mode call nobody makes yet (Sprint 6.2 scope; submission logic is Sprint 6.5).

Results are appended to three CSV logs under `logs/` (not gitignored, committed automatically by the workflow after every run):

- `logs/availability.csv` — one row per run × raw series × window (`critical` or `training`), with coverage counts and a `status` column for quick reading.
- `logs/challenge_catalog.csv` — one row per run, a snapshot of the Arena's own challenge format (including a `spec_sha256` hash that changes if the Arena changes anything about the challenge, even fields this repo doesn't read).
- `logs/audit_runs.csv` — one row per run with load/timing metrics (`n_client_calls`, `duration_s`) and status-count summaries.

Two things worth knowing before reading `availability.csv`:

- For `DA_FORECAST` series (`load_forecast_day_ahead`, `wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`) on `window_kind == "critical"` only, the actual fetch reaches one calendar day further back than the window the coverage numbers are computed over — `window_start_local`/`window_end_local` always describe the *evaluation* window, never the wider fetch range. This lets `latest_target_local`/`latest_target_offset_days` distinguish "not yet published for tomorrow" (`offset_days == -1`, coverage `0` — normal, just early) from "genuinely stalled."
- `latest_target_offset_days` is a **three-valued, censored indicator** (`0`, `-1`, or empty), not a continuous age measure: empty means no value was found anywhere in the two-day fetch range, not "the series failed" and not "the last value is older than a day." Reading it as a magnitude gives wrong conclusions.

Fallback policy, submit/no-submit thresholds, and leaderboard monitoring are all Sprint 6.5 — this audit only measures and logs.
