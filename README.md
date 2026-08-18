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
