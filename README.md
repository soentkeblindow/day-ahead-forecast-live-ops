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
| **A** — file-hash diff of `src/` and `tests/` against the source repo | Every line, not a sample | Byte-identical (42/42 files in `src/`, 26/26 in `tests/`) except the documented omissions above and one later, scoped patch to `entsoe_client.py` (see below) |
| **B** — copied test suite (`uv run pytest`) | Behavior on the source repo's own tests | 447 passed, 1 deselected (integration marker), no test code changed. `mlflow` was pinned to `3.11.1` (matching the source repo's resolved version) after an initial run surfaced a hard failure caused by newer mlflow's `file://`-store behavior change |
| **C** — golden fixture (`tests/test_provenance.py`) | Default `LGBMForecaster()` behavior on a fixed, in-code synthetic dataset | Bit-identical match against the frozen fixture; regression anchor for all later changes starting with 6.3 |
| **D** — production backtest (`scripts/backtest.py --model lgbm --alpha 0.5 --window rolling --train-span-days 90 --refit-every 1 --test-start 2021-01-01 --test-end 2025-12-31`) | End-to-end reproduction of the documented headline metric | MAE = 15.3988 EUR/MWh vs. the documented reference MAE ≈ 15.40 EUR/MWh (`model_validation_report.md` in the source repo). RMSE 26.5926 vs. 26.59; WAPE 0.1285 vs. 0.129. MLflow run under experiment `arena_models`, tag `study=provenance`. Reviewed and accepted by the owner (2026-08-18) |

**Dependency pinning:** `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `pyarrow` are pinned to the exact versions resolved in the source repo's `uv.lock` (`4.6.0`, `2.3.5`, `2.3.3`, `1.8.0`, `23.0.1`). `mlflow` was pinned as a sixth package (`3.11.1`) after Beleg B surfaced the version-drift failure described above.

**Post-copy patch (2026-08-26):** `entsoe_client.py`'s `EntsoePandasClient(...)` call got an explicit `timeout=30`. The source repo left it at the library default (`None`, i.e. no timeout at all), which let a stalled ENTSO-E request block indefinitely — this caused a real production incident, hanging the daily audit job for 6 hours and, via the `audit.yml` concurrency group, cascading into 3 skipped cron runs on 2026-08-25. This is the one intentional deviation from Beleg A's byte-identical copy; everything else in `src/`, `tests/`, and `scripts/` remains untouched.

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

## Objective and scoring metric

**Claim.** The Energy Arena's point challenge (ID 2) is scored by RMSE, whose optimal estimator is the conditional *mean*. The inherited model (`objective="quantile", alpha=0.5`, Sprint 3 Entscheidung 2) predicts the conditional *median* instead — coherent with the quantile grid and the calibration work of the source project, but not RMSE-optimal on right-skewed day-ahead prices, where the mean sits systematically above the median. `scripts/backtest.py --objective l2` (Sprint 6.3) adds the RMSE-aligned alternative alongside the unchanged default.

**Evidence.** Both objectives were run under the identical production configuration (`--window rolling --train-span-days 90 --refit-every 1 --test-start 2021-01-01 --test-end 2025-12-31`, untuned defaults), then compared by regime and by two time periods — `full` (the whole 5-year window) and `recent` (the last 12 months) — via `scripts/compare_objectives.py`, with a Diebold-Mariano significance test on the squared-error loss differential:

| Period | RMSE median | RMSE l2 | Δ (l2 − median) | DM test p-value (daily / hourly) |
|---|---|---|---|---|
| `full` (2021–2025) | 26.5926 | 26.5261 | −0.0665 (l2 nominally better) | 0.901 / 0.891 — **not significant** |
| `recent` (last 12 months) | 20.6808 | 22.2888 | +1.6080 (median better) | 0.092 / 0.083 — marginal, not significant at 5% |

Full results: [`outputs/results/objective_comparison.csv`](outputs/results/objective_comparison.csv) (regime × period breakdown), [`outputs/results/dm_test_objective.csv`](outputs/results/dm_test_objective.csv) (significance tests).

**Consequence.** The full-window RMSE edge for `l2` is not statistically distinguishable from noise, and in the period that actually matters for the Arena — now — the inherited median objective is *better* on both MAE and RMSE (with marginal, not conventionally significant, DM-test evidence). This is a valid, documented negative result, not an inconclusive one: there is no robust case for switching the default objective on this model and this data. Sprint 6.4 (the Arena bridge) therefore stays on the existing median/quantile default. The `--objective l2` flag remains in the codebase regardless — Sprint 8's quantile challenges need the explicit quantile/l2 separation this step introduced either way.

## Hourly-to-quarter-hourly bridge

**Claim.** The Arena's point challenge (ID 2) submits 92/96/100 quarter-hourly values per delivery day, but the existing model only forecasts on an hourly grid. Sprint 6 Entscheidung 1 ("bridge-first") bets on expanding the hourly forecast with an **additive intra-hour Shape Profile** — the last N days' average deviation from the hourly mean, per (hour, quarter-of-hour) cell — rather than building a native quarter-hourly model up front (that is Sprint 7). This step measures whether that bet pays off: does the expansion carry information over the Arena's own persistence baseline, and does the shape profile itself add anything over the trivial flat expansion (repeating each hourly value four times)?

This is an **upper-bound** measurement, not a live-capability claim: it runs with the full engineered feature set, including `wind_onshore_forecast` / `wind_offshore_forecast` / `solar_forecast` / `residual_load_forecast` columns that the 6.2 availability audit found are **not** published at gate closure. The real live-capability gate, with only what is actually available in time, is Sprint 6.6.

**Evidence.** Walk-forward over the full evaluable window (295 delivery days, 2025-10-29 to 2026-08-20, derived from the data — the first day with 28 full prior days of quarter-hourly price history), production model configuration (LightGBM, `objective="quantile"`, `alpha=0.5`, untuned, full feature set), three candidates compared against the realised quarter-hourly price:

| Candidate | MAE | RMSE | WAPE |
|---|---|---|---|
| `bridge_shape` (hourly model + shape profile, N=28) | 14.3886 | 27.2874 | 0.1391 |
| `bridge_flat` (hourly model, flat expansion) | 15.5813 | 28.8270 | 0.1507 |
| `baseline` (Arena persistence replica) | 28.0315 | 45.5587 | 0.2710 |

Diebold-Mariano tests on the squared-error loss differential, daily-block primary and native quarter-hourly resolution as a robustness check, both periods (`full` and `post_changeover`, i.e. the window whose entire 90-day training span post-dates the switch to the quarter-hourly auction product) — every one of the eight tests significant at p < 1e-10:

- `bridge_shape` vs. `baseline`: decisively better, as expected — day-ahead persistence has large day-to-day error and the hourly model already clears MAE 26.6 EUR/MWh on Beleg D.
- `bridge_shape` vs. `bridge_flat`: **also significant**, not just noise. The shape profile itself carries information beyond the bridge architecture's basic level/form split.

A sensitivity sweep over the shape window N ∈ {7, 14, 56} (against the preregistered headline N=28) is flat — MAE varies by only a few hundredths across the whole range, well within what the differing evaluable windows per N would explain on their own — so N=28 is not a fragile choice.

Full results: [`outputs/results/arena_bridge_backtest.csv`](outputs/results/arena_bridge_backtest.csv) (period × day-type × candidate breakdown), [`outputs/results/dm_test_bridge.csv`](outputs/results/dm_test_bridge.csv) (significance tests), [`outputs/results/shape_profile_sensitivity.csv`](outputs/results/shape_profile_sensitivity.csv) (N sweep).

**A note on the baseline itself.** The persistence baseline replica (`models/arena_baseline.py`) is verified two ways: by hand against calculated examples covering both DST directions (`tests/test_arena_baseline.py`), and — since the Arena exposes no live baseline score to check against before Sprint 6.5's leaderboard client exists — by comparing its output directly against the Arena starter kit's own reference persistence implementation (`energy-arena-participate/_starter_core.py`). The two agree exactly on every ordinary day of the year; on the two DST-transition days, the starter kit's reference code is itself not DST-safe (it refuses to build a payload at all, and its underlying shift logic is structurally unable to produce the correct value count either way), while this repo's replica handles both transitions correctly by construction. This closes the spec's mandatory baseline-verification requirement via a documented Rückfrage and code-level comparison rather than a live platform score — the residual gap (confirming the Arena *server's* actual scoring logic, not just a client-side reference model) stays open until 6.5.

**Consequence.** The bridge architecture carries real information over the Arena baseline, and the shape profile earns its place over the cheaper flat-expansion alternative — both at high significance, not a marginal or ambiguous result. This validates Entscheidung 1 ("bridge-first") well enough to build Sprint 6.5's own renewables forecast on top of it, rather than pivoting to a native quarter-hourly model first. The number above is a best case, not an operating point: with only the features actually available at gate closure, Sprint 6.6 measures what this architecture can really do live, and that number will be lower. The 100-value fall-back-DST day is not yet in today's evaluable window (the next one is 2026-10-25) and remains covered only by the synthetic tests in `tests/test_bridge.py` / `tests/test_arena_baseline.py` until then.

## Weather data

**Claim.** Sprint 6.5's own renewables forecast needs day-ahead weather features that are honestly available at gate closure, not just historically accurate ones. The only architecture that satisfies this is a **fixed run with a fixed lead time**: the 00-UTC Open-Meteo Single Runs API run of D−1, read out at lead times +22 to +48h to cover local delivery day D (model `ecmwf_ifs`, 9 km native resolution). Two more convenient Open-Meteo archives — the *Historical Forecast API* and the *Previous Runs API* — are deliberately never used anywhere in this repo, not in production code, not in tests, not even for a quick comparison.

**Evidence.** Both convenient archives leak information past the gate-closure deadline in a way no backtest metric would ever surface:

| Archive | What it actually returns | Why it leaks |
|---|---|---|
| Historical Forecast API | Stitches together the *first hours* of consecutive runs | For delivery day D at 18:00, the value comes from a run initialized **on D itself** — near-analysis, not a forecast |
| Previous Runs API (`_previous_day1`) | A fixed 24h lead time relative to the valid timestamp | For D's 23:00 hour, that's a run from 23:00 on D−1 — eleven hours **after** this project's gate-closure deadline |

A model trained on either would silently look better in every backtest (it would learn the relationship between near-observed weather and generation) while being served a genuine 22-48h forecast live — a train/serve skew that is invisible in any historical metric, not a smaller version of the same problem. The Single Runs API's `run=` parameter is the only Open-Meteo endpoint that lets a run be pinned explicitly rather than resolved implicitly by the clock — the same "calendar, not clock" rule already enforced in code by `run_init_for_target_day`.

The price of this rule is real and worth naming: Open-Meteo's Historical Forecast API reaches back to 2016/2017 for this model, while the Single Runs archive used here starts only at **2024-03-14** — the provider's own archive start, not negotiable. The historical weather artefact (`data/interim/weather_ifs_run00.parquet`, built 2026-09-01) covers 2024-03-14 through 2026-08-30: 64,512 rows, 896 of 900 calendar days with a run. The 4 missing days (2025-08-05, -06, -08, -09) are provider-side gaps (`modelRunUnavailable` from Open-Meteo itself), not a client bug.

**Consequence.** Sprint 6.5.2's training window for the renewables model is bounded by this archive start, not by how far back ENTSO-E or Yahoo Finance data goes — a real, named cost rather than a silent one. `.github/workflows/weather_availability_probe.yml` runs at several odd-minute morning UTC slots (`23 5,6,7,8,9 * * *`, deliberately off the audit workflow's `:00`/`:45` cron to avoid dispatch congestion) and measures, live, how early the 00-UTC run of D−1 actually becomes available each morning. That is a handover to Sprint 6.7: the submission workflow needs **multiple cron runs spread across the morning**, not one — a single too-early look must not cost the whole day, and GitHub Actions' own schedule-cron drift (observed repeatedly on this repo's `audit.yml`) makes a single well-timed run unreliable regardless. The concrete slot times are 6.7's to derive from whatever the availability log has accumulated by then, not fixed here. Also handed over explicitly: this step builds **no fallback** for a missing run — `fetch_run`/`run_init_for_target_day` raise `WeatherRunUnavailable` and propagate it; catching that exception and deciding not to submit is Sprint 6.7's job (Entscheidung 5b), not this one's.

Weather data by [Open-Meteo](https://open-meteo.com/), used under their free non-commercial terms. Calls stay to what's operationally needed: every run (historical or live) is fetched at most once, via the per-run cache under `data/cache/weather_single_runs/`.

## Capacity anchor table

`data/capacity_anchors_public_registry.csv` holds monthly installed-capacity
support points for solar, onshore wind and offshore wind. They normalise the
renewables target into a capacity factor, so that the model learns weather
rather than build-out. The file is a checked-in snapshot of a public registry
source (Energy-Charts, Fraunhofer ISE), not a live lookup: installed capacity
moves by a fraction of a percent per month and is not worth a runtime
dependency at gate closure.

It does have a shelf life. Values are interpolated between anchors and
extrapolated at most four monthly intervals past the last usable anchor — the
two most recent months are discarded, because the registry still revises them.
Past that limit `installed_capacity_at` raises instead of quietly extending a
stale trend. Every run logs how long the current table is still valid and warns
21 days before it expires. Refresh it roughly monthly:

    uv run python scripts/build_capacity_anchors.py
