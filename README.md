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
| **A** — file-hash diff of `src/` and `tests/` against the source repo | Every line, not a sample | Byte-identical (42/42 files in `src/`, 26/26 in `tests/`) except the documented omissions above and three later, scoped patches (see below) |
| **B** — copied test suite (`uv run pytest`) | Behavior on the source repo's own tests | 447 passed, 1 deselected (integration marker), no test code changed. `mlflow` was pinned to `3.11.1` (matching the source repo's resolved version) after an initial run surfaced a hard failure caused by newer mlflow's `file://`-store behavior change |
| **C** — golden fixture (`tests/test_provenance.py`) | Default `LGBMForecaster()` behavior on a fixed, in-code synthetic dataset | Bit-identical match against the frozen fixture; regression anchor for all later changes starting with 6.3 |
| **D** — production backtest (`scripts/backtest.py --model lgbm --alpha 0.5 --window rolling --train-span-days 90 --refit-every 1 --test-start 2021-01-01 --test-end 2025-12-31`) | End-to-end reproduction of the documented headline metric | MAE = 15.3988 EUR/MWh vs. the documented reference MAE ≈ 15.40 EUR/MWh (`model_validation_report.md` in the source repo). RMSE 26.5926 vs. 26.59; WAPE 0.1285 vs. 0.129. MLflow run under experiment `arena_models`, tag `study=provenance`. Reviewed and accepted by the owner (2026-08-18) |

**Dependency pinning:** `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `pyarrow` are pinned to the exact versions resolved in the source repo's `uv.lock` (`4.6.0`, `2.3.5`, `2.3.3`, `1.8.0`, `23.0.1`). `mlflow` was pinned as a sixth package (`3.11.1`) after Beleg B surfaced the version-drift failure described above.

**Three intentional deviations from Beleg A's byte-identical copy** (everything else in `src/`, `tests/`, and `scripts/` remains untouched):

1. **Post-copy patch (2026-08-26):** `entsoe_client.py`'s `EntsoePandasClient(...)` call got an explicit `timeout=30`. The source repo left it at the library default (`None`, i.e. no timeout at all), which let a stalled ENTSO-E request block indefinitely — this caused a real production incident, hanging the daily audit job for 6 hours and, via the `audit.yml` concurrency group, cascading into 3 skipped cron runs on 2026-08-25.
2. **Sprint 6.7.1a (2026-09-11):** `entsoe_client.py`'s six `fetch_*` functions gained a purely additive `use_cache: bool = True` keyword, and `_entsoe_cache.py::cached_fetch` now merges into whatever's already on disk instead of replacing it outright (and writes atomically) — fixes a still-open-month cache file being fully overwritten on every maintenance run. Default behavior unchanged; existing callers unaffected.
3. **`load_actual`-weight fallback (2026-09-23):** `normalize.py::to_hourly_vwap`, used to compute the hourly `day_ahead_price` as a `load_actual`-weighted VWAP, either fell back cleanly to a simple mean when an entire hour's weight was missing, or — undetected until now — silently zero-weighted just the affected quarter-hour(s) when only part of an hour's weight was missing, still calling the result a VWAP. This is structurally guaranteed for "today"'s not-yet-elapsed hours every single live run (`load_actual` for the future can't exist yet) and occasionally hits already-elapsed hours when ENTSO-E's actuals publication lags. Fix: `to_hourly_vwap` gained an optional `weight_fallback` parameter (per-quarter-hour fallback to a second weight series, forcing the whole hour to the simple mean if a gap survives even that); `to_hourly()` now passes `load_forecast_day_ahead` as that fallback for the price column. Omitting the parameter (every other caller) reproduces the exact prior behavior. Measured against 8,568 already-settled historical hours: the fix changes exactly 14 of them (all on one already-identified outlier day, 2026-09-22), the rest of the historical window is untouched — not fully bitgleich for the past, unlike deviations 1 and 2, but the affected set is small and precisely known. See `docs/bugs_in_live_system.md` entry 2 for the full measurement.

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

## Live-capable feature set

**Claim.** The official TSO forecast columns (`wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`, and the `residual_load_forecast` term derived from them) are not published in time for the Arena's own gate-closure deadline. This is not an assumption: the first real `audit.yml` run (2026-08-19, ~15:59 local — well after the 12:00 deadline) measured `coverage_ratio=0` and `latest_target_offset_days=-1` for all three wind/solar series for delivery day D, while `load_forecast_day_ahead` was already complete (Sprint 6.2, `docs/sprint6_step6_2_log.md` Schritt 8).

**They are reconstructed from weather model data instead.** Sprint 6.5.1's Open-Meteo Single Runs archive (fixed 00-UTC run of D−1, the only architecture that doesn't leak past gate closure — see "Weather data" above) feeds Sprint 6.5.2's own capacity-factor models, which stand in for the missing TSO series wherever the 6.2 audit found them unavailable (`features/nwp_fundamentals.py`, `_nwp`-suffixed columns, spec 6.5.3).

**What that costs.** Walk-forward over the full evaluable window, two independently-trained hourly LightGBM models (production config, untuned) built per delivery day from two disjoint feature sets — `live` (only what's available at gate closure) and `original` (the full TSO-fundamentals set, an upper-bound reference line, never itself a go-live candidate):

| Resolution | Days evaluated | `original` RMSE (upper bound) | `live` RMSE |
|---|---|---|---|
| Quarter-hourly (binding, 28-day shape profile) | 305 of 312 candidates | 28.7183 | 30.0207 |
| Hourly (secondary, non-binding) | 428 of 451 candidates | 26.0418 | 27.3454 |

Reconstructing from weather data instead of reading the TSO forecast costs about **1.3 EUR/MWh of RMSE** at both resolutions. In physical units (MW, against the realised TSO series over the same window, `scripts/compare_residual_load_mw.py`, `outputs/results/residual_load_reconstruction_mw.csv`): residual load MAE 1993 MW / RMSE 2829 MW (mean error −509 MW, i.e. the reconstruction runs slightly low on average); wind onshore MAE 1210 MW; wind offshore MAE 398 MW; solar MAE 1089 MW.

**What's left.** `live` against the Arena persistence baseline replica — the actual go-live criterion (Entscheidung 24: RMSE(live) < RMSE(baseline) **and** a one-sided Diebold-Mariano test at p < 0.10 on the binding quarter-hourly resolution):

| Candidate | RMSE (quarter-hourly, binding) |
|---|---|
| `live` | 30.0207 |
| Arena persistence baseline | 45.1685 |

DM test p ≈ 0.0000 (squared daily-block loss, HAC lag 192, quarter-hourly horizon 96). **Result: PASS.** Even with only what's honestly available at gate closure, the reconstructed feature set beats the Arena's own baseline decisively — the cost measured above does not erase the advantage found in Sprint 6.4's upper-bound bridge measurement.

Full results: [`outputs/results/arena_live_gate.csv`](outputs/results/arena_live_gate.csv) (both resolutions, both periods, three candidates), [`outputs/results/dm_test_live_gate.csv`](outputs/results/dm_test_live_gate.csv), [`outputs/results/residual_load_reconstruction_mw.csv`](outputs/results/residual_load_reconstruction_mw.csv).

**A number that must never sit next to this one without its difference stated (Entscheidung 8):** the source project's own production-backtest MAE (15.3988 EUR/MWh, Sprint 6.1's provenance proof) is an **hourly, full-feature, 2021–2025** measurement — a different resolution, a different feature set, and a different evaluation window from every number on this page. They answer different questions and are not a before/after comparison.

## Data store and scheduling

**Claim.** GitHub Actions' own `schedule:` trigger is not reliable enough to time this repo's daily jobs, so Sprint 6.7.1 replaces it with an external timekeeper (a Cloudflare Worker Cron Trigger) and moves the raw input data itself into a versioned, independently-validated store — because it can no longer assume a same-day GitHub Actions cache is fresh.

**Evidence — why the timer moved.** Measured over 85 real `audit.yml` runs and 67 real `weather_availability_probe.yml` runs since the 08-27 regime change (`docs/cron_jobs.md`, gitignored, local-only): median dispatch delay **+242 min** (mean +225, max +693) for `audit.yml`, median **+266 min** (mean +273, max +382) for the weather probe — none of the 6.6-era `audit.yml` runs landed before the Arena's own gate-closure deadline. `ops/trigger/` (a Cloudflare Worker, real deploy verified end-to-end via a genuine `workflow_dispatch` call reaching GitHub) re-derives Europe/Berlin local time from the tz database on every 5-minute tick and dispatches `workflow_dispatch` at five fixed local slots — GitHub Actions stays the executor, only the timing decision moved.

**What's in the store, and what deliberately is not.** `src/energy_price_forecast/ops/store.py` packs and versions exactly the *raw*, externally-fetched series this repo depends on — ENTSO-E (day-ahead prices, load, wind/solar forecasts, generation by type, scheduled exchanges, physical cross-border flows), the Open-Meteo weather archive, and TTF gas / EUA CO2 — as a single tar, published as a GitHub Release Asset (not a repo commit, not Actions Cache: this data changes daily and would otherwise bloat the repository's history forever). It **never** contains anything derived (`data/interim/hourly.parquet`, `data/processed/features.parquet`, the renewables artefact, or any `.bak` file) — features are always rebuilt fresh from the raw store on every run (Entscheidung 19), never cached themselves. A positive-list glob (`_SOURCE_GLOBS`, keyed by source name) enforces this: what isn't named doesn't travel.

**Consequence — why every update passes an independent control before it can enter the store.** In Sprint 6.6, `data/_entsoe_cache.py`'s own completeness check silently accepted a stale, roughly 4×-too-short monthly cache file because it assumed hourly resolution on a series that had been quarter-hourly since 2025-09-30 — a real incident, not a hypothetical one, and the direct reason `ops/store.py` never trusts a fetch's own cache-write logic. `validate_source()`/`validate_weather_run()` independently re-check column set, index sanity, gaplessness (for a definitely-over period), and per-column NaN fraction before anything is allowed to advance the published manifest; a rejected fetch reverts the on-disk cache to its exact pre-fetch state (byte-identical, verified — `docs/sprint6_step6_7_1_log.md` "Schritt A11") rather than silently keeping a half-corrupted file. The rebuild path (`scripts/rebuild_store.py`, years of history) and the daily incremental path (`scripts/sync_store.py`, hours to days) still use genuinely different tolerances for the same underlying data — discovered the hard way when a single already-known permanent gap, negligible against six years of history, spuriously failed the very first live sync run at ~5% of that day's tiny checked window.

**Claim (6.7.1a) — the store gates structural integrity, not per-column data quality.** Two further real findings, both from the first days of unattended automatic operation (2026-09-11), separated what the store's write-time gate is actually responsible for from what belongs to the daily submission decision. **Evidence:** the live sync's own NaN check used to measure a fraction over whatever window the current sync gap happened to be — anywhere from a few hours to hundreds of days — so a fixed percentage threshold never meant the same thing twice; and two real, fully automatic `maintain_store.yml` runs went red over a genuine, persisting ENTSO-E gap in `gen_hard_coal`, a raw column no feature, model, or evaluation code anywhere actually reads (confirmed by a source-scan test, not assumed), freezing the three `generation`-group columns the price model *does* need (`gen_wind_onshore`/`_offshore`/`_solar`, via `features/lags.py`'s forecast-error lags) along with it. **Consequence:** `EXPECTATION_TABLE` now separates CHECKED columns (present-required, NaN-gated) from CARRIED ones (fetched and stored because ENTSO-E returns them in the same response anyway, but read by nothing — never a blocking reason, in either mode); the live-mode NaN check runs over a fixed 7-day window regardless of the actual sync gap and only blocks on gross corruption (≥90% NaN) or a structural finding (bad schema, shrinking range, a broken index) — a real but partial gap in a checked column is now recorded (manifest + `logs/store_sync.csv`) as a non-blocking hint with an absolute cell count, not silently discarded and not a reason to halt the store. Judging whether *this specific submission* has what it needs from those checked columns is explicitly left to 6.7.2's own per-column freshness check, which reads the exact counts this step now records rather than re-deriving them.

## Daily submission

**Claim.** The full submission path — check, fit, predict, build the payload, log, submit — runs live on the real schedule. Sprint 6.7.2 built the pipeline with the `POST` structurally off and produced the dry-run evidence for it; Sprint 6.7.3 flipped it on. `.github/workflows/submit.yml` fires at three fixed local slots (10:40, 11:15, 11:40 Europe/Berlin, dispatched by the same Cloudflare Worker as everything else here) ahead of the Arena's 12:00 gate closure. Every slot that produces a complete payload submits — there is no separate "already submitted today" gate — and the Arena scores whichever submission for a delivery day was the last one accepted before the deadline, so a later slot correcting or replacing an earlier one is the intended behavior. Every run writes a protocol line to `logs/submissions.jsonl` plus, when it got that far, the built payload to `logs/submission_payloads/<delivery day>.json`.

**The live switch is a repository variable, not a code path.** `ARENA_LIVE` (a GitHub repository *variable*, not a secret — visible and flippable from the repo UI without a commit or a deploy) gates the single POST call site in `arena/submit.py`; `arena/config.py::is_live_enabled()` requires the exact literal `"true"` — unset, empty, or any other value keeps a run in dry-run mode. That default is fail-safe by design: a misconfigured or missing variable silences the platform call rather than sending something. The same variable is the kill switch — setting it back to anything but `"true"` stops live submissions from the next scheduled run onward, no deploy needed. A one-off, unconditional real submission independent of the switch also exists (`mode=smoke` on a manual dispatch, a validated persistence-baseline payload for D+2 that never triggers a model fit) — for confirming a new API key or a platform change before trusting the scheduled path again, not for routine use.

**The maintenance job that feeds this path carries its own soft time budget.** `maintain_store.yml`'s sync step checks an injectable clock against a 900-second internal budget (`SYNC_SOFT_BUDGET_SECONDS`, derived from 12 real run times — worst observed successful run 11.9 minutes, comfortably under the 20-minute hard job timeout) before starting each new source or heal step; on overrun it skips the remaining work, logs which sources were skipped, and still reaches `store.publish_store()`. That last part is the point: on 2026-09-16, before this existed, a hung ENTSO-E source let the hard 20-minute timeout kill the job mid-heal, discarding the entire run's already-successful work — including a same-run weather fetch that had already completed — because publishing sat after the expensive part. The budget can no longer let a slow upstream source erase work that already succeeded; it only ever costs the skipped sources their own freshness for that one run.

**Evidence — freshness is measured where staleness actually bites.** The obvious design is a table of maximum ages per source. This repo deliberately has none, because such a table has to be guessed and then drifts away from what the code really reads. Instead, the feature row the model would predict from *is* the check: the pipeline builds it, and every feature the selected candidate declares must be present and non-NaN. A price history that only reaches D−3 leaves the 24h and 48h lags empty and the row fails; a load forecast missing for D empties its own column and the row fails. This is strictly more precise in both directions than an age table — a source that is old but still sufficient for the features actually used (commodities over a weekend) passes correctly, and a source that is fresh but whose cell is empty for some other reason is caught correctly. Two gates that can contradict each other never come into existence. Ages per source are still recorded in the protocol, because Sprint 6.8 needs them to reconstruct failure *patterns* ("ENTSO-E lagged two days") rather than only their symptoms ("feature X was NaN").

Three things the built row cannot see are checked separately: the weather run for D−1 must exist and pass `validate_weather_run()` before any fitting starts (an HTTP-200-but-all-NaN file was found for real in 6.7.1); the capacity anchor table and the holiday calendar must still cover the delivery day (a calendar that ends turns a real holiday into a plain weekday — no NaN, no error, just a wrong value); and the 90-day training window must reach as far as it should and carry the expected number of rows, which is the one failure mode the target row cannot show. That last check exists because of a real incident on 2026-09-11, where a frozen source would have shortened the window by five days without producing a single NaN.

The one place where forward-filling makes staleness invisible — TTF gas and EUA CO2, which do not trade on weekends and holidays — was measured rather than assumed: the real history contains one 5-day gap in gas and ten in EUA CO2. The fill limit was raised from 4 to 7 days accordingly, then to 14 (Sprint 6.9: the owner's own re-check of the full history still found no real gap longer than 4 days, so the higher limit is bounded headroom against a future outage, not evidence of one already seen) — still bounded either way, so a permanently dead feed still turns into NaN and still silences the day — with a non-blocking warning that stays at the *original* 4-day limit, which marks exactly the cases that would have been a silent day before either change.

**Consequence — the system stays silent rather than submitting something worse.** A missing submission is scored with the Arena's own persistence baseline, so the floor of the fallback ladder costs exactly what the baseline costs. Submitting with a feature missing does not: LightGBM routes a NaN into a direction learned at training time, where the column was present, and the result is not a slightly worse forecast but one of unknown quality. A degraded model is therefore a model trained and measured on its own reduced feature set (Sprint 6.9, "Fallback ladder" below), never the full model with columns missing at prediction time. It is a loop over rows, not an `if complete: submit else: skip`, because provisional branches harden and the submission path is the last place anyone wants to rebuild under load.

Silence is made visible instead of being hidden: a delivery day that produced no submission by its last slot turns that run red (`SILENCE_STREAK_THRESHOLD = 1`, an explicit policy parameter, not a code detail), while earlier incomplete runs of the same day stay green — they still have later attempts. A live system that quietly does nothing is worse than no live system at all.

**What building this actually found.** Nine real findings across nine sessions, none of which any synthetic test fixture had shown, because fixtures get built at convenient resolutions and over convenient periods. Four of them turned out to be one class, and it is the interesting result of this step: **the backtest machinery is label-driven — a day exists because there is a value to measure against — while the live path needs a prediction for a day that has no label by definition.** The delivery-day price does not exist at submission time (that is the whole point of forecasting it); the TSO's own renewables forecast for that day is published at 18:00 on D−1, after gate closure; and "today", the most recent training day, is structurally only half finished at 10:40 in the morning. Each of these silently dropped a day that the pipeline needed, and each stayed invisible for the same reason: every wiring probe until then had used a *past* delivery day, for which the world happened to already be complete. The standing rule that came out of it — every probe pins `as_of` to a real submission slot and never reads the wall clock — is worth more than any of the individual fixes.

A real end-to-end run for delivery day 2026-09-14 (`as_of` pinned to 10:40 local) produces a complete payload: 96 values, 116.36–284.16 EUR/MWh, mean 196.04, with a plausible German day-ahead shape — morning peak 6–8, midday solar trough 12–15, evening peak 18–21. The only excluded training days are the two already-documented corrupt weather runs, which the extent check tolerates by name rather than by a blanket allowance.

**Status.** The dry-run phase (three real delivery days with a complete payload from the real trigger and workflow, 2026-09-15 to -17) is closed. The smoke test (2026-09-17, a real accepted baseline submission, `submission_id=70554`) and the live switch (`ARENA_LIVE=true`, set 2026-09-17T13:33:19Z) both happened for real before the first model-based live submission — 2026-09-18, target day 2026-09-19, `submission_id=70706` — replaced that smoke baseline exactly as designed. The Arena's own leaderboard confirmed the same day: model RMSE 33.01 against a benchmark of 93.92, rank 6 for that single delivery day (a daily figure, not a cumulative ranking). Full narrative, including the nine real findings from building this and the first live day's own findings: `docs/sprint6_step6_7_3_log.md`.

## Fallback ladder

**Claim.** The single-row design above ("one model or silence") turned any one missing input — a load forecast that hadn't published yet, a dead commodity feed, an expired capacity anchor table — into a silent day, scored at the Arena's own (much weaker) persistence baseline. Sprint 6.9 replaces it with three independently gemessen rows, tried in rank order; the first one whose own required columns are actually present for the delivery day wins.

| Rank | Row | Reads | Load source | Measured RMSE |
|---|---|---|---|---|
| 1 | `core_gas` | calendar, NWP residual-load reconstruction, price lags, TTF gas | ENTSO-E `load_forecast_day_ahead` | 28.97 |
| 2 | `core_gas_loadpatch` | identical to rank 1 | Similar-Day-Patch (below) | 29.01 |
| 3 | `base` | calendar, price lags only — **no gas** | — | 38.39 |
| — | silence | — | — | Arena persistence baseline, 45.17 |

**Rank 2 is not a second model.** It is rank 1's own model and training, unchanged — only the target day's own `load_forecast_day_ahead` cell is replaced, by a same-weekday reference (yesterday for Tue–Fri, last week for Mon/Sat/Sun, the last Sunday before a holiday), escalating a week further back if that reference is itself a holiday or its own forecast is incomplete (`src/energy_price_forecast/arena/load_patch.py`). Measured against the real Endregel (holiday/availability escalation plus the exact 2 a.m. DST handling for both the 23h and 25h transition days) rather than the cruder weekday heuristic first sounded out: RMSE 29.01, indistinguishable from rank 1's own real-load RMSE at any reasonable significance (DM p ≈ 0.75) — the patch costs essentially nothing on the days it's actually needed.

**Rank 3 is deliberately gas-free.** Keeping TTF gas out of the floor row removes the one single point of failure a dead commodity feed used to be: rank 1/2 fail without it, but the system no longer goes silent — it drops to rank 3 instead, still comfortably ahead of the baseline (38.39 vs. 45.17).

**`checked` → `carried`, timed to this exact step.** Store columns no row above reads any more (`load_actual`, all twelve cross-border-flow columns each way, `generation`'s per-fuel-type breakdown, EUA CO2) moved from gating a store sync to merely being carried alongside it — their absence is now a hint, never a red run. The reclassification landed together with the model that stopped needing them, not before: an earlier model still relying on the old single row would have gone silent on exactly the columns this change stops checking.

**Downgrade protection.** A later slot's own row must be at least as good (numerically lower or equal rank) as whatever was already accepted live for the same delivery day before it is allowed to send — a later slot succeeding at rank 3 must never overwrite an earlier slot's already-accepted rank 1. A blocked send still archives its payload and keeps the run green (`KEPT`, spec Entscheidung — read from `logs/submissions.jsonl`, not a separate lock file).

**`full` left live operation, and why that's a valid trade.** The 6.6-gated single-row model (`full`, the full-fundamentals reconstruction measured and gated for go-live in Sprint 6.6) is no longer served by the live path — plain rank 1 (`core_gas`) is. This is allowed because `core_gas` clears the exact same go-live bar 6.6 used (Entscheidung 24: RMSE below the Arena's own persistence baseline, one-sided DM test p < 0.10) in its own real production configuration, not a relaxed one — replacing `full` costs nothing against the criterion that gated it into live operation in the first place.

**Energy-Charts as a price mirror, not a second load source.** `day_ahead_price_ec` (Fraunhofer ISE's independent day-ahead price series) feeds every price consumer — lags, labels, the shape profile, the persistence fallback — via a single `coalesce_price()` merge point: ENTSO-E wins whenever both are present, EC only fills a genuine ENTSO-E gap, silently (a rounding-convention change on either side must never take the system down). Across the full measured history this fires on exactly one cell (an archive-start boundary artifact, zero real conflicts). `load_forecast_day_ahead_ec` is collected the same way but read by no row: Sprint 6.8's own availability research found Energy-Charts' load forecast was never available *before* ENTSO-E's on any day it checked (and on 2026-09-25 was still missing at 11:36 local), so a second live source buys no timing advantage for exactly the failure mode it would need to cover — the Similar-Day-Patch above already handles "ENTSO-E's load forecast is missing" almost as well (29.01 vs. a measured 29.04 for an EC-load candidate) without a second data dependency. Each maintenance run still records the first slot at which the EC load forecast for a given delivery day was complete (`logs/store_sync.csv`, `ec_load_target_day`/`ec_load_complete_for_target_day`); if a real timing advantage shows up over time, `core_gas_ec` (already measured, RMSE 29.04, clears the same gate) can return as its own row — not part of this step.

**Limits, stated rather than hidden.** The Similar-Day-Patch escalates up to `MAX_LOAD_PATCH_WEEKS_BACK=4` weeks back, a deliberately generous, *unmeasured* bound — a longer real outage hits the training-window tolerance first. The combination of an active patch with a multi-day training-window gap, and a renewables label-edge gap beyond `MAX_RNW_LABEL_EDGE_AGE_DAYS=7` days, are both left unmeasured on purpose: the gap to rank 3 (≈30 vs. 38 EUR/MWh RMSE) makes either measurement low-value for the cost of building it. Loosening the price model's own "whole day excluded on any training gap" policy stays explicitly out of scope.

Full narrative, the real historical days where the Similar-Day-Patch's escalation logic actually fired (including a genuine two-step case), and the Parity Check proving each row's own builder is a byte-identical column subset of the production feature function: `docs/sprint6_step6_9_log.md` (gitignored, local-only).
