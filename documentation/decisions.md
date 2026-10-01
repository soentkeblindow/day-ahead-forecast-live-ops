# Design decisions

The ten decisions behind this repository's architecture, each in 2-3 sentences. Full narrative and measurements for all of these live in the (gitignored, local) sprint logs this README and `removed_modules.md` point to; this file exists so a public reader doesn't need those logs to understand *why*.

1. **Bridge architecture instead of a native quarter-hourly model.** The Arena submits quarter-hourly values, but the underlying price model forecasts hourly. Rather than building a native 15-minute model first, an hourly forecast is expanded with an additive intra-hour shape profile (the last 28 days' average deviation per hour/quarter-hour cell). Measured against the Arena's own persistence baseline and against a flat (unshaped) expansion, both the bridge itself and the shape profile on top of it carry real, statistically significant information (`outputs/results/arena_bridge_backtest.csv`, `dm_test_bridge.csv`).

2. **Reconstruct wind/solar/residual load from weather data, not the TSO's own forecast.** ENTSO-E's day-ahead wind/solar forecast (14.1.D) isn't published until roughly 18:00 the day before delivery — after the Arena's 12:00 gate closure. Training on it anyway (as many backtests implicitly do) would be invisible overfitting to information a live system never has. This repository instead builds its own capacity-factor models from ECMWF weather data, at a measured, named cost (~1.3 EUR/MWh RMSE) rather than a silent one.

3. **An external Cloudflare Worker drives the daily schedule, not GitHub Actions' own `schedule:` trigger.** Measured dispatch delay on this repository's own cron-triggered workflows reached a median of several hours, well past the Arena's gate closure — not an edge case, the normal case. A small Worker re-derives local time from the tz database every 5 minutes and dispatches `workflow_dispatch` at fixed slots instead.

4. **The raw data store is a GitHub Release asset, encrypted, not a git commit.** Committing daily-changing raw data would make the repository's history grow forever and conflicts with keeping it public (one input series, TTF gas via Yahoo Finance, isn't redistributable). A single encrypted tar, re-downloaded fresh by every run and never cached, avoids both problems without carving out individual columns.

5. **A three-row fallback ladder instead of "one model or silence."** The original design treated any single missing input — a late load forecast, a dead commodity feed — as a reason to submit nothing, which the Arena then scores at its own, much weaker persistence baseline. Three independently-measured rows, tried in rank order, mean a missing input degrades the forecast instead of silencing it outright.

6. **A missing load forecast gets patched from a similar day, not treated as fatal.** Rank 2 of the ladder is rank 1's own model and training, unchanged — only the target day's own load-forecast cell is replaced by a same-weekday reference (yesterday, last week, or the last Sunday before a holiday, escalating further back if that reference is itself incomplete). Measured RMSE cost of the patch itself is statistically indistinguishable from using the real forecast.

7. **The floor row is deliberately gas-free.** Keeping TTF gas out of rank 3 removes the one single point of failure a dead commodity feed would otherwise be for the entire ladder — a dead gas feed now costs the two upper rows, not the whole system.

8. **Energy-Charts is a price/load mirror, not a second model input.** `day_ahead_price_ec` fills a genuine ENTSO-E gap via a single merge point where ENTSO-E always wins on conflict — measured to fire on essentially one cell across the full price history. It is not wired in as its own candidate row because the measured timing advantage over ENTSO-E's own load forecast wasn't large enough to justify a second live data dependency.

9. **A later, worse submission may never overwrite an already-accepted better one.** Later slots in the same day can retry with fresher data, but only ever upgrade — not downgrade — what's already been accepted for that delivery day.

10. **Silence is preferred over a degraded, unknown-quality prediction.** A missing feature is never silently filled and fed to the model anyway (a gradient-boosted tree routes a NaN down whichever branch training happened to prefer, which is not "slightly worse," it's unknown). A candidate either has what it actually needs, or the ladder moves to the next row, or the day is silent — visibly, not quietly.
