# Operations notes

A short, public-facing extract. The full internal runbook (incident history, exact schedule table, per-source diagnosis steps) stays in the project's own local notes — this covers what an outside reader needs to understand how the live system is operated.

## The live switch

Whether the submission job actually posts to the Arena is a GitHub repository *variable* (`ARENA_LIVE`), not a code path — flippable from the repository settings without a commit or a deploy. Fail-safe by design: anything other than the exact string `true` keeps a run in dry-run mode (it still builds and validates a payload, just never sends it).

## If the key is ever lost

The encrypted data store's key (`STORE_ENCRYPTION_KEY`) is a symmetric Fernet key. If it's ever lost, the store doesn't need to be decrypted to recover — `scripts/rebuild_store.py` rebuilds the entire store from the original data sources instead, from a fresh key. This is the deliberate escape hatch, not a last resort improvised after the fact.

## What a red run means

Not every red run is a problem. A completely silent delivery day (every fallback row's inputs incomplete) is reported in the normal course of operation — the Arena scores it with its own persistence baseline, and in the early phase of running this system a silent day is itself useful information, so it's intentionally still flagged (`SILENCE_STREAK_THRESHOLD`). A genuine platform rejection, a network/transport error, or an unhandled exception are different: those are real problems, independent of how many delivery days have gone well before them, and turn red on every attempted slot, not just the last one of the day.

## Manual commands

```
gh workflow run "Maintain data store"     # trigger a maintenance sync by hand
gh workflow run "Daily submission"        # trigger a submission attempt by hand (respects ARENA_LIVE)
gh workflow run "Daily submission" -f mode=smoke   # unconditional baseline submission, for confirming a new API key
```

A manually-triggered submission run after the Arena's own gate closure exits cleanly rather than failing — that's the first check every run performs on itself, not an error.
