# External trigger (Cloudflare Workers Cron Triggers)

Replaces GitHub Actions' own `schedule:` trigger as the timekeeper for this
repo (spec 6.7.1, Entscheidung 20). GitHub's own `schedule:` dispatch was
measured at a median delay of +242min (max +693min) since 2026-08-27 --
`docs/cron_jobs.md` section 2 has the full measurement. GitHub Actions stays
the *executor*; this worker is the *scheduler*.

Lives in the repo (not just in the Cloudflare dashboard) so the next person
reading this repo can see what actually runs, without needing access to a
third-party account.

## What it does

One UTC cron trigger, polling every 5 minutes (see `wrangler.toml` and
`src/index.ts::POLL_INTERVAL_MINUTES`). On each tick, the worker re-derives
the actual Europe/Berlin local time from the event's own timestamp and
checks it against all five slots in `src/index.ts::SLOTS`, dispatching a
`workflow_dispatch` for any slot whose half-open time window
`[intended, intended + 5min)` contains the current tick. Almost every tick
matches no slot and silently no-ops -- this is expected, not a bug (see the
code comments for why this has to be stateless).

**Revised during the O4 rollout** from an original ten-entry design (one
fixed UTC cron per local slot, doubled for CET/CEST) after Cloudflare
Workers Free rejected the ninth/tenth entry with "Workers Free limit of 5
cron triggers per account" (error code 10072) -- a live platform limit not
accounted for at design time. The poll-based design needs only one
registered trigger regardless of slot count, since DST correctness came
from recomputing local time every tick, not from which fixed UTC cron
fired -- the CET/CEST doubling was never actually load-bearing for that
property. Full narrative: `docs/sprint6_step6_7_1_log.md`, step A2
addendum.

Current slot table (spec 6.7.1 section 5.1):

| Local time (Europe/Berlin) | Workflow |
|---|---|
| 06:30 | `weather_availability_probe.yml` |
| 09:30 | `audit.yml` |
| 10:10 | `maintain_store.yml` |
| 11:05 | `maintain_store.yml` |
| 11:30 | `audit.yml` |

6.7.2 adds submission slots as additional rows to this table and to
`CRON_TO_SLOT` -- the structure is meant to be extended, not rebuilt.

## Rollout (owner tasks, spec 6.7.1 section 3.5)

1. **O1 -- Cloudflare account.** Free tier is far more than this needs (ten
   scheduled invocations/day against a 100k-requests/day free quota).
2. **O2 -- GitHub token.** Create a **fine-grained personal access token**,
   scoped to **only this repository**, with **Actions: read and write**
   permission, and an **explicit expiration date**. Do not use a
   classic/unscoped token.
3. **O3 -- tell the implementer the token's expiration date.** It goes into
   the store's deadline guard (spec section 5.5) so a maintenance run warns
   `DEADLINE_WARN_DAYS` before the token silently stops working.
4. Install dependencies once: `npm install` (from this directory).
5. Authenticate wrangler: `npx wrangler login`.
6. Store the token as a Worker secret -- **never in a file, never committed**:
   ```
   npx wrangler secret put GITHUB_DISPATCH_TOKEN
   ```
   (paste the token from O2 when prompted; wrangler stores it in Cloudflare's
   secret store, not in this repo).
7. Deploy: `npx wrangler deploy`.
8. **O4, last step -- trigger once by hand** to confirm the deployment works
   before relying on the cron schedule:
   ```
   npx wrangler deployments list   # confirm the deploy landed
   ```
   A full scheduled-event dry run without waiting for the next real cron tick
   can be done locally against `wrangler dev --test-scheduled` (see the
   Wrangler docs for the exact invocation, since the CLI flag has changed
   across versions) -- confirm the console log shows a dispatch attempt, then
   check the Actions tab of the repo for the resulting run.

## What must never live in this repo

The token from O2/step 6 above. Not as a value, not as a placeholder in an
example file, not commented out. This README names the **secret variable
name** (`GITHUB_DISPATCH_TOKEN`) so a reader knows what to configure --
never a value.

## Rotating the token

When the deadline guard warns (or the token is due to expire regardless):
repeat O2 (new fine-grained PAT, same scope) and step 6
(`wrangler secret put GITHUB_DISPATCH_TOKEN` overwrites the old value). No
code change needed. Update the expiry date in the store's deadline table
(spec section 5.5) to match the new token.

## Local development

`npm run dev` starts `wrangler dev` for iterating on the handler logic.
Cron Triggers cannot be fired by an incoming HTTP request in local dev the
way a `fetch` handler can -- use Wrangler's scheduled-event testing flag
(`--test-scheduled`, see the Wrangler CLI docs for your installed version)
and inspect the console output; nothing is dispatched to the real GitHub
repo unless `GITHUB_DISPATCH_TOKEN` is actually set in the dev environment.
