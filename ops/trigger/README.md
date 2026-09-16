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
checks it against all slots in `src/index.ts::SLOTS`, dispatching a
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

Current slot table (spec 6.7.3 section 2.2, owner-confirmed 2026-09-16):
`audit.yml` has retired from the schedule entirely (`workflow_dispatch`
only from here on, manual diagnosis) and `weather_availability_probe.yml`
no longer has any slot inside the 10:00-12:00 maintenance/submission
window (section 2.2: it must never take priority over either).

| Local time (Europe/Berlin) | Workflow |
|---|---|
| 08:30 | `weather_availability_probe.yml` |
| 09:30 | `weather_availability_probe.yml` |
| 10:10 | `maintain_store.yml` |
| 10:40 | `submit.yml` |
| 10:55 | `maintain_store.yml` |
| 11:15 | `submit.yml` |
| 11:25 | `maintain_store.yml` |
| 11:40 | `submit.yml` |
| 18:30 | `weather_availability_probe.yml` |

The three remaining `weather_availability_probe.yml` slots (owner-specified
2026-09-15, originally five, reduced by the 10:00-12:00 exclusion above)
exist for the Energy-Charts knowledge-time probe riding along in the same
workflow (`scripts/probe_energy_charts_forecast.py`), not for the weather
probe itself -- weather's own run has historically been available well
before any of these times. Two samples in the pre-gate-closure (12:00
local) morning, plus one past 18:00 (an independent timing check, not
useful for a submission itself).

The three submission slots are deliberately interleaved with, not appended
after, the maintenance slots: Pflege 10:10 -> Einreichung 10:40 -> Pflege
10:55 (a second chance for the load forecast and the weather run) ->
Einreichung 11:15 -> Pflege 11:25 (a third chance) -> Einreichung 11:40
(the day's last submission attempt, ~20 minutes clear of the 12:00 gate
closure even in the worst case -- spec 6.7.3 section 2.2's own
worst-case-timing derivation). The real day-ahead window is only two hours
wide because the load forecast is only guaranteed at 10:00 local. Each
`submit.yml` dispatch now carries `nominal_slot`/`is_last_slot_of_day` as
`workflow_dispatch` inputs (`Slot.inputs` in `src/index.ts`) instead of
`submit.yml` re-deriving them from its own wall-clock start time.

## Rollout (owner tasks, spec 6.7.1 section 3.5)

1. **O1 -- Cloudflare account.** Free tier is far more than this needs (288
   poll invocations/day, one every 5 minutes, against a 100k-requests/day
   free quota -- the poll design, not the original ten-fixed-cron one; see
   "What it does" above).
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
8. **O4, last step -- confirm the deployment works before relying on the
   cron schedule.** **Cron Trigger changes take up to 15 minutes to
   propagate globally (Cloudflare's own docs) -- deploy once, then wait a
   full comfortable window (an hour-plus lead to a round local time worked
   well) before checking, rather than iterating with fast redeploys.** A
   real rollout (2026-09-09) burned most of a session on exactly this: a
   messy local `wrangler dev --test-scheduled` detour, then several rapid
   production redeploys nudging a temporary test slot a few minutes into
   the future each time -- every one of which likely reset the propagation
   window before it ever completed, so the whole fast-iteration campaign
   never had a real chance to succeed. One clean deploy plus one patient
   wait is what actually worked. Full narrative:
   `docs/sprint6_step6_7_1_log.md`, step O4.
   ```
   npx wrangler deployments list   # confirm the deploy landed
   ```
   Then check the Actions tab of the repo (or `gh run list --workflow=<name>`)
   for the resulting `workflow_dispatch` run once the wait is over.

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
