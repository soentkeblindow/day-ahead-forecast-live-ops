/**
 * External scheduler for the sbl-energy-forecast Arena repo (spec 6.7.1,
 * Entscheidung 20 / section 5.1). GitHub Actions' own `schedule:` trigger
 * measured a median dispatch delay of +242min (max +693min, docs/cron_jobs.md
 * section 2) -- of 72 audit.yml runs since the 08-27 regime change, none
 * landed before gate closure. This worker replaces `schedule:` as the sole
 * timekeeper. GitHub Actions stays the executor: this worker's only job is a
 * well-timed workflow_dispatch call.
 *
 * DST handling (spec 6.7.1 section 2.1): Cloudflare Cron Triggers, like
 * GitHub's, run in UTC only. The ORIGINAL design gave each local slot TWO
 * UTC cron entries in wrangler.toml (one for CET, one for CEST), ten total,
 * with this handler re-deriving actual local time and picking whichever
 * entry matched. REVISED during O4 rollout (docs/sprint6_step6_7_1_log.md,
 * step A2 addendum): Cloudflare Workers Free caps an account at 5
 * registered cron triggers -- the tenth entry failed to deploy (error code
 * 10072), a live platform limit the original design didn't account for.
 * wrangler.toml now registers a single poll cron (every POLL_INTERVAL_MINUTES);
 * this handler re-derives actual Europe/Berlin local time on every tick and
 * checks it against ALL entries in SLOTS below, so the CET/CEST doubling is
 * no longer needed at all -- DST correctness came from recomputing local
 * time per tick, not from which cron string fired, so the fixed-UTC-time
 * encoding was never actually load-bearing for that property. No stored
 * "already fired today" flag anywhere: a slot whose window has already
 * passed for the day simply doesn't match on the next tick and no-ops.
 * Stateless by design -- a second fault mode (a stuck or incorrect flag)
 * would be worse than the drift problem this replaces.
 */

interface Env {
  GITHUB_DISPATCH_TOKEN: string;
}

interface Slot {
  /** HH:MM in Europe/Berlin, the intended local fire time. */
  localTime: string;
  workflow: string;
  /** Extra workflow_dispatch inputs to send with this slot's dispatch
   * (spec 6.7.3 section 2.2) -- submit.yml reads nominal_slot/
   * is_last_slot_of_day from these instead of deriving them from wall-clock
   * time at execution. The old derivation trusted the job's own start time
   * to still fall in the intended slot's window, which a run queued behind
   * a slow maintain_store.yml run (the whole reason for section 2.1's soft
   * budget) is no longer guaranteed to do -- this worker already knows
   * which slot it meant to fire, no need to re-guess it downstream. Omitted
   * for slots that need no inputs (maintain_store.yml/
   * weather_availability_probe.yml). */
  inputs?: Record<string, string>;
}

const REPO_OWNER = "soentkeblindow";
const REPO_NAME = "sbl-energy-forecast";

// Must match wrangler.toml's `crons = ["*/5 * * * *"]`. Deliberately kept
// as one constant instead of two independent numbers, because the
// half-open window check below (inSlotWindow) only guarantees exactly one
// dispatch per slot per day when the window width equals the poll cadence
// -- see that function's comment.
const POLL_INTERVAL_MINUTES = 5;

// Rebuilt for spec 6.7.3 section 2.2 (owner-confirmed 2026-09-16, an
// addition to the spec's own baseline three-maintenance-pass schedule --
// see docs/sprint6_step6_7_3.md section 2.2 for the full worst-case-timing
// derivation): audit.yml's slots are gone entirely (workflow_dispatch-only
// from here on, manual diagnosis only -- the workflow file itself already
// dropped its schedule: trigger back in 6.7.1). weather_availability_probe.yml
// originally kept a pre-gate-closure morning slot (09:30) and a post-18:00
// confirmation slot (18:30) -- both since replaced by maintenance/submission
// passes of their own (see the 2026-09-24 and 2026-09-26 comments below);
// as of 2026-09-26, weather_availability_probe.yml has no scheduled slot
// left in SLOTS at all (workflow_dispatch-only, same as audit.yml).
//
// The 2026-09-21/-22 in-window probe slots (10:35/10:50/11:10/11:35) that
// used to sit here were removed again 2026-09-26 (spec 6.9 section 2.10,
// Schritt 13): the Energy-Charts `load` knowledge-time question they
// existed to narrow is now answered for free by the maintenance job itself
// -- the EC load source is fetched on every one of the four maintain_store.yml
// passes below regardless (spec section 2.4), and scripts/sync_store.py now
// records, every run, whether load_forecast_day_ahead_ec is already
// complete for the next delivery day (logs/store_sync.csv's
// ec_load_target_day/ec_load_complete_for_target_day columns). A dedicated
// probe inside the 10:00-12:00 window is no longer the only way to observe
// this, so 6.7.3's original "never a probe in that window" rule (caution
// about an unrelated third-party call competing for the window, not a real
// resource conflict) applies again without exception. The 18:30 evening
// confirmation slot, outside that window, is unaffected and stays.
//
// Fourth maintenance/submission pass added 2026-09-24 (owner instruction,
// same day as the incident that motivated it): the 2026-09-25 delivery day's
// ENTSO-E `load_forecast_day_ahead` only landed in the store between the
// 08:56 and 09:25 UTC maintain runs (~10:56-11:25 Berlin,
// docs/data_sources_for_live_model_use.md section 1.2's 2026-09-24 update)
// -- past the 10:55 maintenance pass, so the 11:15 submission slot still saw
// an incomplete row and only the 11:40 slot (which caught the 11:25
// maintenance pass) went through. Pflege 11:45 -> Einreichung 11:55 gives a
// fourth, later chance for exactly this failure mode; 11:40 is no longer the
// last slot of the day (is_last_slot_of_day now on 11:55). This narrows the
// safety margin to gate closure from the previous ~20 minutes to ~5 minutes
// in the worst case -- an explicit, owner-confirmed trade-off (2026-09-24),
// not an oversight; a genuinely slow fit/predict pass at 11:55 now has much
// less room before 12:00 than any earlier slot ever had.
//
// The 09:30 morning weather_availability_probe.yml slot was removed the same
// day (owner instruction) to make room for this without adding a slot --
// the 18:30 evening probe was, at the time, the only remaining coverage
// away from the 10:00-12:00 window (the in-window probes mentioned above
// were themselves removed later, 2026-09-26). Losing 09:30 meant a
// broken/missing weather run for the day no longer got an early-morning
// signal, only the 18:30 confirmation -- an accepted reduction in
// early-warning lead time, not a beneficial side effect. The 18:30 probe
// itself was in turn replaced by a maintain_store.yml pass 2026-09-26 (see
// the comment above the 18:30 entry in SLOTS) -- weather-run early-warning
// coverage is now gone entirely in favour of the earlier 09:00/09:30 pair
// and a same-evening store refresh, an explicit owner trade-off, not an
// oversight.
// sync_mode inputs on the maintain_store.yml slots below (docs/
// bugs_in_live_system.md entry 6, 2026-10-01): a real near-miss on that
// job's hard timeout (1085s against a then-20-minute limit) root-caused to
// a live API cache-miss cascade for cross_border_flows -- a source fully
// carried and read by no live code. The four passes inside the
// gate-closure-adjacent window (10:10/10:55/11:25/11:45) now get
// sync_mode: "relevant-only", which skips that source entirely; the two
// passes outside the window (09:00, 18:30) keep sync_mode: "all", so the
// store still gets a fully current refresh twice a day, just shifted away
// from the hours that matter for submission timing. Pending owner approval
// of this table before the next `wrangler deploy` (2026-10-01).
const SLOTS: Slot[] = [
  // Fifth, earlier maintenance/submission pass added 2026-09-26 (owner
  // instruction, same day as the in-window probe removal above): an early
  // 09:00/09:30 pair ahead of the existing 10:10 pass, for the case where
  // the day's data (ENTSO-E load, EC load, weather run) is already complete
  // well before the main window -- the earlier ladder rows (core_gas) can
  // then submit that much sooner instead of waiting for 10:40 regardless.
  // Not the last slot of the day (is_last_slot_of_day stays on 11:55) --
  // this only ever adds an earlier opportunity, never removes a later one.
  { localTime: "09:00", workflow: "maintain_store.yml", inputs: { sync_mode: "all" } },
  {
    localTime: "09:30",
    workflow: "submit.yml",
    inputs: { nominal_slot: "09:30", is_last_slot_of_day: "false" },
  },
  // Maintenance/submission window (spec 6.7.3 section 2.2, extended to four
  // passes 2026-09-24 per the comment above; in-window probes removed
  // 2026-09-26, see the comment above SLOTS): Pflege 10:10 -> Einreichung
  // 10:40 -> Pflege 10:55 (a second chance for the load forecast and the
  // weather run) -> Einreichung 11:15 -> Pflege 11:25 (a third chance) ->
  // Einreichung 11:40 (no longer the last slot) -> Pflege 11:45 (a fourth
  // chance) -> Einreichung 11:55 (the day's actual last submission attempt
  // now, only ~5 minutes clear of the 12:00 gate closure even in the worst
  // case). submit.yml's own nominal_slot/is_last_slot_of_day inputs replace
  // its former wall-clock derivation -- this worker already knows which
  // slot it meant to fire.
  { localTime: "10:10", workflow: "maintain_store.yml", inputs: { sync_mode: "relevant-only" } },
  {
    localTime: "10:40",
    workflow: "submit.yml",
    inputs: { nominal_slot: "10:40", is_last_slot_of_day: "false" },
  },
  { localTime: "10:55", workflow: "maintain_store.yml", inputs: { sync_mode: "relevant-only" } },
  {
    localTime: "11:15",
    workflow: "submit.yml",
    inputs: { nominal_slot: "11:15", is_last_slot_of_day: "false" },
  },
  { localTime: "11:25", workflow: "maintain_store.yml", inputs: { sync_mode: "relevant-only" } },
  {
    localTime: "11:40",
    workflow: "submit.yml",
    inputs: { nominal_slot: "11:40", is_last_slot_of_day: "false" },
  },
  { localTime: "11:45", workflow: "maintain_store.yml", inputs: { sync_mode: "relevant-only" } },
  {
    localTime: "11:55",
    workflow: "submit.yml",
    inputs: { nominal_slot: "11:55", is_last_slot_of_day: "true" },
  },
  // The 18:30 weather_availability_probe.yml confirmation slot was replaced
  // with a maintain_store.yml pass 2026-09-26 (owner instruction, same
  // change as the 09:00/09:30 addition above) -- an evening maintenance
  // pass keeps the store current for the next day's early 09:00 pass
  // instead of only confirming weather-run availability. This is also the
  // point at which the header-migrating maintenance run for Schritt 13's
  // new logs/store_sync.csv columns is expected to actually land live.
  { localTime: "18:30", workflow: "maintain_store.yml", inputs: { sync_mode: "all" } },
];

function localHHMM(utcMillis: number): string {
  // Intl with a named IANA zone re-resolves the UTC offset for the given
  // instant from the tz database itself -- no fixed-offset arithmetic on a
  // timestamp, which is the exact DST-bug class named in the spec's
  // Implementation Notes (four occurrences already this sprint).
  const formatter = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Berlin",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  return formatter.format(new Date(utcMillis));
}

function minutesSinceMidnight(hhmm: string): number {
  // Regex + non-null assertions rather than a bare split().map(Number):
  // tsconfig's noUncheckedIndexedAccess makes a plain destructure of
  // hhmm.split(":") type as (string | undefined)[] even though the input
  // is always a well-formed "HH:MM" literal (from SLOTS or the Intl
  // formatter) -- never caught before because this project's TS code had
  // never actually been run through tsc (found during the O4 rollout).
  const match = /^(\d{2}):(\d{2})$/.exec(hhmm);
  if (!match) {
    throw new Error(`Invalid HH:MM value: ${hhmm}`);
  }
  return Number(match[1]!) * 60 + Number(match[2]!);
}

function inSlotWindow(actualLocal: string, intendedLocal: string): boolean {
  // Half-open [intended, intended + POLL_INTERVAL_MINUTES) rather than a
  // symmetric +/-tolerance: a symmetric window wider than half the poll
  // cadence causes adjacent ticks to double-dispatch the same slot (e.g.
  // ticks at 10:05/10:10/10:15 would all fall within +/-5min of a 10:10
  // slot). A half-open window exactly as wide as the poll interval is
  // guaranteed to contain precisely one tick per day, because ticks land on
  // a fixed POLL_INTERVAL_MINUTES grid. Wraps across midnight via mod 1440
  // for slots that could sit near 00:00 in the future, though none do today.
  const actual = minutesSinceMidnight(actualLocal);
  const intended = minutesSinceMidnight(intendedLocal);
  const delta = (actual - intended + 1440) % 1440;
  return delta < POLL_INTERVAL_MINUTES;
}

async function dispatchWorkflow(
  workflow: string,
  token: string,
  inputs?: Record<string, string>
): Promise<Response> {
  const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/${workflow}/dispatches`;
  // GitHub Actions workflow_dispatch inputs are always strings regardless of
  // their declared `type:` in the workflow YAML (a `type: boolean` input
  // still arrives in ${{ inputs.x }} as the literal string "true"/"false")
  // -- Slot.inputs' Record<string, string> type matches that directly, no
  // JSON boolean/number encoding to get right here.
  const body: { ref: string; inputs?: Record<string, string> } = { ref: "main" };
  if (inputs) {
    body.inputs = inputs;
  }
  return fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "sbl-energy-forecast-trigger",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export default {
  async scheduled(event: ScheduledEvent, env: Env, _ctx: ExecutionContext): Promise<void> {
    const actualLocal = localHHMM(event.scheduledTime);
    const matches = SLOTS.filter((slot) => inSlotWindow(actualLocal, slot.localTime));

    if (matches.length === 0) {
      // Almost every tick lands here -- five slots out of 288 ticks/day is
      // the expected, silent no-op path, not an error condition.
      console.log(`Tick at ${actualLocal} local -- no slot in window, no dispatch.`);
      return;
    }

    for (const slot of matches) {
      // No retry on failure here -- repetition is the schedule's job, not
      // the process's (spec section 5.1: "Kein Retry auf Zeitbasis"). Same
      // argument as weather_client.py's WeatherRunUnavailable: one attempt,
      // no waiting -- the next slot tries again tomorrow.
      const response = await dispatchWorkflow(slot.workflow, env.GITHUB_DISPATCH_TOKEN, slot.inputs);
      if (!response.ok) {
        console.error(
          `Dispatch failed for ${slot.workflow}: HTTP ${response.status} ${await response.text()}`
        );
        continue;
      }
      console.log(`Dispatched ${slot.workflow} at ${actualLocal} local (slot ${slot.localTime}).`);
    }
  },
};
