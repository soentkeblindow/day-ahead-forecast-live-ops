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
}

const REPO_OWNER = "soentkeblindow";
const REPO_NAME = "sbl-energy-forecast";

// Must match wrangler.toml's `crons = ["*/5 * * * *"]`. Deliberately kept
// as one constant instead of two independent numbers, because the
// half-open window check below (inSlotWindow) only guarantees exactly one
// dispatch per slot per day when the window width equals the poll cadence
// -- see that function's comment.
const POLL_INTERVAL_MINUTES = 5;

// weather_availability_probe.yml's own five slots (08:30/09:30/10:30/11:30/
// 18:30, owner-specified 2026-09-15, replacing the original single 06:30
// slot entirely) exist for the Energy-Charts knowledge-time probe now
// riding along in the same workflow (scripts/probe_energy_charts_forecast.py,
// docs/sprint6_auftrag_energy_charts_backup.md section 4): four spread
// across the pre-gate-closure (12:00 local) morning -- a single very-early
// check only shows whether a forecast published overnight, never whether
// one published later in the morning would still land in time -- plus the
// section-4-mandated post-18:00 check (independent confirmation that a
// late-arriving series really is 14.1.D-timed, not useful for a submission
// itself). Same local time as an existing audit.yml slot (09:30, 11:30) is
// not a conflict -- SLOTS supports several workflows per tick, each
// dispatched independently.
const SLOTS: Slot[] = [
  { localTime: "08:30", workflow: "weather_availability_probe.yml" },
  { localTime: "09:30", workflow: "weather_availability_probe.yml" },
  { localTime: "09:30", workflow: "audit.yml" },
  { localTime: "10:10", workflow: "maintain_store.yml" },
  { localTime: "10:30", workflow: "weather_availability_probe.yml" },
  // Submission slots (spec 6.7.2, section 3.4) -- deliberately interleaved
  // with the maintenance/audit slots, not appended after them: Pflege 10:10
  // -> Einreichung 10:40 -> Pflege 11:05 (a second chance for the load
  // forecast and the weather run) -> Einreichung 11:25 -> Pflege 11:40 (a
  // third chance, right before the final submission attempt) -> Einreichung
  // 11:50. The real day-ahead window is only two hours wide because the
  // load forecast is only guaranteed at 10:00 local. Transitional 9-slot
  // day (audit.yml/weather_availability_probe.yml retire in 6.7.3).
  { localTime: "10:40", workflow: "submit.yml" },
  { localTime: "11:05", workflow: "maintain_store.yml" },
  { localTime: "11:25", workflow: "submit.yml" },
  { localTime: "11:30", workflow: "weather_availability_probe.yml" },
  { localTime: "11:30", workflow: "audit.yml" },
  { localTime: "11:40", workflow: "maintain_store.yml" },
  { localTime: "11:50", workflow: "submit.yml" },
  { localTime: "18:30", workflow: "weather_availability_probe.yml" },
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

async function dispatchWorkflow(workflow: string, token: string): Promise<Response> {
  const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/${workflow}/dispatches`;
  return fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "sbl-energy-forecast-trigger",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main" }),
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
      const response = await dispatchWorkflow(slot.workflow, env.GITHUB_DISPATCH_TOKEN);
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
