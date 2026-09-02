# Fixture provenance

## `arena_public_example.json`

- **Retrieved:** 2026-08-19T13:21:52Z
- **Endpoint:** `GET /api/v1/challenges/2` (the `latest_public_example.payload` field of the
  response; the endpoint itself returns the full challenge detail, of which this is one
  sub-object — see `docs/sprint6_step6_2_spec.md` §6.4)
- **Command:**

  ```sh
  curl -s -H "X-API-Key: $ARENA_API_KEY" "$ARENA_API_BASE_URL/api/v1/challenges/2"
  # then extract .latest_public_example.payload
  ```

This is the Arena's own most recent accepted submission for challenge 2 (Day-Ahead Prices |
Germany-Luxembourg | Point Forecast) at retrieval time, not a payload we constructed. It is
compared *structurally*, not numerically, in `tests/test_arena_payload.py` — the values belong
to a different model and are expected to differ from anything we submit; what must match is the
key set, the flat `values` list, the value count for the challenge's resolution, and the
`target_start` format. Re-running the retrieval command later will return a different
`target_start`/`values` (the API updates `latest_public_example` as new submissions land) but the
same structure — that's the point of comparing structurally rather than freezing a specific day's
numbers.

Golden Fixture (a), the payload example from the `energy-arena-participate` starter repo, was not
retrieved (spec §3, Entscheidung 3) — Fixture (b) above is authoritative because it comes from the
Arena API itself, not from a reference implementation.

## `energy_charts_installed_power.json`

- **Retrieved:** 2026-09-02 (source's own `last_update` field: 2026-09-01T13:03:43 UTC)
- **Endpoint:** `GET https://api.energy-charts.info/installed_power?country=de&time_step=monthly&installation_decommission=false`
- **Command:**

  ```sh
  curl -s "https://api.energy-charts.info/installed_power?country=de&time_step=monthly&installation_decommission=false"
  ```

Full, unmodified real API response (18.7 KB) used by
`tests/test_capacity.py`'s `scripts/build_capacity_anchors.py` tests (spec 6.5.2a section 5.4) --
no network access in the tests themselves. Covers `01.2002` through `09.2026`; the tests fix
`as_of=2026-09-02` so the retained-range logic (through the last *complete* calendar month) is
deterministic regardless of when the tests run.
