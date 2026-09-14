"""Shared, single-source-of-truth home for the known ENTSO-E cross-border
hourly-vs-quarter-hourly resolution-transition windows (A9 finding,
2026-09-10). A leaf module with zero internal imports so both
``ops/store.py`` (the write-time NaN-gate, which already imports
``data/entsoe_client.py``) and ``data/_entsoe_cache.py``/``data/entsoe_client.py``
(the cache-hit heuristic, which must not import ``ops/store.py`` --
that would be circular) can reference the exact same values without
either duplicating them or inverting the existing import direction.

Values must stay identical to what ``ops/store.py::EXPECTATION_TABLE``
originally carried inline -- moved here verbatim, not re-derived.
"""

from __future__ import annotations

SCHEDULED_EXCHANGES_LOW_RESOLUTION_WINDOWS: dict[str, tuple[str, str]] = {
    "scheduled_net_de_to_at": ("2024-06-01", "2025-08-01"),
    "scheduled_net_de_to_ch": ("2024-06-01", "2025-07-01"),
    "scheduled_net_de_to_nl": ("2024-06-01", "2025-07-01"),
    "scheduled_net_de_to_pl": ("2024-06-01", "2025-04-01"),
    "scheduled_net_de_to_dk_1": ("2024-06-01", "2025-04-01"),
}

CROSS_BORDER_FLOWS_LOW_RESOLUTION_WINDOWS: dict[str, tuple[str, str]] = {
    "physical_net_de_to_fr": ("2021-08-01", "2025-05-01"),
    "physical_net_de_to_pl": ("2021-08-01", "2024-07-01"),
    "physical_net_de_to_nl": ("2021-08-01", "2021-09-01"),
    "physical_net_de_to_at": ("2021-08-01", "2021-09-01"),
    "physical_net_de_to_ch": ("2021-08-01", "2021-09-01"),
}
