"""Reproducible builder for the public-registry capacity anchor table
(spec 6.5.2a, section 5.4, optional per section 2.1).

Calls the same Energy-Charts endpoint with the same parameters as the
original 6.5.2 step 2 fetch and writes the same file structure (long form,
MW, four ``#`` comment lines) to
``data/capacity_anchors_public_registry.csv``. Only ``Solar DC``,
``Wind onshore`` and ``Wind offshore`` are used -- ``Solar AC`` is
deliberately not the target series (see ``data/capacity.py``'s module
docstring for why). Anchors run from 2015-01 through the last calendar
month strictly before the fetch date; the API includes the current
in-progress month too, but it is never complete (observed live: it just
repeats the prior month's value verbatim until the month closes).

Fail-fast, not silent repair: a missing target series, a value gap inside
the retained range, a non-monotonically-increasing series, a gap in the
monthly cadence, or a value for an anchor older than
``_REVISION_CHECK_MONTHS`` that differs from the currently committed file
all abort with no write. The last one is the revision check from spec
6.5.2a section 2.2: the registry revises its most recent one to two months
upward as later reports arrive (this is exactly why
``TRAILING_ANCHORS_DISCARDED`` exists), so a changed *older* value is a
genuine event -- worth a deliberate look, not an unattended overwrite.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from typing import Final

import numpy as np
import pandas as pd
import requests

from energy_price_forecast.data.capacity import PUBLIC_REGISTRY_ANCHORS_PATH, ProductionType

logger = logging.getLogger(__name__)

_ENDPOINT: Final = "https://api.energy-charts.info/installed_power"
_PARAMS: Final = {"country": "de", "time_step": "monthly", "installation_decommission": "false"}
_TIMEOUT_S: Final = 30

_SOURCE_SERIES_NAME: Final[dict[ProductionType, str]] = {
    ProductionType.SOLAR: "Solar DC",
    ProductionType.WIND_ONSHORE: "Wind onshore",
    ProductionType.WIND_OFFSHORE: "Wind offshore",
}

_ANCHOR_START: Final = pd.Timestamp("2015-01-31", tz="UTC")

# Revision-check floor (spec 6.5.2a section 2.2 / 5.4): anchors older than
# this many months are assumed stable; a changed value there is a revision
# event, not routine monthly movement, and aborts the refresh.
_REVISION_CHECK_MONTHS: Final[int] = 3

_COMMENT_LINES: Final = (
    "# Source: Energy-Charts API (Fraunhofer ISE), {url}",
    "# Fetched: {fetched_date} (source last_update: {last_update})",
    "# Definition: installed power at the end of each calendar month, in GW per the source "
    "API (converted to MW here); 'solar' is the source's 'Solar DC' series (module/nameplate "
    "DC capacity, not inverter-limited AC)",
    "# Fixed as the permanent capacity source (spec 6.5.2a section 5.5): ENTSO-E 14.1.A is a "
    "rejected alternative, not a pending second source; see the data/capacity.py module "
    "docstring and docs/sprint6_step6_5_2a_log.md for the rationale",
)


def fetch_response() -> dict[str, object]:
    resp = requests.get(_ENDPOINT, params=_PARAMS, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _parse_month(label: str) -> pd.Timestamp:
    """'MM.YYYY' -> end-of-month UTC timestamp, matching the anchor_date convention."""
    month, year = label.split(".")
    return pd.Timestamp(year=int(year), month=int(month), day=1, tz="UTC") + pd.offsets.MonthEnd(0)


def _last_complete_month_end(as_of: pd.Timestamp) -> pd.Timestamp:
    """The last calendar month strictly before ``as_of``'s month -- the
    current in-progress month is never complete."""
    first_of_this_month = pd.Timestamp(year=as_of.year, month=as_of.month, day=1, tz="UTC")
    return first_of_this_month - pd.Timedelta(days=1)


def parse_anchors(response: dict[str, object], *, as_of: pd.Timestamp) -> pd.DataFrame:
    """Long-form (anchor_date, production_type, capacity_mw), 2015-01
    through the last complete calendar month before ``as_of``, GW converted
    to MW. Raises on a missing target series or a null value inside the
    retained range."""
    time_labels: list[str] = response["time"]  # type: ignore[assignment]
    production_types: list[dict[str, object]] = response["production_types"]  # type: ignore[assignment]
    by_name: dict[str, list[float | None]] = {
        str(p["name"]): p["data"]  # type: ignore[misc]
        for p in production_types
    }

    last_complete = _last_complete_month_end(as_of)

    rows: list[dict[str, object]] = []
    for production_type, series_name in _SOURCE_SERIES_NAME.items():
        if series_name not in by_name:
            raise ValueError(
                f"target series {series_name!r} missing from the Energy-Charts response "
                f"for production_type={production_type.value!r}"
            )
        values = by_name[series_name]
        if len(values) != len(time_labels):
            raise ValueError(
                f"series {series_name!r} has {len(values)} points, expected "
                f"{len(time_labels)} (matching 'time')"
            )
        for label, value_gw in zip(time_labels, values, strict=True):
            anchor_date = _parse_month(label)
            if anchor_date < _ANCHOR_START or anchor_date > last_complete:
                continue
            if value_gw is None:
                raise ValueError(
                    f"null value for {series_name!r} at {label!r} within the retained "
                    "range -- the Energy-Charts response has an unexpected gap"
                )
            rows.append(
                {
                    "anchor_date": anchor_date,
                    "production_type": production_type.value,
                    # The source reports GW to 3 decimal places, so MW is an
                    # exact integer mathematically; round away the float64
                    # representation noise (e.g. 65.236 * 1000.0 evaluating
                    # to 65236.00000000001) rather than committing it.
                    "capacity_mw": float(round(float(value_gw) * 1000.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["production_type", "anchor_date"]).reset_index(drop=True)


def validate_monotonic_cadence(anchors: pd.DataFrame) -> None:
    """Fail-fast on a value decrease or a gap in the monthly cadence,
    per production type (spec 6.5.2a section 5.4)."""
    for production_type, group in anchors.groupby("production_type"):
        dates = pd.DatetimeIndex(group["anchor_date"])
        values = group["capacity_mw"].to_numpy()

        decreases = values[1:] < values[:-1]
        if decreases.any():
            bad = int(np.argmax(decreases))
            raise ValueError(
                f"production_type={production_type!r} is not monotonically non-decreasing "
                f"at {dates[bad + 1].date()}: {values[bad]} -> {values[bad + 1]} MW"
            )

        expected = pd.date_range(dates[0], dates[-1], freq="ME")
        if len(expected) != len(dates) or not (expected == dates).all():
            raise ValueError(
                f"production_type={production_type!r} has a gap in the monthly cadence "
                f"between {dates[0].date()} and {dates[-1].date()}"
            )


def validate_against_committed(
    anchors: pd.DataFrame, *, as_of: pd.Timestamp, force: bool = False
) -> None:
    """Fail-fast if any anchor older than ``_REVISION_CHECK_MONTHS`` differs
    from the currently committed file (spec 6.5.2a section 2.2's revision
    check) -- and log the new-vs-overlapping split either way, so the "does
    the new file only differ in newly added months" question (spec section
    5.4) is answered on every run, not assumed.

    ``force=True`` downgrades the abort to a logged warning (full changed
    table included) and lets the caller proceed to write anyway -- for a
    deliberate, owner-reviewed refresh, not a routine one. The default stays
    exact-equality fail-fast; ``force`` never silently loosens the check for
    unattended runs, it only exists as an explicit, visible override."""
    if not PUBLIC_REGISTRY_ANCHORS_PATH.exists():
        logger.info("no committed anchor table yet -- skipping the revision check")
        return

    old = pd.read_csv(PUBLIC_REGISTRY_ANCHORS_PATH, comment="#")
    old["anchor_date"] = pd.to_datetime(old["anchor_date"], utc=True)

    merged = old.merge(
        anchors, on=["anchor_date", "production_type"], suffixes=("_old", "_new"), how="inner"
    )
    revision_floor = as_of.normalize() - pd.DateOffset(months=_REVISION_CHECK_MONTHS)
    stable = merged.loc[merged["anchor_date"] < revision_floor]
    changed = stable.loc[~np.isclose(stable["capacity_mw_old"], stable["capacity_mw_new"])]
    if not changed.empty:
        detail = changed[["anchor_date", "production_type", "capacity_mw_old", "capacity_mw_new"]]
        message = (
            "Energy-Charts revised anchor(s) older than the "
            f"{_REVISION_CHECK_MONTHS}-month floor -- a revision event (spec 6.5.2a "
            f"section 2.2), not routine refresh:\n{detail}"
        )
        if not force:
            raise ValueError(message + "\naborting without writing (pass --force to override)")
        logger.warning("%s\nproceeding anyway: --force was passed", message)

    new_pairs = set(zip(anchors["anchor_date"], anchors["production_type"], strict=True))
    old_pairs = set(zip(old["anchor_date"], old["production_type"], strict=True))
    added = new_pairs - old_pairs
    logger.info(
        "refresh check: %d (anchor_date, production_type) pairs are new relative to the "
        "committed file, %d overlap and match within tolerance, 0 revisions beyond the "
        "%d-month floor",
        len(added),
        len(merged) - len(changed),
        _REVISION_CHECK_MONTHS,
    )


def build_csv_text(
    anchors: pd.DataFrame, response: dict[str, object], *, fetched_at: pd.Timestamp
) -> str:
    last_update_ts: int = response["last_update"]  # type: ignore[assignment]
    last_update = dt.datetime.fromtimestamp(last_update_ts, tz=dt.UTC)
    url = requests.Request("GET", _ENDPOINT, params=_PARAMS).prepare().url

    header = "\n".join(_COMMENT_LINES).format(
        url=url,
        fetched_date=fetched_at.date().isoformat(),
        last_update=f"{last_update.isoformat().replace('+00:00', '')} UTC",
    )
    body = anchors.assign(anchor_date=anchors["anchor_date"].dt.date.astype(str))[
        ["anchor_date", "production_type", "capacity_mw"]
    ]
    return header + "\n" + body.to_csv(index=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="proceed past a failed revision check (spec 6.5.2a section 2.2) after a "
        "deliberate, owner-reviewed look at the logged diff -- not for unattended use",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    as_of = pd.Timestamp.now(tz="UTC")

    response = fetch_response()
    anchors = parse_anchors(response, as_of=as_of)
    validate_monotonic_cadence(anchors)
    validate_against_committed(anchors, as_of=as_of, force=args.force)

    csv_text = build_csv_text(anchors, response, fetched_at=as_of)
    PUBLIC_REGISTRY_ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_REGISTRY_ANCHORS_PATH.write_text(csv_text, encoding="utf-8", newline="\n")
    logger.info("wrote %d rows to %s", len(anchors), PUBLIC_REGISTRY_ANCHORS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
