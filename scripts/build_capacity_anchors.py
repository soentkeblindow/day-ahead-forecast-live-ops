"""Reproducible builder for the public-registry capacity anchor table
(spec 6.5.2a, section 5.4, optional per section 2.1; vintage layout per
spec 7.4.1 section 3.5).

Calls the same Energy-Charts endpoint with the same parameters as the
original 6.5.2 step 2 fetch and writes the same file structure (long form,
MW, four ``#`` comment lines) as a **new** vintage file
``data/capacity_anchors/public_registry_<Abrufdatum>.csv`` -- an existing
vintage file is never overwritten (spec 7.4.1 section 3.1); the builder
aborts if the target filename already exists. ``SHA256SUMS`` in the same
directory gets a new line appended. Only ``Solar DC``, ``Wind onshore`` and
``Wind offshore`` are used -- ``Solar AC`` is deliberately not the target
series (see ``data/capacity.py``'s module docstring for why). Anchors run
from 2015-01 through the last calendar month strictly before the fetch
date; the API includes the current in-progress month too, but it is never
complete (observed live: it just repeats the prior month's value verbatim
until the month closes).

Fail-fast, not silent repair: a missing target series, a value gap inside
the retained range, a non-monotonically-increasing series, or a gap in the
monthly cadence all still abort with no write. A value for an anchor older
than ``_REVISION_CHECK_MONTHS`` that differs from the latest already-committed
vintage (spec 6.5.2a section 2.2's revision check) no longer aborts as of
spec 7.4.1 section 3.5 -- it is only reported (which months, which series,
old and new value), since every fetch now becomes its own permanent,
never-overwritten vintage rather than replacing the one file every
consumer reads live.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
import requests

from energy_price_forecast.data.capacity import ProductionType
from energy_price_forecast.data.capacity_vintages import (
    CAPACITY_ANCHORS_DIR,
    SHA256SUMS_FILENAME,
    available_vintages,
)

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
    time_labels = cast("list[str]", response["time"])
    production_types = cast("list[dict[str, object]]", response["production_types"])
    by_name: dict[str, list[float | None]] = {
        str(p["name"]): cast("list[float | None]", p["data"]) for p in production_types
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


def validate_against_committed(anchors: pd.DataFrame, *, as_of: pd.Timestamp) -> None:
    """Report (never raise, spec 7.4.1 section 3.5) any anchor older than
    ``_REVISION_CHECK_MONTHS`` that differs from the latest already-committed
    vintage in ``data/capacity_anchors/`` (spec 6.5.2a section 2.2's revision
    check) -- and log the new-vs-overlapping split either way, so the "does
    the new file only differ in newly added months" question (spec section
    5.4) is answered on every run, not assumed.

    Before spec 7.4.1, a revision this old aborted the refresh outright
    (``--force`` was the explicit override). Since every fetch is now its
    own permanent vintage file rather than a replacement of the one file
    every consumer reads live, there is nothing left to protect by
    aborting -- the old vintage stays exactly as it was, unmodified."""
    try:
        _, latest_path = available_vintages()[-1]
    except ValueError:
        logger.info("no committed vintage yet -- skipping the revision check")
        return

    old = pd.read_csv(latest_path, comment="#")
    old["anchor_date"] = pd.to_datetime(old["anchor_date"], utc=True)

    merged = old.merge(
        anchors, on=["anchor_date", "production_type"], suffixes=("_old", "_new"), how="inner"
    )
    revision_floor = as_of.normalize() - pd.DateOffset(months=_REVISION_CHECK_MONTHS)
    stable = merged.loc[merged["anchor_date"] < revision_floor]
    changed = stable.loc[~np.isclose(stable["capacity_mw_old"], stable["capacity_mw_new"])]
    if not changed.empty:
        detail = changed[["anchor_date", "production_type", "capacity_mw_old", "capacity_mw_new"]]
        logger.warning(
            "Energy-Charts revised anchor(s) older than the %d-month floor relative to "
            "vintage %s -- a revision event (spec 6.5.2a section 2.2), not routine "
            "refresh, reported only (spec 7.4.1 section 3.5):\n%s",
            _REVISION_CHECK_MONTHS,
            latest_path.name,
            detail,
        )

    new_pairs = set(zip(anchors["anchor_date"], anchors["production_type"], strict=True))
    old_pairs = set(zip(old["anchor_date"], old["production_type"], strict=True))
    added = new_pairs - old_pairs
    logger.info(
        "refresh check vs. vintage %s: %d (anchor_date, production_type) pairs are new, "
        "%d overlap and match within tolerance, %d revision(s) beyond the %d-month floor",
        latest_path.name,
        len(added),
        len(merged) - len(changed),
        len(changed),
        _REVISION_CHECK_MONTHS,
    )


def build_csv_text(
    anchors: pd.DataFrame, response: dict[str, object], *, fetched_at: pd.Timestamp
) -> str:
    last_update_ts = cast("int", response["last_update"])
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


def new_vintage_path(as_of: pd.Timestamp, *, anchors_dir: Path = CAPACITY_ANCHORS_DIR) -> Path:
    """The vintage path this run would write to -- raises if it already
    exists (spec 7.4.1 section 3.5: a vintage file is never overwritten)."""
    path = anchors_dir / f"public_registry_{as_of.date().isoformat()}.csv"
    if path.exists():
        raise ValueError(
            f"{path} already exists -- a vintage file is never overwritten "
            "(spec 7.4.1 section 3.1); re-run on a later Abrufdatum instead"
        )
    return path


def _append_checksum(
    path: Path, sums_path: Path = CAPACITY_ANCHORS_DIR / SHA256SUMS_FILENAME
) -> None:
    """Append ``path``'s sha256 to ``sums_path`` (``sha256sum``-compatible
    format), sorted by filename for a deterministic, reviewable diff."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries: dict[str, str] = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing_digest, _, name = line.partition(" *")
            entries[name] = existing_digest
    entries[path.name] = digest
    sums_path.write_text(
        "".join(f"{entries[name]} *{name}\n" for name in sorted(entries)), encoding="utf-8"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    as_of = pd.Timestamp.now(tz="UTC")
    new_path = new_vintage_path(as_of)

    response = fetch_response()
    anchors = parse_anchors(response, as_of=as_of)
    validate_monotonic_cadence(anchors)
    validate_against_committed(anchors, as_of=as_of)

    csv_text = build_csv_text(anchors, response, fetched_at=as_of)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(csv_text, encoding="utf-8", newline="\n")
    _append_checksum(new_path)
    logger.info("wrote %d rows to %s", len(anchors), new_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
