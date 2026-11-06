"""Capacity-anchor vintage selection (spec 7.4.1, section 3.2).

Every live run, OOS-store computation, and backfill needs to pick exactly
one historical vintage of ``data/capacity_anchors/public_registry_*.csv``
for a given delivery day -- the one the live run would actually have seen
on the day it ran. This module is the single place that rule is defined;
``ops/store.py``'s own "one rule, one place" convention for the equivalent
live/OOS split.

Rule (spec 7.4.1 section 3.2): a vintage dated ``V`` is valid starting the
day *after* ``V``. For a delivery (target) day ``D``, the run day
("Lauftag") is ``D - 1`` -- both are Europe/Berlin local calendar dates
already, by the convention used throughout this codebase (e.g.
``ops/windows.py::next_delivery_day``), so no timezone conversion is
needed here, only calendar-date arithmetic. Days before the first
available vintage take the first vintage in full (owner decision, spec
section 3.2) -- there is no "no vintage available" case.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Final

from energy_price_forecast.config import PROJECT_ROOT

CAPACITY_ANCHORS_DIR: Final = PROJECT_ROOT / "data" / "capacity_anchors"
SHA256SUMS_FILENAME: Final = "SHA256SUMS"

_VINTAGE_FILENAME_RE: Final = re.compile(r"^public_registry_(\d{4}-\d{2}-\d{2})\.csv$")


def available_vintages(anchors_dir: Path = CAPACITY_ANCHORS_DIR) -> list[tuple[dt.date, Path]]:
    """All (vintage_date, path) pairs found in ``anchors_dir``, sorted ascending.

    Raises ``ValueError`` if the directory has no recognizable vintage
    files -- there is always at least one vintage once the store is
    seeded, so an empty result means a real setup problem, not a case to
    handle gracefully.
    """
    vintages: list[tuple[dt.date, Path]] = []
    for path in sorted(anchors_dir.glob("public_registry_*.csv")):
        match = _VINTAGE_FILENAME_RE.match(path.name)
        if match is None:
            continue
        vintages.append((dt.date.fromisoformat(match.group(1)), path))
    if not vintages:
        raise ValueError(f"no capacity-anchor vintage files found in {anchors_dir}")
    return sorted(vintages, key=lambda item: item[0])


def vintage_for_lauftag(
    lauftag: dt.date, *, anchors_dir: Path = CAPACITY_ANCHORS_DIR
) -> tuple[dt.date, Path]:
    """The capacity vintage valid on ``lauftag`` (spec 7.4.1 section 3.2).

    Returns the latest vintage ``V`` with ``V < lauftag``, or the earliest
    available vintage if ``lauftag`` is before the first one even applies.
    """
    vintages = available_vintages(anchors_dir)
    applicable = [vintage for vintage in vintages if vintage[0] < lauftag]
    if applicable:
        return applicable[-1]
    return vintages[0]


def vintage_for_target_day(
    target_day: dt.date, *, anchors_dir: Path = CAPACITY_ANCHORS_DIR
) -> tuple[dt.date, Path]:
    """The capacity vintage a computation for delivery day ``target_day``
    uses (spec 7.4.1 section 3.2): the vintage valid on the Lauftag
    ``target_day - 1`` day.
    """
    lauftag = target_day - dt.timedelta(days=1)
    return vintage_for_lauftag(lauftag, anchors_dir=anchors_dir)


def verify_checksums(anchors_dir: Path = CAPACITY_ANCHORS_DIR) -> None:
    """Raises if any vintage file in ``anchors_dir`` has changed since it
    was recorded in ``SHA256SUMS``, or has no entry at all (spec 7.4.1
    section 3.1: "eine Datei, die einmal da ist, wird nie mehr verändert").

    Does not check for a ``SHA256SUMS`` entry with no corresponding file --
    that is a stale-entry bookkeeping issue, not evidence that a committed
    vintage was tampered with, which is what this guards against.
    """
    sums_path = anchors_dir / SHA256SUMS_FILENAME
    recorded: dict[str, str] = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition(" *")
            recorded[name] = digest

    for _vintage_date, path in available_vintages(anchors_dir):
        if path.name not in recorded:
            raise ValueError(f"{path.name} has no entry in {sums_path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != recorded[path.name]:
            raise ValueError(
                f"{path.name} has changed since it was recorded in {sums_path} "
                f"(recorded {recorded[path.name]}, actual {actual}) -- a committed "
                "vintage file must never be modified (spec 7.4.1 section 3.1)"
            )
