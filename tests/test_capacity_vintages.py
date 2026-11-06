"""Tests for ``data/capacity_vintages.py`` (spec 7.4.1 section 3.2, 7)."""

import datetime as dt
import hashlib
from pathlib import Path

import pytest

from energy_price_forecast.data.capacity_vintages import (
    CAPACITY_ANCHORS_DIR,
    available_vintages,
    verify_checksums,
    vintage_for_lauftag,
    vintage_for_target_day,
)


def _make_vintage(anchors_dir: Path, date_str: str) -> Path:
    path = anchors_dir / f"public_registry_{date_str}.csv"
    path.write_text(f"# vintage {date_str}\n", encoding="utf-8")
    return path


def test_available_vintages_sorted_ascending(tmp_path: Path) -> None:
    _make_vintage(tmp_path, "2026-10-06")
    _make_vintage(tmp_path, "2026-09-01")
    _make_vintage(tmp_path, "2026-09-02")

    vintages = available_vintages(tmp_path)

    assert [v[0] for v in vintages] == [
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
        dt.date(2026, 10, 6),
    ]


def test_available_vintages_ignores_non_matching_files(tmp_path: Path) -> None:
    _make_vintage(tmp_path, "2026-09-01")
    (tmp_path / "SHA256SUMS").write_text("deadbeef *public_registry_2026-09-01.csv\n")
    (tmp_path / "readme.md").write_text("not a vintage file\n")

    vintages = available_vintages(tmp_path)

    assert len(vintages) == 1


def test_available_vintages_raises_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no capacity-anchor vintage files found"):
        available_vintages(tmp_path)


def test_vintage_for_lauftag_boundary_day_is_not_yet_valid(tmp_path: Path) -> None:
    """A vintage dated V is valid starting the day *after* V (spec 3.2) --
    on V itself, the previous vintage (or the first one) still applies."""
    _make_vintage(tmp_path, "2026-09-01")
    _make_vintage(tmp_path, "2026-10-06")

    # On the vintage's own date, it is not yet valid.
    date, _ = vintage_for_lauftag(dt.date(2026, 10, 6), anchors_dir=tmp_path)
    assert date == dt.date(2026, 9, 1)

    # The day after, it is.
    date, _ = vintage_for_lauftag(dt.date(2026, 10, 7), anchors_dir=tmp_path)
    assert date == dt.date(2026, 10, 6)


def test_vintage_for_lauftag_before_first_vintage_takes_the_first(tmp_path: Path) -> None:
    _make_vintage(tmp_path, "2026-09-01")
    _make_vintage(tmp_path, "2026-10-06")

    date, _ = vintage_for_lauftag(dt.date(2020, 1, 1), anchors_dir=tmp_path)

    assert date == dt.date(2026, 9, 1)


def test_vintage_for_lauftag_with_multiple_vintages_picks_the_latest_applicable(
    tmp_path: Path,
) -> None:
    _make_vintage(tmp_path, "2026-09-01")
    _make_vintage(tmp_path, "2026-09-02")
    _make_vintage(tmp_path, "2026-10-06")

    date, _ = vintage_for_lauftag(dt.date(2026, 9, 15), anchors_dir=tmp_path)

    assert date == dt.date(2026, 9, 2)


def test_vintage_for_target_day_uses_lauftag_d_minus_1(tmp_path: Path) -> None:
    _make_vintage(tmp_path, "2026-09-01")
    _make_vintage(tmp_path, "2026-10-06")

    # Target day 2026-10-07 -> Lauftag 2026-10-06 -> vintage not yet valid on
    # its own date -> falls back to the 2026-09-01 vintage.
    date, _ = vintage_for_target_day(dt.date(2026, 10, 7), anchors_dir=tmp_path)
    assert date == dt.date(2026, 9, 1)

    # Target day 2026-10-08 -> Lauftag 2026-10-07 -> 2026-10-06 is valid.
    date, _ = vintage_for_target_day(dt.date(2026, 10, 8), anchors_dir=tmp_path)
    assert date == dt.date(2026, 10, 6)


def test_vintage_for_target_day_across_a_dst_transition(tmp_path: Path) -> None:
    """DST transitions change local wall-clock offsets, not calendar-date
    labels -- Lauftag = target_day - 1 calendar day is correct either way,
    no timezone-aware arithmetic needed (spec 7.4.1 section 3.2)."""
    _make_vintage(tmp_path, "2026-09-01")

    # 2026-10-25 is the real Europe/Berlin DST fallback day (25h local day).
    date, path = vintage_for_target_day(dt.date(2026, 10, 26), anchors_dir=tmp_path)

    assert date == dt.date(2026, 9, 1)
    assert path == tmp_path / "public_registry_2026-09-01.csv"


def _write_sha256sums(anchors_dir: Path, entries: dict[str, str]) -> None:
    (anchors_dir / "SHA256SUMS").write_text(
        "".join(f"{digest} *{name}\n" for name, digest in entries.items()), encoding="utf-8"
    )


def test_verify_checksums_passes_when_everything_matches(tmp_path: Path) -> None:
    path = _make_vintage(tmp_path, "2026-09-01")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sha256sums(tmp_path, {path.name: digest})

    verify_checksums(tmp_path)  # must not raise


def test_verify_checksums_raises_on_a_changed_file(tmp_path: Path) -> None:
    path = _make_vintage(tmp_path, "2026-09-01")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sha256sums(tmp_path, {path.name: digest})

    path.write_text("# tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has changed"):
        verify_checksums(tmp_path)


def test_verify_checksums_raises_on_a_file_without_an_entry(tmp_path: Path) -> None:
    _make_vintage(tmp_path, "2026-09-01")
    _write_sha256sums(tmp_path, {})  # no entry for the file that exists

    with pytest.raises(ValueError, match="no entry"):
        verify_checksums(tmp_path)


def test_verify_checksums_against_the_real_committed_vintages() -> None:
    """Real guard, not synthetic: the actual committed
    ``data/capacity_anchors/`` must still match its own ``SHA256SUMS`` --
    spec 7.4.1 section 3.1's "never modified again" promise, checked
    against the real files, not just the logic in isolation."""
    verify_checksums(CAPACITY_ANCHORS_DIR)
