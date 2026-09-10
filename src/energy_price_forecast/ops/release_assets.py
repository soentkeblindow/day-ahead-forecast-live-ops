"""GitHub Releases as the transport layer for the persistent data store
(spec 6.7.1, Entscheidung 17 / section 5.2).

Knows nothing about what an asset contains -- packing, manifests and store
semantics live in ops/store.py. Every function is a thin wrapper around one
GitHub REST API call, so a bad response from GitHub is always attributable
to exactly one operation.

Auth: reads ``GITHUB_TOKEN`` from the environment (the built-in Actions
token inside the maintenance workflow, permissions: contents: write; a
personal access token with the same scope for local runs, e.g.
scripts/rebuild_store.py -- spec section 3.4, no new secret is introduced).
Repo coordinates come from ``GITHUB_REPOSITORY`` (set automatically inside
Actions) or fall back to this repo's own known owner/name for local runs --
the same two constants also named in ops/trigger/src/index.ts, since that
worker lives outside the Python package and can't import this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import requests

_API_BASE: Final[str] = "https://api.github.com"
_UPLOAD_BASE: Final[str] = "https://uploads.github.com"
_API_VERSION: Final[str] = "2022-11-28"
_TIMEOUT_S: Final[float] = 30.0
_UPLOAD_TIMEOUT_S: Final[float] = 120.0  # a store-sized tar can take longer than a plain API call
_USER_AGENT: Final[str] = "sbl-energy-forecast-store"

_DEFAULT_OWNER: Final[str] = "soentkeblindow"
_DEFAULT_REPO: Final[str] = "sbl-energy-forecast"


class ReleaseAssetsError(RuntimeError):
    """A GitHub Releases API call did not succeed as expected."""


@dataclass(frozen=True)
class ReleaseRef:
    release_id: int
    tag: str


@dataclass(frozen=True)
class AssetRef:
    asset_id: int
    name: str
    release_id: int
    uploaded: bool
    size_bytes: int


def _repo() -> tuple[str, str]:
    combined = os.getenv("GITHUB_REPOSITORY")
    if combined and "/" in combined:
        owner, _, repo = combined.partition("/")
        if owner and repo:
            return owner, repo
    return _DEFAULT_OWNER, _DEFAULT_REPO


def _token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ReleaseAssetsError(
            "GITHUB_TOKEN is not set. Inside GitHub Actions this is the built-in "
            "token (permissions: contents: write); for local runs export a "
            "personal access token with the same scope."
        )
    return token


def _headers(*, accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": accept,
        "X-GitHub-Api-Version": _API_VERSION,
        "User-Agent": _USER_AGENT,
    }


def ensure_release(tag: str) -> ReleaseRef:
    """Return the release for ``tag``, creating it if it does not exist.

    Idempotent on purpose: the very first maintenance run must not require a
    manual setup step in the GitHub UI, or the rebuild path in
    scripts/rebuild_store.py cannot run unattended either.
    """
    owner, repo = _repo()
    get_resp = requests.get(
        f"{_API_BASE}/repos/{owner}/{repo}/releases/tags/{tag}",
        headers=_headers(),
        timeout=_TIMEOUT_S,
    )
    if get_resp.status_code == 200:
        body = get_resp.json()
        return ReleaseRef(release_id=body["id"], tag=tag)
    if get_resp.status_code != 404:
        raise ReleaseAssetsError(
            f"GET release {tag!r} failed: HTTP {get_resp.status_code} {get_resp.text[:200]}"
        )

    create_resp = requests.post(
        f"{_API_BASE}/repos/{owner}/{repo}/releases",
        headers=_headers(),
        json={"tag_name": tag, "name": tag, "draft": False, "prerelease": False},
        timeout=_TIMEOUT_S,
    )
    if create_resp.status_code not in (200, 201):
        raise ReleaseAssetsError(
            f"Creating release {tag!r} failed: HTTP {create_resp.status_code} "
            f"{create_resp.text[:200]}"
        )
    body = create_resp.json()
    return ReleaseRef(release_id=body["id"], tag=tag)


def list_assets(release: ReleaseRef) -> tuple[AssetRef, ...]:
    """All assets of the release, newest name first.

    Assets whose upload has not completed are skipped: a reader must never
    see a half-written store (spec section 2.5). GitHub reports a completed
    upload as ``state == "uploaded"``; a zero-byte asset (the observable
    symptom of an interrupted upload) is treated as incomplete regardless of
    the reported state, rather than trusted.

    Names are the ``store-YYYYMMDDTHHMMSSZ.tar`` timestamp convention
    (ops/store.py), so a plain descending string sort is also a descending
    chronological sort -- no separate timestamp parsing needed here.
    """
    owner, repo = _repo()
    assets: list[AssetRef] = []
    page = 1
    while True:
        resp = requests.get(
            f"{_API_BASE}/repos/{owner}/{repo}/releases/{release.release_id}/assets",
            headers=_headers(),
            params={"per_page": 100, "page": page},
            timeout=_TIMEOUT_S,
        )
        if resp.status_code != 200:
            raise ReleaseAssetsError(
                f"Listing assets for release {release.tag!r} failed: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
        body: list[dict[str, Any]] = resp.json()
        if not body:
            break
        for item in body:
            uploaded = item.get("state") == "uploaded" and item.get("size", 0) > 0
            assets.append(
                AssetRef(
                    asset_id=item["id"],
                    name=item["name"],
                    release_id=release.release_id,
                    uploaded=uploaded,
                    size_bytes=item.get("size", 0),
                )
            )
        if len(body) < 100:
            break
        page += 1

    complete = [a for a in assets if a.uploaded]
    return tuple(sorted(complete, key=lambda a: a.name, reverse=True))


def download_asset(asset: AssetRef, target: Path) -> None:
    """Download asset's raw bytes to target.

    Uses the API asset endpoint with ``Accept: application/octet-stream``,
    not ``browser_download_url``: the store repo is private (Entscheidung
    22), and this endpoint authenticates with the same GITHUB_TOKEN used
    everywhere else here, while browser_download_url needs a separate
    signed-URL redirect.
    """
    owner, repo = _repo()
    resp = requests.get(
        f"{_API_BASE}/repos/{owner}/{repo}/releases/assets/{asset.asset_id}",
        headers=_headers(accept="application/octet-stream"),
        timeout=_UPLOAD_TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise ReleaseAssetsError(
            f"Downloading asset {asset.name!r} failed: HTTP {resp.status_code} {resp.text[:200]}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)


def upload_asset(release: ReleaseRef, path: Path) -> AssetRef:
    """Upload path as a new asset of release.

    Never overwrites an existing asset (spec section 2.5) -- the caller is
    responsible for a timestamped name that cannot collide (ops/store.py's
    ``store-YYYYMMDDTHHMMSSZ.tar`` convention).
    """
    owner, repo = _repo()
    with path.open("rb") as f:
        resp = requests.post(
            f"{_UPLOAD_BASE}/repos/{owner}/{repo}/releases/{release.release_id}/assets",
            headers={**_headers(), "Content-Type": "application/x-tar"},
            params={"name": path.name},
            data=f,
            timeout=_UPLOAD_TIMEOUT_S,
        )
    if resp.status_code not in (200, 201):
        raise ReleaseAssetsError(
            f"Uploading asset {path.name!r} failed: HTTP {resp.status_code} {resp.text[:200]}"
        )
    body = resp.json()
    uploaded = body.get("state") == "uploaded" and body.get("size", 0) > 0
    return AssetRef(
        asset_id=body["id"],
        name=body["name"],
        release_id=release.release_id,
        uploaded=uploaded,
        size_bytes=body.get("size", 0),
    )


def delete_asset(asset: AssetRef) -> None:
    owner, repo = _repo()
    resp = requests.delete(
        f"{_API_BASE}/repos/{owner}/{repo}/releases/assets/{asset.asset_id}",
        headers=_headers(),
        timeout=_TIMEOUT_S,
    )
    if resp.status_code != 204:
        raise ReleaseAssetsError(
            f"Deleting asset {asset.name!r} failed: HTTP {resp.status_code} {resp.text[:200]}"
        )


def prune_assets(release: ReleaseRef, keep: int) -> tuple[AssetRef, ...]:
    """Delete all but the newest ``keep`` complete assets of release.

    Composes list_assets (already newest-first, already filtered to
    complete uploads) with delete_asset -- pruning is a pure
    release-assets-level operation (it never inspects what's inside an
    asset), so it lives here rather than in ops/store.py.

    Returns the AssetRef objects that were deleted, for logging.
    """
    assets = list_assets(release)
    to_delete = assets[keep:]
    for asset in to_delete:
        delete_asset(asset)
    return to_delete
