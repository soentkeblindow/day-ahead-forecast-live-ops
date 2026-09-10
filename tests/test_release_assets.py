"""Unit tests for ops/release_assets.py -- the GitHub Releases transport
layer for the persistent data store (spec 6.7.1, section 5.2/7).

All requests.{get,post,delete} calls are monkeypatched against a small fake
response, the same house style as tests/test_weather_client.py -- nothing in
this file touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from energy_price_forecast.ops import release_assets as ra


class _FakeResponse:
    """Just enough of requests.Response for release_assets.py's code paths."""

    def __init__(self, status_code: int, body: Any = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else ""

    def json(self) -> Any:
        return self._body


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someowner/somerepo")


def _asset_body(
    asset_id: int, name: str, *, state: str = "uploaded", size: int = 1024
) -> dict[str, Any]:
    return {"id": asset_id, "name": name, "state": state, "size": size}


# ---------------------------------------------------------------------------
# ensure_release
# ---------------------------------------------------------------------------


def test_ensure_release_existing_is_not_recreated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append("get")
        return _FakeResponse(200, {"id": 42})

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append("post")
        raise AssertionError("must not create a release that already exists")

    monkeypatch.setattr(ra.requests, "get", fake_get)
    monkeypatch.setattr(ra.requests, "post", fake_post)

    ref = ra.ensure_release("data-store")

    assert ref == ra.ReleaseRef(release_id=42, tag="data-store")
    assert calls == ["get"]


def test_ensure_release_missing_is_created(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(404, {"message": "Not Found"})

    created: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        created.update(kwargs["json"])
        return _FakeResponse(201, {"id": 7})

    monkeypatch.setattr(ra.requests, "get", fake_get)
    monkeypatch.setattr(ra.requests, "post", fake_post)

    ref = ra.ensure_release("data-store")

    assert ref == ra.ReleaseRef(release_id=7, tag="data-store")
    assert created["tag_name"] == "data-store"
    assert created["draft"] is False


# ---------------------------------------------------------------------------
# list_assets
# ---------------------------------------------------------------------------


def test_list_assets_sorted_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    body = [
        _asset_body(1, "store-20260901T100000Z.tar"),
        _asset_body(2, "store-20260903T100000Z.tar"),
        _asset_body(3, "store-20260902T100000Z.tar"),
    ]

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        page = kwargs["params"]["page"]
        return _FakeResponse(200, body if page == 1 else [])

    monkeypatch.setattr(ra.requests, "get", fake_get)

    assets = ra.list_assets(ra.ReleaseRef(release_id=99, tag="data-store"))

    assert [a.name for a in assets] == [
        "store-20260903T100000Z.tar",
        "store-20260902T100000Z.tar",
        "store-20260901T100000Z.tar",
    ]


def test_list_assets_skips_incomplete_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = [
        _asset_body(1, "store-20260901T100000Z.tar", state="uploaded", size=1024),
        _asset_body(2, "store-20260902T100000Z.tar", state="uploaded", size=0),  # interrupted
        _asset_body(3, "store-20260903T100000Z.tar", state="starter", size=1024),  # not finished
    ]

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        page = kwargs["params"]["page"]
        return _FakeResponse(200, body if page == 1 else [])

    monkeypatch.setattr(ra.requests, "get", fake_get)

    assets = ra.list_assets(ra.ReleaseRef(release_id=99, tag="data-store"))

    assert [a.name for a in assets] == ["store-20260901T100000Z.tar"]


# ---------------------------------------------------------------------------
# prune_assets
# ---------------------------------------------------------------------------


def test_prune_assets_keeps_exactly_n_newest_deletes_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [f"store-2026090{d}T100000Z.tar" for d in range(1, 6)]  # 5 assets, 01..05
    body = [_asset_body(i, name) for i, name in enumerate(names, start=1)]

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        page = kwargs["params"]["page"]
        return _FakeResponse(200, body if page == 1 else [])

    deleted_ids: list[int] = []

    def fake_delete(url: str, **kwargs: Any) -> _FakeResponse:
        asset_id = int(url.rsplit("/", 1)[-1])
        deleted_ids.append(asset_id)
        return _FakeResponse(204)

    monkeypatch.setattr(ra.requests, "get", fake_get)
    monkeypatch.setattr(ra.requests, "delete", fake_delete)

    deleted = ra.prune_assets(ra.ReleaseRef(release_id=99, tag="data-store"), keep=2)

    # newest-first: 05, 04, 03, 02, 01 -- keep=2 keeps 05 and 04, deletes 03, 02, 01
    assert [a.name for a in deleted] == [
        "store-20260903T100000Z.tar",
        "store-20260902T100000Z.tar",
        "store-20260901T100000Z.tar",
    ]
    assert set(deleted_ids) == {3, 2, 1}


# ---------------------------------------------------------------------------
# download/upload/delete round-trip mechanics
# ---------------------------------------------------------------------------


def test_download_asset_writes_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        assert kwargs["headers"]["Accept"] == "application/octet-stream"
        return _FakeResponse(200, content=b"tar-bytes-here")

    monkeypatch.setattr(ra.requests, "get", fake_get)

    target = tmp_path / "nested" / "store.tar"
    ra.download_asset(ra.AssetRef(1, "store.tar", 99, True, 14), target)

    assert target.read_bytes() == b"tar-bytes-here"


def test_upload_asset_sends_name_as_query_param(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["params"] = kwargs["params"]
        captured["content_type"] = kwargs["headers"]["Content-Type"]
        return _FakeResponse(201, {"id": 5, "name": "store-x.tar", "state": "uploaded", "size": 3})

    monkeypatch.setattr(ra.requests, "post", fake_post)

    src = tmp_path / "store-x.tar"
    src.write_bytes(b"abc")
    asset = ra.upload_asset(ra.ReleaseRef(release_id=99, tag="data-store"), src)

    assert "uploads.github.com" in captured["url"]
    assert captured["params"]["name"] == "store-x.tar"
    assert captured["content_type"] == "application/x-tar"
    assert asset.uploaded is True
    assert asset.name == "store-x.tar"


def test_upload_asset_incomplete_response_is_flagged_not_uploaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(201, {"id": 5, "name": "store-x.tar", "state": "starter", "size": 0})

    monkeypatch.setattr(ra.requests, "post", fake_post)

    src = tmp_path / "store-x.tar"
    src.write_bytes(b"abc")
    asset = ra.upload_asset(ra.ReleaseRef(release_id=99, tag="data-store"), src)

    assert asset.uploaded is False


def test_delete_asset_raises_on_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_delete(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(404, {"message": "Not Found"})

    monkeypatch.setattr(ra.requests, "delete", fake_delete)

    with pytest.raises(ra.ReleaseAssetsError):
        ra.delete_asset(ra.AssetRef(1, "store.tar", 99, True, 10))


# ---------------------------------------------------------------------------
# repo/token resolution
# ---------------------------------------------------------------------------


def test_missing_token_raises_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(ra.ReleaseAssetsError, match="GITHUB_TOKEN"):
        ra.ensure_release("data-store")


def test_repo_falls_back_to_known_coordinates_outside_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert ra._repo() == ("soentkeblindow", "sbl-energy-forecast")
