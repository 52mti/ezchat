from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import update_release


class _FakeResponse:
    def __init__(self, url: str, text: str, *, status_code: int = 200) -> None:
        self.url = url
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise update_release.requests.HTTPError(f"{self.status_code} error")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def get(self, url: str, timeout=None):  # noqa: ANN001
        assert url == self.response.url
        return self.response


def test_fetch_release_manifest_uses_defaults_for_optional_urls():
    payload = {
        "app_version": "2026.06.07.1",
        "runtime_version": "2026.06.02-full-package",
        "requires_full_package": False,
    }
    session = _FakeSession(
        _FakeResponse(
            update_release.DEFAULT_RELEASE_MANIFEST_URL,
            json.dumps(payload, ensure_ascii=False),
        )
    )

    manifest = update_release.fetch_release_manifest(session=session)

    assert manifest.app_version == "2026.06.07.1"
    assert manifest.runtime_version == "2026.06.02-full-package"
    assert manifest.update_share_url == "https://liaotianbao.lanzouq.com/b006712hbg"
    assert manifest.update_file_name == ""
    assert manifest.update_url == ""
    assert manifest.full_package_url == "https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d"


def test_fetch_release_manifest_raises_on_http_error():
    session = _FakeSession(
        _FakeResponse(update_release.DEFAULT_RELEASE_MANIFEST_URL, "denied", status_code=403)
    )

    with pytest.raises(update_release.ReleaseManifestError):
        update_release.fetch_release_manifest(session=session)


def test_fetch_release_manifest_reads_cos_update_url():
    payload = {
        "app_version": "2026.06.07.1",
        "runtime_version": "2026.06.02-full-package",
        "update_url": "https://example.cos.ap-shanghai.myqcloud.com/packages/update.zip",
    }
    session = _FakeSession(
        _FakeResponse(update_release.DEFAULT_RELEASE_MANIFEST_URL, json.dumps(payload))
    )

    manifest = update_release.fetch_release_manifest(session=session)

    assert manifest.update_url == payload["update_url"]


def test_load_local_release_state_reads_runtime_manifest_fields(tmp_path):
    (tmp_path / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "version": "2026.06.02-full-package",
                "appVersion": "2026.06.07.0",
                "runtimeVersion": "2026.06.02-full-package",
                "modelVersion": "2026.06.02-lite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = update_release.load_local_release_state(tmp_path)

    assert state.app_version == "2026.06.07.0"
    assert state.runtime_version == "2026.06.02-full-package"
    assert state.model_version == "2026.06.02-lite"


def test_load_local_release_state_falls_back_to_runtime_manifest_version(tmp_path):
    (tmp_path / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "version": "2026.06.02-full-package",
                "models": {"defaultTier": "lite"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = update_release.load_local_release_state(tmp_path)

    assert state.app_version == "2026.06.02-full-package"
    assert state.runtime_version == "2026.06.02-full-package"
    assert state.model_version == "2026.06.02-full-package-lite"


def test_load_local_release_state_marks_partial_update_as_incomplete(tmp_path):
    (tmp_path / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "version": "2026.06.02-full-package",
                "appVersion": "20260831-05",
                "runtimeVersion": "2026.06.02-full-package",
                "modelVersion": "2026.06.02-full-package-lite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "app_modules.zip").write_bytes(b"stale")
    (tmp_path / "package_manifest.json").write_text(
        json.dumps({"files": [{"path": "app_modules.zip", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )

    state = update_release.load_local_release_state(tmp_path)

    assert state.app_version == "20260831-05"
    assert state.package_consistent is False


def test_evaluate_release_manifest_returns_incremental_for_newer_app_version():
    remote = update_release.ReleaseManifest(
        app_version="2026.06.07.1",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
        requires_full_package=False,
        update_share_url="https://liaotianbao.lanzouq.com/b006712hbg",
        update_file_name="jixiebijiaoben_update_20260607_0904.zip",
        full_package_url="https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d",
        release_notes="fixes",
    )
    local = update_release.LocalReleaseState(
        package_root=Path("."),
        app_version="2026.06.07.0",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
    )

    decision = update_release.evaluate_release_manifest(remote, local)

    assert decision.action == "incremental"
    assert decision.target_app_version == "2026.06.07.1"
    assert decision.update_file_name == "jixiebijiaoben_update_20260607_0904.zip"


def test_evaluate_release_manifest_returns_full_package_on_runtime_change():
    remote = update_release.ReleaseManifest(
        app_version="2026.06.07.1",
        runtime_version="2026.06.08-full-package",
        model_version="2026.06.02-lite",
        requires_full_package=False,
        update_share_url="https://liaotianbao.lanzouq.com/b006712hbg",
        update_file_name="jixiebijiaoben_update_20260608_0904.zip",
        full_package_url="https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d",
        release_notes="runtime change",
    )
    local = update_release.LocalReleaseState(
        package_root=__import__("pathlib").Path("."),
        app_version="2026.06.07.0",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
    )

    decision = update_release.evaluate_release_manifest(remote, local)

    assert decision.action == "full_package"
    assert "Runtime version changed" in decision.reason


def test_evaluate_release_manifest_returns_none_when_app_version_matches_even_if_runtime_differs():
    remote = update_release.ReleaseManifest(
        app_version="2026.06.07.1",
        runtime_version="2026.06.08-full-package",
        model_version="2026.06.03-lite",
        requires_full_package=True,
        update_share_url="https://liaotianbao.lanzouq.com/b006712hbg",
        update_file_name="jixiebijiaoben_update_20260608_0904.zip",
        full_package_url="https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d",
        release_notes="runtime change",
    )
    local = update_release.LocalReleaseState(
        package_root=Path("."),
        app_version="2026.06.07.1",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
    )

    decision = update_release.evaluate_release_manifest(remote, local)

    assert decision.action == "none"
    assert "matches the remote" in decision.reason
