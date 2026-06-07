from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .update_sources import (
    DEFAULT_FULL_PACKAGE_PAGE_URL,
    DEFAULT_UPDATE_SHARE_URL,
)


DEFAULT_RELEASE_MANIFEST_URL = "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/release.json"
DEFAULT_RELEASE_TIMEOUT_SECONDS = 15


class ReleaseManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseManifest:
    app_version: str
    runtime_version: str
    model_version: str
    requires_full_package: bool
    update_share_url: str
    update_file_name: str
    full_package_url: str
    release_notes: str


@dataclass(frozen=True)
class LocalReleaseState:
    package_root: Path
    app_version: str
    runtime_version: str
    model_version: str


@dataclass(frozen=True)
class ReleaseDecision:
    action: str
    current_app_version: str
    target_app_version: str
    reason: str
    update_share_url: str
    update_file_name: str
    full_package_url: str
    release_notes: str


def fetch_release_manifest(
    *,
    url: str = DEFAULT_RELEASE_MANIFEST_URL,
    timeout_seconds: int = DEFAULT_RELEASE_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> ReleaseManifest:
    http = session or requests.Session()
    response = http.get(url, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ReleaseManifestError(f"Failed to fetch release manifest from {url}: {exc}") from exc

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError(f"Release manifest at {url} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ReleaseManifestError(f"Release manifest at {url} must be a JSON object.")

    app_version = str(payload.get("app_version", "") or "").strip()
    runtime_version = str(payload.get("runtime_version", "") or "").strip()
    if not app_version:
        raise ReleaseManifestError("Release manifest is missing app_version.")
    if not runtime_version:
        raise ReleaseManifestError("Release manifest is missing runtime_version.")

    update_share_url = str(payload.get("update_share_url", "") or "").strip() or DEFAULT_UPDATE_SHARE_URL
    full_package_url = str(payload.get("full_package_url", "") or "").strip() or DEFAULT_FULL_PACKAGE_PAGE_URL

    return ReleaseManifest(
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=str(payload.get("model_version", "") or "").strip(),
        requires_full_package=bool(payload.get("requires_full_package", False)),
        update_share_url=update_share_url,
        update_file_name=str(payload.get("update_file_name", "") or "").strip(),
        full_package_url=full_package_url,
        release_notes=str(payload.get("release_notes", "") or "").strip(),
    )


def load_local_release_state(package_root: str | Path | None = None) -> LocalReleaseState:
    root = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parent.parent
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.is_file():
        raise ReleaseManifestError(f"Local runtime manifest was not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"Local runtime manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseManifestError(f"Local runtime manifest must be a JSON object: {manifest_path}")

    base_version = str(payload.get("version", "") or "").strip()
    app_version = str(payload.get("appVersion", "") or "").strip() or base_version
    runtime_version = str(payload.get("runtimeVersion", "") or "").strip() or base_version
    model_version = str(payload.get("modelVersion", "") or "").strip()
    if not model_version:
        default_tier = str((payload.get("models", {}) or {}).get("defaultTier", "") or "").strip()
        if default_tier and runtime_version:
            model_version = f"{runtime_version}-{default_tier}"
    if not app_version or not runtime_version:
        raise ReleaseManifestError(f"Local runtime manifest is missing version fields: {manifest_path}")

    return LocalReleaseState(
        package_root=root,
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
    )


def evaluate_release_manifest(
    remote: ReleaseManifest,
    local: LocalReleaseState,
) -> ReleaseDecision:
    if remote.app_version == local.app_version:
        return ReleaseDecision(
            action="none",
            current_app_version=local.app_version,
            target_app_version=remote.app_version,
            reason="Local app_version already matches the remote release manifest.",
            update_share_url=remote.update_share_url,
            update_file_name=remote.update_file_name,
            full_package_url=remote.full_package_url,
            release_notes=remote.release_notes,
        )

    if remote.requires_full_package:
        return ReleaseDecision(
            action="full_package",
            current_app_version=local.app_version,
            target_app_version=remote.app_version,
            reason="Remote release manifest requires a full package upgrade.",
            update_share_url=remote.update_share_url,
            update_file_name=remote.update_file_name,
            full_package_url=remote.full_package_url,
            release_notes=remote.release_notes,
        )

    if remote.runtime_version and local.runtime_version and remote.runtime_version != local.runtime_version:
        return ReleaseDecision(
            action="full_package",
            current_app_version=local.app_version,
            target_app_version=remote.app_version,
            reason="Runtime version changed and must be upgraded with the full package.",
            update_share_url=remote.update_share_url,
            update_file_name=remote.update_file_name,
            full_package_url=remote.full_package_url,
            release_notes=remote.release_notes,
        )

    if (
        remote.model_version
        and local.model_version
        and remote.model_version != local.model_version
    ):
        return ReleaseDecision(
            action="full_package",
            current_app_version=local.app_version,
            target_app_version=remote.app_version,
            reason="Model version changed and must be upgraded with the full package.",
            update_share_url=remote.update_share_url,
            update_file_name=remote.update_file_name,
            full_package_url=remote.full_package_url,
            release_notes=remote.release_notes,
        )

    if remote.app_version != local.app_version:
        return ReleaseDecision(
            action="incremental",
            current_app_version=local.app_version,
            target_app_version=remote.app_version,
            reason="A different app_version is available and can be delivered through the update package.",
            update_share_url=remote.update_share_url,
            update_file_name=remote.update_file_name,
            full_package_url=remote.full_package_url,
            release_notes=remote.release_notes,
        )

    return ReleaseDecision(
        action="none",
        current_app_version=local.app_version,
        target_app_version=remote.app_version,
        reason="Local release state is already up to date.",
        update_share_url=remote.update_share_url,
        update_file_name=remote.update_file_name,
        full_package_url=remote.full_package_url,
        release_notes=remote.release_notes,
    )


def resolve_release_decision(
    *,
    package_root: str | Path | None = None,
    url: str = DEFAULT_RELEASE_MANIFEST_URL,
    timeout_seconds: int = DEFAULT_RELEASE_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> ReleaseDecision:
    remote = fetch_release_manifest(url=url, timeout_seconds=timeout_seconds, session=session)
    local = load_local_release_state(package_root)
    return evaluate_release_manifest(remote, local)

