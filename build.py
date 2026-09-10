from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
import py_compile
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urlparse

import requests


PACKAGE_NAME = "jixiebijiaoben"
LAUNCHER_EXE = "机械臂脚本.exe"
LAUNCHER_ALIAS_EXE = "jixiebijiaoben_launcher.exe"
START_CMD = "start_jixiebijiaoben.cmd"
DEFAULT_FULL_PACKAGE_NVIDIA_LANE = "cu118"
DEFAULT_RELEASE_MANIFEST_URL = "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/release.json"
DEFAULT_UPDATE_PACKAGE_URL_PREFIX = "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/packages"
DEFAULT_UPDATE_SHARE_URL = "https://liaotianbao.lanzouq.com/b006712hbg"
DEFAULT_FULL_PACKAGE_PAGE_URL = "https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d"
RELEASE_FILE_NAME = "release.json"
DEFAULT_COS_UPLOAD_TIMEOUT_SECONDS = 30
DEFAULT_COS_SIGNATURE_TTL_SECONDS = 600
DEFAULT_LANZOU_TIMEOUT_SECONDS = 30
COS_SECRET_ID_ENV = "TENCENT_COS_SECRET_ID"
COS_SECRET_KEY_ENV = "TENCENT_COS_SECRET_KEY"
COS_SESSION_TOKEN_ENV = "TENCENT_COS_SESSION_TOKEN"
LANZOU_USERNAME_ENV = "LANZOU_USERNAME"
LANZOU_PASSWORD_ENV = "LANZOU_PASSWORD"
LANZOU_UPDATE_FOLDER_ID_ENV = "LANZOU_UPDATE_FOLDER_ID"
LANZOU_UPDATE_FOLDER_NAME_ENV = "LANZOU_UPDATE_FOLDER_NAME"
LOCAL_ENV_DIR_NAME = ".env"
LOCAL_ENV_FILE_NAMES = (".env",)
LOCAL_ENV_DIR_PATTERNS = ("*.env", "*.txt")

INCLUDE_PATHS = [
    "apps",
    "vision",
    "services",
    "ui",
    "workers",
    "tts",
    "auto_glm",
    "lgb",
    "main.py",
    "launcher_entry.py",
    "ocr_subprocess_entry.py",
    "update_apply.py",
    "runtime_layout.py",
    "requirements.txt",
    "pyproject.toml",
    "config.json.example",
    "runtime_manifest.json",
]

BUNDLED_DATA_FILES = [
    "runtime/greetings/pingtai.txt",
]
PIPER_SOURCE_ASSETS_DIR = "tts/piper"

RUNTIME_SKELETON_DIRS = [
    "engine/app",
    "engine/ocr_cpu",
    "engine/ocr_nvidia",
    "engine/ocr_directml",
    "runtime/logs",
    "runtime/errors",
    "runtime/memory",
    "runtime/reports",
    "runtime/tmp/ocr_requests",
    "runtime/greetings",
    "models/ocr/paddlex_named_models",
    "models/ocr/metadata",
    "photo",
    "photo/errors",
]

# 发布包只保护本工程的业务代码。模板、配置、模型和只读外部依赖保持原样。
PROTECTED_SOURCE_PATHS = (
    "apps",
    "vision",
    "services",
    "ui",
    "workers",
    "tts",
    "main.py",
    "launcher_entry.py",
    "ocr_subprocess_entry.py",
    "update_apply.py",
    "runtime_layout.py",
    "auto_glm",
    "lgb",
)

APPLICATION_ENTRY_PYC = "app_core_entry.pyc"
APPLICATION_MODULE_ARCHIVE = "app_modules.zip"
TTS_SERVER_PYC = "tts_server.pyc"
UPDATE_APPLY_PYC = "update_apply.pyc"
OCR_BRIDGE_PYC = "ocr_subprocess_entry.pyc"
OCR_BRIDGE_ARCHIVE = "ocr_bridge_modules.zip"

_IGNORE_NAMES = {
    ".git",
    ".gitignore",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".env",
    "dist",
    "build",
    "wheelhouse",
    "_bootstrap",
    "outbox.json",
    "install_state.json",
    "AGENTS.md",
}
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".tmp", ".md")
_STORED_SUFFIXES = {".zip", ".jpg", ".jpeg", ".png", ".gz", ".7z", ".exe"}

RUNTIME_CACHE_INPUTS = (
    "runtime_builder.py",
    "runtime_manifest.json",
    "requirements.txt",
    "runtime_layout.py",
    "ocr_subprocess_entry.py",
)
PACKAGE_CACHE_EXTRA_INPUTS = (
    "build.py",
    "runtime_builder.py",
    "app_core_entry.py",
)
DEPRECATED_PACKAGE_PATHS = (
    "apps/wechat",
)


def _requirements_specs(root: Path) -> list[str]:
    path = root / "requirements.txt"
    if not path.is_file():
        return []
    specs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        specs.append(stripped)
    return specs


def _runtime_manifest_payload(root: Path) -> dict:
    default = {
        "sharedPackages": [
            {"spec": "paddleocr==3.4.0"},
            {"spec": "paddlex[ocr-core]==3.4.3"},
        ],
        "backends": {},
    }
    path = root / "runtime_manifest.json"
    if not path.is_file():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return payload if isinstance(payload, dict) else default


def _runtime_manifest_path(root: Path) -> Path:
    return root / "runtime_manifest.json"


def _write_runtime_manifest_payload(root: Path, payload: dict) -> None:
    _runtime_manifest_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _shared_runtime_specs(root: Path) -> set[str]:
    payload = _runtime_manifest_payload(root)
    shared = payload.get("sharedPackages", [])
    specs: set[str] = set()
    if isinstance(shared, list):
        for item in shared:
            if isinstance(item, dict):
                spec = str(item.get("spec", "") or "").strip()
                if spec:
                    specs.add(spec)
    return specs


def _bundled_dependency_specs(root: Path) -> list[str]:
    shared = _shared_runtime_specs(root)
    return [spec for spec in _requirements_specs(root) if spec not in shared]


def _manifest_default_model_spec(root: Path, *, backend_key: str, model_tier: str) -> dict[str, str]:
    payload = _runtime_manifest_payload(root)
    models = payload.get("models", {})
    if not isinstance(models, dict):
        return {}
    tiers = models.get("tiers", {})
    if not isinstance(tiers, dict):
        return {}
    tier_payload = tiers.get(model_tier, {})
    if not isinstance(tier_payload, dict):
        return {}
    backends = tier_payload.get("backends", {})
    if not isinstance(backends, dict):
        return {}
    backend_payload = backends.get(backend_key, {})
    if not isinstance(backend_payload, dict):
        return {}
    return {
        "provider": str(backend_payload.get("provider", "") or "").strip(),
        "det": str(backend_payload.get("detModelName", "") or "").strip(),
        "rec": str(backend_payload.get("recModelName", "") or "").strip(),
        "det_version": str(backend_payload.get("detVersion", "") or "").strip(),
        "rec_version": str(backend_payload.get("recVersion", "") or "").strip(),
    }


def _runtime_version(root: Path) -> str:
    payload = _runtime_manifest_payload(root)
    return str(payload.get("runtimeVersion", "") or payload.get("version", "") or "").strip()


def _model_version(root: Path, *, model_tier: str) -> str:
    payload = _runtime_manifest_payload(root)
    configured = str(payload.get("modelVersion", "") or "").strip()
    if configured:
        return configured
    runtime_version = _runtime_version(root)
    tier = str(model_tier or payload.get("models", {}).get("defaultTier", "") or "").strip()
    return f"{runtime_version}-{tier}" if runtime_version and tier else runtime_version


def _next_release_app_version(
    root: Path,
    *,
    now: datetime | None = None,
    include_remote: bool = True,
) -> str:
    """Return a simple calendar release number such as ``20260830-02``."""
    date_text = (now or datetime.now()).strftime("%Y%m%d")
    sequence = 0

    def consider(version: object, update_file_name: object = "") -> None:
        nonlocal sequence
        version_text = str(version or "").strip()
        matched = re.fullmatch(rf"{re.escape(date_text)}-(\d+)", version_text)
        if matched:
            sequence = max(sequence, int(matched.group(1)))
            return
        # Older hash-based releases still count as the first publication when
        # their update archive was created today.
        if str(update_file_name or "").startswith(f"{PACKAGE_NAME}_update_{date_text}_"):
            sequence = max(sequence, 1)

    for manifest_path in (root / "runtime_manifest.json", root / "dist" / RELEASE_FILE_NAME):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            consider(payload.get("appVersion") or payload.get("app_version"), payload.get("update_file_name"))

    if include_remote:
        try:
            response = requests.get(DEFAULT_RELEASE_MANIFEST_URL, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                consider(payload.get("app_version"), payload.get("update_file_name"))
        except (requests.RequestException, ValueError):
            pass
    return f"{date_text}-{sequence + 1:02d}"


def _write_package_release_metadata(
    package_root: Path,
    *,
    app_version: str,
    runtime_version: str,
    model_version: str,
) -> None:
    payload = _runtime_manifest_payload(package_root)
    payload["appVersion"] = app_version
    payload["runtimeVersion"] = runtime_version or str(payload.get("version", "") or "").strip()
    payload["modelVersion"] = model_version
    _write_runtime_manifest_payload(package_root, payload)


def _release_manifest_payload(
    *,
    app_version: str,
    runtime_version: str,
    model_version: str,
    update_file_name: str,
    update_url: str = "",
    requires_full_package: bool = False,
) -> dict[str, object]:
    return {
        "app_version": app_version,
        "runtime_version": runtime_version,
        "model_version": model_version,
        "requires_full_package": requires_full_package,
        "update_share_url": DEFAULT_UPDATE_SHARE_URL,
        "update_file_name": update_file_name,
        "update_url": update_url or f"{DEFAULT_UPDATE_PACKAGE_URL_PREFIX}/{quote(update_file_name, safe='-_.~')}",
        "full_package_url": DEFAULT_FULL_PACKAGE_PAGE_URL,
        "release_notes": "",
        "release_manifest_url": DEFAULT_RELEASE_MANIFEST_URL,
    }


def _write_release_manifest(
    dist_root: Path,
    *,
    app_version: str,
    runtime_version: str,
    model_version: str,
    update_file_name: str,
    update_url: str = "",
    requires_full_package: bool = False,
) -> Path:
    payload = _release_manifest_payload(
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
        update_file_name=update_file_name,
        update_url=update_url,
        requires_full_package=requires_full_package,
    )
    dist_root.mkdir(parents=True, exist_ok=True)
    path = dist_root / RELEASE_FILE_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _release_manifest_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _cache_stable_runtime_manifest_bytes(path: Path) -> bytes:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path.read_bytes()
    if not isinstance(payload, dict):
        return path.read_bytes()
    payload = dict(payload)
    payload.pop("appVersion", None)
    payload.pop("runtimeVersion", None)
    payload.pop("modelVersion", None)
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _cache_input_bytes(relative: str, path: Path) -> bytes:
    normalized = relative.replace("\\", "/")
    if normalized == "runtime_manifest.json":
        return _cache_stable_runtime_manifest_bytes(path)
    return path.read_bytes()


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _iter_local_env_paths(root: Path) -> Iterator[Path]:
    for name in LOCAL_ENV_FILE_NAMES:
        path = root / name
        if path.is_file():
            yield path
    env_dir = root / LOCAL_ENV_DIR_NAME
    if env_dir.is_dir():
        seen: set[Path] = set()
        for pattern in LOCAL_ENV_DIR_PATTERNS:
            for path in sorted(env_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    yield path


def _apply_local_env_files(root: Path) -> None:
    merged: dict[str, str] = {}
    for path in _iter_local_env_paths(root):
        merged.update(_parse_env_file(path))
    for key, value in merged.items():
        os.environ.setdefault(key, value)


def _required_env(name: str) -> str:
    value = str(os.environ.get(name, "") or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set COS credentials or pass --skip-release-upload for local-only builds."
        )
    return value


def _cos_credentials_from_env() -> tuple[str, str, str]:
    return (
        _required_env(COS_SECRET_ID_ENV),
        _required_env(COS_SECRET_KEY_ENV),
        str(os.environ.get(COS_SESSION_TOKEN_ENV, "") or "").strip(),
    )


def _lanzou_credentials_from_env() -> tuple[str, str, int | None, str]:
    folder_id_text = str(os.environ.get(LANZOU_UPDATE_FOLDER_ID_ENV, "") or "").strip()
    folder_id = int(folder_id_text) if folder_id_text else None
    return (
        _required_env(LANZOU_USERNAME_ENV),
        _required_env(LANZOU_PASSWORD_ENV),
        folder_id,
        str(os.environ.get(LANZOU_UPDATE_FOLDER_NAME_ENV, "") or "").strip(),
    )


def _cos_quote(value: str) -> str:
    return quote(value, safe="-_.~")


def _cos_object_path(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    return path if path.startswith("/") else "/" + path


def _cos_signature_headers(headers: dict[str, str]) -> tuple[str, str]:
    normalized: list[tuple[str, str]] = []
    for key, value in headers.items():
        normalized.append((key.strip().lower(), " ".join(str(value).strip().split())))
    normalized.sort(key=lambda item: item[0])
    header_list = ";".join(key for key, _ in normalized)
    header_pairs = "&".join(f"{key}={_cos_quote(value)}" for key, value in normalized)
    return header_list, header_pairs


def _cos_authorization(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    secret_id: str,
    secret_key: str,
    now: int | None = None,
    ttl_seconds: int = DEFAULT_COS_SIGNATURE_TTL_SECONDS,
) -> str:
    timestamp = int(time.time() if now is None else now)
    sign_time = f"{timestamp};{timestamp + ttl_seconds}"
    header_list, header_pairs = _cos_signature_headers(headers)
    http_string = "\n".join(
        [
            method.lower(),
            _cos_object_path(url),
            "",
            header_pairs,
            "",
        ]
    )
    sha1_http_string = hashlib.sha1(http_string.encode("utf-8")).hexdigest()
    string_to_sign = "\n".join(["sha1", sign_time, sha1_http_string, ""])
    sign_key = hmac.new(secret_key.encode("utf-8"), sign_time.encode("utf-8"), hashlib.sha1).hexdigest()
    signature = hmac.new(sign_key.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).hexdigest()
    return (
        f"q-sign-algorithm=sha1&q-ak={secret_id}&q-sign-time={sign_time}"
        f"&q-key-time={sign_time}&q-header-list={header_list}"
        f"&q-url-param-list=&q-signature={signature}"
    )


def _upload_release_manifest(
    *,
    payload: dict[str, object],
    url: str = DEFAULT_RELEASE_MANIFEST_URL,
    secret_id: str,
    secret_key: str,
    session_token: str = "",
    public_read: bool = True,
    timeout_seconds: int = DEFAULT_COS_UPLOAD_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> None:
    parsed = urlparse(url)
    host = parsed.netloc.strip()
    if not host:
        raise RuntimeError(f"Invalid release manifest URL: {url}")
    body = _release_manifest_bytes(payload)
    headers = {
        "Host": host,
        "Content-Type": "application/json; charset=utf-8",
    }
    if public_read:
        headers["x-cos-acl"] = "public-read"
    if session_token:
        headers["x-cos-security-token"] = session_token
    headers["Authorization"] = _cos_authorization(
        method="PUT",
        url=url,
        headers=headers,
        secret_id=secret_id,
        secret_key=secret_key,
    )

    http = session or requests.Session()
    put_response = http.put(url, data=body, headers=headers, timeout=timeout_seconds)
    put_response.raise_for_status()

    verify_response = http.get(url, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
    verify_response.raise_for_status()


def _upload_update_package_to_cos(
    update_zip: Path,
    *,
    timeout_seconds: int = DEFAULT_COS_UPLOAD_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> str:
    """上传更新包到 COS，并返回公开下载地址。"""
    url = f"{DEFAULT_UPDATE_PACKAGE_URL_PREFIX}/{quote(update_zip.name, safe='-_.~')}"
    secret_id, secret_key, session_token = _cos_credentials_from_env()
    parsed = urlparse(url)
    headers = {
        "Host": parsed.netloc,
        "Content-Type": "application/zip",
        "Content-Length": str(update_zip.stat().st_size),
        "x-cos-acl": "public-read",
    }
    if session_token:
        headers["x-cos-security-token"] = session_token
    headers["Authorization"] = _cos_authorization(
        method="PUT",
        url=url,
        headers=headers,
        secret_id=secret_id,
        secret_key=secret_key,
    )
    http = session or requests.Session()
    with update_zip.open("rb") as handle:
        response = http.put(url, data=handle, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    verify = http.head(url, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
    verify.raise_for_status()
    return url


def _lanzou_login_session(
    *,
    username: str,
    password: str,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> tuple[requests.Session, int]:
    try:
        from services.update_sources import _get_with_acw
    except Exception as exc:
        raise RuntimeError(f"Could not import Lanzou helpers: {exc}") from exc

    http = session or requests.Session()
    login_url = "https://accounts.woozooo.com/accounts.php?action=login&ref=pc.woozooo.com"
    user_agent = "Mozilla/5.0"
    _get_with_acw(http, login_url, timeout_seconds=timeout_seconds)
    login_response = http.post(
        "https://accounts.woozooo.com/accounts.php",
        data={
            "task": "uselogin",
            "username": username,
            "password": password,
            "ref": "pc.woozooo.com",
        },
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": login_url,
            "Origin": "https://accounts.woozooo.com",
            "User-Agent": user_agent,
        },
        timeout=timeout_seconds,
    )
    login_response.raise_for_status()
    payload = login_response.json()
    if int(payload.get("zt", 0)) != 1:
        raise RuntimeError(f"Lanzou login failed: {payload!r}")
    http.get(payload["msgs"], headers={"Referer": login_url, "User-Agent": user_agent}, timeout=timeout_seconds)
    _get_with_acw(http, "https://pc.woozooo.com/mydisk.php", timeout_seconds=timeout_seconds, referer=login_url)
    uid_text = str(http.cookies.get_dict().get("ylogin", "") or "").strip()
    if not uid_text:
        raise RuntimeError("Lanzou login succeeded but no ylogin cookie was returned.")
    return http, int(uid_text)


def _lanzou_json_post(
    session: requests.Session,
    *,
    url: str,
    data: dict[str, object],
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> dict:
    response = session.post(
        url,
        data=data,
        headers={
            "Referer": "https://pc.woozooo.com/mydisk.php",
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Lanzou response payload: {payload!r}")
    return payload


def _lanzou_list_folders(
    session: requests.Session,
    *,
    uid: int,
    parent_folder_id: int = -1,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> list[dict]:
    payload = _lanzou_json_post(
        session,
        url=f"https://pc.woozooo.com/doupload.php?uid={uid}",
        data={"task": 47, "folder_id": parent_folder_id},
        timeout_seconds=timeout_seconds,
    )
    folders = payload.get("text", [])
    return folders if isinstance(folders, list) else []


def _lanzou_find_folder_id(folders: list[dict], folder_name: str) -> int | None:
    target = str(folder_name or "").strip()
    if not target:
        return None
    for folder in folders:
        if not isinstance(folder, dict):
            continue
        if str(folder.get("name", "") or "").strip() == target:
            folder_id_text = str(folder.get("fol_id", "") or "").strip()
            if folder_id_text:
                return int(folder_id_text)
    return None


def _lanzou_resolve_update_folder_id(
    session: requests.Session,
    *,
    uid: int,
    folder_id: int | None,
    folder_name: str,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> int:
    if folder_id is not None:
        return folder_id
    folders = _lanzou_list_folders(
        session,
        uid=uid,
        parent_folder_id=-1,
        timeout_seconds=timeout_seconds,
    )
    resolved = _lanzou_find_folder_id(folders, folder_name)
    if resolved is None:
        raise RuntimeError(f"Could not find Lanzou update folder: {folder_name}")
    return resolved


def _lanzou_list_files(
    session: requests.Session,
    *,
    folder_id: int,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> list[dict]:
    files: list[dict] = []
    page = 1
    while True:
        payload = _lanzou_json_post(
            session,
            url="https://pc.woozooo.com/doupload.php",
            data={"task": 5, "folder_id": folder_id, "pg": page},
            timeout_seconds=timeout_seconds,
        )
        items = payload.get("text", [])
        if isinstance(items, list):
            files.extend(item for item in items if isinstance(item, dict))
        if int(payload.get("info", 0) or 0) == 0:
            break
        page += 1
    return files


def _lanzou_delete_file(
    session: requests.Session,
    *,
    file_id: int,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> None:
    payload = _lanzou_json_post(
        session,
        url="https://pc.woozooo.com/doupload.php",
        data={"task": 6, "file_id": file_id},
        timeout_seconds=timeout_seconds,
    )
    if int(payload.get("zt", 0) or 0) != 1:
        raise RuntimeError(f"Failed to delete Lanzou file {file_id}: {payload!r}")


def _lanzou_upload_file(
    session: requests.Session,
    *,
    file_path: Path,
    folder_id: int,
    timeout_seconds: int = 3600,
) -> None:
    headers = {
        "Referer": "https://pc.woozooo.com/mydisk.php",
        "User-Agent": "Mozilla/5.0",
    }
    # 蓝奏云新网页使用 html5up.php；旧版接口仍保留回退，兼容旧账户页面。
    endpoints = ("https://pc.woozooo.com/html5up.php", "https://pc.woozooo.com/fileup.php")
    response = None
    for endpoint in endpoints:
        with file_path.open("rb") as handle:
            response = session.post(
                endpoint,
                data={
                    "task": "1",
                    "vie": "2",
                    "ve": "2",
                    "folder_id": str(folder_id),
                    "folder_id_bb_n": str(folder_id),
                    "name": file_path.name,
                },
                files={
                    "upload_file": (file_path.name, handle, "application/octet-stream"),
                },
                headers=headers,
                timeout=timeout_seconds,
            )
        if response.status_code != 404:
            break
    assert response is not None
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("zt", 0) or 0) != 1:
        raise RuntimeError(f"Failed to upload Lanzou file {file_path.name}: {payload!r}")


def _publish_update_package_to_lanzou(
    update_zip: Path,
    *,
    timeout_seconds: int = DEFAULT_LANZOU_TIMEOUT_SECONDS,
) -> None:
    username, password, folder_id, folder_name = _lanzou_credentials_from_env()
    session, uid = _lanzou_login_session(
        username=username,
        password=password,
        timeout_seconds=timeout_seconds,
    )
    resolved_folder_id = _lanzou_resolve_update_folder_id(
        session,
        uid=uid,
        folder_id=folder_id,
        folder_name=folder_name,
        timeout_seconds=timeout_seconds,
    )

    existing_before = _lanzou_list_files(session, folder_id=resolved_folder_id, timeout_seconds=timeout_seconds)
    _lanzou_upload_file(session, file_path=update_zip, folder_id=resolved_folder_id)
    existing_after = _lanzou_list_files(session, folder_id=resolved_folder_id, timeout_seconds=timeout_seconds)
    uploaded_names = {str(item.get("name_all", "") or "").strip() for item in existing_after}
    if update_zip.name not in uploaded_names:
        raise RuntimeError(f"Lanzou upload completed but {update_zip.name} is not visible in the target folder.")

    for item in existing_after:
        name = str(item.get("name_all", "") or "").strip()
        file_id_text = str(item.get("id", "") or "").strip()
        if not file_id_text or name == update_zip.name:
            continue
        _lanzou_delete_file(session, file_id=int(file_id_text), timeout_seconds=timeout_seconds)

    remaining = _lanzou_list_files(session, folder_id=resolved_folder_id, timeout_seconds=timeout_seconds)
    remaining_names = [str(item.get("name_all", "") or "").strip() for item in remaining]
    if remaining_names != [update_zip.name]:
        raise RuntimeError(f"Unexpected Lanzou files after publish: {remaining_names!r}")


def _publish_release_manifest(
    *,
    app_version: str,
    runtime_version: str,
    model_version: str,
    update_file_name: str,
    requires_full_package: bool = False,
    update_zip: Path | None = None,
) -> str:
    try:
        from services.update_sources import UpdateSourceError, resolve_lanzou_update_package
    except Exception as exc:
        raise RuntimeError(f"Could not import Lanzou update resolver: {exc}") from exc

    try:
        resolve_lanzou_update_package(
            share_url=DEFAULT_UPDATE_SHARE_URL,
            file_name=update_file_name,
        )
    except UpdateSourceError as exc:
        raise RuntimeError(
            "The update package is not available in the configured Lanzou share yet. "
            f"Upload {update_file_name} to {DEFAULT_UPDATE_SHARE_URL} before publishing release.json. "
            f"Resolver error: {exc}"
        ) from exc

    update_url = _upload_update_package_to_cos(update_zip) if update_zip is not None else ""
    payload = _release_manifest_payload(
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
        update_file_name=update_file_name,
        update_url=update_url,
        requires_full_package=requires_full_package,
    )
    secret_id, secret_key, session_token = _cos_credentials_from_env()
    _upload_release_manifest(
        payload=payload,
        secret_id=secret_id,
        secret_key=secret_key,
        session_token=session_token,
    )
    return DEFAULT_RELEASE_MANIFEST_URL


def _relative_matches_prefix(relative: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return relative == normalized or relative.startswith(normalized + "/")


def _is_app_customer_config_asset(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) < 3 or parts[0] != "apps":
        return False
    if (
        parts[2] == "ocr_config.py"
        or parts[2].endswith("_ocr_config.py")
    ):
        return True
    return False


def _archive_excludes(
    relative: str,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> bool:
    # 客户更新由当前启动器拉起，启动器自身仍被 Windows 占用，更新包不能携带
    # 这两个 EXE，否则更新程序会尝试覆盖正在运行的文件并失败。
    if exclude_full_runtimes and relative in {LAUNCHER_EXE, LAUNCHER_ALIAS_EXE}:
        return True
    if exclude_runtime_state and relative == "runtime/install_state.json":
        return True
    if exclude_app_customer_config_assets and _is_app_customer_config_asset(relative):
        return True
    if exclude_full_runtimes and (
        # engine/ 是基础服务（Python 解释器 + OCR 运行时），只随全量包分发。
        _relative_matches_prefix(relative, "engine")
        or _relative_matches_prefix(relative, "runtime/app")
        or _relative_matches_prefix(relative, "runtime/ocr_cpu")
        or _relative_matches_prefix(relative, "runtime/ocr_nvidia")
        or _relative_matches_prefix(relative, "runtime/ocr_directml")
        or _relative_matches_prefix(relative, "runtime/tts/piper")
        # 顶层 tts/piper 是引擎安装源（约 119MB）。客户机由全量包安装到
        # runtime/tts/piper 并被更新器保留；更新包携带它只增加下载体积，
        # 运行时也永远优先使用 runtime/tts/piper 下的副本。
        or _relative_matches_prefix(relative, "tts/piper")
        or _relative_matches_prefix(relative, "models/ocr")
    ):
        return True
    if _relative_matches_prefix(relative, "runtime/.paddlex"):
        return True
    parts = relative.split("/")
    if "__pycache__" in parts:
        return True
    protected_bytecode = {
        APPLICATION_ENTRY_PYC,
        TTS_SERVER_PYC,
        UPDATE_APPLY_PYC,
        OCR_BRIDGE_PYC,
    }
    return relative.endswith((".pyc", ".pyo")) and relative not in protected_bytecode


def _iter_archive_dirs(
    package_root: Path,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> Iterator[str]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_dir():
            continue
        relative = path.relative_to(package_root).as_posix()
        if not relative:
            continue
        if _archive_excludes(
            relative,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            continue
        yield f"{PACKAGE_NAME}/{relative}/"


def _iter_archive_files(
    package_root: Path,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> Iterator[tuple[Path, str]]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if _archive_excludes(
            relative,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            continue
        yield path, f"{PACKAGE_NAME}/{relative}"


def _write_archive(
    package_root: Path,
    dist_root: Path,
    suffix: str,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
    timestamp: str = "",
) -> Path:
    dist_root.mkdir(parents=True, exist_ok=True)
    archive_path = dist_root / f"{PACKAGE_NAME}{suffix}{'_' + timestamp if timestamp else ''}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname in _iter_archive_dirs(
            package_root,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            archive.writestr(arcname, "")
        for path, arcname in _iter_archive_files(
            package_root,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            compression = zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, arcname, compress_type=compression)
    return archive_path


def _write_full_package_archive(package_root: Path, dist_root: Path, *, timestamp: str = "") -> Path:
    return _write_archive(
        package_root,
        dist_root,
        "_full",
        exclude_runtime_state=False,
        exclude_full_runtimes=False,
        exclude_app_customer_config_assets=False,
        timestamp=timestamp,
    )


def _write_customer_update_archive(package_root: Path, dist_root: Path, *, timestamp: str = "") -> Path:
    return _write_archive(
        package_root,
        dist_root,
        "_update",
        exclude_runtime_state=True,
        exclude_full_runtimes=True,
        exclude_app_customer_config_assets=True,
        timestamp=timestamp,
    )


def _ignore(_dir_path: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in _IGNORE_NAMES}
    ignored.update(name for name in names if any(name.endswith(suffix) for suffix in _IGNORE_SUFFIXES))
    return ignored


def _copy_item(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignore, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _copy_sources(root: Path, package_root: Path) -> None:
    for relative in INCLUDE_PATHS:
        source = root / relative
        if not source.exists():
            continue
        _copy_item(source, package_root / relative)
    for relative in DEPRECATED_PACKAGE_PATHS:
        deprecated_path = package_root / relative
        if deprecated_path.exists():
            _remove_path(deprecated_path)
    for skeleton in RUNTIME_SKELETON_DIRS:
        (package_root / skeleton).mkdir(parents=True, exist_ok=True)
    (package_root / "runtime" / ".paddlex").mkdir(parents=True, exist_ok=True)
    root_config = package_root / "config.json"
    if root_config.exists():
        root_config.unlink()


def _copy_bundled_data(root: Path, package_root: Path) -> None:
    for relative in BUNDLED_DATA_FILES:
        source = root / relative
        if source.is_file():
            _copy_item(source, package_root / relative)
    # TTS 引擎（tts/piper）随包顶层分发并直接在该位置运行；
    # runtime/ 只保留运行数据，不再复制引擎副本。


def _run_build_step(command: list[str], *, cwd: Path) -> None:
    print("[build-step] " + " ".join(str(part) for part in command))
    subprocess.check_call(command, cwd=cwd)


def _install_runtime_into_package(
    root: Path,
    package_root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None = None,
    model_tier: str = "lite",
    cuda_lane: str | None = None,
    skip_validation: bool = False,
    bundle_models: bool = False,
    allow_missing_hardware: bool = False,
) -> None:
    command = [
        sys.executable,
        str(root / "runtime_builder.py"),
        "--package-root",
        str(package_root),
        "--runtime-name",
        runtime_name,
        "--install-scope",
        install_scope,
        "--model-tier",
        model_tier,
        "--skip-state-write",
    ]
    if backend is not None:
        command.extend(["--backend", backend])
    if cuda_lane is not None:
        command.extend(["--cuda-lane", cuda_lane])
    if skip_validation:
        command.append("--skip-validation")
    if bundle_models:
        command.append("--bundle-models")
    if allow_missing_hardware:
        command.append("--allow-missing-hardware")
    _run_build_step(command, cwd=root)


def _cleanup_packaged_ocr_cache(package_root: Path) -> None:
    cache_root = package_root / "runtime" / ".paddlex"
    if cache_root.exists():
        shutil.rmtree(cache_root, onexc=_handle_remove_readonly)


def _cleanup_packaged_transients(package_root: Path) -> None:
    _cleanup_packaged_ocr_cache(package_root)
    for cache_dir in package_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)


def _cleanup_python_caches(package_root: Path) -> None:
    for path in sorted(package_root.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, onexc=_handle_remove_readonly)
    for path in package_root.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for path in package_root.rglob("*.pyo"):
        path.unlink(missing_ok=True)


def _write_full_package_runtime_state(package_root: Path, *, model_tier: str, nvidia_cuda_lane: str) -> None:
    state_path = package_root / "runtime" / "install_state.json"
    model_spec = _manifest_default_model_spec(package_root, backend_key="cpu", model_tier=model_tier)
    payload = {
        "schema_version": 5,
        "backend": "auto",
        "model_tier": model_tier,
        "execution_mode": "subprocess",
        "package_mode": "full",
        "runtime_name": "app",
        "cuda_lane": nvidia_cuda_lane,
        "cache_home": "runtime/.paddlex",
        "status": "installed",
        "python_executable": "engine/app/python.exe",
        "runtime_path": "engine/app",
        "message": "Full package runtimes assembled by build.py",
        "validation": "ok",
        "validation_error": "",
        "ocr_device": "",
        "hardware": {},
        "common_packages": _bundled_dependency_specs(package_root),
        "packages": [],
        "installed_backends": ["cpu", "paddle_cuda", "onnx_directml"],
        "runtime_inventory": {
            "app": "engine/app",
            "cpu": "engine/ocr_cpu",
            "paddle_cuda": "engine/ocr_nvidia",
            "onnx_directml": "engine/ocr_directml",
        },
        "models": model_spec,
        "validation_details": {
            "build_mode": "full_package",
            "bundled_models_root": "models/ocr/paddlex_named_models",
            "nvidia_cuda_lane": nvidia_cuda_lane,
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime_cache_root(root: Path) -> Path:
    configured = str(os.environ.get("JXB_RUNTIME_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        system_drive = os.environ.get("SystemDrive", "C:")
        return Path(system_drive + os.sep) / "jxb_rt_cache"
    return Path(tempfile.gettempdir()) / "jxb_rt_cache"


def _runtime_cache_key(
    root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None,
    model_tier: str,
    cuda_lane: str | None,
    skip_validation: bool,
    bundle_models: bool,
) -> str:
    payload = {
        "runtime_name": runtime_name,
        "install_scope": install_scope,
        "backend": backend,
        "model_tier": model_tier,
        "cuda_lane": cuda_lane,
        "skip_validation": skip_validation,
        "bundle_models": bundle_models,
        "python": sys.version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for relative in RUNTIME_CACHE_INPUTS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        digest.update(_cache_input_bytes(relative, path))
    return digest.hexdigest()[:16]


def _runtime_cache_dir(root: Path, *, runtime_name: str, cache_key: str) -> Path:
    return _runtime_cache_root(root) / runtime_name / cache_key


def _restore_runtime_from_cache(package_root: Path, cache_dir: Path, *, runtime_name: str) -> bool:
    runtime_source = cache_dir / "engine" / runtime_name
    if not runtime_source.is_dir():
        return False
    runtime_target = package_root / "engine" / runtime_name
    if runtime_target.exists():
        shutil.rmtree(runtime_target, onexc=_handle_remove_readonly)
    shutil.copytree(runtime_source, runtime_target)
    models_source = cache_dir / "models"
    if models_source.is_dir():
        shutil.copytree(models_source, package_root / "models", dirs_exist_ok=True)
    return True


def _store_runtime_cache(package_root: Path, cache_dir: Path, *, runtime_name: str, include_models: bool) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)
    (cache_dir / "engine").mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_root / "engine" / runtime_name, cache_dir / "engine" / runtime_name)
    if include_models:
        models_root = package_root / "models"
        if models_root.is_dir():
            shutil.copytree(models_root, cache_dir / "models")


def _ensure_runtime_available(
    root: Path,
    package_root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None = None,
    model_tier: str = "lite",
    cuda_lane: str | None = None,
    skip_validation: bool = False,
    bundle_models: bool = False,
    allow_missing_hardware: bool = False,
    use_cache: bool = True,
) -> None:
    cache_key = _runtime_cache_key(
        root,
        runtime_name=runtime_name,
        install_scope=install_scope,
        backend=backend,
        model_tier=model_tier,
        cuda_lane=cuda_lane,
        skip_validation=skip_validation,
        bundle_models=bundle_models,
    )
    cache_dir = _runtime_cache_dir(root, runtime_name=runtime_name, cache_key=cache_key)
    if use_cache and _restore_runtime_from_cache(package_root, cache_dir, runtime_name=runtime_name):
        print(f"[runtime-cache] reuse {runtime_name} ({cache_key})")
        return
    _install_runtime_into_package(
        root,
        package_root,
        runtime_name=runtime_name,
        install_scope=install_scope,
        backend=backend,
        model_tier=model_tier,
        cuda_lane=cuda_lane,
        skip_validation=skip_validation,
        bundle_models=bundle_models,
        allow_missing_hardware=allow_missing_hardware,
    )
    if use_cache:
        _store_runtime_cache(package_root, cache_dir, runtime_name=runtime_name, include_models=bundle_models)
        print(f"[runtime-cache] store {runtime_name} ({cache_key})")


def _build_full_package_runtimes(
    root: Path,
    package_root: Path,
    *,
    model_tier: str,
    nvidia_cuda_lane: str,
    use_cache: bool,
) -> None:
    _ensure_runtime_available(root, package_root, runtime_name="app", install_scope="app", model_tier=model_tier, use_cache=use_cache)
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_cpu",
        install_scope="backend",
        backend="cpu",
        model_tier=model_tier,
        bundle_models=True,
        use_cache=use_cache,
    )
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier=model_tier,
        cuda_lane=nvidia_cuda_lane,
        skip_validation=True,
        allow_missing_hardware=True,
        use_cache=use_cache,
    )
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_directml",
        install_scope="backend",
        backend="onnx_directml",
        model_tier=model_tier,
        skip_validation=True,
        allow_missing_hardware=True,
        use_cache=use_cache,
    )


def _staging_root(dist_root: Path) -> Path:
    return dist_root / "_staging"


def _next_staging_session_root(dist_root: Path) -> Path:
    staging_root = _staging_root(dist_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = staging_root / f"build_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a free staging directory under dist/_staging.")


def _prepare_staging_package_root(dist_root: Path) -> Path:
    package_root = _next_staging_session_root(dist_root) / PACKAGE_NAME
    package_root.mkdir(parents=True, exist_ok=True)
    return package_root


def _package_cache_root(root: Path) -> Path:
    configured = str(os.environ.get("JXB_PACKAGE_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        return Path(root.anchor) / "jxb_pkg_cache"
    return Path(tempfile.gettempdir()) / "jxb_pkg_cache"


def _iter_hashable_input_files(root: Path, relatives: list[str]) -> Iterator[tuple[str, Path]]:
    for relative in relatives:
        source = root / relative
        if not source.exists():
            continue
        if source.is_file():
            yield relative.replace("\\", "/"), source
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.name in _ignore(str(path.parent), [path.name]):
                continue
            yield path.relative_to(root).as_posix(), path


def _package_cache_key(
    root: Path,
    *,
    model_tier: str,
    nvidia_cuda_lane: str,
    include_full_payload: bool,
    app_version_salt: str = "",
) -> str:
    digest = hashlib.sha256()
    payload = {
        "model_tier": model_tier,
        "nvidia_cuda_lane": nvidia_cuda_lane,
        "include_full_payload": include_full_payload,
        "app_version_salt": str(app_version_salt or "").strip(),
        "python": sys.version,
    }
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    relatives = [*INCLUDE_PATHS, *BUNDLED_DATA_FILES, *PACKAGE_CACHE_EXTRA_INPUTS]
    for relative, path in _iter_hashable_input_files(root, relatives):
        digest.update(relative.encode("utf-8"))
        digest.update(_cache_input_bytes(relative, path))
    return digest.hexdigest()[:16]


def _package_cache_dir(root: Path, cache_key: str) -> Path:
    return _package_cache_root(root) / cache_key


def _package_cache_complete(
    cache_dir: Path,
    *,
    require_package_dir: bool,
    require_full_zip: bool,
) -> bool:
    if require_package_dir and not (cache_dir / PACKAGE_NAME).is_dir():
        return False
    if not (cache_dir / f"{PACKAGE_NAME}_update.zip").is_file():
        return False
    if require_full_zip and not (cache_dir / f"{PACKAGE_NAME}_full.zip").is_file():
        return False
    return True


def _link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        if source.anchor.lower() == target.anchor.lower():
            os.link(source, target)
            return
    except OSError:
        pass
    shutil.copy2(source, target)


def _remove_path(target: Path) -> None:
    target_str = str(target)
    if not os.path.lexists(target_str):
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "rmdir", "/S", "/Q", target_str],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if completed.returncode == 0 or not os.path.lexists(target_str):
            return
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
        return
    shutil.rmtree(target, onexc=_handle_remove_readonly)


def _junction_or_copy_dir(source: Path, target: Path) -> None:
    if os.path.lexists(str(target)):
        _remove_path(target)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if completed.returncode == 0:
            return
    shutil.copytree(source, target)


def _publish_package_directory(staged_package_root: Path, dist_root: Path) -> Path:
    published_root = dist_root / PACKAGE_NAME
    try:
        _remove_path(published_root)
        shutil.copytree(staged_package_root, published_root)
        return published_root
    except PermissionError:
        locked_target = dist_root / f"{PACKAGE_NAME}_built"
        _remove_path(locked_target)
        shutil.copytree(staged_package_root, locked_target)
        print(f"[warn] publish dir busy, kept new build at: {locked_target}")
        return locked_target


def _cleanup_staging(dist_root: Path) -> None:
    staging_root = _staging_root(dist_root)
    if staging_root.exists():
        _remove_path(staging_root)


def _remove_dist_archives(dist_root: Path) -> None:
    for archive_path in dist_root.glob(f"{PACKAGE_NAME}_*.zip"):
        archive_path.unlink(missing_ok=True)


def _publish_cached_package_directory(cached_package_root: Path, dist_root: Path) -> Path:
    target = dist_root / PACKAGE_NAME
    try:
        _remove_path(target)
        shutil.copytree(cached_package_root, target)
        return target
    except PermissionError:
        fallback = dist_root / f"{PACKAGE_NAME}_built"
        _remove_path(fallback)
        shutil.copytree(cached_package_root, fallback)
        print(f"[warn] publish dir busy, kept cached build at: {fallback}")
        return fallback


def _restore_package_from_cache(root: Path, dist_root: Path, cache_dir: Path) -> tuple[Path, Path, Path]:
    dist_root.mkdir(parents=True, exist_ok=True)
    full_zip = dist_root / f"{PACKAGE_NAME}_full.zip"
    update_zip = dist_root / f"{PACKAGE_NAME}_update.zip"
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_full.zip", full_zip)
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_update.zip", update_zip)
    published_root = _publish_cached_package_directory(cache_dir / PACKAGE_NAME, dist_root)
    return published_root, full_zip, update_zip


def _restore_archives_from_cache(
    dist_root: Path,
    cache_dir: Path,
    *,
    include_full: bool,
    timestamp: str = "",
) -> tuple[Path | None, Path]:
    dist_root.mkdir(parents=True, exist_ok=True)
    full_zip: Path | None = None
    update_zip = dist_root / f"{PACKAGE_NAME}_update{'_' + timestamp if timestamp else ''}.zip"
    if include_full:
        cached_full = cache_dir / f"{PACKAGE_NAME}_full.zip"
        if cached_full.is_file():
            full_zip = dist_root / f"{PACKAGE_NAME}_full{'_' + timestamp if timestamp else ''}.zip"
            _link_or_copy_file(cached_full, full_zip)
    elif not timestamp:
        target_full = dist_root / f"{PACKAGE_NAME}_full.zip"
        if target_full.exists():
            target_full.unlink(missing_ok=True)
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_update.zip", update_zip)
    return full_zip, update_zip


def _store_package_cache(
    root: Path,
    published_root: Path,
    full_zip: Path | None,
    update_zip: Path,
    *,
    cache_key: str,
) -> Path:
    cache_dir = _package_cache_dir(root, cache_key)
    if cache_dir.exists():
        shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(published_root, cache_dir / PACKAGE_NAME)
    if full_zip is not None and full_zip.is_file():
        shutil.copy2(full_zip, cache_dir / f"{PACKAGE_NAME}_full.zip")
    shutil.copy2(update_zip, cache_dir / f"{PACKAGE_NAME}_update.zip")
    return cache_dir


def _current_package_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _write_windows_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\r\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _copy_executable_aliases(package_root: Path) -> None:
    source = package_root / LAUNCHER_EXE
    if source.is_file():
        shutil.copy2(source, package_root / LAUNCHER_ALIAS_EXE)


def _write_cmd_launchers(package_root: Path) -> None:
    _write_windows_script(
        package_root / START_CMD,
        [
            "@echo off",
            "setlocal",
            'cd /d "%~dp0"',
            "",
            f'set "LAUNCHER_EXE=%~dp0{LAUNCHER_ALIAS_EXE}"',
            "",
            'if exist "%LAUNCHER_EXE%" goto start_exe',
            "echo Missing protected launcher executable.",
            "pause",
            "exit /b 1",
            "",
            ":start_exe",
            'start "" "%LAUNCHER_EXE%"',
            "exit /b 0",
        ],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_package_manifest(package_root: Path) -> None:
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if relative.startswith("runtime/.paddlex/"):
            continue
        files.append({"path": relative, "sha256": _sha256(path), "size": path.stat().st_size})
    payload = {
        "schema_version": 2,
        "package_name": PACKAGE_NAME,
        "contains_private_config": False,
        "files": files,
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _exe_cache_fresh(cached_exe: Path, root: Path, deps: list[str]) -> bool:
    if not cached_exe.is_file():
        return False
    exe_mtime = cached_exe.stat().st_mtime
    for dep in deps:
        dep_path = root / dep
        if not dep_path.is_file() or dep_path.stat().st_mtime > exe_mtime:
            return False
    return True


def _source_signature(root: Path) -> str:
    """Build-cache key for frozen executables; includes every protected source file."""
    digest = hashlib.sha256()
    for relative in [*PROTECTED_SOURCE_PATHS, "launcher_entry.py", "app_core_entry.py"]:
        source = root / relative
        if source.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(source.read_bytes())
        elif source.is_dir():
            for path in sorted(source.rglob("*.py")):
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _freeze_executable(root: Path, cache: Path, work: Path, *, name: str, entry: str, args: list[str]) -> Path:
    target = cache / name
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name",
        Path(name).stem,
        "--distpath",
        str(cache),
        "--workpath",
        str(work / Path(name).stem),
        "--specpath",
        str(work),
        *args,
        str(root / entry),
    ]
    print("[pyinstaller] " + " ".join(command))
    subprocess.check_call(command, cwd=root)
    if not target.is_file():
        raise RuntimeError(f"PyInstaller did not create {target}")
    return target


def _freeze_executables(root: Path, package_root: Path, *, rebuild: bool) -> None:
    cache = root / "build" / "exe_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached_launcher = cache / LAUNCHER_EXE
    deps = ["launcher_entry.py", "runtime_layout.py"]
    if rebuild and not _exe_cache_fresh(cached_launcher, root, deps):
        try:
            import PyInstaller  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("PyInstaller is required. Run `pip install -r requirements-dev.txt`.") from exc
        _freeze_executable(root, cache, root / "build" / "_pyinstaller", name=LAUNCHER_EXE, entry="launcher_entry.py", args=[])
    if not cached_launcher.is_file():
        raise RuntimeError(f"Missing cached launcher executable: {cached_launcher}")
    shutil.copy2(cached_launcher, package_root / LAUNCHER_EXE)

    # Compile business modules into a zip archive. Python can import pyc-only
    # modules from it, while customers cannot open the source files directly.
    temporary = Path(tempfile.mkdtemp(prefix="jxb_modules_"))
    try:
        archive = package_root / APPLICATION_MODULE_ARCHIVE
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for relative in PROTECTED_SOURCE_PATHS:
                source = root / relative
                files = [source] if source.is_file() else sorted(source.rglob("*.py")) if source.is_dir() else []
                if relative == "lgb" and source.is_dir() and not (source / "__init__.py").is_file():
                    # lgb is a read-only dependency and has no package marker.
                    # Add the marker only to the protected bytecode archive so
                    # imports such as ``from lgb.lgb import RobotInterface``
                    # work after release Python sources are removed.
                    init_source = temporary / "lgb_package_init.py"
                    init_source.write_text("", encoding="utf-8")
                    init_target = temporary / "lgb" / "__init__.pyc"
                    init_target.parent.mkdir(parents=True, exist_ok=True)
                    py_compile.compile(str(init_source), cfile=str(init_target), doraise=True)
                    output.write(init_target, "lgb/__init__.pyc")
                for file_path in files:
                    target = temporary / file_path.relative_to(root).with_suffix(".pyc")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    py_compile.compile(str(file_path), cfile=str(target), doraise=True)
                    output.write(target, target.relative_to(temporary).as_posix())
        for name, source in ((APPLICATION_ENTRY_PYC, "app_core_entry.py"), (TTS_SERVER_PYC, "tts/server.py"), (UPDATE_APPLY_PYC, "update_apply.py")):
            target = package_root / name
            py_compile.compile(str(root / source), cfile=str(target), doraise=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _build_ocr_bridge_bytecode(root: Path, package_root: Path) -> None:
    """Keep OCR on its selected runtime while removing readable bridge sources."""
    if not (root / "ocr_subprocess_entry.py").is_file() or not (root / "vision").is_dir():
        return
    temporary = Path(tempfile.mkdtemp(prefix="jxb_ocr_bridge_"))
    try:
        entry_target = package_root / OCR_BRIDGE_PYC
        py_compile.compile(str(root / "ocr_subprocess_entry.py"), cfile=str(entry_target), doraise=True)
        archive = package_root / OCR_BRIDGE_ARCHIVE
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for relative in ("vision", "runtime_layout.py"):
                source = root / relative
                files = [source] if source.is_file() else sorted(source.rglob("*.py"))
                for file_path in files:
                    compiled = temporary / file_path.relative_to(root).with_suffix(".pyc")
                    compiled.parent.mkdir(parents=True, exist_ok=True)
                    py_compile.compile(str(file_path), cfile=str(compiled), doraise=True)
                    output.write(compiled, compiled.relative_to(temporary).as_posix())
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _remove_packaged_python_sources(package_root: Path) -> None:
    """Release directories retain templates/configuration only; code lives in executables.

    ``update_apply.py`` 是唯一例外：旧版客户 UI 会直接执行这个源码文件。保留它
    让旧 UI 能先完成一次更新，之后再切换到新版的 ``update_apply.pyc``。
    """
    for relative in PROTECTED_SOURCE_PATHS:
        if relative == "update_apply.py":
            continue
        target = package_root / relative
        if target.is_file() and target.suffix == ".py":
            target.unlink(missing_ok=True)
        elif target.is_dir():
            for source_file in target.rglob("*.py"):
                source_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the full jixiebijiaoben package.")
    parser.add_argument("--full", action="store_true", help="also build dist/jixiebijiaoben_full.zip")
    parser.add_argument("--skip-freeze", action="store_true", help="reuse cached launcher exe")
    parser.add_argument("--no-runtime-cache", action="store_true", help="do not reuse cached runtimes")
    parser.add_argument("--publish-dir", action="store_true", help="also publish the unpacked package directory into dist/")
    parser.add_argument("--skip-release-upload", action="store_true", help="skip uploading release.json to Tencent COS")
    parser.add_argument("--model-tier", default="lite")
    parser.add_argument("--nvidia-cuda-lane", default=DEFAULT_FULL_PACKAGE_NVIDIA_LANE)
    parser.add_argument("--app-version-salt", default="", help="force a different app_version without changing runtime/model versions")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    _apply_local_env_files(root)
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    include_full_payload = bool(args.full or args.publish_dir)
    package_timestamp = _current_package_timestamp()
    _remove_dist_archives(dist)

    app_version = _next_release_app_version(root, include_remote=not args.skip_release_upload)
    package_cache_key = _package_cache_key(
        root,
        model_tier=args.model_tier,
        nvidia_cuda_lane=args.nvidia_cuda_lane,
        include_full_payload=include_full_payload,
        app_version_salt=f"{args.app_version_salt}:{app_version}",
    )
    runtime_version = _runtime_version(root)
    model_version = _model_version(root, model_tier=args.model_tier)
    cache_dir = _package_cache_dir(root, package_cache_key)
    if _package_cache_complete(
        cache_dir,
        require_package_dir=include_full_payload,
        require_full_zip=False,
    ):
        print(f"[package-cache] hit {package_cache_key}")
        full_zip, update_zip = _restore_archives_from_cache(
            dist,
            cache_dir,
            include_full=args.full,
            timestamp=package_timestamp,
        )
        if args.full and full_zip is None:
            full_zip = _write_full_package_archive(cache_dir / PACKAGE_NAME, dist, timestamp=package_timestamp)
        if args.publish_dir:
            published_root = _publish_cached_package_directory(cache_dir / PACKAGE_NAME, dist)
            print(f"[done] package={published_root}")
        if full_zip is not None:
            print(f"[done] full={full_zip}")
        if args.skip_release_upload:
            release_manifest = _write_release_manifest(
                dist,
                app_version=app_version,
                runtime_version=runtime_version,
                model_version=model_version,
                update_file_name=update_zip.name,
            )
            print(f"[done] release(local-only)={release_manifest}")
        else:
            _publish_update_package_to_lanzou(update_zip)
            release_manifest_url = _publish_release_manifest(
                app_version=app_version,
                runtime_version=runtime_version,
                model_version=model_version,
                update_file_name=update_zip.name,
                update_zip=update_zip,
            )
            print(f"[done] release(uploaded)={release_manifest_url}")
        _write_package_release_metadata(
            root,
            app_version=app_version,
            runtime_version=runtime_version,
            model_version=model_version,
        )
        print(f"[done] update={update_zip}")
        return 0

    package_root = _prepare_staging_package_root(dist)
    _copy_sources(root, package_root)
    _copy_bundled_data(root, package_root)
    _write_package_release_metadata(
        package_root,
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
    )
    if include_full_payload:
        _build_full_package_runtimes(
            root,
            package_root,
            model_tier=args.model_tier,
            nvidia_cuda_lane=args.nvidia_cuda_lane,
            use_cache=not args.no_runtime_cache,
        )
        _cleanup_packaged_transients(package_root)
        _cleanup_python_caches(package_root)
        _write_full_package_runtime_state(package_root, model_tier=args.model_tier, nvidia_cuda_lane=args.nvidia_cuda_lane)
    _freeze_executables(root, package_root, rebuild=not args.skip_freeze)
    _build_ocr_bridge_bytecode(root, package_root)
    _remove_packaged_python_sources(package_root)
    _copy_executable_aliases(package_root)
    _write_cmd_launchers(package_root)
    _write_package_manifest(package_root)

    full_zip: Path | None = None
    if args.full:
        full_zip = _write_full_package_archive(package_root, dist, timestamp=package_timestamp)
    update_zip = _write_customer_update_archive(package_root, dist, timestamp=package_timestamp)
    cache_dir = _store_package_cache(root, package_root, full_zip, update_zip, cache_key=package_cache_key)
    published_root: Path | None = None
    if args.publish_dir:
        published_root = _publish_package_directory(package_root, dist)
    _cleanup_staging(dist)

    print(f"[package-cache] store {package_cache_key} -> {cache_dir}")
    if published_root is not None:
        print(f"[done] package={published_root}")
    if full_zip is not None:
        print(f"[done] full={full_zip}")
    if args.skip_release_upload:
        release_manifest = _write_release_manifest(
            dist,
            app_version=app_version,
            runtime_version=runtime_version,
            model_version=model_version,
            update_file_name=update_zip.name,
        )
        print(f"[done] release(local-only)={release_manifest}")
    else:
        _publish_update_package_to_lanzou(update_zip)
        release_manifest_url = _publish_release_manifest(
            app_version=app_version,
            runtime_version=runtime_version,
            model_version=model_version,
            update_file_name=update_zip.name,
            update_zip=update_zip,
        )
        print(f"[done] release(uploaded)={release_manifest_url}")
    _write_package_release_metadata(
        root,
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
    )
    print(f"[done] update={update_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
