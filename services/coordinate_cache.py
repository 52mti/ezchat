"""按电脑、设备、App、应用版本和透视签名隔离的坐标持久化缓存。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

COORDINATE_SECTIONS = ("template_points", "screen_anchors", "ocr_regions")
DEFAULT_ROOT = Path("runtime") / "coordinate_cache"
_LOCK = threading.RLock()
_APP_VERSION_CACHE: list[str] = []


def _part(value: Any) -> str:
    text = str(value or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24] if text else "unknown"


def _app_version() -> str:
    """读包根 runtime_manifest.json 的 appVersion；版本变了坐标缓存整体作废。

    更新可能调整模板区域/阈值，旧版本缓存的屏幕坐标不能跨版本复用。
    """
    if _APP_VERSION_CACHE:
        return _APP_VERSION_CACHE[0]
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    root = Path(configured).resolve() if configured else Path(__file__).resolve().parents[1]
    version = ""
    try:
        payload = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            version = str(payload.get("appVersion", "") or "").strip()
    except (OSError, json.JSONDecodeError, ValueError):
        version = ""
    _APP_VERSION_CACHE[:] = [version]
    return version


def _scope(config: dict[str, Any]) -> tuple[str, str, str, str, str]:
    runtime = config.get("_runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    machine = hashlib.sha256(f"{platform.node()}:{uuid.getnode()}".encode("utf-8")).hexdigest()[:24]
    return (
        machine,
        _part(runtime.get("device_ip")),
        _part(runtime.get("app_name")),
        _part(_app_version()),
        _part(runtime.get("perspective_signature")),
    )


def _path(config: dict[str, Any], root: str | Path = DEFAULT_ROOT) -> Path:
    machine, device, app, app_version, perspective = _scope(config)
    return Path(root) / machine / device / app / app_version / f"{perspective}.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_into_runtime(config: dict[str, Any]) -> bool:
    try:
        payload = json.loads(_path(config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    cache = config.setdefault("_runtime_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        config["_runtime_cache"] = cache
    with _LOCK:
        for section in COORDINATE_SECTIONS:
            if isinstance(payload.get(section), dict):
                cache[section] = copy.deepcopy(payload[section])
    return True


def load_latest_into_runtime(config: dict[str, Any], root: str | Path = DEFAULT_ROOT) -> bool:
    """加载当前设备/App/版本下最近一次透视缓存，供设置界面查看。"""
    machine, device, app, _app_version, _perspective = _scope(config)
    scope_dir = Path(root) / machine / device / app / _app_version
    try:
        candidates = [path for path in scope_dir.glob("*.json") if path.is_file()]
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    cache = config.setdefault("_runtime_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        config["_runtime_cache"] = cache
    with _LOCK:
        for section in COORDINATE_SECTIONS:
            if isinstance(payload.get(section), dict):
                cache[section] = copy.deepcopy(payload[section])
    return True


def persist_from_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("_runtime", {})
    cache = config.get("_runtime_cache", {})
    if not isinstance(runtime, dict) or not runtime.get("perspective_signature") or not isinstance(cache, dict):
        return
    payload: dict[str, Any] = {
        "schema_version": 1,
        "perspective_signature": runtime["perspective_signature"],
        "app_version": _app_version(),
    }
    for section in COORDINATE_SECTIONS:
        if isinstance(cache.get(section), dict) and cache[section]:
            payload[section] = copy.deepcopy(cache[section])
    with _LOCK:
        _atomic_write(_path(config), payload)
