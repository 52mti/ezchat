"""按电脑、设备、App、应用版本和透视签名隔离的坐标持久化缓存。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

COORDINATE_SECTIONS = ("template_points", "screen_anchors", "ocr_regions")
DEFAULT_ROOT = Path("runtime") / "coordinate_cache"
_LOCK = threading.RLock()
_APP_VERSION_CACHE: list[str] = []

# 落盘防抖：运行期坐标学习（ocr_regions 抖动等）每写一个点就全量 fsync，
# Windows+杀软下单次 10~50ms 且全部串行在全局锁上，多设备时把 UI 线程
# （设置界面读缓存走同一把锁）饿出可感知卡顿。改为：内存立即生效，
# 落盘合并到 0.5s 窗口后的单次写；标定等关键路径走 immediate=True 同步落盘。
PERSIST_DEBOUNCE_SECONDS = 0.5
_DIRTY: dict[str, tuple[dict[str, Any], float]] = {}
_PERSIST_THREAD: list[threading.Thread] = []


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


def _path(config: dict[str, Any], root: str | Path | None = None) -> Path:
    # root 运行时取值（而非默认参数绑定），测试可整体重定向缓存根目录。
    machine, device, app, app_version, perspective = _scope(config)
    return Path(root if root is not None else DEFAULT_ROOT) / machine / device / app / app_version / f"{perspective}.json"


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
    path = _path(config)
    try:
        stat = path.stat()
        stamp: tuple[Any, ...] | None = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    runtime = config.get("_runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    # 每次截图都会调本函数同步坐标缓存；文件未变化时凭 (路径,mtime,size)
    # 短路，跳过整读+deepcopy（多设备并发截图时的主要读盘放大点）。
    if stamp is not None and runtime.get("_coordinate_cache_stamp") == stamp:
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        runtime["_coordinate_cache_stamp"] = None
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
    runtime["_coordinate_cache_stamp"] = stamp
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


def _build_payload(config: dict[str, Any]) -> dict[str, Any] | None:
    """在锁内快照当前运行缓存；worker 并发改 dict 时重试几次。"""
    runtime = config.get("_runtime", {})
    cache = config.get("_runtime_cache", {})
    if not isinstance(runtime, dict) or not runtime.get("perspective_signature") or not isinstance(cache, dict):
        return None
    for _ in range(3):
        try:
            payload: dict[str, Any] = {
                "schema_version": 1,
                "perspective_signature": runtime["perspective_signature"],
                "app_version": _app_version(),
            }
            for section in COORDINATE_SECTIONS:
                if isinstance(cache.get(section), dict) and cache[section]:
                    payload[section] = copy.deepcopy(cache[section])
            return payload
        except RuntimeError:
            continue
    return None


def _persist_now(config: dict[str, Any]) -> None:
    payload = _build_payload(config)
    if payload is None:
        return
    _atomic_write(_path(config), payload)


def _ensure_persist_thread() -> None:
    if _PERSIST_THREAD:
        return

    def loop() -> None:
        while True:
            time.sleep(0.1)
            now = time.monotonic()
            due: list[dict[str, Any]] = []
            with _LOCK:
                for key in list(_DIRTY):
                    config, deadline = _DIRTY[key]
                    if deadline <= now:
                        due.append(config)
                        _DIRTY.pop(key, None)
            for config in due:
                try:
                    _persist_now(config)
                except Exception:
                    continue

    thread = threading.Thread(target=loop, name="coordinate-cache-persist", daemon=True)
    thread.start()
    _PERSIST_THREAD[:] = [thread]


def flush_persist_debounce() -> None:
    """立即落盘全部待写缓存（worker 停止等收尾时机调用）。"""
    with _LOCK:
        pending = [config for config, _ in _DIRTY.values()]
        _DIRTY.clear()
    for config in pending:
        try:
            _persist_now(config)
        except Exception:
            continue


def persist_from_runtime(config: dict[str, Any], *, immediate: bool = False) -> None:
    runtime = config.get("_runtime", {})
    cache = config.get("_runtime_cache", {})
    if not isinstance(runtime, dict) or not runtime.get("perspective_signature") or not isinstance(cache, dict):
        return
    if immediate:
        _persist_now(config)
        return
    with _LOCK:
        _DIRTY[str(_path(config))] = (config, time.monotonic() + PERSIST_DEBOUNCE_SECONDS)
    _ensure_persist_thread()
