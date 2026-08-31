"""视觉模型调用次数的持久化计数。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

# 始终以程序目录为基准，不能依赖用户从哪个快捷方式/工作目录启动程序。
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "vision_usage.json"
_LOCK = threading.RLock()


def _usage_path(config: dict[str, Any]) -> Path:
    runtime = config.get("_runtime", {})
    raw = runtime.get("vision_usage_path") if isinstance(runtime, dict) else None
    path = Path(str(raw)) if raw else DEFAULT_PATH
    if not path.is_absolute():
        package_root = os.environ.get("JIXIEBI_PACKAGE_ROOT", "")
        root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
        path = root / path
    return path.resolve()


def scope_key(config: dict[str, Any]) -> str:
    runtime = config.get("_runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    raw = "|".join(
        (
            f"{platform.node()}:{uuid.getnode()}",
            str(runtime.get("device_ip", "")),
            str(runtime.get("app_name", "xinyu")),
            str(runtime.get("perspective_signature", "")),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "scopes": {}}
    return value if isinstance(value, dict) and isinstance(value.get("scopes"), dict) else {"schema_version": 1, "scopes": {}}


def acquire(config: dict[str, Any], *, limit: int) -> tuple[bool, int]:
    """原子地占用一次调用额度；进程重启后仍延续计数。"""
    path = _usage_path(config)
    key = scope_key(config)
    with _LOCK:
        payload = _load(path)
        scopes = payload["scopes"]
        current = max(0, int(scopes.get(key, 0) or 0))
        if current >= limit:
            return False, current
        scopes[key] = current + 1
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
        return True, current + 1


def rule_learning_status(config: dict[str, Any]) -> dict[str, Any]:
    """读取当前设备/App/透视的连续无新增规则状态。"""
    path = _usage_path(config)
    key = scope_key(config)
    with _LOCK:
        payload = _load(path)
        status = payload.get("rule_learning", {}).get(key, {})
        return dict(status) if isinstance(status, dict) else {}


def record_rule_learning(
    config: dict[str, Any],
    *,
    added_count: int,
    disable_after_zero: int,
) -> dict[str, Any]:
    """记录一次成功的视觉识别结果，并返回更新后的停用状态。"""
    path = _usage_path(config)
    key = scope_key(config)
    threshold = max(1, int(disable_after_zero or 3))
    with _LOCK:
        payload = _load(path)
        states = payload.setdefault("rule_learning", {})
        state = states.setdefault(key, {})
        if int(added_count or 0) > 0:
            zero_count = 0
            disabled = False
        else:
            zero_count = max(0, int(state.get("zero_count", 0) or 0)) + 1
            disabled = zero_count >= threshold
        state.update(
            {
                "zero_count": zero_count,
                "disabled": disabled,
                "threshold": threshold,
            }
        )
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
        return dict(state)
