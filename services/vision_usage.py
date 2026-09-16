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

# 不能在模块导入时从 __file__ 推导根目录：模块发布在 app_modules.zip 中时，
# __file__.parents[1] 会落到 app_modules.zip，后续 mkdir 会触发 WinError 183。
def _default_path() -> Path:
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).resolve() / "runtime" / "vision_usage.json"
    return DEFAULT_PATH


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "vision_usage.json"


_LOCK = threading.RLock()


def _usage_path(config: dict[str, Any]) -> Path:
    runtime = config.get("_runtime", {})
    raw = runtime.get("vision_usage_path") if isinstance(runtime, dict) else None
    path = Path(str(raw)) if raw else _default_path()
    if not path.is_absolute():
        package_root = os.environ.get("JIXIEBI_PACKAGE_ROOT", "")
        root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
        path = root / path
    return path.resolve()


def _scope_key_raw(identity: str, perspective: str, generation: int) -> str:
    raw = "|".join((identity, perspective, str(generation)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _generation_for(payload: dict[str, Any], identity: str) -> int:
    return int(payload.get("reset_generations", {}).get(identity, 0) or 0)


def _runtime_perspective(config: dict[str, Any]) -> str:
    runtime = config.get("_runtime", {})
    return str(runtime.get("perspective_signature", "")) if isinstance(runtime, dict) else ""


def scope_key(config: dict[str, Any]) -> str:
    identity = _scope_identity(config)
    with _LOCK:
        generation = _generation_for(_load(_usage_path(config)), identity)
    return _scope_key_raw(identity, _runtime_perspective(config), generation)


def _scope_identity(config: dict[str, Any]) -> str:
    """返回不含透视参数的设备-App 标识，供手工重置整组视觉状态使用。"""
    runtime = config.get("_runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    return "|".join(
        (
            f"{platform.node()}:{uuid.getnode()}",
            str(runtime.get("device_ip", "")),
            str(runtime.get("app_name", "xinyu")),
        )
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "scopes": {}}
    return value if isinstance(value, dict) and isinstance(value.get("scopes"), dict) else {"schema_version": 1, "scopes": {}}


def _write(path: Path, payload: dict[str, Any]) -> None:
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


def acquire(config: dict[str, Any], *, limit: int) -> tuple[bool, int]:
    """原子地占用一次调用额度；进程重启后仍延续计数。"""
    path = _usage_path(config)
    identity = _scope_identity(config)
    perspective = _runtime_perspective(config)
    with _LOCK:
        # 锁内单次读盘同时取得代次与计数（原先 scope_key + acquire 各读一次）。
        payload = _load(path)
        key = _scope_key_raw(identity, perspective, _generation_for(payload, identity))
        scopes = payload["scopes"]
        current = max(0, int(scopes.get(key, 0) or 0))
        if current >= limit:
            return False, current
        scopes[key] = current + 1
        _write(path, payload)
        return True, current + 1


def rule_learning_status(config: dict[str, Any]) -> dict[str, Any]:
    """读取当前设备/App/透视的连续无新增规则状态。"""
    path = _usage_path(config)
    identity = _scope_identity(config)
    with _LOCK:
        payload = _load(path)
        key = _scope_key_raw(identity, _runtime_perspective(config), _generation_for(payload, identity))
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
    identity = _scope_identity(config)
    threshold = max(1, int(disable_after_zero or 3))
    with _LOCK:
        payload = _load(path)
        key = _scope_key_raw(identity, _runtime_perspective(config), _generation_for(payload, identity))
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
        _write(path, payload)
        return dict(state)


def reset(config: dict[str, Any]) -> None:
    """清空当前设备-App 的视觉调用额度及规则学习停用状态。

    透视签名由设备四点参数决定，主界面在未启动设备时并不一定能取得它。
    因此使用递增代次使这个设备-App 的所有旧透视状态同时失效，其他设备和
    其他 App 的状态保持不变。
    """
    path = _usage_path(config)
    identity = _scope_identity(config)
    with _LOCK:
        payload = _load(path)
        generations = payload.setdefault("reset_generations", {})
        generations[identity] = max(0, int(generations.get(identity, 0) or 0)) + 1
        _write(path, payload)
