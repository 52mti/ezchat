"""监控墙数据源：worker 每次截图请求的最近一帧（docs/13 主UI 扩展，2026-09-13）。

photo_capture.capture_transform_image 是 worker 取图的必经通道，成功即发布到这里；
按设备 IP 限频（默认 1 秒/帧）防止刷爆 UI；UI 侧定时拉取 snapshots()。
只服务于监控展示，不落盘、不进事件队列。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterable

_LOCK = threading.Lock()
_LATEST: dict[str, dict[str, Any]] = {}  # device_ip -> {"app", "image", "label", "ts"}
MIN_INTERVAL_SECONDS = 1.0
# 订阅者：worker 截图发布成功后立即回调（回调发生在 worker 线程，
# 订阅方自行负责跨线程投递，如 Qt 信号）。监控墙由此从"定时轮询"
# 改为"截图即达"。
_LISTENERS: list = []


def add_capture_listener(callback) -> None:
    with _LOCK:
        _LISTENERS.append(callback)


def publish_capture(device_ip: str, app_name: str, image_bytes: bytes, *, label: str = "") -> None:
    ip = str(device_ip or "").strip()
    if not ip or not image_bytes:
        return
    now = time.time()
    with _LOCK:
        prev = _LATEST.get(ip)
        if prev is not None and now - float(prev.get("ts", 0.0)) < MIN_INTERVAL_SECONDS:
            return
        _LATEST[ip] = {
            "app": str(app_name or ""),
            "image": bytes(image_bytes),
            "label": str(label or ""),
            "ts": now,
        }
        listeners = list(_LISTENERS)
    for callback in listeners:
        try:
            callback(ip)
        except Exception:
            continue


def snapshots() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {ip: dict(data) for ip, data in _LATEST.items()}


def retain_devices(device_ips: Iterable[str]) -> None:
    """只保留设备列表中仍存在的设备，防止已删设备的画面残留。"""
    keep = {str(ip).strip() for ip in device_ips if str(ip or "").strip()}
    with _LOCK:
        for ip in list(_LATEST):
            if ip not in keep:
                _LATEST.pop(ip, None)


def reset_for_tests() -> None:
    with _LOCK:
        _LATEST.clear()
        _LISTENERS.clear()
