"""本机控制台：从原 PyQt 主窗口抽出的设备启停、扫描、配置和事件消费。

不创建窗口，也不改 DeviceWorker。页面和 HTTP 只调用这里。
"""

from __future__ import annotations

import copy
import io
import os
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from apps import get_default_app_name, get_registered_apps, resolve_app_name
from apps.visual_assets import machine_template_id
from services.config_manager import ConfigManager
from services.device_backend_proxy import DeviceBackendProxyManager
from services.license_client import LicenseClient
from services.local_tts import LocalTtsServer
from services.update_release import ReleaseDecision, ReleaseManifestError, resolve_release_decision
from services.update_sources import UpdateSourceError, resolve_lanzou_update_package
from workers.device_worker import DeviceWorker, WorkerEvent, is_stale_worker_event


STATUS_COORDINATE_RE = re.compile(r"\s*[：:,-]?\s*\(?\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*\)?\s*$")
UI_EVENT_DRAIN_LIMIT = 120
UI_EVENT_DRAIN_SECONDS = 0.02
UNATTENDED_REFRESH_SECONDS = 1.0
LOG_BUFFER_LIMIT = 1000
UPDATE_DOWNLOAD_CHUNK_SIZE = 1024 * 256
INSTANCE_LOCK_PATH = Path("runtime") / "main_ui.lock"

APP_SETTING_KEYS = (
    "ocr",
    "ocr_modes",
    "templates",
    "match_threshold",
    "popup_match_threshold",
    "template_match_regions",
    "template_match_thresholds",
)
RUNTIME_SETTING_KEYS = (
    "poll_interval",
    "step_timeout",
    "keep_awake_interval_seconds",
    "exclude_ocr_wait_from_step_timeout",
)
DEVICE_APP_SETTING_KEYS = (
    "photo_sent_cooldown_seconds",
    "online_photo_send_count",
    "persona_profile_json",
    "greetings_txt",
    "judge_enabled",
    "greeting_ai_enabled",
    "history_turns",
    "summary_interval",
)
PERSONA_TEXT_KEYS = ("基础设定", "人物信息", "边界条件", "审核标准")


def mask_secret(value: str) -> str:
    if not value:
        return ""
    return "****" + value[-4:] if len(value) > 4 else "****"


def strip_status_coordinates(detail: str) -> str:
    cleaned = STATUS_COORDINATE_RE.sub("", str(detail or "").strip()).strip()
    return cleaned.rstrip("：:,，- ").strip()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class InstanceLock:
    """占住 runtime/main_ui.lock，避免两个进程各起一套值守线程。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or INSTANCE_LOCK_PATH
        self._handle: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


@dataclass
class DeviceView:
    status: str = "stopped"
    detail: str = ""
    health: str = "unknown"
    balance: str = ""
    average_reply_seconds: float | None = None
    reply_round_count: int = 0
    unattended_started_at: float | None = None
    unattended_elapsed: float | None = None
    notice: str = ""


@dataclass
class ConsoleApp:
    config_manager: ConfigManager
    events: queue.Queue = field(default_factory=queue.Queue)
    update_events: queue.Queue = field(default_factory=queue.Queue)
    workers: dict[str, Any] = field(default_factory=dict)
    views: dict[str, DeviceView] = field(default_factory=dict)
    logs: list[tuple[str, str]] = field(default_factory=list)
    local_tts_server: LocalTtsServer | None = None
    device_backend_proxies: DeviceBackendProxyManager = field(default_factory=DeviceBackendProxyManager)
    worker_factory: Callable[..., Any] | None = None
    scan_func: Callable[..., list[str]] | None = None
    license_client_factory: Callable[..., LicenseClient] | None = None
    package_root: Path | None = None
    template_root: Path | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _listeners: list[queue.Queue] = field(default_factory=list)
    _drain_stop: threading.Event = field(default_factory=threading.Event)
    _drain_thread: threading.Thread | None = None
    _scan_running: bool = False
    _scan_message: str = ""
    _tts_bind_lock: threading.Lock = field(default_factory=threading.Lock)
    _tts_bind_in_progress: set[str] = field(default_factory=set)
    _last_unattended_refresh_at: float = 0.0
    _update_check_in_progress: bool = False
    _download_cancel: threading.Event | None = None
    update_state: dict[str, Any] = field(default_factory=dict)
    calibration_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    screenshots: dict[str, bytes] = field(default_factory=dict)
    on_exit: Callable[[], None] | None = None
    _decision: ReleaseDecision | None = None

    def __post_init__(self) -> None:
        if self.local_tts_server is None:
            self.local_tts_server = LocalTtsServer()
        self.refresh_rows()

    def start(self) -> None:
        if self._drain_thread and self._drain_thread.is_alive():
            return
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, name="console-events", daemon=True)
        self._drain_thread.start()

    def shutdown(self) -> None:
        self._drain_stop.set()
        for worker in list(self.workers.values()):
            stop = getattr(worker, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        if self.local_tts_server is not None:
            try:
                self.local_tts_server.stop()
            except Exception:
                pass
        try:
            self.device_backend_proxies.close_all()
        except Exception:
            pass

    def subscribe(self) -> queue.Queue:
        listener: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: queue.Queue) -> None:
        with self._lock:
            self._listeners = [item for item in self._listeners if item is not listener]

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener.put_nowait(payload)
            except queue.Full:
                try:
                    listener.get_nowait()
                except queue.Empty:
                    pass
                try:
                    listener.put_nowait(payload)
                except queue.Full:
                    pass

    def _drain_loop(self) -> None:
        while not self._drain_stop.wait(0.1):
            changed = self.drain_once()
            if changed:
                self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})

    def drain_once(self) -> bool:
        started_at = time.monotonic()
        processed = 0
        changed = False
        while processed < UI_EVENT_DRAIN_LIMIT:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            changed = True
            try:
                self.handle_worker_event(event)
            except Exception as exc:
                device_ip = str(getattr(event, "device_ip", "unknown") or "unknown")
                self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: 事件处理失败：{exc}")
            if time.monotonic() - started_at >= UI_EVENT_DRAIN_SECONDS:
                break
        now = time.monotonic()
        if now - self._last_unattended_refresh_at >= UNATTENDED_REFRESH_SECONDS:
            self._last_unattended_refresh_at = now
            if self._refresh_unattended():
                changed = True
        if self._drain_update_events():
            changed = True
        return changed

    def _drain_update_events(self) -> bool:
        changed = False
        processed = 0
        while processed < UI_EVENT_DRAIN_LIMIT:
            try:
                event_type, payload = self.update_events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            changed = True
            self._handle_update_event(str(event_type), payload if isinstance(payload, dict) else {})
        return changed

    def handle_worker_event(self, event: WorkerEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        event_ip = str(event.device_ip or "")
        message = str(event.message or "")
        level = str(event.level or "")
        if message == "scan_complete":
            devices = [str(ip) for ip in payload.get("devices", []) if str(ip).strip()]
            self._finish_device_scan(devices)
            return
        if message == "scan_error":
            self._finish_device_scan([], error=str(payload.get("error", "") or "扫描失败"))
            return
        if is_stale_worker_event(event, self.workers.get(event_ip)):
            return
        if level != "status":
            self.append_log(event_ip, f"[{time.strftime('%H:%M:%S')}] [{level}] {event_ip}: {message}")
        if level == "status":
            view = self._view(event_ip)
            if view.status in {"stopping", "stopped"}:
                worker = self.workers.get(event_ip)
                if worker is None or not worker.is_alive() or getattr(worker, "stop_requested", False):
                    return
            is_open = bool(payload.get("is_circuit_open"))
            current_step = str(payload.get("current_step", "") or "")
            detail = strip_status_coordinates(str(payload.get("current_step_label", "") or message))
            view.reply_round_count = int(payload.get("reply_round_count") or 0)
            average = payload.get("average_reply_seconds")
            view.average_reply_seconds = None if average in (None, "") else float(average)
            if "phone_screen_health" in payload:
                view.health = str(payload.get("phone_screen_health", "unknown") or "unknown")
            if is_open:
                view.status = "circuit_open"
            elif current_step.startswith(("black_screen", "no_phone")):
                view.status = "abnormal"
            else:
                view.status = "running"
            view.detail = detail
        elif message in {"paused", "stopped", "stopping"}:
            if message == "stopped":
                self._finalize_unattended(event_ip)
            self._view(event_ip).status = message
            self._view(event_ip).detail = str(payload.get("detail", "") or "")
        elif message == "xinyu_balance":
            self._view(event_ip).balance = str(payload.get("balance", "") or "")

    def _handle_update_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "license_verified":
            self._finish_license_verify(payload)
            return
        if event_type in {"check_complete", "check_error"}:
            self._update_check_in_progress = False
        if event_type == "check_error":
            self.update_state = {"phase": "error", "message": str(payload.get("message", "") or "检查更新失败")}
            self.publish({"type": "update", "update": self.update_state})
            return
        if event_type == "check_complete":
            decision = payload.get("decision")
            if not isinstance(decision, ReleaseDecision):
                return
            self._decision = decision
            self.update_state = {
                "phase": "ready" if decision.action != "none" else "current",
                "action": decision.action,
                "current_app_version": decision.current_app_version,
                "target_app_version": decision.target_app_version,
                "reason": decision.reason,
                "release_notes": decision.release_notes,
                "message": "当前已经是最新版本。" if decision.action == "none" else "发现可更新版本。",
            }
            self.publish({"type": "update", "update": self.update_state})
            return
        if event_type in {"download_resolved", "download_started", "download_progress"}:
            progress = dict(self.update_state)
            progress["phase"] = "downloading"
            progress.update({key: payload.get(key) for key in ("file_name", "size", "downloaded", "total") if key in payload})
            self.update_state = progress
            self.publish({"type": "update", "update": self.update_state})
            return
        if event_type == "download_cancelled":
            self.update_state = {**self.update_state, "phase": "cancelled", "message": "已取消本次更新下载。"}
            self.publish({"type": "update", "update": self.update_state})
            return
        if event_type == "download_error":
            self.update_state = {"phase": "error", "message": str(payload.get("message", "") or "下载更新失败")}
            self.publish({"type": "update", "update": self.update_state})
            return
        if event_type == "download_complete":
            archive_path = str(payload.get("archive_path", "") or "")
            self.update_state = {**self.update_state, "phase": "downloaded", "archive_path": archive_path, "message": "更新包已下载完成。"}
            self.publish({"type": "update", "update": self.update_state})

    def list_devices(self) -> list[dict[str, Any]]:
        rows = []
        for device in self.config_manager.devices():
            if not isinstance(device, dict):
                continue
            ip = str(device.get("ip", "") or "")
            view = self.views.get(ip, DeviceView())
            seconds = self._unattended_seconds(ip)
            rows.append(
                {
                    "ip": ip,
                    "selected_app": resolve_app_name(device.get("selected_app")),
                    "license_masked": mask_secret(str(device.get("license_key", "") or "")),
                    "license_expires": str(device.get("license_expires", "") or ""),
                    "status": view.status,
                    "detail": view.detail,
                    "health": view.health,
                    "balance": view.balance or "--",
                    "average_reply_seconds": view.average_reply_seconds,
                    "reply_round_count": view.reply_round_count,
                    "unattended_seconds": None if seconds is None else max(0.0, float(seconds)),
                    "unattended_label": format_duration(seconds),
                    "notice": view.notice,
                }
            )
        return rows

    def status_snapshot(self) -> dict[str, Any]:
        devices = self.list_devices()
        running = sum(1 for item in devices if item["status"] == "running")
        alerts = sum(1 for item in devices if item["status"] in {"circuit_open", "abnormal", "license_error"})
        return {
            "device_count": len(devices),
            "running_count": running,
            "alert_count": alerts,
            "scan_running": self._scan_running,
            "scan_message": self._scan_message,
            "ocr_mode": self._ocr_mode_label(),
            "ocr_waiting": self._ocr_waiting(),
            "pool_size": self._pool_size(),
            "apps": [
                {"name": name, "label": getattr(app_cls, "name", name)}
                for name, app_cls in get_registered_apps().items()
            ],
        }

    def refresh_rows(self) -> None:
        """增删设备后按存活 worker 恢复行状态，避免活线程被标成已停止。"""
        known = {str(device.get("ip", "")) for device in self.config_manager.devices() if isinstance(device, dict)}
        for ip in list(self.views):
            if ip not in known:
                self.views.pop(ip, None)
        for ip in known:
            view = self.views.setdefault(ip, DeviceView())
            view.status = self._restored_row_status(ip)

    def _restored_row_status(self, device_ip: str) -> str:
        worker = self.workers.get(device_ip)
        if worker is None or not worker.is_alive():
            return "stopped"
        if getattr(worker, "stop_requested", False):
            return "stopping"
        if getattr(worker, "is_paused", False):
            return "paused"
        return "running"

    def _view(self, device_ip: str) -> DeviceView:
        return self.views.setdefault(device_ip, DeviceView())

    def append_log(self, device_ip: str, line: str) -> None:
        self.logs.append((device_ip, line))
        if len(self.logs) > LOG_BUFFER_LIMIT:
            del self.logs[: len(self.logs) - LOG_BUFFER_LIMIT]
        view = self.views.get(device_ip)
        if view is not None and "[warning]" in line:
            view.notice = line
        self.publish({"type": "log", "device_ip": device_ip, "line": line})

    def logs_for(self, device_ip: str = "") -> list[str]:
        selected = str(device_ip or "")
        return [line for event_ip, line in self.logs if not selected or event_ip == selected]

    def add_device(self, device_ip: str) -> bool:
        created = self.config_manager.add_device(self._new_device(device_ip))
        if created:
            self.refresh_rows()
            self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})
        return created

    def delete_device(self, device_ip: str) -> bool:
        worker = self.workers.get(device_ip)
        if worker is not None:
            stop = getattr(worker, "stop", None)
            if callable(stop):
                stop()
            self.workers.pop(device_ip, None)
        view = self.views.get(device_ip)
        if view is not None:
            view.unattended_started_at = None
        removed = self.config_manager.remove_device(device_ip)
        if removed:
            self.refresh_rows()
            self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: device removed")
            self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})
        return removed

    def set_selected_app(self, device_ip: str, app_name: str) -> None:
        worker = self.workers.get(device_ip)
        alive = worker is not None and bool(getattr(worker, "is_alive", lambda: False)())
        if alive and not bool(getattr(worker, "stop_requested", False)):
            raise ValueError("设备正在运行，停止后再切换 App")
        self.config_manager.update_device(device_ip, {"selected_app": resolve_app_name(app_name)})

    def start_device(self, device_ip: str) -> dict[str, Any]:
        device = self.config_manager.find_device(device_ip)
        if not device:
            return {"ok": False, "error": "设备不存在"}
        existing = self.workers.get(device_ip)
        view = self._view(device_ip)
        if existing is not None and existing.is_alive():
            if view.status == "paused":
                resume = getattr(existing, "resume", None)
                if callable(resume):
                    resume()
                view.status = "running"
                view.notice = ""
                return {"ok": True, "status": "running"}
            if getattr(existing, "stop_requested", False) or view.status == "stopping":
                self.append_log(
                    device_ip,
                    f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: 旧进程正在退出，请等状态变为已停止后再启动",
                )
                return {"ok": False, "error": "旧进程正在退出，请等状态变为已停止后再启动"}
            return {"ok": True, "status": view.status or "running"}
        selected_app = resolve_app_name(device.get("selected_app"))
        device["selected_app"] = selected_app
        self.config_manager.update_device(device_ip, {"selected_app": selected_app})
        self._start_and_bind_local_tts([device_ip])
        factory = self.worker_factory or DeviceWorker
        worker = factory(device, events=self.events, config_manager=self.config_manager)
        self.workers[device_ip] = worker
        worker.start()
        view.unattended_started_at = time.monotonic()
        view.unattended_elapsed = 0.0
        view.status = "running"
        view.notice = ""
        self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})
        return {"ok": True, "status": "running"}

    def stop_device(self, device_ip: str) -> dict[str, Any]:
        worker = self.workers.get(device_ip)
        if worker is None:
            self._view(device_ip).status = "stopped"
            return {"ok": True, "status": "stopped"}
        worker.stop()
        self._view(device_ip).status = "stopping"
        self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})
        return {"ok": True, "status": "stopping"}

    def scan_devices(self) -> dict[str, Any]:
        if self._scan_running:
            return {"ok": True, "running": True}
        self._scan_running = True
        self._scan_message = "正在扫描同一网段的控制盒…"

        def scan() -> None:
            try:
                finder = self.scan_func
                if finder is None:
                    from lgb.lgb import scan_devices as finder
                found = finder(timeout=0.3, max_workers=50)
            except Exception as exc:
                self.events.put(WorkerEvent("", "warning", "scan_error", {"error": str(exc)}))
                return
            self.events.put(WorkerEvent("", "info", "scan_complete", {"devices": list(found)}))

        threading.Thread(target=scan, name="device-scan", daemon=True).start()
        return {"ok": True, "running": True}

    def _finish_device_scan(self, devices: list[str], *, error: str = "") -> None:
        self._scan_running = False
        if error:
            self._scan_message = error
            self.publish({"type": "scan", "ok": False, "message": error})
            return
        added = 0
        for device_ip in devices:
            if self.config_manager.add_device(self._new_device(device_ip)):
                added += 1
        self.refresh_rows()
        self._scan_message = f"发现 {len(devices)} 台，新增 {added} 台"
        self.publish(
            {
                "type": "scan",
                "ok": True,
                "found": len(devices),
                "added": added,
                "message": self._scan_message,
                "devices": self.list_devices(),
                "status": self.status_snapshot(),
            }
        )

    def _new_device(self, device_ip: str) -> dict[str, Any]:
        return {
            "ip": str(device_ip).strip(),
            "license_key": "",
            "license_expires": "",
            "selected_app": get_default_app_name(),
            "persona": {},
            "app_overrides": {app_name: {} for app_name in get_registered_apps()},
        }

    def _start_and_bind_local_tts(self, device_ips: list[str] | None = None) -> None:
        start_only = device_ips is None
        targets = list(dict.fromkeys(str(ip).strip() for ip in (device_ips or []) if str(ip).strip()))
        with self._tts_bind_lock:
            new_targets = [ip for ip in targets if ip not in self._tts_bind_in_progress]
            self._tts_bind_in_progress.update(new_targets)
        if not new_targets and not start_only:
            return
        server = self.local_tts_server
        if server is None:
            return

        def bind() -> None:
            try:
                if start_only:
                    server.start()
                    self.events.put(WorkerEvent("本地TTS", "info", "本地 TTS 服务已启动"))
                    return
                results = server.bind_devices(new_targets)
                for device_ip, tts_url, error in results:
                    if error:
                        self.events.put(WorkerEvent(device_ip, "warning", f"本地 TTS 绑定失败：{error}"))
                    else:
                        self.events.put(WorkerEvent(device_ip, "info", f"本地 TTS 已绑定：{tts_url}"))
            except Exception as exc:
                self.events.put(WorkerEvent("本地TTS", "warning", f"本地 TTS 服务启动失败：{exc}"))
            finally:
                with self._tts_bind_lock:
                    self._tts_bind_in_progress.difference_update(new_targets)

        threading.Thread(target=bind, name="local-tts-bind", daemon=True).start()

    def bind_license(self, device_ip: str, license_key: str) -> dict[str, Any]:
        device = self.config_manager.find_device(device_ip)
        if not device:
            return {"ok": False, "error": "设备不存在"}
        license_key = license_key.strip()
        if not license_key:
            return {"ok": False, "error": "请先输入卡密"}
        for other in self.config_manager.devices():
            if other.get("ip") != device_ip and other.get("license_key") == license_key:
                return {"ok": False, "error": "卡密已被其他设备使用"}
        global_config = self.config_manager.get("global", default={})
        if not isinstance(global_config, dict):
            global_config = {}
        factory = self.license_client_factory or LicenseClient
        client = factory(
            api_url=str(global_config.get("license_api_url", "")),
            project_secret=str(global_config.get("project_secret", "")),
        )
        self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: 卡密校验中…")

        def verify() -> None:
            try:
                result = client.verify_with_offline(
                    device_ip,
                    license_key,
                    str(device.get("license_expires", "")),
                    machine_id=machine_template_id(),
                )
            except Exception as exc:
                self.update_events.put(
                    ("license_verified", {"device_ip": device_ip, "license_key": license_key, "error": str(exc)})
                )
                return
            self.update_events.put(
                (
                    "license_verified",
                    {
                        "device_ip": device_ip,
                        "license_key": license_key,
                        "ok": bool(result.ok),
                        "message": str(result.message),
                        "expires_at": str(result.expires_at or ""),
                        "status": str(result.status or ""),
                    },
                )
            )

        threading.Thread(target=verify, name="license-verify", daemon=True).start()
        return {"ok": True, "pending": True}

    def _finish_license_verify(self, payload: dict[str, Any]) -> None:
        device_ip = str(payload.get("device_ip", "") or "")
        license_key = str(payload.get("license_key", "") or "")
        view = self._view(device_ip)
        if payload.get("error"):
            view.status = "license_error"
            view.notice = f"校验异常：{payload['error']}"
            self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: {view.notice}")
            return
        if not payload.get("ok"):
            view.status = "license_error"
            view.notice = str(payload.get("message", "") or "校验失败")
            self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: {view.notice}")
            return
        self.config_manager.update_device(
            device_ip,
            {"license_key": license_key, "license_expires": str(payload.get("expires_at", "") or "")},
        )
        status_value = str(payload.get("status", "") or "")
        view.status = "normal" if status_value == "ok" else (status_value or "normal")
        view.notice = "卡密已绑定"
        self.publish({"type": "devices", "devices": self.list_devices(), "status": self.status_snapshot()})

    def set_pool_size(self, value: int) -> int:
        size = min(16, max(1, int(value or 2)))
        self.config_manager.set("global", "ocr_engine_pool_size", value=size)
        return size

    def get_settings(self, device_ip: str, app_name: str | None = None) -> dict[str, Any]:
        device = self.config_manager.find_device(device_ip) or {}
        app = resolve_app_name(app_name or device.get("selected_app"))
        merged = self.config_manager.get_app_config(device_ip, app, {})
        persona_path = str(merged.get("persona_profile_json", "") or "")
        texts = self._read_persona_texts(persona_path)
        if not any(texts.values()):
            texts = self._read_persona_texts(str((self.template_root or Path("apps")) / app / "persona_default.json"))
        return {
            "device_ip": device_ip,
            "app_name": app,
            "runtime": {key: merged.get(key) for key in RUNTIME_SETTING_KEYS},
            "device_app": {key: merged.get(key) for key in DEVICE_APP_SETTING_KEYS},
            "app": {key: merged.get(key) for key in APP_SETTING_KEYS if key in merged},
            "persona_texts": texts,
            "persona_chat_model": str(self.config_manager.get("global", "persona_chat_model", default="") or ""),
            "has_api_key": self._has_api_key(),
        }

    def save_settings(self, device_ip: str, app_name: str, values: dict[str, Any]) -> dict[str, Any]:
        app = resolve_app_name(app_name)
        payload = dict(values)
        texts = payload.pop("persona_texts", None)
        if isinstance(texts, dict):
            payload["persona_profile_json"] = self._write_persona_texts(
                app,
                str(payload.get("persona_profile_json", "") or ""),
                texts,
            )
        app_updates = {key: copy.deepcopy(payload[key]) for key in APP_SETTING_KEYS if key in payload}
        runtime_updates = {key: copy.deepcopy(payload[key]) for key in RUNTIME_SETTING_KEYS if key in payload}
        device_app_updates = {key: copy.deepcopy(payload[key]) for key in DEVICE_APP_SETTING_KEYS if key in payload}
        if app_updates:
            self.config_manager.update_app_config(app, app_updates)
        if runtime_updates:
            self.config_manager.update_device_runtime(device_ip, runtime_updates)
        if device_app_updates:
            self.config_manager.update_device_app_overrides(device_ip, app, device_app_updates)
        model = str(payload.get("persona_chat_model", "") or "").strip()
        if model:
            self.config_manager.update_global("persona_chat_model", model)
        api_key = str(payload.get("deepseek_api_key", "") or "").strip()
        if api_key:
            from services.deepseek_client import write_local_api_key

            write_local_api_key(None, api_key)
        return self.get_settings(device_ip, app)

    def reset_vision(self, device_ip: str, app_name: str) -> None:
        from services.vision_usage import reset as reset_vision_usage

        app = resolve_app_name(app_name)
        reset_vision_usage({"_runtime": {"device_ip": device_ip, "app_name": app}})
        self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: {app} 视觉识别已重置")

    def open_backend(self, device_ip: str) -> str:
        return self.device_backend_proxies.open_url(device_ip)

    def capture_screenshot(self, device_ip: str) -> bytes:
        image = self._capture_image(device_ip)
        data = _image_jpeg(image)
        self.screenshots[device_ip] = data
        return data

    def start_calibration(self, device_ip: str, app_name: str | None = None) -> dict[str, Any]:
        device = self.config_manager.find_device(device_ip)
        if not device:
            return {"ok": False, "error": "设备不存在"}
        app = resolve_app_name(app_name or device.get("selected_app"))
        current = self.calibration_state.get(device_ip) or {}
        if current.get("phase") == "running":
            return {"ok": False, "error": "上一次标定还在执行，请稍候。"}
        self.calibration_state[device_ip] = {"phase": "running", "app_name": app}
        self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: 开始视觉+OCR标定...")

        def run() -> None:
            try:
                from services.defaults import with_app_defaults

                app_config = with_app_defaults(self.config_manager.get_app_config(device_ip, app, {}))
                app_config["_runtime"] = {"device_ip": device_ip, "app_name": app}
                app_config["_runtime_cache"] = {}
                image = self._capture_image(device_ip, app_config=app_config, label="vision_calibration")
                portrait = _ensure_portrait(image)
                jpeg = _image_jpeg(portrait)
                self.screenshots[device_ip] = jpeg
                from apps.xinyu import steps as xinyu_steps
                from lgb.lgb import RobotInterface

                result = xinyu_steps.calibrate_dialogue_chat_area(
                    RobotInterface(device_ip),
                    app_config,
                    image=portrait,
                )
                public = {key: value for key, value in result.items() if key != "image"}
                self.calibration_state[device_ip] = {
                    "phase": "done",
                    "app_name": app,
                    "image_ready": True,
                    "result": _json_ready(public),
                }
                if result.get("ok"):
                    self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: 视觉标定完成")
                else:
                    self.append_log(
                        device_ip,
                        f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: 视觉标定未更新：{result.get('reason', '')}",
                    )
            except Exception as exc:
                self.calibration_state[device_ip] = {"phase": "error", "app_name": app, "error": str(exc)}
                self.append_log(device_ip, f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: 视觉标定失败：{exc}")
            self.publish({"type": "calibration", "device_ip": device_ip, "calibration": self.calibration_state[device_ip]})

        threading.Thread(target=run, name="vision-calibration", daemon=True).start()
        return {"ok": True, "phase": "running"}

    def list_templates(self, app_name: str) -> list[dict[str, str]]:
        app = resolve_app_name(app_name)
        items: list[dict[str, str]] = []
        for group in ("buttons", "screens"):
            root = self._template_dir(app, group)
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file():
                    items.append({"group": group, "name": path.name, "path": str(path)})
        return items

    def save_template(
        self,
        app_name: str,
        group: str,
        name: str,
        image_bytes: bytes,
        region: list[float] | None = None,
    ) -> dict[str, Any]:
        app = resolve_app_name(app_name)
        if group not in {"buttons", "screens"}:
            raise ValueError("模板分组只能是 buttons 或 screens")
        if not image_bytes:
            raise ValueError("没有模板图片")
        safe = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(name or "").strip()).strip("._") or "template"
        if not safe.lower().endswith((".jpg", ".jpeg", ".png")):
            safe += ".jpg"
        root = self._template_dir(app, group)
        root.mkdir(parents=True, exist_ok=True)
        path = root / safe
        path.write_bytes(image_bytes)
        saved_region = None
        if region and len(region) == 4:
            stem = path.stem
            key = f"popup.{stem}" if group == "buttons" else f"screen.{stem}"
            saved_region = [float(item) for item in region]
            self.config_manager.update_app_config(app, {"template_match_regions": {key: saved_region}})
        return {"path": str(path), "name": path.name, "group": group, "region": saved_region}

    def chat_contacts(self, device_ip: str, app_name: str | None = None) -> list[dict[str, Any]]:
        from services.chat_history_store import ChatHistoryStore

        app = resolve_app_name(app_name or (self.config_manager.find_device(device_ip) or {}).get("selected_app"))
        return ChatHistoryStore().list_contacts(device_ip=device_ip, app_name=app)

    def chat_messages(self, device_ip: str, app_name: str | None, contact_name: str) -> list[dict[str, Any]]:
        from services.chat_history_store import ChatHistoryStore

        app = resolve_app_name(app_name or (self.config_manager.find_device(device_ip) or {}).get("selected_app"))
        return ChatHistoryStore().get_messages(device_ip=device_ip, app_name=app, contact_name=contact_name)

    def monitor_meta(self) -> list[dict[str, Any]]:
        from services.monitor_feed import snapshots

        frames = snapshots()
        items = []
        for device in self.list_devices():
            frame = frames.get(device["ip"])
            items.append(
                {
                    "ip": device["ip"],
                    "status": device["status"],
                    "has_image": bool(frame and frame.get("image")),
                    "label": str(frame.get("label", "") if frame else ""),
                    "ts": frame.get("ts") if frame else None,
                }
            )
        return items

    def monitor_image(self, device_ip: str) -> bytes | None:
        from services.monitor_feed import snapshots

        frame = snapshots().get(device_ip)
        if not frame:
            return self.screenshots.get(device_ip)
        image = frame.get("image")
        return bytes(image) if image else None

    def check_update(self) -> dict[str, Any]:
        if self._update_check_in_progress:
            return {"ok": True, "running": True}
        self._update_check_in_progress = True
        root = self._package_root()

        def run() -> None:
            try:
                decision = resolve_release_decision(package_root=root)
            except ReleaseManifestError as exc:
                self.update_events.put(("check_error", {"message": str(exc)}))
                return
            self.update_events.put(("check_complete", {"decision": decision}))

        threading.Thread(target=run, name="release-check", daemon=True).start()
        return {"ok": True, "running": True}

    def download_update(self) -> dict[str, Any]:
        decision = self._decision
        if decision is None or decision.action == "none":
            return {"ok": False, "error": "没有可下载的更新"}
        if self.update_state.get("phase") == "downloading":
            return {"ok": True, "running": True}
        cancel_event = threading.Event()
        self._download_cancel = cancel_event
        threading.Thread(
            target=self._run_update_download,
            kwargs={"decision": decision, "cancel_event": cancel_event},
            name="release-download",
            daemon=True,
        ).start()
        return {"ok": True, "running": True}

    def cancel_download(self) -> None:
        if self._download_cancel is not None:
            self._download_cancel.set()

    def apply_update(self) -> dict[str, Any]:
        archive_path = str(self.update_state.get("archive_path", "") or "").strip()
        if not archive_path:
            return {"ok": False, "error": "还没有下载完成的更新包"}
        root = self._package_root()
        updater = root / ("update_apply.pyc" if os.name == "nt" else "update_apply.py")
        if not updater.is_file():
            updater = root / "update_apply.py"
        if not updater.is_file():
            return {"ok": False, "error": f"缺少更新脚本：{updater}"}
        command = self._update_apply_command(root=root, updater=updater, archive_path=archive_path)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen(command, cwd=root, creationflags=creationflags)
        except OSError as exc:
            return {"ok": False, "error": f"无法启动更新程序：{exc}"}
        if self.local_tts_server is not None:
            try:
                self.local_tts_server.stop()
            except Exception:
                pass
        if self.on_exit is not None:
            self.on_exit()
        return {"ok": True}

    def _run_update_download(self, *, decision: ReleaseDecision, cancel_event: threading.Event) -> None:
        target_path: Path | None = None
        try:
            with requests.Session() as http:
                resolved = None
                if decision.update_url:
                    try:
                        response = http.head(decision.update_url, timeout=30, allow_redirects=True)
                        response.raise_for_status()
                        resolved = type("Resolved", (), {})()
                        resolved.file_name = decision.update_file_name or Path(decision.update_url).name
                        resolved.size = response.headers.get("Content-Length", "")
                        resolved.download_url = decision.update_url
                        resolved.detail_url = decision.update_url
                    except (requests.RequestException, OSError):
                        resolved = None
                if resolved is None:
                    resolved = resolve_lanzou_update_package(
                        share_url=decision.update_share_url,
                        file_name=decision.update_file_name,
                        session=http,
                    )
                self.update_events.put(("download_resolved", {"file_name": resolved.file_name, "size": resolved.size}))
                target_dir = self._package_root() / "runtime" / "tmp" / "updates"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / resolved.file_name
                with http.get(
                    resolved.download_url,
                    stream=True,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": resolved.detail_url},
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length", "0") or "0")
                    self.update_events.put(("download_started", {"file_name": resolved.file_name, "total": total}))
                    downloaded = 0
                    with target_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=UPDATE_DOWNLOAD_CHUNK_SIZE):
                            if cancel_event.is_set():
                                raise RuntimeError("cancelled")
                            if not chunk:
                                continue
                            handle.write(chunk)
                            downloaded += len(chunk)
                            self.update_events.put(("download_progress", {"downloaded": downloaded, "total": total}))
                if not zipfile.is_zipfile(target_path):
                    raise UpdateSourceError("下载到的更新包不是有效的 zip 文件。")
            self.update_events.put(("download_complete", {"archive_path": str(target_path)}))
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                if target_path is not None:
                    target_path.unlink(missing_ok=True)
                self.update_events.put(("download_cancelled", {}))
                return
            self.update_events.put(("download_error", {"message": str(exc)}))
        except (UpdateSourceError, requests.RequestException, OSError) as exc:
            if target_path is not None:
                target_path.unlink(missing_ok=True)
            self.update_events.put(("download_error", {"message": str(exc)}))

    def _update_apply_command(self, *, root: Path, updater: Path, archive_path: str) -> list[str]:
        if os.name != "nt":
            return [sys.executable, "-s", str(updater), "--package-root", str(root), "--archive", archive_path, "--pid", str(os.getpid())]
        from update_orchestration import write_orchestration_script

        script_path = write_orchestration_script(Path(os.environ.get("TEMP", ".")) / "jxb_update_apply")
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-PackageRoot",
            str(root),
            "-ArchivePath",
            archive_path,
            "-PidToWait",
            str(os.getpid()),
        ]

    def _template_dir(self, app_name: str, group: str) -> Path:
        base = self.template_root or Path("apps")
        return base / app_name / "templates" / group

    def _package_root(self) -> Path:
        if self.package_root is not None:
            return self.package_root
        return Path(__file__).resolve().parent.parent

    def _capture_image(self, device_ip: str, app_config: dict[str, Any] | None = None, label: str = "console_capture") -> Any:
        from lgb.lgb import RobotInterface
        from services.defaults import with_app_defaults
        from services.photo_capture import capture_transform_image

        device = self.config_manager.find_device(device_ip) or {}
        app = resolve_app_name(device.get("selected_app"))
        config = app_config
        if config is None:
            config = with_app_defaults(self.config_manager.get_app_config(device_ip, app, {}))
            config["_runtime"] = {"device_ip": device_ip, "app_name": app}
            config["_runtime_cache"] = {}
        return capture_transform_image(RobotInterface(device_ip), config, pre_move=True, label=label)

    def _unattended_seconds(self, device_ip: str) -> float | None:
        view = self.views.get(device_ip)
        if view is None:
            return None
        if view.unattended_started_at is not None:
            return time.monotonic() - view.unattended_started_at
        return view.unattended_elapsed

    def _finalize_unattended(self, device_ip: str) -> None:
        view = self._view(device_ip)
        seconds = self._unattended_seconds(device_ip)
        view.unattended_started_at = None
        if seconds is not None:
            view.unattended_elapsed = max(0.0, float(seconds))

    def _refresh_unattended(self) -> bool:
        changed = False
        for device_ip, view in self.views.items():
            if view.unattended_started_at is not None:
                changed = True
                _ = device_ip
        return changed

    def _ocr_mode_label(self) -> str:
        path = self._package_root() / "runtime" / "install_state.json"
        if not path.is_file():
            return "OCR: 未安装"
        try:
            import json

            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "OCR: 未知"
        if not isinstance(state, dict):
            return "OCR: 未知"
        backend = str(state.get("backend", "") or "")
        device = str(state.get("ocr_device", "") or "")
        if backend == "paddle_cuda" or device.startswith("gpu"):
            return "OCR: GPU"
        if backend == "cpu" or device == "cpu":
            return "OCR: CPU"
        return "OCR: 自动"

    def _ocr_waiting(self) -> int:
        try:
            from vision import get_ocr_queue_status

            status = get_ocr_queue_status()
            return max(0, int(status.get("waiting", 0) or 0)) if isinstance(status, dict) else 0
        except Exception:
            return 0

    def _pool_size(self) -> int:
        try:
            return min(16, max(1, int(self.config_manager.get("global", "ocr_engine_pool_size", default=2) or 2)))
        except (TypeError, ValueError):
            return 2

    def _has_api_key(self) -> bool:
        path = Path(".env") / "deepseek_api_key.txt"
        try:
            return bool(path.is_file() and path.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    def _read_persona_texts(self, path_text: str) -> dict[str, str]:
        path = Path(path_text) if path_text else None
        if path is None or not path.is_file():
            return {key: "" for key in PERSONA_TEXT_KEYS}
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {key: "" for key in PERSONA_TEXT_KEYS}
        if not isinstance(payload, dict):
            return {key: "" for key in PERSONA_TEXT_KEYS}
        return {key: str(payload.get(key, "") or "") for key in PERSONA_TEXT_KEYS}

    def _write_persona_texts(self, app_name: str, path_text: str, texts: dict[str, Any]) -> str:
        import json

        cleaned = {key: str(texts.get(key, "") or "") for key in PERSONA_TEXT_KEYS}
        if not path_text:
            defaults = self._read_persona_texts(str((self.template_root or Path("apps")) / app_name / "persona_default.json"))
            if cleaned == defaults or not any(cleaned.values()):
                return ""
            guided = Path(self.config_manager.app_config_root) / app_name / "persona_profiles"
            guided.mkdir(parents=True, exist_ok=True)
            path_text = str(guided / f"persona_{time.strftime('%Y%m%d_%H%M%S')}.json")
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)
        Path(path_text).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        return path_text


def _ensure_portrait(image: Any) -> Any:
    from vision.screen import ensure_portrait_image

    return ensure_portrait_image(image)


def _image_jpeg(image: Any) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    buffer = io.BytesIO()
    converted = image.convert("RGB") if hasattr(image, "convert") else image
    converted.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
