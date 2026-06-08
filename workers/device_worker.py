from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from apps import get_registered_apps, resolve_app_name
from apps.defaults import with_app_defaults
from apps.visual_assets import back_button_template_paths
from lgb.lgb import RobotInterface
from services import (
    AIClient,
    ConfigManager,
    LicenseClient,
    MemoryStore,
    MessageProcessor,
    setup_runtime_logging,
)
from services.license_client import _is_future_time


@dataclass(frozen=True)
class WorkerEvent:
    device_ip: str
    level: str
    message: str
    payload: dict[str, Any] | None = None


class DeviceWorker(threading.Thread):
    def __init__(
        self,
        device: dict[str, Any],
        app_config: dict[str, Any] | None = None,
        events: "queue.Queue[WorkerEvent] | None" = None,
        config_manager: ConfigManager | None = None,
        robot: Any | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.config_manager = config_manager or ConfigManager()
        self.events = events or queue.Queue()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        app_registry = get_registered_apps()
        requested_app_name = str(device.get("selected_app", "") or "").strip()
        if requested_app_name in app_registry:
            app_name = requested_app_name
        else:
            app_name = resolve_app_name(requested_app_name)
            if app_name not in app_registry:
                app_name = next(iter(app_registry))
        self.app_name = app_name
        self.app_config = with_app_defaults(app_config or self.config_manager.get_app_config(str(device["ip"]), app_name, {}))
        global_config = self.config_manager.get("global", default={})
        try:
            ocr_max_concurrency = int(global_config.get("ocr_max_concurrency", 1) or 1)
        except (TypeError, ValueError):
            ocr_max_concurrency = 1
        self.app_config["ocr_max_concurrency"] = min(16, max(1, ocr_max_concurrency))
        ocr_config = self.app_config.get("ocr", {})
        if isinstance(ocr_config, dict):
            ocr_config["max_concurrency"] = self.app_config["ocr_max_concurrency"]
        runtime_context = self.app_config.setdefault("_runtime", {})
        if isinstance(runtime_context, dict):
            runtime_context["device_ip"] = str(device["ip"])
            runtime_context["app_name"] = app_name
            runtime_context["should_stop"] = self._stop_event.is_set
        self.robot = robot or RobotInterface(str(device["ip"]))
        try:
            setattr(self.robot, "_codex_click_jitter_enabled", True)
        except Exception:
            pass
        try:
            setattr(self.robot, "_codex_should_stop", self._stop_event.is_set)
        except Exception:
            pass
        self.ai_client = AIClient(
            api_base=str(global_config.get("chat_api_base", "http://liaotianbao.icu")),
            chat_bearer=str(global_config.get("chat_bearer", "")),
            credits_bearer=str(global_config.get("credits_bearer", "")),
        )
        self.license_client = LicenseClient(
            api_url=str(global_config.get("license_api_url", "")),
            project_uuid=str(global_config.get("project_uuid", "")),
            project_secret=str(global_config.get("project_secret", "")),
        )
        self.memory_store = MemoryStore()
        self.message_processor = MessageProcessor(
            device=device,
            app_name=app_name,
            config_manager=self.config_manager,
            ai_client=self.ai_client,
            memory_store=self.memory_store,
            logger=self._emit,
        )
        app_cls = app_registry[app_name]
        self.app = app_cls(
            self.robot,
            self.app_config,
            reply_provider=self.message_processor.prepare_reply,
            sent_callback=self.message_processor.mark_sent,
        )
        if hasattr(self.app, "set_status_callback"):
            self.app.set_status_callback(self._emit_app_status)
        if hasattr(self.app, "set_log_callback"):
            self.app.set_log_callback(self._emit)
        self._last_license_check = 0.0

    def pause(self, detail: str = "") -> None:
        self._pause_event.clear()
        self._emit("info", "paused", {"detail": detail} if detail else None)

    def resume(self) -> None:
        self._pause_event.set()
        if self.app.status.is_circuit_open:
            self.app.status.is_circuit_open = False
            self.app.status.error_count = 0
        self._emit("info", "resumed")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._emit("info", "stopping")

    def _emit(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.events.put(WorkerEvent(str(self.device["ip"]), level, message, payload))
        safe_payload = "" if payload is None else f" {payload}"
        try:
            setup_runtime_logging().log(
                {"error": 40, "warning": 30, "info": 20, "status": 20}.get(level, 20),
                "%s %s%s",
                self.device["ip"],
                message,
                safe_payload,
            )
        except Exception:
            return

    def _emit_app_status(self, payload: dict[str, Any]) -> None:
        if self._stop_event.is_set():
            return
        detail = str(payload.get("current_step_label", "") or payload.get("current_step", "") or "progress")
        log_detail = str(payload.get("current_step_log_label", "") or detail)
        self._emit("info", "runtime_step", {"step": payload.get("current_step", ""), "detail": log_detail})
        self._emit("status", f"progress: {detail}", payload)

    def _license_allows_run(self) -> bool:
        expires = str(self.device.get("license_expires", "") or "")
        if expires and not _is_future_time(expires):
            self._emit("error", "license expired", {"license_expires": expires})
            self.pause()
            return False

        now = time.monotonic()
        if now - self._last_license_check < 300:
            return True
        self._last_license_check = now

        license_api_url = self.config_manager.get("global", "license_api_url", default="")
        license_key = str(self.device.get("license_key", "") or "")
        if not license_api_url:
            self._emit("warning", "license_api_url not configured; skipping online license check")
            return True
        result = self.license_client.verify_with_offline(str(self.device["ip"]), license_key, expires)
        if not result.ok:
            self._emit("error", result.message, {"status": result.status})
            self.pause()
            return False
        if result.expires_at and result.expires_at != expires:
            self.device["license_expires"] = result.expires_at
            self.config_manager.update_device(str(self.device["ip"]), {"license_expires": result.expires_at})
            self._emit("info", f"license {result.status}", {"expires_at": result.expires_at})
        return True

    def _poll_interval_for_status(self, status: dict[str, Any], default_interval: float) -> float:
        step = str(status.get("current_step", "") or "")
        if step != "empty_list_target":
            return default_interval
        idle_interval = float(self.app_config.get("idle_poll_interval_seconds", 5) or 5)
        return max(default_interval, idle_interval)

    def _required_back_button_templates_available(self) -> bool:
        return bool(back_button_template_paths(getattr(self.app, "config", self.app_config)))

    def _ensure_startup_templates(self) -> bool:
        if self._required_back_button_templates_available():
            return True
        detail = "没有返回按钮"
        if hasattr(self.app, "status"):
            self.app.status.last_error = detail
            self.app.status.is_circuit_open = True
        if hasattr(self.app, "set_runtime_step"):
            self.app.set_runtime_step("missing_back_button_template", detail)
        self._emit("error", detail, {"app_name": self.app_name})
        return False

    def _fallback_status(self) -> dict[str, Any]:
        status_obj = getattr(self.app, "status", None)
        if hasattr(status_obj, "as_dict"):
            try:
                status = status_obj.as_dict()
                if isinstance(status, dict):
                    return status
            except Exception:
                pass
        return {
            "last_error": str(getattr(status_obj, "last_error", "") or ""),
            "error_count": int(getattr(status_obj, "error_count", 0) or 0),
            "is_circuit_open": bool(getattr(status_obj, "is_circuit_open", False)),
            "current_step": str(getattr(status_obj, "current_step", "") or ""),
            "current_step_label": str(getattr(status_obj, "current_step_label", "") or ""),
        }

    def _safe_get_status(self) -> dict[str, Any]:
        try:
            status = self.app.get_status()
            if isinstance(status, dict):
                return status
        except Exception as exc:
            self._emit("error", "worker_status_error", {"error": str(exc), "app_name": self.app_name})
        return self._fallback_status()

    def _record_unhandled_app_exception(self, exc: Exception) -> dict[str, Any]:
        status_obj = getattr(self.app, "status", None)
        try:
            current_count = int(getattr(status_obj, "error_count", 0) or 0)
        except (TypeError, ValueError):
            current_count = 0
        error_count = current_count + 1
        if status_obj is not None:
            with contextlib.suppress(Exception):
                status_obj.error_count = error_count
            with contextlib.suppress(Exception):
                status_obj.last_error = str(exc)
            threshold = int(getattr(self.app, "circuit_breaker_threshold", 5) or 5)
            if error_count >= threshold:
                with contextlib.suppress(Exception):
                    status_obj.is_circuit_open = True
        status = self._fallback_status()
        status["error_count"] = error_count
        status["last_error"] = str(exc)
        label = str(status.get("current_step_label", "") or status.get("current_step", "") or "run_once")
        self._emit(
            "error",
            "app_error",
            {
                "step": status.get("current_step", ""),
                "label": label,
                "error": str(exc),
                "error_count": error_count,
                "app_name": self.app_name,
                "thread_isolated": True,
            },
        )
        return status

    def run(self) -> None:
        self._emit("info", "started")
        try:
            poll_interval = float(self.app_config.get("poll_interval", 1) or 1)
            try:
                if not self._ensure_startup_templates():
                    return
            except Exception as exc:
                status = self._record_unhandled_app_exception(exc)
                status["checked_at"] = datetime.now().isoformat(timespec="seconds")
                self._emit("status", "tick", status)
                return
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                try:
                    if not self._license_allows_run():
                        continue
                    self.app.run_once()
                    status = self._safe_get_status()
                except Exception as exc:
                    status = self._record_unhandled_app_exception(exc)
                if self._stop_event.is_set():
                    break
                status["checked_at"] = datetime.now().isoformat(timespec="seconds")
                self._emit("status", "tick", status)
                if status.get("is_circuit_open"):
                    self._emit("error", "circuit open", status)
                    self.pause(str(status.get("current_step_label", "") or status.get("current_step", "") or ""))
                wait_interval = self._poll_interval_for_status(status, poll_interval)
                if self._stop_event.wait(wait_interval):
                    break
        finally:
            self._emit("info", "stopped")
