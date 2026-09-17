from __future__ import annotations

import contextlib
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from apps import get_registered_apps, resolve_app_name
from services.defaults import with_app_defaults
from apps.visual_assets import back_button_template_paths
from lgb.lgb import RobotInterface
from services import (
    ChatHistoryStore,
    ConfigManager,
    DeepSeekClient,
    LicenseClient,
    MessageProcessor,
    PersonaChat,
    PersonaSessionStore,
    setup_runtime_logging,
    write_error_incident,
)
from services.cancellation import OperationCancelled, raise_if_cancelled
from services.license_client import _is_future_time


@dataclass(frozen=True)
class WorkerEvent:
    device_ip: str
    level: str
    message: str
    payload: dict[str, Any] | None = None
    worker_id: str = ""


RECENT_LOG_BUFFER_LIMIT = 100
PRE_ERROR_LOG_LIMIT = 10
TIMELINE_LOG_LIMIT = 200

# lgb 的 HTTP 请求全部没有超时（只读保护区改不了），设备断网/半死时一个
# get_status 就能把工作线程挂住几分钟。守卫层按方法补上单次调用上限：
# 动作类（move_click/swipe 走 send_command 且 wait=True）要等机械动作
# 完成，上限放宽；查询/截图类收紧。
ROBOT_CALL_TIMEOUT_SECONDS = {
    "get_status": 8.0,
    "get_capture_image": 20.0,
    "get_transform_image": 20.0,
    "send_command": 45.0,
    "move_click": 45.0,
    "swipe": 45.0,
    "play_audio": 30.0,
    "reset_device": 30.0,
    "restart_system": 10.0,
}
DEFAULT_ROBOT_CALL_TIMEOUT_SECONDS = 20.0


class _RobotCallGuard:
    """机械臂调用守卫：停止后快速失败 + 单次调用限时。

    1. 停止请求发出后，任何新的机械臂调用立即抛 OperationCancelled，
       工作线程不必等上一个长步骤走完才开始退出；
    2. 实际调用提交到本守卫的有界线程池并限时等待，超时后放弃等待
       （指令仍只会发一次，不会重复执行动作），避免无限挂起。
       池化替代原先"每次调用新建 daemon 线程"：多设备高频调用不再
       反复创建线程，超时挂住的调用占用的是有界 worker 而不是无限
       堆积的新线程（设备半死时的线程泄漏源）。
    """

    EXECUTOR_WORKERS = 4

    def __init__(self, robot: Any, stop_check: Any) -> None:
        self._robot = robot
        self._stop_check = stop_check
        self._executor = ThreadPoolExecutor(
            max_workers=self.EXECUTOR_WORKERS,
            thread_name_prefix="robot-call",
        )

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._robot, name)
        if not callable(target):
            return target
        timeout = float(ROBOT_CALL_TIMEOUT_SECONDS.get(name, DEFAULT_ROBOT_CALL_TIMEOUT_SECONDS))

        def call(*args: Any, **kwargs: Any) -> Any:
            raise_if_cancelled(self._robot)
            future = self._executor.submit(target, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                # 不 cancel：已发出的 HTTP 让它自然跑完，线程归池复用。
                if self._stop_check():
                    raise OperationCancelled("operation cancelled")
                raise TimeoutError(f"机械臂调用超时：{name} 超过 {timeout:.0f}s 未返回")

        return call


def is_stale_worker_event(event: WorkerEvent, current: Any) -> bool:
    """旧 worker 迟到的事件应被 UI 忽略。

    典型竞态：停止旧 worker 期间（退出可能要几十秒）又点了启动，新
    worker 已经跑起来，旧 worker 退出时迟到的 "stopped" 事件会把行状态
    打回"已停止"。带 worker_id 的事件只信当前注册的 worker。
    """
    worker_id = str(getattr(event, "worker_id", "") or "")
    if not worker_id:
        return False
    if current is None:
        return True
    return str(getattr(current, "worker_id", "") or "") != worker_id


def _ocr_backend_label_from_state() -> str:
    state_path = Path("runtime") / "install_state.json"
    if not state_path.is_file():
        return ""
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    device = str(payload.get("ocr_device", "") or "").strip().lower()
    backend = str(payload.get("backend", "") or "").strip().lower()
    if device.startswith(("gpu", "cuda", "dml")) or backend in {"paddle_cuda", "onnx_directml", "gpu", "cuda", "directml"}:
        return "GPU"
    if device == "cpu" or backend == "cpu":
        return "CPU"
    return ""


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
        # 日志缓冲/事故上下文必须先于任何 _emit 初始化：构造早期的 DeepSeek key
        # 警告曾在客户机触发 AttributeError → Qt 槽内未处理异常 → PyQt qFatal 闪退。
        self._recent_logs: deque[dict[str, Any]] = deque(maxlen=RECENT_LOG_BUFFER_LIMIT)
        self._active_incident: dict[str, Any] | None = None
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
        # 引擎池大小注入：主 UI「引擎并发数」（global.ocr_engine_pool_size，
        # 默认 2）→ app 配置 ocr 子树；app 级显式值优先。旧 ocr_max_concurrency
        # 注入已移除——它作为显式并发覆盖会把并发槽压回 1，强制全串行。
        try:
            engine_pool_size = int(global_config.get("ocr_engine_pool_size", 2) or 2)
        except (TypeError, ValueError):
            engine_pool_size = 2
        engine_pool_size = min(16, max(1, engine_pool_size))
        ocr_config = self.app_config.setdefault("ocr", {})
        if not ocr_config.get("ocr_engine_pool_size"):
            ocr_config["ocr_engine_pool_size"] = engine_pool_size
        runtime_context = self.app_config.setdefault("_runtime", {})
        if isinstance(runtime_context, dict):
            runtime_context["device_ip"] = str(device["ip"])
            runtime_context["app_name"] = app_name
            runtime_context["should_stop"] = self._stop_event.is_set
            runtime_context["ocr_backend"] = _ocr_backend_label_from_state()
            # App 运行期间用于保存视觉模型学习到的 App 级 OCR 规则。
            # _runtime 不会写入 config.json。
            runtime_context["config_manager"] = self.config_manager
            runtime_context["rule_learning"] = global_config.get("rule_learning", {}) if isinstance(global_config, dict) else {}
        self.worker_id = f"{str(device['ip'])}-{uuid.uuid4().hex[:8]}"
        self.robot_inner = robot or RobotInterface(str(device["ip"]))
        try:
            setattr(self.robot_inner, "_codex_click_jitter_enabled", True)
        except Exception:
            pass
        try:
            setattr(self.robot_inner, "_codex_should_stop", self._stop_event.is_set)
        except Exception:
            pass
        self.robot = _RobotCallGuard(self.robot_inner, self._stop_event.is_set)
        # 人设 AI 新链路（docs/13）：DeepSeek 官方直连，会话记忆在本地 persona_sessions。
        # 打包环境模块经 zipimport 导入，__file__ 落在 app_modules.zip 内：直接
        # parents[1] 会把 workspace_root 指向 zip 文件，persona_sessions 首次落库
        # mkdir 时触发 WinError 183（文件已存在），AI 回复整轮失败（客户机
        # 20260914 事故）。必须优先取 JIXIEBI_PACKAGE_ROOT。
        configured_root = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
        workspace_root = (
            str(Path(configured_root).resolve()) if configured_root else str(Path(__file__).resolve().parents[1])
        )
        self.persona_client = DeepSeekClient(workspace_root=workspace_root)
        self.persona_store = PersonaSessionStore(root=str(Path(workspace_root) / "runtime" / "persona_sessions"))
        self.persona_chat = PersonaChat(
            client=self.persona_client,
            store=self.persona_store,
            workspace_root=workspace_root,
            read_settings=self._read_persona_settings,
            logger=self._emit,
        )
        if not self.persona_client.api_key:
            self._emit("warning", "DeepSeek key 未配置，人设 AI 回复将失败（.env/deepseek_api_key.txt）", {})
        try:
            removed_sessions = self.persona_store.cleanup_inactive_sessions(str(device["ip"]), app_name)
            if removed_sessions:
                self._emit("info", "清理 30 天未活跃人设会话", {"count": removed_sessions})
        except Exception:
            pass
        self.license_client = LicenseClient(
            api_url=str(global_config.get("license_api_url", "")),
            project_uuid=str(global_config.get("project_uuid", "")),
            project_secret=str(global_config.get("project_secret", "")),
        )
        self.chat_history_store = ChatHistoryStore()
        self.message_processor = MessageProcessor(
            device=device,
            app_name=app_name,
            config_manager=self.config_manager,
            persona_chat=self.persona_chat,
            chat_history_store=self.chat_history_store,
            logger=self._emit,
        )
        app_cls = app_registry[app_name]
        app_kwargs: dict[str, Any] = {
            "reply_provider": self.message_processor.prepare_reply,
            "sent_callback": self.message_processor.mark_sent,
        }
        if getattr(app_cls, "supports_dialogue_continuity", False):
            app_kwargs["continuity_required_provider"] = self.message_processor.needs_continuity_context
        if getattr(app_cls, "supports_greeting", False):
            app_kwargs["greeting_provider"] = self._provide_greeting
            app_kwargs["greeting_sent_notifier"] = self._notify_synthetic_greeting
        self.app = app_cls(self.robot, self.app_config, **app_kwargs)
        if hasattr(self.app, "set_status_callback"):
            self.app.set_status_callback(self._emit_app_status)
        if hasattr(self.app, "set_log_callback"):
            self.app.set_log_callback(self._emit)
        self._last_license_check = 0.0
        self._last_progress_event_key: tuple[str, str, str] | None = None
        self._last_progress_event_at = 0.0
        self._incident_sequence = 0
        self._last_error_count_seen = 0
        self._live_config_fingerprint = self._config_fingerprint(self.app_config)

    def _read_persona_settings(self) -> dict[str, Any]:
        """人设客户设置（docs/13 五/六节）：device.app_overrides[app] + global.persona_chat_model。"""
        settings: dict[str, Any] = {}
        try:
            overrides = (self.device.get("app_overrides") or {}).get(self.app_name) or {}
            if isinstance(overrides, dict):
                settings.update(overrides)
            model = str(self.config_manager.get("global", "persona_chat_model", default="") or "")
            if model:
                settings["model"] = model
        except Exception:
            return {}
        return settings

    def _provide_greeting(self, contact_name: str) -> str:
        return self.persona_chat.generate_greeting(
            device_ip=str(self.device.get("ip", "")),
            app_name=self.app_name,
            contact_name=str(contact_name or ""),
        )

    def _notify_synthetic_greeting(self, contact_name: str) -> None:
        try:
            self.persona_chat.record_synthetic_greeting(
                device_ip=str(self.device.get("ip", "")),
                app_name=self.app_name,
                contact_name=str(contact_name or ""),
            )
        except Exception as exc:
            self._emit("warning", "平台新招呼占位落库失败", {"contact": contact_name, "error": str(exc)})

    @staticmethod
    def _config_fingerprint(config: dict[str, Any]) -> str:
        try:
            payload = {key: value for key, value in config.items() if key not in {"_runtime", "_runtime_cache"}}
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return repr(config)

    def _reload_live_app_config(self) -> None:
        """重新加载当前设备当前 App 配置，避免运行中新增模板范围要重启才生效。"""
        fresh = with_app_defaults(
            self.config_manager.get_app_config(str(self.device["ip"]), self.app_name, {})
        )
        fingerprint = self._config_fingerprint(fresh)
        if fingerprint == self._live_config_fingerprint:
            return

        old_runtime = self.app_config.get("_runtime", {})
        old_cache = self.app_config.get("_runtime_cache", {})
        fresh["_runtime"] = old_runtime if isinstance(old_runtime, dict) else {}
        fresh["_runtime_cache"] = old_cache if isinstance(old_cache, dict) else {}
        global_config = self.config_manager.get("global", default={})
        runtime = fresh.setdefault("_runtime", {})
        if isinstance(runtime, dict):
            runtime["device_ip"] = str(self.device["ip"])
            runtime["app_name"] = self.app_name
            runtime["should_stop"] = self._stop_event.is_set
            runtime["ocr_backend"] = _ocr_backend_label_from_state()
            runtime["config_manager"] = self.config_manager
            runtime["rule_learning"] = global_config.get("rule_learning", {}) if isinstance(global_config, dict) else {}
        # 引擎池大小随全局配置热生效（主 UI 改「引擎并发数」下一轮起作用）。
        try:
            engine_pool_size = int(global_config.get("ocr_engine_pool_size", 2) or 2)
        except (TypeError, ValueError):
            engine_pool_size = 2
        engine_pool_size = min(16, max(1, engine_pool_size))
        ocr_config = fresh.setdefault("ocr", {})
        if not ocr_config.get("ocr_engine_pool_size"):
            ocr_config["ocr_engine_pool_size"] = engine_pool_size

        self.app_config.clear()
        self.app_config.update(fresh)
        self._live_config_fingerprint = fingerprint
        self._emit("info", "当前设备当前 App 配置已热刷新", {"app_name": self.app_name})

    def pause(self, detail: str = "") -> None:
        self._pause_event.clear()
        self._emit("info", "paused", {"detail": detail} if detail else None)

    def resume(self) -> None:
        self._pause_event.set()
        if self.app.status.is_circuit_open:
            self.app.status.is_circuit_open = False
            self.app.status.error_count = 0
            self._last_error_count_seen = 0
        self._emit("info", "resumed")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._emit("info", "stopping")

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        # UI 表格重建时按存活 worker 恢复行状态需要区分"在跑/已暂停"。
        return not self._pause_event.is_set()

    def _emit(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self._record_log_entry(self._log_entry(level, message, payload))
        self.events.put(WorkerEvent(str(self.device["ip"]), level, message, payload, self.worker_id))
        safe_payload = "" if payload is None else f" {payload}"
        log_level = {"error": 40, "warning": 30, "info": 20, "status": 20}.get(level, 20)
        if log_level < 30:
            return
        try:
            setup_runtime_logging().log(
                log_level,
                "%s %s%s",
                self.device["ip"],
                message,
                safe_payload,
            )
        except Exception:
            return

    def _log_entry(self, level: str, message: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": str(level),
            "message": str(message),
            "payload": dict(payload) if isinstance(payload, dict) else {},
        }

    def _should_buffer_log(self, entry: dict[str, Any]) -> bool:
        level = str(entry.get("level", "") or "")
        message = str(entry.get("message", "") or "")
        if level == "status" and message == "tick":
            return False
        return level in {"info", "warning", "error", "status"}

    def _record_log_entry(self, entry: dict[str, Any]) -> None:
        if not self._should_buffer_log(entry):
            return
        self._recent_logs.append(entry)
        if self._active_incident is None:
            return
        timeline = self._active_incident.setdefault("timeline_logs", [])
        timeline.append(entry)
        if len(timeline) > TIMELINE_LOG_LIMIT:
            del timeline[:-TIMELINE_LOG_LIMIT]

    def _runtime_context(self) -> dict[str, Any]:
        runtime = self.app_config.get("_runtime")
        if not isinstance(runtime, dict):
            runtime = {}
            self.app_config["_runtime"] = runtime
        return runtime

    def _runtime_snapshot(self) -> dict[str, Any]:
        runtime = self._runtime_context()
        related_capture_paths = runtime.get("related_capture_paths", [])
        template_paths = runtime.get("template_paths", [])
        if not isinstance(related_capture_paths, list):
            related_capture_paths = []
        if not isinstance(template_paths, list):
            template_paths = []
        return {
            "ocr_backend": str(runtime.get("ocr_backend", "") or ""),
            "error_screenshot": str(runtime.get("error_screenshot", "") or runtime.get("last_error_capture_path", "") or ""),
            "current_capture_path": str(runtime.get("current_capture_path", "") or runtime.get("last_capture_path", "") or ""),
            "related_capture_paths": [str(item) for item in related_capture_paths[-10:]],
            "template_paths": [str(item) for item in template_paths[-10:]],
            "last_template_path": str(runtime.get("last_template_path", "") or ""),
            "last_template_match_key": str(runtime.get("last_template_match_key", "") or ""),
        }

    def _merge_incident_context(self, incident: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if snapshot.get("ocr_backend"):
            incident["ocr_backend"] = snapshot["ocr_backend"]
        if snapshot.get("error_screenshot"):
            incident["error_screenshot"] = snapshot["error_screenshot"]
        if snapshot.get("current_capture_path"):
            incident["current_capture_path"] = snapshot["current_capture_path"]
        related = incident.setdefault("related_capture_paths", [])
        for item in snapshot.get("related_capture_paths", []):
            if item and item not in related:
                related.append(item)
        templates = incident.setdefault("template_paths", [])
        for item in snapshot.get("template_paths", []):
            if item and item not in templates:
                templates.append(item)
        last_template = str(snapshot.get("last_template_path", "") or "")
        if last_template and last_template not in templates:
            templates.append(last_template)
        if snapshot.get("last_template_match_key"):
            incident["last_template_match_key"] = snapshot["last_template_match_key"]

    def _ensure_incident_started(self) -> dict[str, Any]:
        if self._active_incident is not None:
            return self._active_incident
        self._incident_sequence += 1
        recent_logs = list(self._recent_logs)
        pre_error_logs = recent_logs[-PRE_ERROR_LOG_LIMIT:]
        timeline_logs: list[dict[str, Any]] = []
        if pre_error_logs and pre_error_logs[-1].get("level") in {"warning", "error"}:
            timeline_logs = [pre_error_logs[-1]]
            pre_error_logs = pre_error_logs[:-1]
        incident = {
            "incident_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._incident_sequence:03d}",
            "device_ip": str(self.device["ip"]),
            "app_name": self.app_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "active",
            "circuit_open": False,
            "error_count_peak": 0,
            "pre_error_logs": pre_error_logs,
            "timeline_logs": timeline_logs,
            "errors": [],
            "ocr_backend": str(self._runtime_context().get("ocr_backend", "") or ""),
            "error_screenshot": "",
            "current_capture_path": "",
            "related_capture_paths": [],
            "template_paths": [],
        }
        self._merge_incident_context(incident, self._runtime_snapshot())
        self._active_incident = incident
        return incident

    def _record_incident_error(self, status: dict[str, Any]) -> None:
        incident = self._ensure_incident_started()
        snapshot = self._runtime_snapshot()
        self._merge_incident_context(incident, snapshot)
        threshold = int(getattr(self.app, "circuit_breaker_threshold", 5) or 5)
        current_count = int(status.get("error_count", 0) or 0)
        incident["error_count_peak"] = max(int(incident.get("error_count_peak", 0) or 0), current_count)
        incident["errors"].append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_count": current_count,
                "threshold": threshold,
                "step": str(status.get("current_step", "") or ""),
                "step_label": str(status.get("current_step_label", "") or ""),
                "error": str(status.get("last_error", "") or ""),
                "circuit_open": bool(status.get("is_circuit_open", False)),
                "error_screenshot": snapshot.get("error_screenshot", ""),
                "current_capture_path": snapshot.get("current_capture_path", ""),
                "template_paths": snapshot.get("template_paths", []),
            }
        )

    def _flush_incident(self, status_name: str, status: dict[str, Any] | None = None) -> None:
        if self._active_incident is None:
            return
        incident = self._active_incident
        self._active_incident = None
        if status is not None:
            self._merge_incident_context(incident, self._runtime_snapshot())
            incident["error_count_peak"] = max(
                int(incident.get("error_count_peak", 0) or 0),
                int(status.get("error_count", 0) or 0),
            )
            incident["circuit_open"] = bool(status.get("is_circuit_open", False))
            incident["last_error"] = str(status.get("last_error", "") or "")
            incident["last_step"] = str(status.get("current_step_label", "") or status.get("current_step", "") or "")
        incident["status"] = status_name
        incident["ended_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            write_error_incident(incident)
        except Exception as exc:
            try:
                setup_runtime_logging().exception("failed to write error incident for %s: %s", self.device["ip"], exc)
            except Exception:
                pass

    def _track_error_window(self, status: dict[str, Any]) -> None:
        try:
            current_count = int(status.get("error_count", 0) or 0)
        except (TypeError, ValueError):
            current_count = 0
        previous_count = self._last_error_count_seen
        if current_count > previous_count:
            self._record_incident_error(status)
        if current_count == 0 and previous_count > 0:
            self._flush_incident("recovered", status)
        elif bool(status.get("is_circuit_open", False)) and current_count > 0:
            self._flush_incident("circuit_open", status)
        self._last_error_count_seen = current_count

    def _emit_app_status(self, payload: dict[str, Any]) -> None:
        if self._stop_event.is_set():
            return
        detail = str(payload.get("current_step_label", "") or payload.get("current_step", "") or "progress")
        log_detail = str(payload.get("current_step_log_label", "") or detail)
        step = str(payload.get("current_step", "") or "")
        event_key = (step, detail, log_detail)
        now = time.monotonic()
        if event_key == self._last_progress_event_key and now - self._last_progress_event_at < 0.8:
            return
        self._last_progress_event_key = event_key
        self._last_progress_event_at = now
        self._emit("info", "runtime_step", {"step": step, "detail": log_detail})
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
        # 仅在错误事件中保存完整调用栈，便于定位客户机上的第三方运行时异常。
        # 正常运行日志仍只显示简短错误，不增加客户日志噪音。
        error_traceback = traceback.format_exc()
        module_path = self._runtime_context().get("module_archive", "")
        if not module_path:
            module_path = str(Path(__file__).resolve().parents[1] / "app_modules.zip")
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
                "traceback": error_traceback,
                "python_executable": sys.executable,
                "module_archive": str(module_path),
                "module_archive_exists": Path(module_path).is_file() if module_path else False,
                "module_archive_is_dir": Path(module_path).is_dir() if module_path else False,
            },
        )
        return status

    def _preload_ocr_engines(self) -> None:
        """启动期后台预热 OCR 引擎池（车道 0..N-1）。

        弱显卡机器首引擎加载可达 77~100s，原先摊在第一轮 OCR 上；挪到
        worker 启动 runway（绑 TTS/返回按钮预缓存等动作期间）并行烧掉。
        与生产调用走同一 cache-key（同池同实例），不会多起子进程；预热
        失败静默——首轮 OCR 会自行拉起，绝不因预热阻断启动。
        """
        try:
            from vision.ocr_runtime import engine_pool_size_for, get_pooled_runtime_engine

            ocr_modes = self.app_config.get("ocr_modes", {})
            ocr_modes = ocr_modes if isinstance(ocr_modes, dict) else {}
            for mode_key in ("list", "dialogue"):
                mode_config = ocr_modes.get(mode_key)
                if not isinstance(mode_config, dict) or not mode_config:
                    continue
                # 与生产调用同构的引擎配置（ocr 子树 + 模式文件引擎键）：
                # 两边必须命中同一池——否则同进程双后端引擎，进程内模式
                # 下 PaddleX 拒绝二次初始化（PDX already initialized）。
                engine_config = dict(self.app_config.get("ocr", {})) if isinstance(self.app_config.get("ocr"), dict) else {}
                for engine_key in ("ocr_backend", "backend", "model_tier", "modelTier", "ocr_engine_pool_size"):
                    if engine_key in mode_config:
                        engine_config[engine_key] = mode_config[engine_key]
                first_engine = None
                for lane in range(engine_pool_size_for(engine_config)):
                    engine = get_pooled_runtime_engine(engine_config, lane=lane)
                    if lane > 0 and engine is first_engine:
                        # 该车道实例创建失败已降级共享 0 号（引擎侧已留痕），
                        # 继续往后预热只会重复空试创建，就此打住。
                        break
                    engine.preload()
                    if lane == 0:
                        first_engine = engine
                break  # 两模式引擎键一致（backend/tier 同池），预热一份即全暖
        except Exception:
            pass

    def run(self) -> None:
        self._emit("info", "started")
        threading.Thread(target=self._preload_ocr_engines, name="ocr-engine-preload", daemon=True).start()
        try:
            poll_interval = float(self.app_config.get("poll_interval", 1) or 1)
            try:
                if not self._ensure_startup_templates():
                    return
            except Exception as exc:
                status = self._record_unhandled_app_exception(exc)
                status["checked_at"] = datetime.now().isoformat(timespec="seconds")
                self._track_error_window(status)
                self._emit("status", "tick", status)
                return
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                try:
                    self._reload_live_app_config()
                    if not self._license_allows_run():
                        continue
                    self.app.run_once()
                    status = self._safe_get_status()
                except Exception as exc:
                    status = self._record_unhandled_app_exception(exc)
                if self._stop_event.is_set():
                    break
                self._track_error_window(status)
                status["checked_at"] = datetime.now().isoformat(timespec="seconds")
                self._emit("status", "tick", status)
                if status.get("is_circuit_open"):
                    self._emit("error", "circuit open", status)
                    self.pause(str(status.get("current_step_label", "") or status.get("current_step", "") or ""))
                wait_interval = self._poll_interval_for_status(status, poll_interval)
                if self._stop_event.wait(wait_interval):
                    break
        finally:
            # 坐标缓存防抖落盘的收尾：worker 退出前把待写缓存刷盘。
            try:
                from services.coordinate_cache import flush_persist_debounce

                flush_persist_debounce()
            except Exception:
                pass
            self._flush_incident("stopped")
            self._emit("info", "stopped")
