from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from apps.cancellation import OperationCancelled, operation_cancelled
from vision.screen import ensure_portrait_image


ERROR_PHOTO_DIR = Path("photo") / "errors"


@dataclass
class AppStatus:
    last_success_at: str = ""
    last_error: str = ""
    error_count: int = 0
    is_circuit_open: bool = False
    current_step: str = ""
    current_step_label: str = ""
    reply_round_count: int = 0
    average_reply_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "is_circuit_open": self.is_circuit_open,
            "current_step": self.current_step,
            "current_step_label": self.current_step_label,
            "reply_round_count": self.reply_round_count,
            "average_reply_seconds": self.average_reply_seconds,
        }


@dataclass(frozen=True)
class ReplyAction:
    text: str = ""
    message_fingerprint: str = ""
    skip: bool = False
    reason: str = ""


class BaseApp:
    name = "base"
    description = ""

    def __init__(self, robot: Any, config: dict[str, Any] | None = None) -> None:
        self.robot = robot
        self.config = config or {}
        self.status = AppStatus()
        self.circuit_breaker_threshold = int(self.config.get("circuit_breaker_threshold", 5))
        self.recover_failure_limit = int(self.config.get("recover_failure_limit", 3))
        self.recover_error_count = 0
        self._reply_round_total_seconds = 0.0
        self._status_callback: Callable[[dict[str, Any]], None] | None = None
        self._log_callback: Callable[[str, str, dict[str, Any] | None], None] | None = None

    def run_once(self) -> None:
        raise NotImplementedError

    def set_status_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._status_callback = callback

    def set_log_callback(self, callback: Callable[[str, str, dict[str, Any] | None], None] | None) -> None:
        self._log_callback = callback

    def set_runtime_step(self, step: str, label: str, log_label: str | None = None) -> None:
        self.status.current_step = str(step)
        self.status.current_step_label = str(label)
        if self._status_callback:
            payload = self.status.as_dict()
            if log_label is not None:
                payload["current_step_log_label"] = str(log_label)
            self._status_callback(payload)

    def emit_runtime_log(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self._log_callback:
            self._log_callback(level, message, payload)

    def recover(self) -> None:
        if hasattr(self.robot, "reset_device"):
            self.robot.reset_device(wait=True)

    def get_status(self) -> dict[str, Any]:
        return self.status.as_dict()

    def mark_success(self) -> None:
        self.status.last_success_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status.last_error = ""
        self.status.error_count = 0
        self.status.is_circuit_open = False
        self.recover_error_count = 0

    def record_reply_round(self, duration_seconds: float) -> None:
        duration = max(0.0, float(duration_seconds))
        self._reply_round_total_seconds += duration
        self.status.reply_round_count += 1
        self.status.average_reply_seconds = self._reply_round_total_seconds / self.status.reply_round_count
        if self._status_callback:
            self._status_callback(self.status.as_dict())

    def on_error(self, exc: Exception) -> None:
        if isinstance(exc, OperationCancelled) or operation_cancelled(self.robot, self.config):
            self.status.last_error = str(exc)
            return
        self.status.error_count += 1
        self.status.last_error = str(exc)
        self.emit_runtime_log(
            "error",
            "app_error",
            {
                "step": self.status.current_step,
                "label": self.status.current_step_label,
                "error": str(exc),
                "error_count": self.status.error_count,
            },
        )
        self._save_error_capture()
        if not self.status.is_circuit_open:
            self._try_recover()
        if self.status.error_count >= self.circuit_breaker_threshold:
            self.status.is_circuit_open = True

    def _try_recover(self) -> None:
        try:
            self.recover()
            self.recover_error_count = 0
        except Exception as recover_exc:
            self.recover_error_count += 1
            self.status.last_error = f"{self.status.last_error}；设备复位失败：{recover_exc}"
            if self.recover_error_count >= self.recover_failure_limit:
                self.status.is_circuit_open = True

    def _save_error_capture(self) -> None:
        from services.retention import cleanup_old_error_captures

        if not hasattr(self.robot, "get_transform_image"):
            return
        host = str(getattr(self.robot, "host", "unknown")).replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (ERROR_PHOTO_DIR / f"{host}_{timestamp}.jpg").resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.robot.get_transform_image(pre_move=True, save_path=path)
            ensure_portrait_image(path).save(path, format="JPEG", quality=95)
            cleanup_old_error_captures(path.parent, retention_days=int(self.config.get("error_capture_retention_days", 7) or 7), max_files=int(self.config.get("error_capture_max_files", 200) or 200))
        except Exception:
            return
