from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable


MonitorEmitter = Callable[[str, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class RobotStatusSnapshot:
    checked_at: str
    ok: bool
    status: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _safe_status(robot: Any) -> RobotStatusSnapshot:
    try:
        if not hasattr(robot, "get_status"):
            return RobotStatusSnapshot(_now(), False, {}, "robot has no get_status")
        status = robot.get_status()
        return RobotStatusSnapshot(_now(), True, status if isinstance(status, dict) else {"value": status})
    except Exception as exc:
        return RobotStatusSnapshot(_now(), False, {}, str(exc))


class RobotArmMonitor:
    """后台轮询机械臂状态，并在复位/避让前记录最近几步状态。"""

    def __init__(
        self,
        robot: Any,
        *,
        device_ip: str,
        emit: MonitorEmitter | None = None,
        interval_seconds: float = 1.0,
        history_size: int = 8,
        enabled: bool = True,
    ) -> None:
        self.robot = robot
        self.device_ip = str(device_ip)
        self.emit = emit
        self.interval_seconds = max(0.2, float(interval_seconds or 1.0))
        self.history: deque[RobotStatusSnapshot] = deque(maxlen=max(1, int(history_size or 8)))
        self.enabled = enabled and hasattr(robot, "get_status")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"RobotArmMonitor-{self.device_ip}",
            daemon=True,
        )
        self._thread.start()
        self._emit("info", "robot_monitor_started", {"interval_seconds": self.interval_seconds})

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self.enabled:
            self._emit("info", "robot_monitor_stopped", None)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample("poll")
            self._stop_event.wait(self.interval_seconds)

    def sample(self, reason: str) -> RobotStatusSnapshot:
        snapshot = _safe_status(self.robot)
        with self._lock:
            self.history.append(snapshot)
        if not snapshot.ok:
            self._emit("warning", "robot_monitor_status_failed", {"reason": reason, "error": snapshot.error})
        return snapshot

    def recent(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.to_dict() for item in self.history]

    def before_operation(self, operation: str, detail: dict[str, Any] | None = None) -> None:
        snapshot = self.sample(f"before_{operation}")
        self._emit(
            "info",
            "robot_monitor_before_operation",
            {
                "operation": operation,
                "phase": "before",
                "detail": detail or {},
                "current_status": snapshot.to_dict(),
                "recent_statuses": self.recent(),
            },
        )

    def after_operation(
        self,
        operation: str,
        *,
        ok: bool,
        elapsed_seconds: float,
        error: str = "",
    ) -> None:
        snapshot = self.sample(f"after_{operation}")
        self._emit(
            "info",
            "robot_monitor_after_operation",
            {
                "operation": operation,
                "phase": "after",
                "ok": bool(ok),
                "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
                "error": error,
                "current_status": snapshot.to_dict(),
                "recent_statuses": self.recent(),
            },
        )

    def _emit(self, level: str, message: str, payload: dict[str, Any] | None) -> None:
        if self.emit is not None:
            self.emit(level, message, payload)


class MonitoredRobot:
    def __init__(self, robot: Any, monitor: RobotArmMonitor) -> None:
        self._robot = robot
        self._monitor = monitor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    @property
    def host(self) -> str:
        return str(getattr(self._robot, "host", self._monitor.device_ip))

    def reset_device(self, *args: Any, **kwargs: Any) -> Any:
        self._monitor.before_operation("reset_device", {"args": list(args), "kwargs": dict(kwargs)})
        started_at = time.monotonic()
        try:
            result = self._robot.reset_device(*args, **kwargs)
        except Exception as exc:
            self._monitor.after_operation(
                "reset_device",
                ok=False,
                elapsed_seconds=time.monotonic() - started_at,
                error=str(exc),
            )
            raise
        self._monitor.after_operation(
            "reset_device",
            ok=True,
            elapsed_seconds=time.monotonic() - started_at,
        )
        return result

    def get_transform_image(self, *args: Any, **kwargs: Any) -> Any:
        pre_move = kwargs.get("pre_move", True)
        if args:
            pre_move = args[0]
        if pre_move:
            self._monitor.before_operation("capture_pre_move", {"args": list(args), "kwargs": dict(kwargs)})
        started_at = time.monotonic()
        try:
            result = self._robot.get_transform_image(*args, **kwargs)
        except Exception as exc:
            if pre_move:
                self._monitor.after_operation(
                    "capture_pre_move",
                    ok=False,
                    elapsed_seconds=time.monotonic() - started_at,
                    error=str(exc),
                )
            raise
        if pre_move:
            self._monitor.after_operation(
                "capture_pre_move",
                ok=True,
                elapsed_seconds=time.monotonic() - started_at,
            )
        return result
