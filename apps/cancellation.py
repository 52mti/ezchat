from __future__ import annotations

import time
from typing import Any


class OperationCancelled(RuntimeError):
    pass


def _checker_stopped(checker: Any) -> bool:
    if checker is None:
        return False
    return bool(checker() if callable(checker) else checker)


def operation_cancelled(robot: Any | None = None, config: dict[str, Any] | None = None) -> bool:
    runtime = config.get("_runtime", {}) if isinstance(config, dict) else {}
    if isinstance(runtime, dict) and _checker_stopped(runtime.get("should_stop")):
        return True
    return _checker_stopped(getattr(robot, "_codex_should_stop", None))


def raise_if_cancelled(robot: Any | None = None, config: dict[str, Any] | None = None) -> None:
    if operation_cancelled(robot, config):
        raise OperationCancelled("operation cancelled")


def cancellable_sleep(
    seconds: float,
    robot: Any | None = None,
    config: dict[str, Any] | None = None,
    *,
    interval: float = 0.1,
) -> None:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        raise_if_cancelled(robot, config)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(interval, remaining))
