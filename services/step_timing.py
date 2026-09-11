from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterator


EXCLUDE_OCR_WAIT_KEY = "exclude_ocr_wait_from_step_timeout"
_EXCLUDED_SECONDS_KEY = "_step_timeout_excluded_seconds"


def _runtime(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    runtime = config.get("_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        config["_runtime"] = runtime
    return runtime


def _excluded_seconds(config: dict[str, Any] | None) -> float:
    runtime = _runtime(config)
    if runtime is None:
        return 0.0
    try:
        return float(runtime.get(_EXCLUDED_SECONDS_KEY, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _add_excluded_seconds(config: dict[str, Any] | None, seconds: float) -> None:
    runtime = _runtime(config)
    if runtime is None:
        return
    runtime[_EXCLUDED_SECONDS_KEY] = _excluded_seconds(config) + max(0.0, float(seconds))


def _exclude_ocr_wait_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return True
    return bool(config.get(EXCLUDE_OCR_WAIT_KEY, True))


@contextmanager
def exclude_ocr_wait_from_step_timeout(
    config: dict[str, Any] | None,
    *,
    now: Callable[[], float] = time.monotonic,
) -> Iterator[None]:
    if not _exclude_ocr_wait_enabled(config):
        yield
        return
    started_at = now()
    try:
        yield
    finally:
        _add_excluded_seconds(config, now() - started_at)


@dataclass
class StepDeadline:
    config: dict[str, Any] | None
    timeout_seconds: float
    started_at: float
    excluded_at_start: float
    now: Callable[[], float] = time.monotonic

    def elapsed_seconds(self) -> float:
        excluded_since_start = _excluded_seconds(self.config) - self.excluded_at_start
        return max(0.0, self.now() - self.started_at - max(0.0, excluded_since_start))

    def expired(self) -> bool:
        return self.elapsed_seconds() >= max(0.0, self.timeout_seconds)

    def remaining_seconds(self) -> float:
        return max(0.0, self.timeout_seconds - self.elapsed_seconds())


def step_deadline(
    config: dict[str, Any] | None,
    timeout_seconds: float,
    *,
    now: Callable[[], float] = time.monotonic,
) -> StepDeadline:
    return StepDeadline(
        config=config,
        timeout_seconds=float(timeout_seconds),
        started_at=now(),
        excluded_at_start=_excluded_seconds(config),
        now=now,
    )
