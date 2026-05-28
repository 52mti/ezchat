from __future__ import annotations

from typing import Any, Callable


Reporter = Callable[..., None]


def format_point(x: float, y: float) -> str:
    return f"({float(x):.3f}, {float(y):.3f})"


def report_step(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    if reporter is not None:
        reporter(step, label, log_label=log_label)


def point_label(action: str, x: float, y: float) -> str:
    return f"{action}：{format_point(x, y)}"


def center_from_bounds(bounds: list[float]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return (left + right) / 2, (top + bottom) / 2
