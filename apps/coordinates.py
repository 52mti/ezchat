from __future__ import annotations

from typing import Any

from apps.cancellation import raise_if_cancelled


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def image_to_robot_point(x: float, y: float) -> tuple[float, float]:
    """Temporarily pass portrait normalized coordinates through directly for live verification."""
    return round(clamp_unit(x), 6), round(clamp_unit(y), 6)


def click_image_point(robot: Any, x: float, y: float) -> None:
    raise_if_cancelled(robot)
    robot_x, robot_y = image_to_robot_point(x, y)
    raise_if_cancelled(robot)
    robot.move_click(robot_x, robot_y)
