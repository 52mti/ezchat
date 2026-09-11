from __future__ import annotations

import random
from typing import Any

from services.cancellation import raise_if_cancelled


CLICK_JITTER_MIN = 0.001
CLICK_JITTER_MAX = 0.003


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def image_to_robot_point(x: float, y: float) -> tuple[float, float]:
    return round(clamp_unit(x), 6), round(clamp_unit(y), 6)


def _robot_click_jitter_enabled(robot: Any) -> bool:
    return bool(getattr(robot, "_codex_click_jitter_enabled", False))


def robot_point_with_jitter(robot: Any, x: float, y: float) -> tuple[float, float]:
    base_x, base_y = image_to_robot_point(x, y)
    if not _robot_click_jitter_enabled(robot):
        return base_x, base_y
    jitter_x = random.uniform(CLICK_JITTER_MIN, CLICK_JITTER_MAX)
    jitter_y = random.uniform(CLICK_JITTER_MIN, CLICK_JITTER_MAX)
    offset_x = jitter_x if random.random() < 0.5 else -jitter_x
    offset_y = jitter_y if random.random() < 0.5 else -jitter_y
    return (
        round(clamp_unit(base_x + offset_x), 6),
        round(clamp_unit(base_y + offset_y), 6),
    )


def click_image_point(robot: Any, x: float, y: float) -> None:
    raise_if_cancelled(robot)
    robot_x, robot_y = robot_point_with_jitter(robot, x, y)
    raise_if_cancelled(robot)
    robot.move_click(robot_x, robot_y)
