"""Map validated AutoGLM actions to this project's mechanical-arm API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from apps.coordinates import click_image_point
from auto_glm.protocol import AutoGLMAction


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str
    requires_user: bool = False


class RobotActionAdapter:
    """Execute safe model actions through explicit project callbacks.

    ``Back`` and text input are app-specific for a camera/robot solution, so
    callers must deliberately provide their implementations. This prevents a
    model fallback from guessing physical coordinates for those actions.
    """

    def __init__(
        self,
        robot: Any,
        *,
        back_handler: Callable[[], bool] | None = None,
        text_handler: Callable[[str], bool] | None = None,
        point_resolver: Callable[[tuple[float, float], AutoGLMAction], tuple[float, float] | None] | None = None,
        max_wait_seconds: float = 10.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.back_handler = back_handler
        self.text_handler = text_handler
        self.point_resolver = point_resolver
        self.max_wait_seconds = max(0.0, float(max_wait_seconds))
        self.sleeper = sleeper

    def execute(self, action: AutoGLMAction) -> ActionResult:
        if action.finished:
            return ActionResult(True, action.message or "模型确认任务完成")
        if action.name in {"Tap", "Double Tap"}:
            x, y = _to_normalized(action.arguments["element"])
            if not self.point_resolver:
                return ActionResult(False, "模型坐标仅是候选，当前 App 未配置局部复核器")
            resolved = self.point_resolver((x, y), action)
            if resolved is None:
                return ActionResult(False, "局部模板/OCR 复核未确认模型点击位置")
            x, y = resolved
            click_image_point(self.robot, x, y)
            if action.name == "Double Tap":
                click_image_point(self.robot, x, y)
            return ActionResult(True, f"机械臂已执行{action.name}：({x:.3f}, {y:.3f})")
        if action.name == "Swipe":
            if not hasattr(self.robot, "swipe"):
                return ActionResult(False, "机械臂未提供滑动接口")
            start_x, start_y = _to_normalized(action.arguments["start"])
            end_x, end_y = _to_normalized(action.arguments["end"])
            self.robot.swipe(start_x, start_y, end_x, end_y, wait=True)
            return ActionResult(True, f"机械臂已执行滑动：({start_x:.3f}, {start_y:.3f}) -> ({end_x:.3f}, {end_y:.3f})")
        if action.name in {"Type", "Type_Name"}:
            if not self.text_handler:
                return ActionResult(False, "当前 App 未配置 AutoGLM 文字输入执行器")
            text = str(action.arguments.get("text", ""))
            return ActionResult(bool(self.text_handler(text)), "已交给 App 的文字输入执行器")
        if action.name == "Back":
            if not self.back_handler:
                return ActionResult(False, "当前 App 未配置 AutoGLM 返回执行器")
            return ActionResult(bool(self.back_handler()), "已交给 App 的返回执行器")
        if action.name == "Wait":
            seconds = min(_wait_seconds(action.arguments.get("duration")), self.max_wait_seconds)
            self.sleeper(seconds)
            return ActionResult(True, f"已等待 {seconds:.1f} 秒")
        if action.name == "Take_over":
            return ActionResult(False, action.message or "模型请求人工接管", requires_user=True)
        return ActionResult(False, f"当前机械臂执行器不支持 AutoGLM 动作：{action.name}")


def _to_normalized(point: list[float]) -> tuple[float, float]:
    return float(point[0]) / 1000, float(point[1]) / 1000


def _wait_seconds(value: Any) -> float:
    try:
        return max(0.0, float(str(value).split()[0]))
    except (TypeError, ValueError, IndexError):
        return 1.0
