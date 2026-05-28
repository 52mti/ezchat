from __future__ import annotations

import pytest

from apps.base import BaseApp
from apps.cancellation import OperationCancelled
from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.utils import tts_input


class CancellableRobot:
    host = "192.168.1.108"

    def __init__(self, stopped: bool = False) -> None:
        self.stopped = stopped
        self.clicks: list[tuple[float, float]] = []
        self.captures = 0
        self.audio: list[str] = []
        self.recoveries = 0

    def _codex_should_stop(self) -> bool:
        return self.stopped

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def get_transform_image(self, pre_move=True, save_path=None) -> bytes:
        self.captures += 1
        return b"not-used"

    def play_audio(self, text: str, wait: bool = True) -> None:
        self.audio.append(text)
        self.stopped = True

    def reset_device(self, wait: bool = True) -> None:
        self.recoveries += 1


class DemoApp(BaseApp):
    def run_once(self) -> None:
        return None


def test_click_is_blocked_after_stop() -> None:
    robot = CancellableRobot(stopped=True)

    with pytest.raises(OperationCancelled):
        click_image_point(robot, 0.5, 0.5)

    assert robot.clicks == []


def test_capture_is_blocked_after_stop() -> None:
    robot = CancellableRobot(stopped=True)

    with pytest.raises(OperationCancelled):
        capture_transform_image(robot, {"_runtime": {"should_stop": robot._codex_should_stop}})

    assert robot.captures == 0


def test_tts_stops_before_next_chunk() -> None:
    robot = CancellableRobot(stopped=False)

    with pytest.raises(OperationCancelled):
        tts_input(robot, "abcdefghijkl", max_chars=5, pause_seconds=0)

    assert robot.audio == ["abcde"]


def test_cancelled_app_error_does_not_recover_robot() -> None:
    robot = CancellableRobot()
    app = DemoApp(robot=robot, config={})

    app.on_error(OperationCancelled("operation cancelled"))

    assert app.get_status()["error_count"] == 0
    assert robot.recoveries == 0


def test_stopped_app_error_does_not_recover_robot() -> None:
    robot = CancellableRobot(stopped=True)
    app = DemoApp(robot=robot, config={})

    app.on_error(RuntimeError("network timeout"))

    assert app.get_status()["error_count"] == 0
    assert robot.recoveries == 0
