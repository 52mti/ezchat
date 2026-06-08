from __future__ import annotations

import random

import pytest

from apps.coordinates import click_image_point
from apps.coordinates import image_to_robot_point
from apps.coordinates import robot_point_with_jitter


def test_image_to_robot_point_passes_portrait_coordinates_through():
    assert image_to_robot_point(0.0, 0.0) == (0.0, 0.0)
    assert image_to_robot_point(1.0, 0.0) == (1.0, 0.0)
    assert image_to_robot_point(0.0, 1.0) == (0.0, 1.0)
    assert image_to_robot_point(1.0, 1.0) == (1.0, 1.0)
    assert image_to_robot_point(0.25, 0.5) == (0.25, 0.5)


class _FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


class _RealRobotLike(_FakeRobot):
    __module__ = "lgb.lgb"
    __name__ = "RobotInterface"


def test_robot_point_with_jitter_keeps_fake_robot_unchanged():
    robot = _FakeRobot()

    assert robot_point_with_jitter(robot, 0.25, 0.5) == (0.25, 0.5)


def test_robot_point_with_jitter_applies_small_offset_for_real_robot_like(monkeypatch):
    robot = _RealRobotLike()
    robot._codex_click_jitter_enabled = True
    monkeypatch.setattr(random, "random", lambda: 0.6)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.002)

    x, y = robot_point_with_jitter(robot, 0.25, 0.5)

    assert x == pytest.approx(0.248)
    assert y == pytest.approx(0.498)


def test_click_image_point_uses_jittered_coordinates_for_real_robot_like(monkeypatch):
    robot = _RealRobotLike()
    robot._codex_click_jitter_enabled = True
    monkeypatch.setattr(random, "random", lambda: 0.6)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.002)

    click_image_point(robot, 0.25, 0.5)

    assert len(robot.clicks) == 1
    assert robot.clicks[0][0] == pytest.approx(0.248)
    assert robot.clicks[0][1] == pytest.approx(0.498)
