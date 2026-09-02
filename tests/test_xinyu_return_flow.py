from copy import deepcopy

import pytest

from apps.xinyu import steps
from apps.xinyu.config import DEFAULT_CONFIG


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


@pytest.mark.parametrize(
    ("microphone_visible", "expected_clicks"),
    [(False, 1), (True, 2)],
)
def test_xinyu_send_return_double_click_requires_microphone(monkeypatch, microphone_visible, expected_clicks):
    config = deepcopy(DEFAULT_CONFIG)
    robot = FakeRobot()
    images = iter(["chat", "chat", "list"])

    monkeypatch.setattr(steps, "ensure_send_cleared", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda *_args, **_kwargs: next(images),
    )
    monkeypatch.setattr(
        steps,
        "_matches_chat_list_screen",
        lambda image, *_args, **_kwargs: image == "list",
    )
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda _image, _config, key, **_kwargs: (
            (0.2, 0.1)
            if key == steps.BACK_BUTTON
            else (0.5, 0.6)
            if microphone_visible and key == steps.MICROPHONE_BUTTON
            else None
        ),
    )
    monkeypatch.setattr(steps.time, "sleep", lambda _seconds: None)

    clicks = steps.go_back(robot, config, double_click_on_first_attempt=True)

    assert clicks == expected_clicks
    assert robot.clicks == [pytest.approx((0.2, 0.1))] * expected_clicks
