from __future__ import annotations

from apps.huanyou import app as huanyou_app


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def test_huanyou_app_reports_detailed_enter_chat_log(monkeypatch):
    robot = FakeRobot()
    events: list[dict[str, object]] = []

    monkeypatch.setattr(huanyou_app.steps, "find_unread", lambda robot, config, reporter=None: (0.2, 0.3))
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "read_message", lambda robot, config, reporter=None: {"contact": "用户", "content": "你好"})
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None: None)

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "回复")
    app.set_status_callback(events.append)

    app.run_once()

    assert robot.clicks == [(0.2, 0.3)]
    assert any(event.get("current_step_label") == "进入聊天界面" for event in events)
    assert any(event.get("current_step_log_label") == "进入聊天界面：(0.200, 0.300)" for event in events)
