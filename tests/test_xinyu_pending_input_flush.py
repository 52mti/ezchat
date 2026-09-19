"""进入聊天后补发输入框残留文本的契约。

背景（20260919）：语音输入轮次在文本已进入输入框后发送失败（异常/复位
中断），下次进入该会话输入框仍有字（发送按钮可匹配）。此时正确动作是
点一下发送、返回列表、本轮完成——不能重复走 TTS 输入，否则同一句话
打进两次。

契约：
1. 发送按钮可匹配 → 点发送 + go_back 返回列表，返回 True；
2. 发送按钮不可匹配 → 不点击不返回，返回 False，走正常回复流程；
3. app 层：补发成功 → 本轮完成（mark_success、不进 AI/TTS、无运行异常）。
"""

from __future__ import annotations

from typing import Any

import apps.xinyu.app as xinyu_app
import apps.xinyu.steps as steps


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def test_flush_pending_input_clicks_send_and_returns(monkeypatch):
    robot = FakeRobot()
    back_calls: list[tuple[Any, ...]] = []

    def fake_template_center(image, cfg, key, group=None):
        assert key == steps.SEND_BUTTON_KEY
        return (0.88, 0.92)

    monkeypatch.setattr(steps, "_template_center", fake_template_center)
    monkeypatch.setattr(steps, "go_back", lambda robot, cfg, reporter=None: back_calls.append((1,)) or 0)

    result = steps.flush_pending_input(robot, {"_runtime_cache": {}}, image="chat")

    assert result is True
    assert robot.clicks == [(0.88, 0.92)]  # 只点一次发送
    assert back_calls == [(1,)]  # 随后返回列表


def test_flush_pending_input_returns_false_without_send_button(monkeypatch):
    robot = FakeRobot()
    monkeypatch.setattr(steps, "_template_center", lambda image, cfg, key, group=None: None)
    monkeypatch.setattr(
        steps, "go_back", lambda *a, **k: (_ for _ in ()).throw(AssertionError("无残留不应返回列表"))
    )

    result = steps.flush_pending_input(robot, {"_runtime_cache": {}}, image="chat")

    assert result is False
    assert robot.clicks == []


def test_app_flushes_pending_input_and_ends_round(monkeypatch, tmp_path):
    config = {"greetings_dir": "", "_runtime_cache": {}}
    robot = FakeRobot()
    logs: list[tuple[str, dict | None]] = []
    target = {"name": "残留君", "content": "在吗", "unread_count": 1, "x": 0.2, "y": 0.3}

    monkeypatch.setattr(xinyu_app.steps, "capture_after_popups", lambda robot, cfg, label=None, reporter=None: "entry")
    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, cfg, reporter=None, image=None: dict(target))
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: None)
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window_after_list_click", lambda robot, cfg, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "flush_pending_input", lambda robot, cfg, image=None, reporter=None: True)
    monkeypatch.setattr(
        xinyu_app.steps,
        "send_new_greeting_if_available",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("补发轮不应再走新招呼检测")),
    )
    monkeypatch.setattr(
        xinyu_app.steps,
        "read_message",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("补发轮不应再读消息")),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload)))

    app.run_once()

    assert any(message == "心遇补发输入框残留文本" for message, _payload in logs)
    assert not any(message == "app_error" for message, _payload in logs)
    assert app.status.error_count == 0
    assert app.status.last_success_at  # 本轮按成功收尾
