from __future__ import annotations

import threading

from apps.wechat.app import WechatApp


class FakeRobot:
    pass


def test_keep_awake_uses_idle_time_before_triggering(monkeypatch):
    calls: list[str] = []
    now = {"value": 100.0}

    monkeypatch.setattr("apps.wechat.app.time.monotonic", lambda: now["value"])
    monkeypatch.setattr("apps.wechat.app.steps.find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(
        "apps.wechat.app.steps.keep_screen_awake",
        lambda robot, config, reporter=None: calls.append("keep_awake") or True,
    )

    app = WechatApp(FakeRobot(), {"keep_awake_interval_seconds": 30})

    now["value"] = 129.0
    app.run_once()

    assert calls == []

    now["value"] = 130.0
    app.run_once()

    assert calls == ["keep_awake"]


def test_run_once_requests_ai_before_waiting_and_focuses_input_box(monkeypatch):
    order: list[str] = []
    ai_started = threading.Event()
    focus_done = threading.Event()

    def fake_reply_provider(message):
        order.append("ai_start")
        ai_started.set()
        assert focus_done.wait(1.0)
        return "回复"

    monkeypatch.setattr("apps.wechat.app.time.monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "apps.wechat.app.steps.find_unread_target",
        lambda robot, config, reporter=None: {"name": "张三", "x": 0.3, "y": 0.4, "unread_count": 1},
    )
    monkeypatch.setattr("apps.wechat.app.steps.enter_chat", lambda robot, x, y: order.append("enter_chat"))
    monkeypatch.setattr(
        "apps.wechat.app.steps.read_message",
        lambda robot, config, target, reporter=None: {"contact": "张三", "content": "你好"},
    )

    def fake_focus_input_box(robot, config, reporter=None):
        assert ai_started.wait(1.0)
        order.append("focus_input_box")
        focus_done.set()

    monkeypatch.setattr("apps.wechat.app.steps.focus_input_box", fake_focus_input_box)
    monkeypatch.setattr("apps.wechat.app.steps.play_voice_reply", lambda robot, text, config, reporter=None: order.append("play_voice_reply"))
    monkeypatch.setattr("apps.wechat.app.steps.send_reply", lambda robot, config, reporter=None: order.append("send_reply"))
    monkeypatch.setattr("apps.wechat.app.steps.go_back", lambda robot, config, reporter=None: order.append("go_back"))

    app = WechatApp(FakeRobot(), {}, reply_provider=fake_reply_provider)

    app.run_once()

    assert order[:3] == ["enter_chat", "ai_start", "focus_input_box"]
    assert order[3:] == ["play_voice_reply", "send_reply", "go_back"]
