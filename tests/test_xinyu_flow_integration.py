"""flow 引擎 v1 接入测试：回复段（PREPARING→VOICE→SENDING→RETURNING）。

steps 层全部打桩，验证编排语义而非视觉逻辑：
- happy path 走完 mark_sent/mark_success；
- 麦克风缺失+已回列表 → 优雅放弃（不计错、留痕、连续 3 次升 warning、
  5 次升级错误、成功后计数清零）；
- AI 失败 → 计 1 次错误但不熔断（阈值内）；
- 开关默认关闭（enable_flow_engine 不在 DEFAULT_CONFIG）。
"""

from __future__ import annotations

from typing import Any

import pytest

import apps.xinyu.steps as steps
from apps.base import ReplyAction
from apps.xinyu.app import XinyuApp


class _FakeDeadline:
    def expired(self) -> bool:
        return False


@pytest.fixture()
def app(monkeypatch):
    robot = type("Robot", (), {"host": "192.168.1.210"})()
    sent: list[tuple[ReplyAction, dict]] = []
    application = XinyuApp(
        robot,
        {"enable_flow_engine": True, "poll_interval": 1},
        reply_provider=lambda message: ReplyAction(text="你好呀"),
        sent_callback=lambda action, message: sent.append((action, message)),
    )
    application.set_log_callback(lambda level, message, payload: logs.append((level, message, payload)))

    logs: list[tuple[str, str, Any]] = []
    application.set_log_callback(lambda level, message, payload: logs.append((level, message, payload)))
    application._test_sent = sent
    application._test_logs = logs

    monkeypatch.setattr(steps, "focus_input_box", lambda *a, **k: None)
    monkeypatch.setattr(steps, "prepare_tts_input", lambda robot, text: object())
    monkeypatch.setattr(steps, "play_prepared_tts_input", lambda robot, chunks: None)
    monkeypatch.setattr(steps, "click_image_point", lambda robot, x, y: None)
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **k: object())
    monkeypatch.setattr(steps, "step_deadline", lambda cfg, seconds: _FakeDeadline())
    monkeypatch.setattr(steps, "go_back", lambda robot, cfg, reporter=None, **k: 0)
    monkeypatch.setattr(steps, "classify_screen", lambda image, cfg: "chat")
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    application._test_monkeypatch = monkeypatch
    return application


def _run_reply_segment(application, *, mic_match=None, screen="chat"):
    mp = application._test_monkeypatch
    mp.setattr(steps, "_template_match", lambda image, cfg, key, group=None: mic_match)
    mp.setattr(steps, "classify_screen", lambda image, cfg: screen)
    mp.setattr(steps, "send_reply", lambda robot, cfg, reporter=None: 1)
    message = {"contact": "月光", "content": "在吗"}
    application._reply_segment_flow(
        message,
        "月光",
        "",
        0.0,
        lambda step, label, **kw: None,
    )


def test_flow_happy_path_marks_sent_and_success(app):
    _run_reply_segment(app, mic_match={"x": 0.1, "y": 0.5, "w": 0.05, "h": 0.05})

    assert len(app._test_sent) == 1
    assert app._test_sent[0][0].text == "你好呀"
    assert app.status.last_success_at != ""
    assert app.status.error_count == 0
    assert app._abandon_streak == 0


def test_flow_mic_missing_back_at_list_abandons_gracefully(app):
    _run_reply_segment(app, mic_match=None, screen="list")

    assert app._test_sent == []                      # 未发送
    assert app.status.error_count == 0               # 不计熔断
    assert app.status.is_circuit_open is False
    assert app._abandon_streak == 1
    assert any(message == "心遇流程优雅放弃本轮" for _level, message, _payload in app._test_logs)


def test_flow_abandon_streak_escalates_at_threshold(app):
    for _ in range(4):
        _run_reply_segment(app, mic_match=None, screen="list")
    assert app._abandon_streak == 4
    warnings = [
        (level, payload)
        for level, message, payload in app._test_logs
        if message == "心遇流程优雅放弃本轮"
    ]
    assert warnings[-1][0] == "warning"  # 第 3 次起升为 warning

    with pytest.raises(RuntimeError, match="连续优雅放弃"):
        _run_reply_segment(app, mic_match=None, screen="list")
    assert app._abandon_streak == 5


def test_flow_success_resets_abandon_streak(app):
    _run_reply_segment(app, mic_match=None, screen="list")
    _run_reply_segment(app, mic_match=None, screen="list")
    assert app._abandon_streak == 2
    _run_reply_segment(app, mic_match={"x": 0.1, "y": 0.5, "w": 0.05, "h": 0.05})
    assert app._abandon_streak == 0


def test_flow_ai_failure_counts_one_error_without_circuit(app, monkeypatch):
    def _boom(message):
        raise RuntimeError("DeepSeek 生成失败")

    monkeypatch.setattr(app, "reply_provider", _boom)
    monkeypatch.setattr(steps, "_template_match", lambda image, cfg, key, group=None: {"x": 0.1, "y": 0.5, "w": 0.05, "h": 0.05})
    monkeypatch.setattr(steps, "send_reply", lambda robot, cfg, reporter=None: 1)

    message = {"contact": "月光", "content": "在吗"}
    app._reply_segment_flow(message, "月光", "", 0.0, lambda step, label, **kw: None)

    assert app.status.error_count == 1
    assert app.status.is_circuit_open is False
    assert app._test_sent == []


def test_flow_engine_disabled_by_default(monkeypatch):
    robot = type("Robot", (), {"host": "192.168.1.210"})()
    application = XinyuApp(robot, {}, reply_provider=lambda message: ReplyAction(text="x"))
    monkeypatch.setattr(
        steps,
        "focus_input_box",
        lambda *a, **k: pytest.fail("开关关闭时不应进入 flow 回复段路径的提前聚焦"),
    )
    assert application._flow_engine_enabled() is False
