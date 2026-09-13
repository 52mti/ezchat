"""未回复视图/对话层对「无可读内容消息」的兜底契约。

背景事故：对方消息是纯表情（👋）时——
- 列表层：行无角标且预览 OCR 为空 → 进不了候选 → 「暂无消息」永久空转；
- 对话层：read_message 找不到 sender1 文字 → 抛异常 → 「运行异常」。

契约：
1. 未回复视图确认后，无角标无预览的行按首行兜底选中（有名字即可）；
2. read_message 区分「对方消息缺失」（可当你好）与「整屏无文字」（保持报错）；
3. app 层接住 DialogueOpponentMessageMissing → 从 pingtai.txt 挑招呼语回复。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import apps.xinyu.app as xinyu_app
import apps.xinyu.steps as steps


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def play_audio(self, chunk: str, wait: bool = True) -> None:
        pass


def _emoji_row() -> dict:
    return {
        "name": "李麦熟",
        "content": "",
        "bounds": [0.177, 0.173, 0.97, 0.222],
        "unread_count": None,
        "favorability": 31,
    }


def test_fallback_unread_view_target_picks_emoji_only_row():
    ocr_result = {"result": {"message_list": [_emoji_row()]}, "ocr_items": []}

    target = steps._fallback_unread_view_target(ocr_result)

    assert target is not None
    assert target["name"] == "李麦熟"
    assert target["unread_count"] == 0
    assert target["x"] == pytest.approx((0.177 + 0.97) / 2)
    assert target["y"] == pytest.approx((0.173 + 0.222) / 2)


def test_fallback_unread_view_target_skips_placeholder_and_nameless_rows():
    rows = [
        {"name": "暂无消息", "content": "", "bounds": [0.1, 0.3, 0.9, 0.35]},
        {"name": "", "content": "", "bounds": [0.1, 0.4, 0.9, 0.45]},  # 无昵称的误检行
        {"name": "无消息", "content": "", "bounds": [0.1, 0.5, 0.9, 0.55]},
    ]

    assert steps._fallback_unread_view_target({"result": {"message_list": rows}}) is None
    assert steps._fallback_unread_view_target({"result": {"message_list": []}}) is None
    assert steps._fallback_unread_view_target({}) is None


def test_find_unread_target_falls_back_to_first_row_in_unread_view(monkeypatch):
    ocr_result = {
        "result": {"message_list": [_emoji_row()]},
        "ocr_items": [],
        "_xinyu_unread_tab_has_red": True,
    }
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, cfg, reporter=None, image=None: "img")
    monkeypatch.setattr(steps, "_run_list_ocr", lambda image, cfg, reporter=None: ocr_result)
    monkeypatch.setattr(
        steps,
        "check_unread_button",
        lambda robot, cfg, image, reporter=None, ocr_result=None: ("img", ocr_result),
    )
    reports: list[tuple[str, str]] = []

    target = steps.find_unread_target(
        FakeRobot(),
        {"_runtime_cache": {}},
        reporter=lambda step, label, log_label=None: reports.append((step, label)),
    )

    assert target is not None and target["name"] == "李麦熟"
    assert any(step == "list_target_fallback" for step, _label in reports)


def test_find_unread_target_prefers_badged_row_without_fallback(monkeypatch):
    badged_row = {
        "name": "葡萄树",
        "content": "在吗",
        "bounds": [0.1, 0.3, 0.9, 0.35],
        "unread_count": 2,
    }
    ocr_result = {
        "result": {"message_list": [_emoji_row(), badged_row]},
        "ocr_items": [],
        "_xinyu_unread_tab_has_red": True,
    }
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, cfg, reporter=None, image=None: "img")
    monkeypatch.setattr(steps, "_run_list_ocr", lambda image, cfg, reporter=None: ocr_result)
    monkeypatch.setattr(
        steps,
        "check_unread_button",
        lambda robot, cfg, image, reporter=None, ocr_result=None: ("img", ocr_result),
    )
    reports: list[tuple[str, str]] = []

    target = steps.find_unread_target(
        FakeRobot(),
        {"_runtime_cache": {}},
        reporter=lambda step, label, log_label=None: reports.append((step, label)),
    )

    assert target is not None and target["name"] == "葡萄树"
    assert not any(step == "list_target_fallback" for step, _label in reports)


def test_read_message_raises_dialogue_missing_when_items_have_no_sender1(monkeypatch):
    config = {"_runtime_cache": {}, "ocr_modes": {"dialogue": {"rules": {}}}}
    monkeypatch.setattr(steps, "_dialogue_engine", lambda cfg: "local")
    monkeypatch.setattr(steps, "_dialogue_message_region", lambda image, cfg: None)
    monkeypatch.setattr(steps, "_update_dynamic_chat_area", lambda cfg, payload, ocr_result: False)
    monkeypatch.setattr(
        steps,
        "_run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"dialogues": [{"sender": "sender0", "content": "怎么啦，取消了啥呀"}]},
            "ocr_items": [{"text": "亲密度 31"}],
        },
    )

    with pytest.raises(steps.DialogueOpponentMessageMissing) as excinfo:
        steps.read_message(FakeRobot(), config, image="chat")
    assert isinstance(excinfo.value, steps.StepUnavailable)


def test_read_message_raises_plain_unavailable_when_screen_blank(monkeypatch):
    config = {"_runtime_cache": {}, "ocr_modes": {"dialogue": {"rules": {}}}}
    monkeypatch.setattr(steps, "_dialogue_engine", lambda cfg: "local")
    monkeypatch.setattr(steps, "_dialogue_message_region", lambda image, cfg: None)
    monkeypatch.setattr(steps, "_update_dynamic_chat_area", lambda cfg, payload, ocr_result: False)
    monkeypatch.setattr(
        steps,
        "_run_ocr_mode",
        lambda image, cfg, mode, region=None: {"result": {"dialogues": []}, "ocr_items": []},
    )

    with pytest.raises(steps.StepUnavailable) as excinfo:
        steps.read_message(FakeRobot(), config, image="chat")
    assert not isinstance(excinfo.value, steps.DialogueOpponentMessageMissing)


def test_app_treats_missing_opponent_message_as_greeting(monkeypatch, tmp_path):
    greetings_dir = tmp_path / "greetings"
    greetings_dir.mkdir()
    (greetings_dir / "pingtai.txt").write_text("嗨，很高兴认识你~\n", encoding="utf-8")
    config = {"greetings_dir": str(greetings_dir), "_runtime_cache": {}}
    robot = FakeRobot()
    logs: list[tuple[str, dict | None]] = []
    reply_calls: list[dict] = []
    played: list[str] = []

    target = _emoji_row() | {"x": 0.5, "y": 0.2}

    def fake_read_message(robot, config, target, image=None, reporter=None):
        raise steps.DialogueOpponentMessageMissing("聊天详情 OCR 未识别到对方消息：目标 sender=sender1")

    monkeypatch.setattr(xinyu_app.steps, "capture_after_popups", lambda robot, cfg, label=None, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, cfg, reporter=None, image=None: target)
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: None)
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, cfg, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, cfg, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "read_message", fake_read_message)
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, cfg, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, cfg, reporter=None: played.append(text))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, cfg, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, cfg, double_click_on_first_attempt=False, reporter=None: 1)

    app = xinyu_app.XinyuApp(
        robot,
        config,
        reply_provider=lambda message: reply_calls.append(message) or "AI回复",
    )
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload)))

    app.run_once()

    assert played == ["嗨，很高兴认识你~"]  # 招呼语被语音发出
    assert reply_calls == []  # 未走 AI 回复
    assert any(message == "对方消息无可读内容，按招呼处理" for message, _payload in logs)
