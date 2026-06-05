from __future__ import annotations

from pathlib import Path

import pytest

from apps.wechat import steps


class FakeRobot:
    def __init__(self, images: list[str]) -> None:
        self.images = list(images)
        self.clicks: list[tuple[float, float]] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return self.images.pop(0) if self.images else "unknown"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"template")


def _config(tmp_path: Path) -> dict[str, dict[str, str]]:
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    for path in [
        buttons_dir / "btn_输入框按钮.jpg",
        buttons_dir / "btn_麦克风按钮.jpg",
        buttons_dir / "btn_发送按钮.jpg",
        buttons_dir / "btn_返回按钮_2.jpg",
        screens_dir / "screen_聊天列表.jpg",
        screens_dir / "screen_聊天界面.jpg",
    ]:
        _touch(path)
    return {
        "templates": {
            "buttons": str(buttons_dir),
            "screens": str(screens_dir),
        },
        "step_timeout": 0.2,
    }


def test_find_unread_target_uses_chat_list_screen_and_list_ocr(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {
                        "name": "张三",
                        "content": "你好",
                        "unread_count": 2,
                        "bounds": [0.1, 0.2, 0.3, 0.4],
                    }
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target == {
        "name": "张三",
        "content": "你好",
        "unread_count": 2,
        "bounds": [0.1, 0.2, 0.3, 0.4],
        "x": pytest.approx(0.2),
        "y": pytest.approx(0.3),
    }


def test_find_unread_target_skips_system_notification_rows(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {
                        "name": "用户通知",
                        "content": "您的好友申请已通过",
                        "unread_count": 99,
                        "bounds": [0.1, 0.1, 0.5, 0.2],
                    },
                    {
                        "name": "无敌阿强",
                        "content": "你看那部新出的电影了吗",
                        "unread_count": 4,
                        "bounds": [0.2, 0.5, 0.6, 0.6],
                    },
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target is not None
    assert target["name"] == "无敌阿强"
    assert target["unread_count"] == 4
    assert target["x"] == pytest.approx(0.4)
    assert target["y"] == pytest.approx(0.55)


def test_read_message_returns_last_sender1_message(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name == "screen_聊天界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "user_name": "李四",
                "dialogues": [
                    {"sender": "sender1", "content": "第一句"},
                    {"sender": "sender2", "content": "收到"},
                    {"sender": "sender1", "content": "最后一句"},
                ],
            },
            "ocr_items": [{"text": "李四"}],
        },
    )

    message = steps.read_message(robot, config, {"name": "李四", "unread_count": 3})

    assert message["contact"] == "李四"
    assert message["content"] == "最后一句"
    assert message["unread_count"] == "3"


def test_input_reply_clicks_input_offset_then_microphone(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["input", "microphone"])
    played: list[str] = []
    sleeps: list[float] = []

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "input" and name == "btn_输入框按钮.jpg":
            return {"x": 0.6, "y": 0.3, "w": 0.2, "h": 0.2, "score": 0.99}
        if image == "microphone" and name == "btn_麦克风按钮.jpg":
            return {"x": 0.2, "y": 0.5, "w": 0.2, "h": 0.2, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "tts_input", lambda robot, text: played.append(text))
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: sleeps.append(seconds))

    steps.input_reply(robot, "你好啊", config)

    assert robot.clicks == [pytest.approx((0.2, 0.38)), pytest.approx((0.3, 0.6))]
    assert played == ["你好啊"]
    assert sleeps == [3.0]


def test_send_reply_clicks_send_once_and_waits_until_disappears(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["send_visible", "send_gone"])

    def fake_match(image, template, threshold=0.85):
        if Path(template).name != "btn_发送按钮.jpg":
            return None
        if image == "send_visible":
            return {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    steps.send_reply(robot, config)

    assert robot.clicks == [pytest.approx((0.6, 0.6))]


def test_go_back_clicks_back_button_twice(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "chat" and name == "btn_返回按钮_2.jpg":
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "chat" and name == "screen_聊天列表.jpg":
            return None
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    steps.go_back(robot, config)

    assert robot.clicks == [pytest.approx((0.15, 0.15)), pytest.approx((0.15, 0.15))]


def test_keep_screen_awake_clicks_bottom_nav_positions_from_list_ocr(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    sleeps: list[float] = []

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(steps, "_ensure_list_screen", lambda robot, cfg, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "微信", "bounds": [0.05, 0.90, 0.15, 0.98]},
                    {"name": "发现", "bounds": [0.42, 0.90, 0.52, 0.98]},
                ]
            },
            "ocr_items": [],
        },
    )

    assert steps.keep_screen_awake(robot, config)
    assert robot.clicks == [pytest.approx((0.10, 0.94))]


def test_template_candidates_include_incremental_variants(tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    for name in ["btn_发送按钮.jpg", "btn_发送按钮_2.jpg"]:
        (buttons_dir / name).write_bytes(b"template")

    candidates = steps._template_candidates(
        {"templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)}},
        steps.SEND_BUTTON_KEY,
        group="buttons",
    )

    assert [path.name for path in candidates] == ["btn_发送按钮.jpg", "btn_发送按钮_2.jpg"]
