from __future__ import annotations

from pathlib import Path

import pytest

from apps.xinyu import app as xinyu_app
from apps.xinyu import steps


class FakeRobot:
    def __init__(self, images: list[object]) -> None:
        self.images = list(images)
        self.clicks: list[tuple[float, float]] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return self.images.pop(0) if self.images else "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"template")


def _config(tmp_path: Path) -> dict[str, object]:
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    for path in [
        buttons_dir / "btn_返回按钮.jpg",
        buttons_dir / "btn_我的按钮.jpg",
        buttons_dir / "btn_消息按钮.jpg",
        screens_dir / "screen_聊天列表.jpg",
        screens_dir / "screen_聊天界面.jpg",
        screens_dir / "screen_我的界面.jpg",
        screens_dir / "screen_黑屏异常.jpg",
    ]:
        _touch(path)
    return {"templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)}, "step_timeout": 0.2}


def test_xinyu_read_my_points_clicks_my_reads_points_and_returns_message(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "list", "my", "list"])
    seen_modes: list[str] = []

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "list" and name == "btn_我的按钮.jpg":
            return {"x": 0.78, "y": 0.86, "w": 0.12, "h": 0.08, "score": 0.99}
        if image == "my" and name == "screen_我的界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "my" and name == "btn_消息按钮.jpg":
            return {"x": 0.56, "y": 0.86, "w": 0.12, "h": 0.08, "score": 0.99}
        return None

    def fake_ocr(image, cfg, mode, region=None):
        seen_modes.append(mode)
        return {"result": {"points": "1288", "balance": "1288"}, "ocr_items": []}

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.read_my_points(robot, config) == "1288"
    assert seen_modes == ["points"]
    assert robot.clicks == [pytest.approx((0.84, 0.9)), pytest.approx((0.62, 0.9))]


def test_xinyu_app_reads_points_once_per_idle_wait_and_reports_balance(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []
    logs: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(
        xinyu_app.steps,
        "keep_screen_awake",
        lambda robot, config, reporter=None: calls.append("keep_awake") or True,
    )
    monkeypatch.setattr(
        xinyu_app.steps,
        "read_my_points",
        lambda robot, config, reporter=None: calls.append("points") or "1288",
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI")
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload)))
    app.last_activity_at -= 31

    app.run_once()
    app.run_once()

    assert calls == ["keep_awake", "points"]
    assert ("xinyu_balance", {"balance": "128", "points": "1288"}) in logs


def test_xinyu_balance_display_truncates_decimals_after_dividing_by_ten():
    assert xinyu_app.XinyuApp._balance_from_points("1013.25") == "101"


def test_xinyu_app_reads_points_again_after_real_message_resets_idle_wait(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 0
    robot = FakeRobot(["list"])
    calls: list[str] = []
    targets = [
        None,
        {"name": "用户", "content": "哦哦", "favorability": 1, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
        None,
    ]

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: targets.pop(0))
    monkeypatch.setattr(xinyu_app.steps, "read_my_points", lambda robot, config, reporter=None: calls.append("points") or "100")
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: None)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI")
    app.run_once()
    app.run_once()
    app.run_once()

    assert calls == ["points", "enter_chat", "points"]
