from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from apps.xinyu import app as xinyu_app
from apps.xinyu import steps


class FakeRobot:
    def __init__(self, images: list[object]) -> None:
        self.images = list(images)
        self.clicks: list[tuple[float, float]] = []
        self.played: list[str] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return self.images.pop(0) if self.images else "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def play_audio(self, chunk: str, wait: bool = True) -> None:
        self.played.append(chunk)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"template")


def _config(tmp_path: Path) -> dict[str, object]:
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    for path in [
        buttons_dir / "btn_返回按钮.jpg",
        buttons_dir / "btn_返回按钮_2.jpg",
        buttons_dir / "btn_未读消息按钮.jpg",
        buttons_dir / "btn_输入框.jpg",
        buttons_dir / "btn_麦克风按钮.jpg",
        buttons_dir / "btn_发送按钮.jpg",
        buttons_dir / "btn_我的按钮.jpg",
        buttons_dir / "btn_消息按钮.jpg",
        buttons_dir / "btn_新招呼按钮.jpg",
        buttons_dir / "popup_custom_1.jpg",
        screens_dir / "screen_聊天列表.jpg",
        screens_dir / "screen_聊天界面.jpg",
        screens_dir / "screen_我的界面.jpg",
        screens_dir / "screen_黑屏异常.jpg",
        screens_dir / "screen_无手机界面.jpg",
    ]:
        _touch(path)
    return {
        "templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)},
        "step_timeout": 0.2,
        "greetings_dir": str(tmp_path / "greetings"),
    }


def test_xinyu_list_ocr_treats_left_badge_as_default_unread(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "白嫖", "favorability": 44, "unread_count": 1, "content": "", "bounds": [0.1, 0.2, 0.4, 0.3]},
                    {"name": "张三", "favorability": 12, "unread_count": 1, "content": "你好", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "白嫖"
    assert target["favorability"] == 44
    assert target["unread_count"] == 1


def test_xinyu_dialogue_region_filter_is_forced_off(tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        "dialogue": {
            "rules": {
                "basic_filter": {
                    "enable_center_y_region_filter": True,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                }
            }
        }
    }

    resolved = steps._config(config)

    assert resolved["ocr_modes"]["dialogue"]["rules"]["basic_filter"]["enable_center_y_region_filter"] is False


def test_xinyu_app_dialogue_region_filter_is_forced_off(tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        "dialogue": {
            "rules": {
                "basic_filter": {
                    "enable_center_y_region_filter": True,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                }
            }
        }
    }

    app = xinyu_app.XinyuApp(FakeRobot([]), config, reply_provider=lambda message: "AI回复")

    assert app.config["ocr_modes"]["dialogue"]["rules"]["basic_filter"]["enable_center_y_region_filter"] is False


def test_xinyu_ensure_list_screen_raises_on_black_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["black"])

    def fake_match(image, template, threshold=0.85):
        if image == "black" and Path(template).name == "screen_黑屏异常.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    with pytest.raises(steps.BlackScreenError):
        steps.ensure_list_screen(robot, config)


def test_xinyu_ensure_list_screen_raises_on_pixel_black_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    black_image = Image.new("RGB", (200, 400), (3, 3, 3))
    robot = FakeRobot([black_image])

    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.BlackScreenError):
        steps.ensure_list_screen(robot, config)
    assert robot.clicks == []


def test_xinyu_ensure_list_screen_raises_on_no_phone_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["no_phone"])

    def fake_match(image, template, threshold=0.85):
        if image == "no_phone" and Path(template).name == "screen_无手机界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    with pytest.raises(steps.NoPhoneScreenError):
        steps.ensure_list_screen(robot, config)
    assert robot.clicks == []


def test_xinyu_ensure_list_screen_prioritizes_returning_to_list_before_popup_scan(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "list"])
    capture_labels: list[str] = []

    def fake_capture(robot, config, pre_move=True, label="screen"):
        capture_labels.append(label)
        return robot.get_transform_image()

    monkeypatch.setattr(steps, "capture_transform_image", fake_capture)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: (
            capture_labels.append(label) or robot.get_transform_image()
        ),
    )
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda image, config, key: (image == "chat" and key == steps.CHAT_WINDOW_SCREEN)
        or (image == "list" and key == steps.CHAT_LIST_SCREEN),
    )
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, config, key, group="buttons": (0.15, 0.15)
        if image == "chat" and key == steps.BACK_BUTTON and group == "buttons"
        else None,
    )

    image = steps.ensure_list_screen(robot, config)

    assert image == "list"
    assert capture_labels[0] == "startup_screen_detect"
    assert "confirm_list" not in capture_labels
    assert robot.clicks == [pytest.approx((0.15, 0.15))]


def test_xinyu_ensure_list_screen_reports_back_clicked_but_not_returned(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "still_chat"])

    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, cfg, pre_move=True, label="screen": robot.get_transform_image())
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: robot.get_transform_image())
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda image, config, key: (image == "chat" and key == steps.CHAT_WINDOW_SCREEN)
        or (image == "still_chat" and key == steps.CHAT_WINDOW_SCREEN),
    )
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, config, key, group="buttons": (0.15, 0.15)
        if image == "chat" and key == steps.BACK_BUTTON and group == "buttons"
        else None,
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    with pytest.raises(steps.StepUnavailable, match="已点击返回按钮，但未回到聊天列表"):
        steps.ensure_list_screen(robot, config)


def test_xinyu_ensure_list_screen_reports_unknown_screen_back_failure(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["unknown"])

    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, cfg, pre_move=True, label="screen": robot.get_transform_image())
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="当前界面无法识别，且未找到返回按钮模板"):
        steps.ensure_list_screen(robot, config)


def test_xinyu_app_black_screen_reports_waiting_without_circuit(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["black"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(steps.BlackScreenError("black screen detected")),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert app.status.is_circuit_open is False
    assert app.status.error_count == 0
    assert "black screen" in app.status.last_error
    assert app.status.current_step == "black_screen_waiting"
    assert app.status.current_step_label == "黑屏异常，等待用户亮屏"
    assert logs[-1][0] == "warning"
    assert logs[-1][1] == "检测到黑屏"


def test_xinyu_app_no_phone_reports_waiting_without_circuit(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["no_phone"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(
            steps.NoPhoneScreenError("no phone detected")
        ),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert app.status.is_circuit_open is False
    assert app.status.error_count == 0
    assert "no phone" in app.status.last_error
    assert app.status.current_step == "no_phone_waiting"
    assert app.status.current_step_label == "无手机异常，等待用户放回手机"
    assert logs[-1][0] == "warning"
    assert logs[-1][1] == "检测到无手机"


def test_xinyu_unread_red_pixel_detection_accepts_pink_red_badge():
    red_badge = Image.new("RGB", (28, 27), (90, 128, 165))
    for x in range(9, 19):
        for y in range(8, 18):
            red_badge.putpixel((x, y), (104, 54, 91))
    no_red = Image.new("RGB", (28, 27), (90, 128, 165))
    full_match = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    assert steps._has_red_pixels(red_badge, full_match) is True
    assert steps._has_red_pixels(no_red, full_match) is False


def test_xinyu_unreplied_filter_clicks_when_no_red(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after"])

    ocr_calls: list[object] = []

    def fake_ocr(image, cfg, mode, region=None):
        ocr_calls.append(image)
        return {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}],
                "message_list": [],
            },
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: False)
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "after")
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    image, _ocr_result = steps.check_unread_button(robot, config, image="list")

    assert image == "after"
    assert robot.clicks == [pytest.approx((0.3, 0.15))]
    assert ocr_calls == ["list", "after"]


def test_xinyu_unreplied_filter_does_not_click_when_red(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after"])

    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}],
                "message_list": [],
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)

    image, _ocr_result = steps.check_unread_button(robot, config, image="list")

    assert image == "list"
    assert robot.clicks == []


def test_xinyu_popup_dismiss_ignores_after_two_clicks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["popup_1", "popup_2", "popup_3"])

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "popup_3"
    assert robot.clicks == [pytest.approx((0.46, 0.23)), pytest.approx((0.46, 0.23))]


def test_xinyu_popup_detection_skips_when_chat_window(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])
    checked_templates: list[str] = []

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        checked_templates.append(template_name)
        if image == "chat" and template_name == "screen_聊天界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "chat" and template_name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "chat"
    assert robot.clicks == []
    assert "screen_聊天界面.jpg" in checked_templates
    assert "popup_custom_1.jpg" not in checked_templates


def test_xinyu_popup_detection_stops_when_after_popup_image_is_chat(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["popup", "chat"])
    checked_templates: list[tuple[str, str]] = []

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        checked_templates.append((str(image), template_name))
        if image == "popup" and template_name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        if image == "chat" and template_name == "screen_聊天界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "chat" and template_name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "chat"
    assert robot.clicks == [pytest.approx((0.46, 0.23))]
    assert ("chat", "screen_聊天界面.jpg") in checked_templates
    assert ("chat", "popup_custom_1.jpg") not in checked_templates


def test_xinyu_unknown_screen_does_not_click_default_back(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["unknown"])

    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="当前界面无法识别，且未找到返回按钮模板"):
        steps.ensure_list_screen(robot, config)
    assert robot.clicks == []


def test_xinyu_go_back_requires_matched_back_button(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="未找到返回按钮模板"):
        steps.go_back(robot, config)
    assert robot.clicks == []


def test_xinyu_input_box_error_is_reported_in_chinese(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "chat")
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="未找到输入框模板"):
        steps.focus_input_box(robot, config)


def test_xinyu_focus_input_box_clamps_negative_offset(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "chat")

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "btn_输入框.jpg":
            return {"x": 0.02, "y": 0.80, "w": 0.11, "h": 0.08, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    steps.focus_input_box(robot, config)

    assert robot.clicks == [(0.12, 0.84)]


def test_xinyu_chat_window_error_is_reported_in_chinese(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "list")
    monkeypatch.setattr(steps, "_matches_screen", lambda image, config, key: False)

    with pytest.raises(steps.StepUnavailable, match="未匹配到聊天界面模板"):
        steps.confirm_chat_window(robot, config)


def test_xinyu_list_selects_highest_favorability_target(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "甲", "favorability": 4, "unread_count": 1, "content": "短消息", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "乙", "favorability": 9, "unread_count": 1, "content": "更高", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "乙"


def test_xinyu_list_falls_back_to_content_when_unread_is_missing(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "user_a", "favorability": 2, "unread_count": 0, "content": "hello", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 7, "unread_count": 0, "content": "reply me", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"
    assert target["unread_count"] == 0


def test_xinyu_list_ignores_empty_placeholder_when_falling_back_to_content(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "\u6682\u65e0\u6d88\u606f", "favorability": 99, "unread_count": 0, "content": "", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 3, "unread_count": 0, "content": "available", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"


def test_xinyu_list_still_prefers_unread_target_over_content_fallback(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "content_only", "favorability": 50, "unread_count": 0, "content": "reply me", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "unread_user", "favorability": 4, "unread_count": 1, "content": "", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "unread_user"
    assert target["unread_count"] == 1


def test_xinyu_list_falls_back_to_content_when_unread_is_missing(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "user_a", "favorability": 2, "unread_count": 0, "content": "hello", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 7, "unread_count": 0, "content": "reply me", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"
    assert target["unread_count"] == 0


def test_xinyu_list_ignores_empty_placeholder_when_falling_back_to_content(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "\u6682\u65e0\u6d88\u606f", "favorability": 99, "unread_count": 0, "content": "", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 3, "unread_count": 0, "content": "available", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"


def test_xinyu_list_still_prefers_unread_target_over_content_fallback(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_has_red_pixels", lambda image, match_result: True)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "message_list": [
                    {"name": "content_only", "favorability": 50, "unread_count": 0, "content": "reply me", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "unread_user", "favorability": 4, "unread_count": 1, "content": "", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "unread_user"
    assert target["unread_count"] == 1


def test_xinyu_app_uses_short_list_content_directly(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "chat", "list"])
    sent: list[tuple[str, str]] = []
    replies: list[dict[str, str]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.3, 0.8))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: replies.append({"text": text}))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))

    events: list[dict[str, object]] = []
    app = xinyu_app.XinyuApp(
        robot,
        config,
        reply_provider=lambda message: "AI回复",
        sent_callback=lambda action, message: sent.append((action.text, message["content"])),
    )
    app.set_status_callback(events.append)

    app.run_once()

    assert replies == [{"text": "AI回复"}]
    assert sent == [("AI回复", "在吗")]
    assert (0.3, 0.8) in robot.clicks
    assert any(event.get("current_step_label") == "进入聊天界面" for event in events)
    assert any(event.get("current_step_log_label") == "进入聊天界面：白嫖 (0.200, 0.300)" for event in events)


def test_xinyu_app_logs_list_content_used_for_reply(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.random, "choice", lambda values: values[0])

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_status_callback(statuses.append)
    app.run_once()

    assert any(
        status.get("current_step") == "direct_content_reply"
        and status.get("current_step_label") == "使用列表内容作为对话输入"
        and status.get("current_step_log_label") == "使用列表内容作为对话输入：在吗"
        for status in statuses
    )


def test_xinyu_app_logs_duplicate_selection(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list", "list", "chat", "chat", "chat", "list"])
    target = {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3}
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: dict(target))
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None, double_click_on_first_attempt=False: None)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()
    app.run_once()

    assert any(message == "心遇重复选中同一会话" for _, message, _ in logs)


def test_xinyu_app_logs_multi_send_clicks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 3)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None, double_click_on_first_attempt=False: None)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert any(message == "心遇发送按钮被连续点击" for _, message, _ in logs)


def test_xinyu_app_skip_reply_still_focuses_input_box_before_back(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    clicks: list[tuple[float, float]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: clicks.append((0.31, 0.79)))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: clicks.append((0.08, 0.06)))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)

    app = xinyu_app.XinyuApp(
        robot,
        config,
        reply_provider=lambda message: xinyu_app.ReplyAction(skip=True, reason="already_processed"),
    )
    app.run_once()

    assert clicks[0] == (0.31, 0.79)
    assert clicks[-1] == (0.08, 0.06)


def test_xinyu_app_empty_target_does_not_keep_awake_before_idle_interval(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "暂无消息",
            "content": "",
            "favorability": None,
            "unread_count": 1,
            "bounds": [0.1, 0.2, 0.3, 0.4],
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(xinyu_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert calls == []
    assert app.status.last_error == ""


def test_xinyu_app_empty_target_keep_awake_only_after_idle_interval(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "暂无消息",
            "content": "",
            "favorability": None,
            "unread_count": 1,
            "bounds": [0.1, 0.2, 0.3, 0.4],
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(xinyu_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.last_activity_at -= 31
    app.run_once()

    assert calls == ["keep_awake"]
    assert app.status.last_error == ""


def test_xinyu_app_uses_greeting_file_for_nihao(monkeypatch, tmp_path):
    config = _config(tmp_path)
    greetings = Path(config["greetings_dir"])
    greetings.mkdir(parents=True, exist_ok=True)
    (greetings / "pingtai.txt").write_text("您好\n你好呀\n", encoding="utf-8")
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    sent: list[str] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "你好", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.31, 0.79))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: sent.append(text))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))
    monkeypatch.setattr(xinyu_app.random, "choice", lambda values: values[0])

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert sent == ["您好"]
    assert (0.31, 0.79) in robot.clicks


def test_xinyu_sends_new_greeting_and_returns(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after_greeting"])

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        if image == "chat" and template_name == "btn_新招呼按钮.jpg":
            return {"x": 0.3, "y": 0.4, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "after_greeting" and template_name == "btn_返回按钮_2.jpg":
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.send_new_greeting_if_available(robot, config, image="chat") is True
    assert robot.clicks == [pytest.approx((0.55, 0.45)), pytest.approx((0.15, 0.15))]


def test_xinyu_new_greeting_missing_continues_normal_flow(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot([])

    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    assert steps.send_new_greeting_if_available(robot, config, image="chat") is False
    assert robot.clicks == []


def test_xinyu_app_new_greeting_skips_ai_flow(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: calls.append("new_greeting") or True)
    monkeypatch.setattr(xinyu_app.steps, "read_message", lambda *args, **kwargs: calls.append("read_message") or {"contact": "白嫖", "content": "在吗"})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda *args, **kwargs: calls.append("focus_input_box"))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda *args, **kwargs: calls.append("play_voice_reply"))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda *args, **kwargs: calls.append("send_reply") or 1)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: calls.append("reply_provider") or "AI回复")
    app.run_once()

    assert calls == ["new_greeting"]
    assert app.status.last_error == ""


def test_xinyu_app_reads_dialogue_sender1_last_message(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "chat", "list"])
    seen: list[dict[str, str]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "这是一个很长的消息内容，需要进详情", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(
        xinyu_app.steps,
        "read_message",
        lambda robot, config, target, image=None, reporter=None: {"contact": target["name"], "content": "最后一条", "message_time": ""},
    )
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.31, 0.79))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: seen.append({"text": text}))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert seen == [{"text": "AI回复"}]


def test_xinyu_read_message_uses_dialogue_target_sender(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        **config.get("ocr_modes", {}),
        "dialogue": {
            "preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True},
            "rules": {
                "min_confidence": 0.3,
                "min_box_height_ratio": 0.02,
                "line_gap": 24,
                "name_gap_ratio": 0.037,
                "sender_detection": {"method": "bubble_color", "sender2_min_x_ratio": 0.52},
                "target_sender": "sender1",
                "ocr_result_excluded_keywords": [],
                "line_excluded_keywords": [],
                "line_and_below_keywords": [],
                "basic_filter": {
                    "enable_time_text_filter": True,
                    "enable_debug_text_filter": True,
                    "enable_reward_text_filter": True,
                    "enable_center_y_region_filter": False,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                    "enable_max_x_filter": True,
                    "max_x_ratio": 0.90,
                    "enable_max_width_filter": True,
                    "max_width_ratio": 0.80,
                    "enable_low_conf_single_char_filter": True,
                    "single_char_confidence_threshold": 0.90,
                    "enable_right_narrow_non_cjk_filter": True,
                    "right_narrow_min_x_ratio": 0.88,
                    "right_narrow_max_width_ratio": 0.12,
                    "enable_non_cjk_non_emoji_filter": True,
                },
            },
        },
    }
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "ocr_items": [{"text": "顶部名字"}],
            "result": {
                "user_name": "",
                "dialogues": [
                    {"sender": "sender1", "content": "更早的对方消息"},
                    {"sender": "sender2", "content": "右侧自己"},
                    {"sender": "sender1", "content": "第一句"},
                    {"sender": "sender1", "content": "第二句"},
                ],
            },
        },
    )

    result = steps.read_message(
        FakeRobot([]),
        config,
        target={"name": "白嫖", "unread_count": 1, "favorability": 12},
        image="chat",
    )

    assert result["content"] == "第一句第二句"
    assert result["sender"] == "sender1"


def test_xinyu_go_back_clicks_back_twice_after_send(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "chat", "list"])

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None: robot.get_transform_image(),
    )

    steps.go_back(robot, config, double_click_on_first_attempt=True)

    assert robot.clicks == [pytest.approx((0.15, 0.15)), pytest.approx((0.15, 0.15))]


def test_xinyu_go_back_single_clicks_when_not_after_send(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "chat", "chat", "chat", "list"])

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None: robot.get_transform_image(),
    )

    steps.go_back(robot, config)

    assert robot.clicks == [pytest.approx((0.15, 0.15)), pytest.approx((0.15, 0.15))]


def test_xinyu_keep_screen_awake_clicks_message_bottom_nav(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "全部消息", "bounds": [0.0, 0.1, 0.2, 0.2]},
                    {"name": "喜欢", "bounds": [0.0, 0.8, 0.2, 1.0]},
                    {"name": "消息", "bounds": [0.6, 0.8, 0.8, 1.0]},
                ]
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.keep_screen_awake(robot, config)
    assert robot.clicks == [pytest.approx((0.7, 0.9))]


def test_xinyu_keep_screen_awake_does_not_click_without_message_bottom_nav(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"bottom_nav": [{"name": "主页", "bounds": [0.8, 0.8, 1.0, 1.0]}]},
            "ocr_items": [],
        },
    )

    assert steps.keep_screen_awake(robot, config) is False
    assert robot.clicks == []


def test_xinyu_play_voice_reply_reclicks_microphone_when_tts_fails(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["tts_retry_attempts"] = 1
    robot = FakeRobot(["microphone", "microphone"])
    played: list[str] = []

    def fake_match(image, template, threshold=0.85):
        if image == "microphone" and Path(template).name == "btn_麦克风按钮.jpg":
            return {"x": 0.4, "y": 0.5, "w": 0.2, "h": 0.2, "score": 0.99}
        return None

    def fake_tts(robot, text):
        played.append(text)
        if len(played) == 1:
            raise RuntimeError("speaker timeout")

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "tts_input", fake_tts)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    steps.play_voice_reply(robot, "你好", config)

    assert played == ["你好", "你好"]
    assert robot.clicks == [pytest.approx((0.5, 0.6)), pytest.approx((0.5, 0.6))]
