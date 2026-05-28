from __future__ import annotations

from pathlib import Path

import pytest

from apps import visual_assets
from apps.huanyou import app as huanyou_app
from apps.huanyou import steps as huanyou_steps
from apps.wechat import app as wechat_app
from apps.wechat import steps as wechat_steps
from apps.xinyu import app as xinyu_app
from apps.xinyu import steps as xinyu_steps


class FakeRobot:
    def __init__(self) -> None:
        self.images = ["popup", "list"]
        self.clicks: list[tuple[float, float]] = []

    def get_transform_image(self, pre_move: bool = True) -> str:
        return self.images.pop(0) if self.images else "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def test_template_names_align_main_program_and_helper_tool_contract(tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()

    targets = {
        rule.key: visual_assets.import_target_path(rule.key, buttons_dir, screens_dir).name
        for rule in visual_assets.import_template_rules()
    }
    tool_rules = visual_assets.asset_rules_for_tool()
    tool_filenames = {rule["key"]: rule["filename"] for rule in tool_rules["templates"]}
    popup_rule = tool_rules["popups"][0]

    assert [rule["label"] for rule in tool_rules["save_types"]] == ["判定界面", "按钮", "弹窗"]
    assert targets[visual_assets.CHAT_LIST_SCREEN] == "chat_list.jpg"
    assert targets[visual_assets.CHAT_WINDOW_SCREEN] == "chat_window.jpg"
    assert targets[visual_assets.NEXT_MESSAGE_BUTTON] == "btn_next_msg.jpg"
    assert targets[visual_assets.VOICE_INPUT_BUTTON] == "btn_voice_input.jpg"
    assert targets[visual_assets.SEND_BUTTON] == "btn_send.jpg"
    assert targets[visual_assets.BACK_BUTTON] == "btn_back.jpg"
    assert targets[visual_assets.CUSTOM_POPUP] == "popup_custom_1.jpg"
    assert tool_filenames[visual_assets.SEND_BUTTON] == targets[visual_assets.SEND_BUTTON]
    assert popup_rule["key"] == visual_assets.CUSTOM_POPUP
    assert popup_rule["multiple"] is True
    assert popup_rule["filename_pattern"] == "popup_custom_{index}.jpg"
    assert popup_rule["match_pattern"] == "popup_*.jpg"
    assert tool_rules["custom_screens"][0]["multiple"] is True
    assert tool_rules["custom_screens"][0]["config_key"] == "screen_states"
    assert tool_rules["custom_screens"][0]["filename_pattern"] == "screen_{name}.jpg"
    assert tool_rules["custom_buttons"][0]["filename_pattern"] == "btn_{name}.jpg"
    assert tool_rules["custom_buttons"][0]["multiple"] is True
    assert "竖屏" in tool_rules["save_types"][1]["description"]
    assert "竖屏" in tool_rules["save_types"][2]["description"]
    assert "竖屏" in tool_rules["custom_buttons"][0]["description"]
    assert "竖屏" in popup_rule["description"]
    assert "regions" not in tool_rules
    assert "screen_roles" not in tool_rules


def test_next_popup_template_path_keeps_popup_pattern(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    (buttons_dir / "popup_custom_1.jpg").write_bytes(b"existing")

    target = visual_assets.import_target_path(visual_assets.CUSTOM_POPUP, buttons_dir, Path())

    assert target.name == "popup_custom_2.jpg"
    assert target.match("popup_*.jpg")
    assert visual_assets.is_popup_template_name(target.name)


def test_runtime_loads_multiple_popup_templates(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    for name in ["popup_custom_2.jpg", "btn_send.jpg", "popup_custom_1.jpg"]:
        (buttons_dir / name).write_bytes(b"template")

    paths = visual_assets.popup_template_paths({"templates": {"buttons": str(buttons_dir)}})

    assert [path.name for path in paths] == ["popup_custom_1.jpg", "popup_custom_2.jpg"]


def test_runtime_detects_user_named_screens(monkeypatch, tmp_path):
    screens_dir = tmp_path / "screens"
    screens_dir.mkdir()
    for name in ["screen_home.jpg", "screen_list.jpg", "screen_chat.jpg"]:
        (screens_dir / name).write_bytes(b"template")

    config = {
        "templates": {"screens": str(screens_dir)},
        "screen_states": {
            "首页": {"filename": "screen_home.jpg"},
            "消息页": {"filename": "screen_list.jpg"},
            "聊天详情": {"filename": "screen_chat.jpg"},
        },
        "screen_roles": {
            "message_list": "消息页",
            "chat_window": "聊天详情",
        },
    }

    def fake_match(image, template, threshold=0.85):
        return {"x": 0, "y": 0, "w": 1, "h": 1, "score": 0.99} if Path(template).name == image else None

    monkeypatch.setattr(wechat_steps, "match", fake_match)

    assert wechat_steps.current_screen("screen_home.jpg", config) == "首页"
    assert wechat_steps.current_screen("screen_list.jpg", config) == "消息页"
    assert wechat_steps.current_screen("screen_chat.jpg", config) == "聊天详情"


def test_each_app_keeps_its_own_step_flow_module():
    assert wechat_app.steps is wechat_steps
    assert xinyu_app.steps is xinyu_steps
    assert huanyou_app.steps is huanyou_steps

    assert wechat_steps.__name__ == "apps.wechat.steps"
    assert xinyu_steps.__name__ == "apps.xinyu.steps"
    assert huanyou_steps.__name__ == "apps.huanyou.steps"
    assert wechat_steps.__name__ != "apps.visual_steps"
    assert xinyu_steps.__name__ != "apps.visual_steps"
    assert huanyou_steps.__name__ != "apps.visual_steps"


def test_popup_is_checked_before_current_screen(monkeypatch, tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    for path in [
        buttons_dir / "popup_custom_1.jpg",
        screens_dir / "chat_list.jpg",
        screens_dir / "chat_window.jpg",
    ]:
        path.write_bytes(b"template")

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "popup" and name.startswith("popup_"):
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        if image == "popup" and name in {"chat_list.jpg", "chat_window.jpg"}:
            raise AssertionError("current screen was checked before popup handling")
        if image == "list" and name == "chat_list.jpg":
            return {"x": 0.2, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}
        return None

    monkeypatch.setattr(wechat_steps, "match", fake_match)
    monkeypatch.setattr(
        wechat_steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"message_list": [{"unread_count": 1, "bounds": [0.2, 0.4, 0.3, 0.6]}]},
            "ocr_items": [],
        },
    )
    robot = FakeRobot()

    pos = wechat_steps.find_unread(
        robot,
        {
            "templates": {
                "buttons": str(buttons_dir),
                "screens": str(screens_dir),
            }
        },
    )

    assert robot.clicks == [pytest.approx((0.46, 0.23))]
    assert pos == pytest.approx((0.25, 0.5))
