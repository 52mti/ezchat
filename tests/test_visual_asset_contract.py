from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from apps import visual_assets, visual_steps
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

    assert [rule["key"] for rule in tool_rules["save_types"]] == ["screen", "button", "popup"]
    assert targets[visual_assets.CHAT_LIST_SCREEN] == "screen_聊天列表.jpg"
    assert targets[visual_assets.CHAT_WINDOW_SCREEN] == "screen_聊天界面.jpg"
    assert targets[visual_assets.MY_SCREEN] == "screen_我的界面.jpg"
    assert targets[visual_assets.NEXT_MESSAGE_BUTTON] == "btn_下一条按钮.jpg"
    assert targets[visual_assets.INPUT_BOX_BUTTON] == "btn_输入框.jpg"
    assert targets[visual_assets.MICROPHONE_BUTTON] == "btn_麦克风按钮.jpg"
    assert targets[visual_assets.SEND_BUTTON] == "btn_发送按钮.jpg"
    assert targets[visual_assets.NEW_GREETING_BUTTON] == "btn_新招呼按钮.jpg"
    assert targets[visual_assets.PHOTO_BUTTON] == "btn_照片按钮.jpg"
    assert targets[visual_assets.COMPLETE_BUTTON] == "btn_完成按钮.jpg"
    assert targets[visual_assets.BACK_BUTTON] == "btn_back.jpg"
    assert targets[visual_assets.CUSTOM_BACK_BUTTON] == visual_assets.next_back_button_template_path(buttons_dir).name
    assert targets[visual_assets.CUSTOM_POPUP] == "popup_custom_1.jpg"
    assert tool_filenames[visual_assets.SEND_BUTTON] == targets[visual_assets.SEND_BUTTON]
    assert tool_filenames[visual_assets.PHOTO_BUTTON] == targets[visual_assets.PHOTO_BUTTON]
    assert tool_filenames[visual_assets.COMPLETE_BUTTON] == targets[visual_assets.COMPLETE_BUTTON]
    assert visual_assets.BACK_BUTTON not in tool_filenames
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
    assert all(rule["match_region_supported"] is True for rule in tool_rules["templates"])
    assert tool_rules["popups"][0]["match_region_supported"] is True
    assert tool_rules["back_buttons"][0]["match_region_supported"] is True



def test_template_region_match_maps_result_back_to_full_image(tmp_path):
    template_path = tmp_path / "btn.jpg"
    template_path.write_bytes(b"template")
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "template_match_regions": {
            visual_assets.SEND_BUTTON: [0.2, 0.3, 0.4, 0.5],
        }
    }
    seen_sizes: list[tuple[int, int]] = []

    def fake_match(cropped_image, template, threshold=0.85):
        seen_sizes.append(cropped_image.size)
        return {"x": 0.25, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}

    result = visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.7,
        match_func=fake_match,
    )

    assert seen_sizes == [(40, 100)]
    assert result == pytest.approx({"x": 0.3, "y": 0.5, "w": 0.04, "h": 0.1, "score": 0.99})



def test_template_region_match_can_expand_configured_region(tmp_path):
    template_path = tmp_path / "btn.jpg"
    template_path.write_bytes(b"template")
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "template_match_region_expand_ratio": 0.05,
        "template_match_regions": {
            visual_assets.SEND_BUTTON: [0.2, 0.3, 0.4, 0.5],
        },
    }
    seen_sizes: list[tuple[int, int]] = []

    def fake_match(cropped_image, template, threshold=0.85):
        seen_sizes.append(cropped_image.size)
        return {"x": 0.25, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}

    visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.7,
        match_func=fake_match,
    )

    assert seen_sizes == [(44, 111)]



def test_template_candidates_score_all_and_return_best_candidate(tmp_path):
    paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in paths:
        path.write_bytes(b"template")
    calls: list[str] = []

    def fake_match(image, template, threshold=0.85):
        calls.append(Path(template).name)
        score = 0.75 if Path(template).name == "first.jpg" else 0.92
        return {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "score": score, "channel": "gray"}

    config: dict[str, object] = {}
    result = visual_assets.match_template_candidates_with_region(
        "image",
        paths,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.7,
        match_func=fake_match,
    )

    assert calls == ["first.jpg", "second.jpg"]
    assert result is not None
    assert result["_template_name"] == "second.jpg"
    assert result["score"] == pytest.approx(0.92)



def test_template_candidates_failure_records_best_diagnostic(tmp_path):
    paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in paths:
        path.write_bytes(b"template")

    def fake_match(image, template, threshold=0.85):
        score = 0.64 if Path(template).name == "first.jpg" else 0.68
        return {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "score": score, "channel": "gray_norm"}

    config: dict[str, object] = {}
    result = visual_assets.match_template_candidates_with_region(
        "image",
        paths,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.7,
        match_func=fake_match,
    )

    assert result is None
    diagnostic = visual_assets.format_template_match_diagnostic(config, visual_assets.SEND_BUTTON)
    assert "second.jpg" in diagnostic
    assert "0.680" in diagnostic
    assert "0.700" in diagnostic
    assert "gray_norm" in diagnostic



def test_template_match_reuses_previous_result_when_same_photo(tmp_path):
    visual_assets._TEMPLATE_MATCH_CACHE.clear()
    template_path = tmp_path / "btn.jpg"
    template_path.write_bytes(b"template")
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "_runtime": {"device_ip": "192.168.1.108", "app_name": "xinyu"},
        "template_match_regions": {
            visual_assets.SEND_BUTTON: [0.2, 0.3, 0.4, 0.5],
        },
    }
    calls = 0

    def fake_match(cropped_image, template, threshold=0.85):
        nonlocal calls
        calls += 1
        return {"x": 0.25, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}

    first = visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.85,
        match_func=fake_match,
    )
    second = visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.85,
        match_func=fake_match,
    )

    assert first == second
    assert calls == 1



def test_template_match_cache_can_be_disabled(tmp_path):
    visual_assets._TEMPLATE_MATCH_CACHE.clear()
    template_path = tmp_path / "btn.jpg"
    template_path.write_bytes(b"template")
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "enable_template_match_cache": False,
        "template_match_regions": {
            visual_assets.SEND_BUTTON: [0.2, 0.3, 0.4, 0.5],
        },
    }
    calls = 0

    def fake_match(cropped_image, template, threshold=0.85):
        nonlocal calls
        calls += 1
        return {"x": 0.25, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}

    visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.85,
        match_func=fake_match,
    )
    visual_assets.match_template_with_region(
        image,
        template_path,
        config,
        visual_assets.SEND_BUTTON,
        threshold=0.85,
        match_func=fake_match,
    )

    assert calls == 2



def test_template_region_config_reports_missing_region_instead_of_global_fallback(tmp_path):
    template_path = tmp_path / "btn.jpg"
    template_path.write_bytes(b"template")

    def fake_match(image, template, threshold=0.85):
        raise AssertionError("global template matching should not run when region config exists")

    with pytest.raises(visual_assets.TemplateMatchRegionMissingError, match="当前类别没有设置范围：button.send"):
        visual_assets.match_template_with_region(
            "screen",
            template_path,
            {"template_match_regions": {}},
            visual_assets.SEND_BUTTON,
            threshold=0.7,
            match_func=fake_match,
        )


@pytest.mark.parametrize("template_key", ["screen.no_phone", "screen.black_screen"])

def test_abnormal_screen_templates_are_global_even_when_region_config_exists(tmp_path, template_key):
    template_path = tmp_path / "screen_abnormal.jpg"
    template_path.write_bytes(b"template")
    calls: list[str] = []

    def fake_match(image, template, threshold=0.85):
        calls.append(str(image))
        return {"x": 0, "y": 0, "w": 1, "h": 1, "score": 0.99}

    result = visual_assets.match_template_with_region(
        "full-screen-image",
        template_path,
        {"template_match_regions": {}},
        template_key,
        threshold=0.7,
        match_func=fake_match,
    )

    assert result is not None
    assert calls == ["full-screen-image"]



def test_next_popup_template_path_keeps_popup_pattern(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    (buttons_dir / "popup_custom_1.jpg").write_bytes(b"existing")

    target = visual_assets.import_target_path(visual_assets.CUSTOM_POPUP, buttons_dir, Path())

    assert target.name == "popup_custom_2.jpg"
    assert target.match("popup_*.jpg")
    assert visual_assets.is_popup_template_name(target.name)



def test_next_back_button_template_path_keeps_app_button_pattern(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    (buttons_dir / "btn_返回按钮.jpg").write_bytes(b"existing")
    (buttons_dir / "btn_返回按钮_2.jpg").write_bytes(b"existing")

    target = visual_assets.import_target_path(visual_assets.CUSTOM_BACK_BUTTON, buttons_dir, Path())

    assert target.name == "btn_返回按钮_3.jpg"
    assert visual_assets.is_back_button_template_name(target.name)



def test_fixed_template_import_uses_incremental_variant_names(tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    (buttons_dir / "btn_发送按钮.jpg").write_bytes(b"existing")
    (buttons_dir / "btn_发送按钮_2.jpg").write_bytes(b"existing")
    (screens_dir / "screen_聊天列表.jpg").write_bytes(b"existing")

    send_target = visual_assets.import_target_path(visual_assets.SEND_BUTTON, buttons_dir, screens_dir)
    list_target = visual_assets.import_target_path(visual_assets.CHAT_LIST_SCREEN, buttons_dir, screens_dir)

    assert send_target.name == "btn_发送按钮_3.jpg"
    assert list_target.name == "screen_聊天列表_2.jpg"



def test_customer_template_import_uses_separate_namespace(tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    (screens_dir / "screen_聊天界面.jpg").write_bytes(b"developer")
    (screens_dir / "screen_聊天界面_2.jpg").write_bytes(b"developer")
    (screens_dir / "screen_聊天界面__customer_1.jpg").write_bytes(b"customer")

    target = visual_assets.import_target_path(
        visual_assets.CHAT_WINDOW_SCREEN,
        buttons_dir,
        screens_dir,
        owner="customer",
    )

    assert target.name == "screen_聊天界面__customer_2.jpg"
    paths = visual_assets.template_variant_paths(screens_dir, "screen_聊天界面.jpg")
    assert [path.name for path in paths] == [
        "screen_聊天界面.jpg",
        "screen_聊天界面_2.jpg",
        "screen_聊天界面__customer_1.jpg",
    ]



def test_template_variant_paths_load_incremental_templates(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    for name in ["btn_发送按钮_10.jpg", "btn_发送按钮.jpg", "btn_发送按钮_2.jpg", "btn_发送按钮_old.jpg"]:
        (buttons_dir / name).write_bytes(b"template")

    paths = visual_assets.template_variant_paths(buttons_dir, "btn_发送按钮.jpg")

    assert [path.name for path in paths] == ["btn_发送按钮.jpg", "btn_发送按钮_2.jpg", "btn_发送按钮_10.jpg"]



def test_photo_button_variant_is_discoverable_without_base_template(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    variant = buttons_dir / "btn_照片按钮_2.jpg"
    variant.write_bytes(b"template")

    paths = visual_assets.template_variant_paths(buttons_dir, "btn_照片按钮.jpg")

    assert paths == [variant]



def test_runtime_loads_multiple_back_button_templates(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    for name in ["btn_返回按钮_10.jpg", "btn_back.jpg", "btn_返回按钮_2.jpg", "btn_返回按钮.jpg"]:
        (buttons_dir / name).write_bytes(b"template")

    paths = visual_assets.back_button_template_paths({"templates": {"buttons": str(buttons_dir)}})

    assert [path.name for path in paths] == [
        "btn_返回按钮.jpg",
        "btn_返回按钮_2.jpg",
        "btn_返回按钮_10.jpg",
        "btn_back.jpg",
    ]



def test_runtime_loads_multiple_popup_templates(tmp_path):
    buttons_dir = tmp_path / "buttons"
    buttons_dir.mkdir()
    for name in ["popup_custom_2.jpg", "btn_send.jpg", "popup_custom_1.jpg"]:
        (buttons_dir / name).write_bytes(b"template")

    paths = visual_assets.popup_template_paths({"templates": {"buttons": str(buttons_dir)}})

    assert [path.name for path in paths] == ["popup_custom_1.jpg", "popup_custom_2.jpg"]



def test_popup_template_region_key_uses_each_template_name(tmp_path):
    path = tmp_path / "popup_custom_2.jpg"

    assert visual_assets.popup_template_region_key(path) == "popup.custom.popup_custom_2"



@pytest.mark.parametrize("steps_module", [visual_steps, xinyu_steps])
def test_popup_match_defaults_to_popup_threshold(monkeypatch, tmp_path, steps_module):
    popup_template = tmp_path / "popup_custom_1.jpg"
    popup_template.write_bytes(b"template")
    seen_thresholds: list[float] = []

    def fake_match(image, template, threshold=0.85):
        seen_thresholds.append(threshold)
        return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}

    monkeypatch.setattr(steps_module, "match", fake_match)
    image = Image.new("RGB", (100, 200), "white")
    config = {"match_threshold": 0.95}

    result = steps_module._popup_match(image, popup_template, config)

    assert result is not None
    assert seen_thresholds == [0.8]



def test_popup_match_drops_results_beyond_right_edge(monkeypatch, tmp_path):
    popup_template = tmp_path / "popup_custom_1.jpg"
    popup_template.write_bytes(b"template")

    def fake_match(image, template, threshold=0.85):
        return {"x": 0.91, "y": 0.2, "w": 0.05, "h": 0.1, "score": 0.99}

    monkeypatch.setattr(xinyu_steps, "match", fake_match)

    assert xinyu_steps._popup_match("screen", popup_template, {}) is None



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

    monkeypatch.setattr(xinyu_steps, "match", fake_match)

    assert xinyu_steps.current_screen("screen_home.jpg", config) == "首页"
    assert xinyu_steps.current_screen("screen_list.jpg", config) == "消息页"
    assert xinyu_steps.current_screen("screen_chat.jpg", config) == "聊天详情"



def test_capture_after_popups_dismisses_popup_and_returns_next_frame(monkeypatch, tmp_path):
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    popup_img = tmp_path / "popup.jpg"
    list_img = tmp_path / "list.jpg"
    Image.new("RGB", (100, 200), "white").save(popup_img)
    Image.new("RGB", (80, 180), "white").save(list_img)
    for path in [
        buttons_dir / "popup_custom_1.jpg",
        screens_dir / "screen_\u804a\u5929\u5217\u8868.jpg",
        screens_dir / "screen_\u804a\u5929\u754c\u9762.jpg",
    ]:
        path.write_bytes(b"template")

    def _is_popup_frame(image) -> bool:
        if isinstance(image, (str, Path)):
            return Path(image).name == "popup.jpg"
        return tuple(getattr(image, "size", ())) == (100, 200)

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if _is_popup_frame(image) and name.startswith("popup_"):
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        if not _is_popup_frame(image) and name == "screen_\u804a\u5929\u5217\u8868.jpg":
            return {"x": 0.2, "y": 0.4, "w": 0.1, "h": 0.2, "score": 0.99}
        return None

    monkeypatch.setattr(xinyu_steps, "match", fake_match)
    monkeypatch.setattr(
        xinyu_steps,
        "_red_pixel_stats",
        lambda _image, _match: {"has_red": True, "red_pixels": 473, "total_pixels": 6732, "threshold": 13, "region": [0.2, 0.1, 0.3, 0.13]},
    )
    def fake_ocr(image, cfg, mode, region=None, **kwargs):
        # 弹窗帧不带列表标题词（消息/通话），列表帧带——这样弹窗不会被误判为列表页
        items = [{"text": "未回复", "x": 0.2, "y": 0.1, "w": 0.1, "h": 0.03}]
        if not _is_popup_frame(image):
            items.append({"text": "消息", "x": 0.1, "y": 0.05, "w": 0.1, "h": 0.03})
            items.append({"text": "通话", "x": 0.3, "y": 0.05, "w": 0.1, "h": 0.03})
        return {
            "result": {"message_list": [{"name": "测试用户", "unread_count": 1, "bounds": [0.2, 0.4, 0.3, 0.6]}]},
            "ocr_items": items,
        }

    monkeypatch.setattr(
        xinyu_steps,
        "run_ocr_mode",
        fake_ocr,
    )
    monkeypatch.setattr(
        xinyu_steps,
        "_capture_image",
        lambda robot, config, pre_move=True, label="screen": robot.images.pop(0) if robot.images else str(list_img),
    )
    robot = FakeRobot()
    robot.images = [str(popup_img), str(list_img)]

    # xinyu 的弹窗契约：capture_after_popups 先关弹窗，再返回弹窗后的帧。
    # 关闭点 = 命中框右上角 (x+w*0.9, y+h*0.1)。
    image = xinyu_steps.capture_after_popups(
        robot,
        {
            "templates": {
                "buttons": str(buttons_dir),
                "screens": str(screens_dir),
            }
        },
        label="test",
    )

    assert robot.clicks == [pytest.approx((0.46, 0.23))]
    assert not _is_popup_frame(image)
