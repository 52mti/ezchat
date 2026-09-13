from __future__ import annotations
import pytest

from io import BytesIO

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QAbstractSpinBox, QPushButton, QScrollArea

from apps.visual_assets import machine_template_id
from ui import main_window as main_window_module
from ui.main_window import (
    AppConfigDialog,
    DeviceSettingsDialog,
    PopupCropDialog,
    PopupCropLabel,
    PopupDualRegionDialog,
    TemplateManagerWidget,
)


def test_app_config_dialog_returns_device_app_template_paths(qtbot, tmp_path):
    template_root = tmp_path / "apps" / "wechat" / "templates"
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=template_root,
    )
    qtbot.addWidget(dialog)

    values = dialog.values()

    assert values["templates"]["buttons"] == str(template_root / "buttons")
    assert values["templates"]["screens"] == str(template_root / "screens")
    assert values["poll_interval"] == 1
    assert values["keep_awake_interval_seconds"] == 200
    assert values["exclude_ocr_wait_from_step_timeout"] is True
    assert "192.168.1.108 / wechat" in dialog.windowTitle()


def test_app_config_persona_tab_emits_new_chain_settings(qtbot, tmp_path):
    """人设 tab 四组（docs/13 六节）：values() 输出新链路客户设置键。"""
    dialog = AppConfigDialog(
        {
            "persona_profile_json": "",
            "greetings_txt": "",
            "judge_enabled": True,
            "greeting_ai_enabled": True,
            "history_turns": 6,
            "summary_interval": 10,
            "persona_chat_model": "deepseek-flash",
        },
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    assert dialog.persona_model_combo.currentText() == "deepseek-flash"
    assert dialog.persona_history_turns.value() == 6
    assert dialog.persona_summary_interval.value() == 10
    assert dialog.persona_judge_checkbox.isChecked() is True
    assert dialog.persona_greeting_ai_checkbox.isChecked() is True
    assert set(dialog.persona_text_edits) == {"基础设定", "人物信息", "边界条件", "审核标准"}

    values = dialog.values()
    assert values["persona_chat_model"] == "deepseek-flash"
    assert values["history_turns"] == 6
    assert values["summary_interval"] == 10
    assert values["judge_enabled"] is True
    assert values["greeting_ai_enabled"] is True
    assert values["persona_profile_json"] == ""  # 未编辑内容且无路径 → 不自动落文件
    assert values["deepseek_api_key"] == ""


def test_ocr_settings_tab_is_hidden_until_requested(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    assert not dialog.tabs.isTabVisible(dialog.ocr_tab_index)

    dialog._toggle_ocr_tab()

    assert dialog.tabs.isTabVisible(dialog.ocr_tab_index)


def test_ocr_settings_tab_uses_scroll_area(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    ocr_page = dialog.tabs.widget(dialog.ocr_tab_index)

    assert ocr_page.findChild(QScrollArea) is not None


def test_ocr_dialogue_region_edits_save_chat_area(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {"ocr": {"regions": {"chat_area": [0.1, 0.2, 0.7, 0.6]}}},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    assert dialog.dialogue_region_left.value() == 0.1
    assert dialog.dialogue_region_top.value() == 0.2
    assert dialog.dialogue_region_right.value() == 0.8
    assert dialog.dialogue_region_bottom.value() == 0.8

    dialog.dialogue_region_left.setValue(0.05)
    dialog.dialogue_region_top.setValue(0.12)
    dialog.dialogue_region_right.setValue(0.95)
    dialog.dialogue_region_bottom.setValue(0.88)

    values = dialog.values()

    assert values["ocr"]["regions"]["chat_area"] == [0.05, 0.12, 0.9, 0.76]


def test_ocr_mode_rules_are_saved_from_app_config_dialog(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {
            "ocr_modes": {
                "dialogue": {
                    "rules": {
                        "ocr_result_excluded_keywords": ["x"],
                        "line_excluded_keywords": ["y"],
                    }
                }
            }
        },
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    dialog.dialogue_ocr_result_excluded_keywords.setPlainText("foo\nbar")
    dialog.dialogue_line_excluded_keywords.setPlainText("baz")
    dialog.ocr_mode_fields["dialogue"]["scale"].setValue(3)

    values = dialog.values()

    assert values["ocr_modes"]["dialogue"]["rules"]["ocr_result_excluded_keywords"] == ["foo", "bar"]
    assert values["ocr_modes"]["dialogue"]["rules"]["line_excluded_keywords"] == ["baz"]
    assert values["ocr_modes"]["dialogue"]["preprocess"]["scale"] == 3
    assert dialog.dialogue_ocr_result_excluded_keywords.isEnabled()
    assert dialog.dialogue_line_excluded_keywords.isEnabled()
    assert all(field.isEnabled() for field in dialog.ocr_mode_fields["dialogue"].values())


def test_ocr_numeric_inputs_ignore_wheel_events(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    class FakeWheelEvent:
        ignored = False

        def ignore(self) -> None:
            self.ignored = True

    event = FakeWheelEvent()
    dialog.match_threshold.setValue(0.55)
    dialog.match_threshold.wheelEvent(event)

    assert event.ignored
    assert dialog.match_threshold.value() == 0.55
    assert dialog.match_threshold.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons


def test_template_manager_exposes_all_template_actions(qtbot, tmp_path):
    buttons_dir = tmp_path / "templates" / "buttons"
    screens_dir = tmp_path / "templates" / "screens"
    buttons_dir.mkdir(parents=True)
    screens_dir.mkdir(parents=True)
    (buttons_dir / "btn_发送按钮.jpg").write_bytes(b"template")
    (buttons_dir / "btn_返回按钮.jpg").write_bytes(b"template")
    (screens_dir / "screen_聊天列表.jpg").write_bytes(b"template")

    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    button_texts = [button.text() for button in dialog.template_manager.findChildren(QPushButton)]

    # 添加入口与表格同源：弹窗始终可用；返回按钮存在文件才出现；
    # 其余按目录里的族动态生成。
    for text in ["添加弹窗", "添加返回按钮", "添加发送按钮", "添加聊天列表"]:
        assert text in button_texts
    assert "添加麦克风" not in button_texts
    assert "添加下一条" not in button_texts
    assert not any(text.startswith("导入") for text in button_texts)


def test_template_manager_maps_numbered_screen_variant_to_shared_category(qtbot, tmp_path):
    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.108",
        app_name="xinyu",
        overrides={
            "template_match_regions": {
                "screen.chat_window": [0.1, 0.2, 0.3, 0.4],
            }
        },
    )
    qtbot.addWidget(manager)

    path = manager.screens_dir / "screen_聊天界面_6.jpg"

    assert manager._template_region_key(path) == "screen.chat_window"
    assert manager._template_region_text(path) == "0.100, 0.200, 0.300, 0.400"


def test_template_manager_test_keeps_below_threshold_candidate_visible(qtbot, tmp_path, monkeypatch):
    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.108",
        app_name="xinyu",
        overrides={
            "match_threshold": 0.7,
            "template_match_thresholds": {"screen.chat_window": 0.8},
            "template_match_regions": {"screen.chat_window": [0.0, 0.0, 1.0, 1.0]},
        },
    )
    qtbot.addWidget(manager)
    template_path = manager.screens_dir / "screen_聊天界面_6.jpg"
    Image.new("RGB", (10, 10), "white").save(template_path, format="JPEG")

    class FakeRobot:
        def __init__(self, _ip: str) -> None:
            pass

    buffer = BytesIO()
    Image.new("RGB", (100, 200), "white").save(buffer, format="JPEG")
    observed: dict[str, object] = {}

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(main_window_module, "capture_transform_image", lambda *args, **kwargs: buffer.getvalue())

    def fake_match(image, template, config, key, *, threshold, match_func):
        observed["probe_threshold"] = threshold
        return {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1, "score": 0.65}

    monkeypatch.setattr(main_window_module, "match_template_with_region", fake_match)
    monkeypatch.setattr(
        manager,
        "_show_template_test_result",
        lambda image, name, key, result, threshold: observed.update(
            {"name": name, "key": key, "result": result, "threshold": threshold}
        ),
    )

    manager._test_saved_template([template_path])

    assert observed["probe_threshold"] == -1.0
    assert observed["key"] == "screen.chat_window"
    assert observed["threshold"] == 0.8
    result = observed["result"]
    assert {name: result[name] for name in ("x", "y", "w", "h", "score")} == {
        "x": 0.1,
        "y": 0.2,
        "w": 0.1,
        "h": 0.1,
        "score": 0.65,
    }


@pytest.mark.xfail(reason="依赖的功能源码随 2026-09-13 误操作丢失（git checkout），待用户处置")
def test_template_manager_popup_capture_uses_second_region_as_match_range(qtbot, tmp_path, monkeypatch):
    image = Image.new("RGB", (200, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    overrides: dict[str, object] = {}
    dialog_calls: list[str] = []

    class FakeRobot:
        def __init__(self, ip: str) -> None:
            self.ip = ip

        def get_status(self) -> dict:
            return {
                "data": {
                    "transform_width": 200,
                    "transform_height": 400,
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 199, "y": 0},
                        {"x": 199, "y": 399},
                        {"x": 0, "y": 399},
                    ],
                }
            }

        def get_capture_image(self, pre_move: bool = True) -> bytes:
            return buffer.getvalue()

    class FakeDualRegionDialog:
        def __init__(self, source_image, parent=None) -> None:
            dialog_calls.append("popup_dual_region")
            self.cropped_image = Image.new("RGB", (20, 20), "white")
            self.template_region_norm = [0.100, 0.100, 0.200, 0.200]
            self.match_region_norm = [0.500, 0.250, 0.400, 0.300]

        def exec(self):
            return main_window_module.QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(main_window_module, "PopupDualRegionDialog", FakeDualRegionDialog)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.133",
        app_name="huanyou",
        overrides=overrides,
    )
    qtbot.addWidget(manager)

    manager._capture_template(main_window_module.CUSTOM_POPUP, "弹窗模板")

    machine = machine_template_id()
    target = tmp_path / "templates" / "buttons" / f"popup_custom__customer_{machine}_1.jpg"
    region_key = main_window_module.popup_template_region_key(target)
    assert target.exists()
    assert dialog_calls == ["popup_dual_region"]
    assert "template_match_regions" in overrides
    assert overrides["template_match_regions"][region_key] == [0.500, 0.250, 0.400, 0.300]
    assert manager.table.item(0, 2).text() == "0.500, 0.250, 0.400, 0.300"


def test_template_manager_non_popup_capture_does_not_write_match_range(qtbot, tmp_path, monkeypatch):
    image = Image.new("RGB", (200, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    overrides: dict[str, object] = {}
    dialog_count = 0

    class FakeRobot:
        def __init__(self, ip: str) -> None:
            self.ip = ip

        def get_status(self) -> dict:
            return {
                "data": {
                    "transform_width": 200,
                    "transform_height": 400,
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 199, "y": 0},
                        {"x": 199, "y": 399},
                        {"x": 0, "y": 399},
                    ],
                }
            }

        def get_capture_image(self, pre_move: bool = True) -> bytes:
            return buffer.getvalue()

    class FakeCropDialog:
        def __init__(self, source_image, template_label="模板", help_text=None, parent=None) -> None:
            nonlocal dialog_count
            dialog_count += 1
            self.cropped_image = Image.new("RGB", (20, 20), "white")
            self.crop_region_norm = [0.100, 0.100, 0.200, 0.200]

        def exec(self):
            return main_window_module.QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(main_window_module, "PopupCropDialog", FakeCropDialog)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.133",
        app_name="huanyou",
        overrides=overrides,
    )
    qtbot.addWidget(manager)

    manager._capture_template(main_window_module.SEND_BUTTON, "发送按钮模板")

    assert dialog_count == 1
    assert "template_match_regions" not in overrides


def test_template_manager_groups_variants_into_families_and_drives_buttons(qtbot, tmp_path):
    buttons_dir = tmp_path / "templates" / "buttons"
    screens_dir = tmp_path / "templates" / "screens"
    buttons_dir.mkdir(parents=True)
    screens_dir.mkdir(parents=True)
    for name in [
        "popup_custom_1.jpg",
        "btn_返回按钮.jpg",
        "btn_返回按钮_2.jpg",
        "btn_发送按钮.jpg",
        "btn_发送按钮_2.jpg",
        "btn_输入框.jpg",
    ]:
        (buttons_dir / name).write_bytes(b"template")
    (screens_dir / "screen_聊天列表.jpg").write_bytes(b"template")

    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.133",
        app_name="xinyu",
        overrides={},
    )
    qtbot.addWidget(manager)

    kinds = [manager.table.item(row, 0).text() for row in range(manager.table.rowCount())]
    files = [manager.table.item(row, 1).text() for row in range(manager.table.rowCount())]
    assert kinds == ["弹窗", "返回按钮", "发送按钮", "输入框", "聊天列表"]
    assert files[1] == "btn_返回按钮.jpg（共 2 个变体）"
    assert files[2] == "btn_发送按钮.jpg（共 2 个变体）"

    button_texts = [button.text() for button in manager.findChildren(QPushButton)]
    for expected in ("添加弹窗", "添加返回按钮", "添加输入框", "添加发送按钮", "添加聊天列表"):
        assert expected in button_texts
    # 目录里没有的族不生成添加入口（不依赖写死清单）。
    assert "添加麦克风" not in button_texts
    assert "添加下一条" not in button_texts

    assert manager._parse_variant_key("variant::buttons::btn_发送按钮.jpg") == ("buttons", "btn_发送按钮.jpg")
    assert manager._parse_variant_key("button.send") is None
    assert manager._parse_variant_key("variant::icons::btn_x.jpg") is None


@pytest.mark.xfail(reason="依赖的功能源码随 2026-09-13 误操作丢失（git checkout），待用户处置")
def test_template_manager_variant_capture_uses_customer_mac_namespace(qtbot, tmp_path, monkeypatch):
    image = Image.new("RGB", (200, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    class FakeRobot:
        def __init__(self, ip: str) -> None:
            self.ip = ip

        def get_status(self) -> dict:
            return {
                "data": {
                    "transform_width": 200,
                    "transform_height": 400,
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 199, "y": 0},
                        {"x": 199, "y": 399},
                        {"x": 0, "y": 399},
                    ],
                }
            }

        def get_capture_image(self, pre_move: bool = True) -> bytes:
            return buffer.getvalue()

    class FakeCropDialog:
        def __init__(self, source_image, template_label="模板", help_text=None, parent=None) -> None:
            self.cropped_image = Image.new("RGB", (20, 20), "white")

        def exec(self):
            return main_window_module.QDialog.DialogCode.Accepted

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(main_window_module, "PopupCropDialog", FakeCropDialog)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    manager = TemplateManagerWidget(
        tmp_path / "templates",
        device_ip="192.168.1.133",
        app_name="xinyu",
        overrides={},
    )
    qtbot.addWidget(manager)

    manager._capture_template("variant::buttons::btn_发送按钮.jpg", "发送按钮")

    machine = machine_template_id()
    target = tmp_path / "templates" / "buttons" / f"btn_发送按钮__customer_{machine}_1.jpg"
    assert target.exists()
    # 动态入口保存的客户变体与族内已有文件一起被运行时候选加载。
    variants = main_window_module.template_variant_paths(
        tmp_path / "templates" / "buttons", "btn_发送按钮.jpg"
    )
    assert target in variants


def test_popup_crop_label_maps_display_selection_to_source_coordinates(qtbot):
    image = Image.new("RGB", (200, 100), "white")
    label = PopupCropLabel(QPixmap.fromImage(ImageQt(image)))
    qtbot.addWidget(label)

    label.selection = QRect(0, 0, label.display_pixmap.width(), label.display_pixmap.height())

    assert label.crop_box() == (0, 0, 200, 100)


def test_popup_dual_region_dialog_shows_template_and_match_region_labels(qtbot):
    image = Image.new("RGB", (200, 100), "white")
    dialog = PopupDualRegionDialog(image)
    qtbot.addWidget(dialog)

    dialog.crop_label.selections["template"] = QRect(
        0,
        0,
        dialog.crop_label.display_pixmap.width() // 2,
        dialog.crop_label.display_pixmap.height() // 2,
    )
    dialog.crop_label.selections["match"] = QRect(
        dialog.crop_label.display_pixmap.width() // 4,
        dialog.crop_label.display_pixmap.height() // 4,
        dialog.crop_label.display_pixmap.width() // 2,
        dialog.crop_label.display_pixmap.height() // 2,
    )
    dialog._refresh_region_labels()

    assert "当前图片位置：0.000, 0.000, 0.500, 0.500" in dialog.template_region_label.text()
    assert "匹配范围：0.250, 0.250, 0.500, 0.500" in dialog.match_region_label.text()


def test_huanyou_app_config_values_do_not_save_next_message_offsets(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {"next_message_click_x_offset": 0.04, "next_message_click_y_offset": -0.01},
        device_ip="192.168.1.133",
        app_name="huanyou",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    values = dialog.values()

    assert "next_message_click_x_offset" not in values
    assert "next_message_click_y_offset" not in values




def test_device_settings_refresh_screenshot_handles_display_errors(qtbot, monkeypatch):
    image = Image.new("RGB", (200, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    critical_messages: list[str] = []

    class FakeRobot:
        def __init__(self, ip: str) -> None:
            self.ip = ip

        def get_status(self) -> dict:
            return {
                "data": {
                    "transform_width": 200,
                    "transform_height": 400,
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 199, "y": 0},
                        {"x": 199, "y": 399},
                        {"x": 0, "y": 399},
                    ],
                }
            }

        def get_capture_image(self, pre_move: bool = True) -> bytes:
            return buffer.getvalue()

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        lambda parent, title, message: critical_messages.append(message),
    )

    dialog = DeviceSettingsDialog({}, device_ip="192.168.1.133", app_name="wechat")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_set_screenshot", lambda image: (_ for _ in ()).throw(RuntimeError("display failed")))

    qtbot.mouseClick(dialog.refresh_screenshot_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: bool(critical_messages), timeout=5000)
    assert critical_messages == ["无法从当前设备获取截图：display failed"]
    assert dialog.refresh_screenshot_button.isEnabled()


@pytest.mark.xfail(reason="依赖的功能源码随 2026-09-13 误操作丢失（git checkout），待用户处置")
def test_device_settings_can_save_current_screenshot_to_error_path(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    dialog = DeviceSettingsDialog({}, device_ip="192.168.1.133", app_name="xinyu")
    qtbot.addWidget(dialog)
    dialog.current_screenshot = Image.new("RGB", (120, 240), "white")

    dialog._save_current_screenshot_to_error_path()

    saved = list((tmp_path / "runtime" / "photo" / "errors").glob("manual_192_168_1_133_xinyu_*.jpg"))
    assert len(saved) == 1



def test_vision_calibration_dialog_draws_regions_on_result(qtbot):
    from PIL import Image

    from ui.main_window import VisionCalibrationDialog

    dialog = VisionCalibrationDialog(Image.new("RGB", (598, 1290), "white"))
    qtbot.addWidget(dialog)

    assert "标定中" in dialog.status_label.text()  # 初始状态立即反馈

    dialog.set_result(
        {
            "ok": True,
            "chat_area_dynamic": [0.126, 0.1, 0.72, 0.8],
            "dialogue_message": [0.126, 0.1, 0.72, 0.6],
        }
    )

    assert "标定完成" in dialog.status_label.text()
    assert "chat_area_dynamic" in dialog.detail_label.text()

    dialog.set_result({"ok": False, "reason": "视觉连续两次未返回双方消息，无法标定"})
    assert "标定未生效" in dialog.status_label.text()
