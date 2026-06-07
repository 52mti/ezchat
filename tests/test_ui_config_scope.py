from __future__ import annotations

from io import BytesIO

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QAbstractSpinBox, QPushButton, QScrollArea

from ui import main_window as main_window_module
from ui.main_window import (
    AppConfigDialog,
    DeviceSettingsDialog,
    PopupCropDialog,
    PopupCropLabel,
    PopupDualRegionDialog,
    TemplateManagerWidget,
)


DEFAULT_PERSONA_ID = "REMOVED-BEARER-TOKEN"


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
    assert values["persona"]["requestPersona"] == DEFAULT_PERSONA_ID
    values["persona"]["name"] = "test"
    assert "persona" in values
    assert "192.168.1.108 / wechat" in dialog.windowTitle()


def test_app_config_persona_selector_uses_preset_and_custom_ids(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {"persona": {"requestPersona": "REMOVED-BEARER-TOKEN"}},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    assert dialog.request_persona_combo.currentText() == "特别素"
    assert dialog.request_persona_custom_input.isEnabled() is False
    assert dialog.values()["persona"]["requestPersona"] == "REMOVED-BEARER-TOKEN"

    dialog.request_persona_combo.setCurrentText("心遇优化版")
    assert dialog.values()["persona"]["requestPersona"] == "REMOVED-BEARER-TOKEN"

    dialog.request_persona_combo.setCurrentText("自定义")
    dialog.request_persona_custom_input.setText("app-custom-persona")

    assert dialog.request_persona_custom_input.isEnabled() is True
    assert dialog.values()["persona"]["requestPersona"] == "app-custom-persona"


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
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    button_texts = [button.text() for button in dialog.template_manager.findChildren(QPushButton)]

    for text in [
        "添加弹窗",
        "添加返回按钮",
        "添加发送按钮",
        "添加麦克风",
        "添加输入框",
        "添加下一条",
        "添加新招呼",
        "添加聊天界面",
        "添加聊天列表",
        "添加我的界面",
    ]:
        assert text in button_texts
    assert not any(text.startswith("导入") for text in button_texts)


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

    target = tmp_path / "templates" / "buttons" / "popup_custom_1.jpg"
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


def test_device_settings_dialog_persists_huanyou_offsets(qtbot):
    dialog = DeviceSettingsDialog(
        {
            "new_greeting_click_x_offset": 0.03,
            "new_greeting_click_y_offset": -0.01,
            "next_message_click_x_offset": 0.04,
            "next_message_click_y_offset": -0.01,
        },
        device_ip="192.168.1.133",
        app_name="huanyou",
    )
    qtbot.addWidget(dialog)

    dialog.new_greeting_click_x_offset.setValue(0.02)
    dialog.new_greeting_click_y_offset.setValue(-0.03)
    dialog.next_message_click_x_offset.setValue(0.05)
    dialog.next_message_click_y_offset.setValue(-0.02)
    values = dialog.values()

    assert values["new_greeting_click_x_offset"] == 0.02
    assert values["new_greeting_click_y_offset"] == -0.03
    assert values["next_message_click_x_offset"] == 0.05
    assert values["next_message_click_y_offset"] == -0.02


def test_device_settings_dialog_can_refresh_screenshot_and_test_click(qtbot, monkeypatch):
    image = Image.new("RGB", (200, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    clicks: list[tuple[float, float]] = []

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

        def move_click(self, x: float, y: float) -> None:
            clicks.append((x, y))

    monkeypatch.setattr(main_window_module, "RobotInterface", FakeRobot)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    dialog = DeviceSettingsDialog(
        {
            "new_greeting_click_x_offset": 0.02,
            "new_greeting_click_y_offset": -0.03,
            "next_message_click_x_offset": 0.05,
            "next_message_click_y_offset": -0.01,
        },
        device_ip="192.168.1.133",
        app_name="huanyou",
    )
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.refresh_screenshot_button, Qt.MouseButton.LeftButton)
    dialog.test_target_combo.setCurrentIndex(0)
    qtbot.mouseClick(
        dialog.crop_label,
        Qt.MouseButton.LeftButton,
        pos=QPoint(dialog.crop_label.width() // 2, dialog.crop_label.height() // 2),
    )

    assert dialog.point_x_label.text() == "0.5000"
    assert dialog.point_y_label.text() == "0.5000"
    assert dialog.click_x_label.text() == "0.5200"
    assert dialog.click_y_label.text() == "0.4700"

    dialog.test_target_combo.setCurrentIndex(1)
    assert dialog.click_x_label.text() == "0.5500"
    assert dialog.click_y_label.text() == "0.4900"

    qtbot.mouseClick(dialog.test_click_button, Qt.MouseButton.LeftButton)

    assert clicks == [(0.55, 0.49)]


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

    assert critical_messages == ["无法从当前设备获取截图：display failed"]
    assert dialog.refresh_screenshot_button.isEnabled()


def test_device_settings_can_save_current_screenshot_to_error_path(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: None)

    dialog = DeviceSettingsDialog({}, device_ip="192.168.1.133", app_name="xinyu")
    qtbot.addWidget(dialog)
    dialog.current_screenshot = Image.new("RGB", (120, 240), "white")

    dialog._save_current_screenshot_to_error_path()

    saved = list((tmp_path / "photo" / "errors").glob("manual_192_168_1_133_xinyu_*.jpg"))
    assert len(saved) == 1

