from __future__ import annotations

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QAbstractSpinBox, QPushButton, QScrollArea

from ui.main_window import AppConfigDialog, PopupCropLabel


DEFAULT_PERSONA_ID = "REMOVED-BEARER-TOKEN"


def test_app_config_dialog_returns_device_app_template_paths(qtbot, tmp_path):
    template_root = tmp_path / "runtime" / "templates" / "wechat"
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
    assert values["persona"]["requestPersona"] == DEFAULT_PERSONA_ID
    values["persona"]["name"] = "test"
    assert "persona" in values
    assert "192.168.1.108 / wechat" in dialog.windowTitle()


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


def test_ocr_dialogue_keyword_editors_save_rules(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {
            "ocr_modes": {
                "dialogue": {
                    "rules": {
                        "ocr_result_excluded_keywords": ["记录点滴"],
                        "line_excluded_keywords": ["已认证"],
                    }
                }
            }
        },
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    assert dialog.dialogue_ocr_result_excluded_keywords.toPlainText() == "记录点滴"
    assert dialog.dialogue_line_excluded_keywords.toPlainText() == "已认证"

    dialog.dialogue_ocr_result_excluded_keywords.setPlainText("折扣商店\n\n活动倒计时")
    dialog.dialogue_line_excluded_keywords.setPlainText("真人认证\n常用语")

    rules = dialog.values()["ocr_modes"]["dialogue"]["rules"]

    assert rules["ocr_result_excluded_keywords"] == ["折扣商店", "活动倒计时"]
    assert rules["line_excluded_keywords"] == ["真人认证", "常用语"]


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


def test_template_manager_only_exposes_add_popup_action(qtbot, tmp_path):
    dialog = AppConfigDialog(
        {},
        device_ip="192.168.1.108",
        app_name="wechat",
        template_root=tmp_path / "templates",
    )
    qtbot.addWidget(dialog)

    button_texts = [button.text() for button in dialog.template_manager.findChildren(QPushButton)]

    assert "添加弹窗" in button_texts
    assert not any(text.startswith("导入") for text in button_texts)


def test_popup_crop_label_maps_display_selection_to_source_coordinates(qtbot):
    image = Image.new("RGB", (200, 100), "white")
    label = PopupCropLabel(QPixmap.fromImage(ImageQt(image)))
    qtbot.addWidget(label)

    label.selection = QRect(0, 0, label.display_pixmap.width(), label.display_pixmap.height())

    assert label.crop_box() == (0, 0, 200, 100)
