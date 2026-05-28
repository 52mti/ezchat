from __future__ import annotations

import json
from unittest.mock import patch

from PyQt6.QtWidgets import QDialog, QHeaderView, QMessageBox

from services.config_manager import ConfigManager
from ui import main_window as main_window_module
from ui.main_window import MainWindow
from workers.device_worker import WorkerEvent


def _window(qtbot, tmp_path, devices):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"devices": devices}, ensure_ascii=False), encoding="utf-8")
    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    window = MainWindow(manager)
    qtbot.addWidget(window)
    return window


def test_device_table_has_checkbox_column_and_batch_selection(qtbot, tmp_path):
    window = _window(
        qtbot,
        tmp_path,
        [
            {"ip": "192.168.1.108", "selected_app": "wechat"},
            {"ip": "192.168.1.109", "selected_app": "wechat"},
        ],
    )

    assert window.table.columnCount() == 8
    assert window.table.horizontalHeaderItem(0).text() == "\u9009\u62e9"
    assert window.table.horizontalHeaderItem(5).text() == "\u5e73\u5747\u7528\u65f6"
    assert window.table.horizontalHeaderItem(6).text() == "\u65e0\u5e72\u9884\u65f6\u957f"
    assert window.table.horizontalHeader().sectionResizeMode(4) == QHeaderView.ResizeMode.Fixed
    assert window.table.horizontalHeader().sectionResizeMode(6) == QHeaderView.ResizeMode.Fixed
    assert window._checked_rows() == []

    window._select_all_devices()
    assert window._checked_rows() == [0, 1]

    window._clear_device_selection()
    assert window._checked_rows() == []


def test_device_table_is_sorted_by_ip(qtbot, tmp_path):
    window = _window(
        qtbot,
        tmp_path,
        [
            {"ip": "192.168.1.20", "selected_app": "wechat"},
            {"ip": "192.168.1.3", "selected_app": "wechat"},
        ],
    )

    assert window.table.item(0, 3).text() == "192.168.1.3"
    assert window.table.item(1, 3).text() == "192.168.1.20"


def test_delete_device_removes_current_row(qtbot, tmp_path):
    window = _window(
        qtbot,
        tmp_path,
        [
            {"ip": "192.168.1.108", "selected_app": "wechat"},
            {"ip": "192.168.1.109", "selected_app": "wechat"},
        ],
    )

    with patch("ui.main_window.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        window._delete_device(0)

    assert window.table.rowCount() == 1
    assert window.table.item(0, 3).text() == "192.168.1.109"


def test_edit_app_config_only_updates_current_device_current_app(qtbot, tmp_path, monkeypatch):
    window = _window(
        qtbot,
        tmp_path,
        [
            {
                "ip": "192.168.1.108",
                "selected_app": "wechat",
                "app_overrides": {
                    "wechat": {"poll_interval": 5},
                    "xinyu": {"poll_interval": 9},
                },
            },
            {
                "ip": "192.168.1.109",
                "selected_app": "wechat",
                "app_overrides": {"wechat": {"poll_interval": 7}},
            },
        ],
    )

    class FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return {"poll_interval": 1, "keep_awake_interval_seconds": 200}

    monkeypatch.setattr("ui.main_window.AppConfigDialog", FakeDialog)
    monkeypatch.setattr("ui.main_window.QMessageBox.information", lambda *args, **kwargs: None)

    window._edit_app_config(0)

    first, second = window.config_manager.devices()
    assert first["app_overrides"]["wechat"]["poll_interval"] == 1
    assert first["app_overrides"]["wechat"]["keep_awake_interval_seconds"] == 200
    assert first["app_overrides"]["xinyu"]["poll_interval"] == 9
    assert second["app_overrides"]["wechat"]["poll_interval"] == 7


def test_status_event_shows_current_step_and_average_reply_time(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    window._set_row_status("192.168.1.108", "running")

    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "status",
            "progress",
            {
                "current_step": "dialogue_ocr_read",
                "current_step_label": "Dialogue OCR Read",
                "is_circuit_open": False,
                "reply_round_count": 3,
                "average_reply_seconds": 9.4,
            },
        )
    )
    window._drain_events()

    status_item = window.table.item(0, 4)
    avg_item = window.table.item(0, 5)
    assert status_item is not None
    assert avg_item is not None
    assert status_item.text() == "Dialogue OCR Read"
    assert avg_item.text() == "9.4\u79d2"
    assert avg_item.data(256) == 3
    assert status_item.data(256) == "running"
    assert status_item.data(257) == "Dialogue OCR Read"


def test_unattended_duration_tracks_each_device(qtbot, tmp_path, monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: current_time[0])
    window = _window(
        qtbot,
        tmp_path,
        [
            {"ip": "192.168.1.108", "selected_app": "wechat"},
            {"ip": "192.168.1.109", "selected_app": "wechat"},
        ],
    )

    assert window.table.item(0, 6).text() == "--"
    assert window.table.item(1, 6).text() == "--"

    window._start_unattended_timer("192.168.1.108")
    current_time[0] = 165.4
    window._refresh_unattended_durations()

    assert window.table.item(0, 6).text() == "00:01:05"
    assert window.table.item(1, 6).text() == "--"

    current_time[0] = 190.0
    window.events.put(WorkerEvent("192.168.1.108", "info", "stopped"))
    window._drain_events()

    assert window.table.item(0, 6).text() == "00:01:30"
    current_time[0] = 230.0
    window._refresh_unattended_durations()
    assert window.table.item(0, 6).text() == "00:01:30"


def test_status_event_hides_coordinate_details(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    window._set_row_status("192.168.1.108", "running")

    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "status",
            "progress",
            {
                "current_step": "enter_chat",
                "current_step_label": "进入聊天：(0.123, 0.456)",
                "is_circuit_open": False,
            },
        )
    )
    window._drain_events()

    status_item = window.table.item(0, 4)
    assert status_item is not None
    assert status_item.text() == "进入聊天"
    assert status_item.data(257) == "进入聊天"
    assert "0.123" not in status_item.text()
    assert "0.456" not in status_item.toolTip()


def test_paused_status_keeps_detail_reason(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])

    window._set_row_status("192.168.1.108", "paused", "黑屏熔断")

    status_item = window.table.item(0, 4)
    assert status_item is not None
    assert status_item.text() == "已暂停-黑屏熔断"
    assert status_item.data(257) == "黑屏熔断"


def test_runtime_step_log_keeps_coordinate_details(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])

    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "info",
            "runtime_step",
            {"step": "enter_chat", "detail": "进入聊天：无敌阿强 (0.123, 0.456)"},
        )
    )
    window._drain_events()

    logs_text = window.logs.toPlainText()
    assert "无敌阿强" in logs_text
    assert "(0.123, 0.456)" in logs_text


def test_stale_status_event_does_not_restart_stopping_row(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    window._set_row_status("192.168.1.108", "stopping")

    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "status",
            "tick",
            {
                "current_step": "list_ocr_scan",
                "current_step_label": "列表OCR扫描",
                "is_circuit_open": False,
            },
        )
    )
    window._drain_events()

    status_item = window.table.item(0, 4)
    assert status_item is not None
    assert status_item.data(256) == "stopping"
    assert status_item.text() == "停止中"


def test_start_worker_resumes_paused_worker(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])

    class FakeWorker:
        def __init__(self) -> None:
            self.resume_calls = 0

        def is_alive(self) -> bool:
            return True

        def resume(self) -> None:
            self.resume_calls += 1

    worker = FakeWorker()
    window.workers["192.168.1.108"] = worker  # type: ignore[assignment]
    window._set_row_status("192.168.1.108", "paused")

    window._start_worker(0)

    assert worker.resume_calls == 1
    assert window.table.item(0, 4).data(256) == "running"


def test_log_panel_shows_ocr_message_and_ai_reply_payload(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])

    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "info",
            "dialogue_ocr_last_message",
            {"contact": "\u65e0\u654c\u963f\u5f3a", "content": "\u4f60\u770b\u90a3\u90e8\u65b0\u51fa\u7684\u7535\u5f71\u4e86\u5417\uff1f"},
        )
    )
    window.events.put(
        WorkerEvent(
            "192.168.1.108",
            "info",
            "ai_reply_text",
            {"contact": "\u65e0\u654c\u963f\u5f3a", "reply_text": "\u6211\u4e5f\u60f3\u770b\uff0c\u4f60\u63a8\u8350\u54ea\u4e00\u573a\uff1f", "skip": False},
        )
    )
    window._drain_events()

    logs_text = window.logs.toPlainText()
    assert "OCR" in logs_text
    assert "\u65e0\u654c\u963f\u5f3a" in logs_text
    assert "AI" in logs_text
    assert "\u6211\u4e5f\u60f3\u770b\uff0c\u4f60\u63a8\u8350\u54ea\u4e00\u573a\uff1f" in logs_text
