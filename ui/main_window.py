from __future__ import annotations

import copy
import importlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QCheckBox,
    QFormLayout,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QRubberBand,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from apps import get_default_app_name, get_registered_apps, resolve_app_name
from apps.defaults import DEFAULT_RUNTIME_CONFIG, with_app_defaults
from apps.coordinates import robot_point_with_jitter
from apps.photo_capture import capture_transform_image
from apps.visual_assets import (
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    CUSTOM_BACK_BUTTON,
    CUSTOM_POPUP,
    INPUT_BOX_BUTTON,
    MICROPHONE_BUTTON,
    MY_SCREEN,
    NEW_GREETING_BUTTON,
    NEXT_MESSAGE_BUTTON,
    SEND_BUTTON,
    BACK_BUTTON,
    TEMPLATE_MATCH_REGIONS_CONFIG_KEY,
    back_button_template_paths,
    fixed_template_rules,
    import_target_path,
    is_back_button_template_name,
    is_popup_template_name,
    popup_template_region_key,
    template_variant_paths,
)
from lgb.lgb import RobotInterface, scan_devices
from services import (
    ConfigManager,
    LicenseClient,
    ReleaseDecision,
    ReleaseManifestError,
    ResolvedUpdatePackage,
    UpdateSourceError,
    resolve_lanzou_update_package,
    resolve_release_decision,
)
from vision.screen import ensure_portrait_image
from workers import DeviceWorker, WorkerEvent


PERSONA_PRESETS: tuple[tuple[str, str], ...] = (
    ("偶尔开车", "REMOVED-BEARER-TOKEN"),
    ("特别素", "REMOVED-BEARER-TOKEN"),
    ("心遇优化版", "REMOVED-BEARER-TOKEN"),
    ("漂流瓶引诱", "REMOVED-BEARER-TOKEN"),
    ("引导视频（概率）", "REMOVED-BEARER-TOKEN"),
    ("引导视频（纯）", "REMOVED-BEARER-TOKEN"),
)
CUSTOM_PERSONA_LABEL = "自定义"


def _pixmap_from_pil_image(image: Image.Image) -> QPixmap:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    qimage = QImage()
    if not qimage.loadFromData(buffer.getvalue(), "PNG"):
        raise ValueError("截图数据无法转换为 Qt 图像")
    pixmap = QPixmap.fromImage(qimage)
    if pixmap.isNull():
        raise ValueError("截图数据无法转换为 Qt 画面")
    return pixmap


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _create_request_persona_selector(current_value: str) -> tuple[QWidget, QComboBox, QLineEdit]:
    current = str(current_value or "").strip()
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    combo = QComboBox()
    for label, persona_id in PERSONA_PRESETS:
        combo.addItem(label, persona_id)
    combo.addItem(CUSTOM_PERSONA_LABEL, "")

    custom_input = QLineEdit()
    custom_input.setPlaceholderText("请输入人设授权 ID")

    matched_index = -1
    for index, (_label, persona_id) in enumerate(PERSONA_PRESETS):
        if persona_id == current:
            matched_index = index
            break
    if matched_index >= 0:
        combo.setCurrentIndex(matched_index)
        custom_input.setText("")
    else:
        combo.setCurrentText(CUSTOM_PERSONA_LABEL)
        custom_input.setText(current)

    def sync_custom_input() -> None:
        is_custom = combo.currentText() == CUSTOM_PERSONA_LABEL
        custom_input.setVisible(is_custom)
        custom_input.setEnabled(is_custom)

    combo.currentIndexChanged.connect(lambda _index: sync_custom_input())
    sync_custom_input()

    layout.addWidget(combo, 0)
    layout.addWidget(custom_input, 1)
    return container, combo, custom_input


def _request_persona_value(combo: QComboBox, custom_input: QLineEdit) -> str:
    selected = combo.currentData()
    if combo.currentText() == CUSTOM_PERSONA_LABEL:
        return custom_input.text().strip()
    return str(selected or "").strip()


def _ocr_device_label_from_backend(backend: object) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized in {"paddle_cuda", "onnx_directml", "gpu", "cuda", "directml"}:
        return "GPU"
    if normalized == "cpu":
        return "CPU"
    return ""


def _ocr_mode_label(config_manager: ConfigManager) -> str:
    config_root = Path(getattr(config_manager, "config_path", Path("config.json"))).resolve().parent
    state_path = config_root / "runtime" / "install_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if isinstance(state, dict):
            device = str(state.get("ocr_device", "") or "").strip().lower()
            if device.startswith(("gpu", "cuda", "dml")):
                return "OCR: GPU"
            if device == "cpu":
                return "OCR: CPU"
            label = _ocr_device_label_from_backend(state.get("backend", ""))
            if label:
                return f"OCR: {label}"

    devices = config_manager.get("devices", default=[])
    device = devices[0] if isinstance(devices, list) and devices and isinstance(devices[0], dict) else {}
    app_name = resolve_app_name(device.get("selected_app", None))
    device_ip = str(device.get("ip", "") or "")
    app_config = config_manager.get_app_config(device_ip, app_name, _app_default_config(app_name))
    ocr_config = app_config.get("ocr", {}) if isinstance(app_config.get("ocr", {}), dict) else {}
    label = _ocr_device_label_from_backend(ocr_config.get("backend", ocr_config.get("ocr_backend", "")))
    if not label:
        label = "GPU" if bool(ocr_config.get("use_gpu", False)) else "CPU"
    return f"OCR: {label}"


def _app_default_config(app_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(f"apps.{app_name}.config")
    except ImportError:
        return with_app_defaults({})
    default_config = getattr(module, "DEFAULT_CONFIG", {})
    return with_app_defaults(default_config if isinstance(default_config, dict) else {})


APP_STYLESHEET = """
QMainWindow {
  background: #f8fafc;
}
QWidget {
  color: #334155;
  font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
  font-size: 13px;
}
QFrame#TopPanel, QFrame#AddPanel, QFrame#SummaryPanel, QFrame#ActionPanel {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}
QFrame#DialogPanel, QFrame#InsetPanel {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}
QLabel#TitleLabel {
  color: #0f172a;
  font-size: 18px;
  font-weight: 700;
}
QLabel#DialogTitle {
  color: #0f172a;
  font-size: 16px;
  font-weight: 700;
}
QLabel#SectionLabel {
  color: #475569;
  font-weight: 600;
}
QLabel#HelperLabel {
  color: #64748b;
  font-size: 12px;
}
QLabel#MetricValue {
  color: #0f172a;
  font-size: 18px;
  font-weight: 700;
}
QLabel#MetricLabel {
  color: #64748b;
  font-size: 12px;
}
QLabel#OcrModeLabel {
  color: #475569;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 6px 10px;
  font-weight: 600;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  min-height: 30px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 3px 8px;
  background: #ffffff;
  selection-background-color: #f97316;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
  border: 1px solid #f97316;
}
QPushButton {
  min-height: 30px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 4px 10px;
  background: #ffffff;
  color: #334155;
}
QPushButton:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}
QPushButton:pressed {
  background: #e2e8f0;
}
QPushButton:disabled {
  color: #94a3b8;
  background: #f8fafc;
}
QPushButton#PrimaryButton {
  background: #f97316;
  border-color: #ea580c;
  color: #ffffff;
  font-weight: 600;
}
QPushButton#PrimaryButton:hover {
  background: #ea580c;
}
QPushButton#DangerButton {
  color: #b91c1c;
  border-color: #fecaca;
  background: #fff7f7;
}
QPushButton#DangerButton:hover {
  background: #fee2e2;
}
QPushButton#QuietButton {
  border-color: transparent;
  background: transparent;
}
QTableWidget {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  gridline-color: #e2e8f0;
  selection-background-color: #ffedd5;
  selection-color: #0f172a;
}
QTableWidget::indicator {
  width: 18px;
  height: 18px;
}
QHeaderView::section {
  background: #f1f5f9;
  color: #475569;
  border: none;
  border-bottom: 1px solid #cbd5e1;
  padding: 8px;
  font-weight: 600;
}
QTextEdit {
  background: #0f172a;
  color: #e2e8f0;
  border: 1px solid #1e293b;
  border-radius: 8px;
  font-family: "Consolas", "Fira Code", monospace;
  font-size: 12px;
}
QTextEdit#KeywordEditor {
  background: #ffffff;
  color: #334155;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
  font-size: 13px;
}
QTextEdit#KeywordEditor:focus {
  border: 1px solid #f97316;
}
QTabWidget::pane {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
}
QTabBar::tab {
  padding: 7px 12px;
  margin-right: 2px;
}
QTabBar::tab:selected {
  color: #ea580c;
  font-weight: 600;
}
"""


STATUS_COLORS = {
    "normal": ("正常", "#dcfce7", "#166534"),
    "running": ("运行中", "#dbeafe", "#1d4ed8"),
    "paused": ("已暂停", "#fef3c7", "#92400e"),
    "stopping": ("停止中", "#f1f5f9", "#475569"),
    "stopped": ("已停止", "#f1f5f9", "#475569"),
    "circuit_open": ("已熔断", "#fee2e2", "#991b1b"),
    "license_error": ("卡密异常", "#fee2e2", "#991b1b"),
    "offline": ("离线授权", "#e0f2fe", "#0369a1"),
}


COL_SELECT = 0
COL_LICENSE = 1
COL_EXPIRES = 2
COL_DEVICE_IP = 3
COL_STATUS = 4
COL_AVG_REPLY_SECONDS = 5
COL_UNATTENDED_SECONDS = 6
COL_RUNNING_APP = 7
COL_BALANCE = 8
STATUS_COLUMN_WIDTH = 150
UNATTENDED_COLUMN_WIDTH = 112
BALANCE_COLUMN_WIDTH = 86

STATUS_COORDINATE_RE = re.compile(r"\s*[：:,-]?\s*\(?\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*\)?\s*$")
UPDATE_DOWNLOAD_CHUNK_SIZE = 1024 * 256


def mask_secret(value: str) -> str:
    if not value:
        return ""
    return "****" + value[-4:] if len(value) > 4 else "****"


def strip_status_coordinates(detail: str) -> str:
    cleaned = STATUS_COORDINATE_RE.sub("", str(detail or "").strip()).strip()
    return cleaned.rstrip("：:,，- ").strip()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        }:
            event.ignore()
            return
        super().keyPressEvent(event)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        }:
            event.ignore()
            return
        super().keyPressEvent(event)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class MainWindow(QMainWindow):
    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue()
        self.update_events: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()
        self.workers: dict[str, DeviceWorker] = {}
        self.log_events: list[WorkerEvent] = []
        self.log_event_times: dict[int, str] = {}
        self.unattended_started_at: dict[str, float] = {}
        self.unattended_elapsed_seconds: dict[str, float] = {}
        self._update_check_in_progress = False
        self._download_cancel_event: threading.Event | None = None
        self._download_progress_dialog: QProgressDialog | None = None
        self._pending_update_decision: ReleaseDecision | None = None
        self._pending_update_package: ResolvedUpdatePackage | None = None
        self.setWindowTitle("机械臂视觉控制系统")
        self.resize(1180, 720)
        self.setStyleSheet(APP_STYLESHEET)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("机械臂视觉控制系统")
        title.setObjectName("TitleLabel")
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        top_panel = QFrame()
        top_panel.setObjectName("TopPanel")
        top_bar = QHBoxLayout(top_panel)
        top_bar.setContentsMargins(12, 10, 12, 10)
        top_bar.setSpacing(8)
        top_bar.addWidget(QLabel("API Key"))
        self.apikey_input = QLineEdit(mask_secret(self.config_manager.get("global", "apikey", default="")))
        self.apikey_input.setEchoMode(QLineEdit.EchoMode.Password)
        top_bar.addWidget(self.apikey_input)
        self.save_apikey_button = self._button("保存", "primary")
        self.save_apikey_button.clicked.connect(self._save_apikey)
        top_bar.addWidget(self.save_apikey_button)
        top_bar.addSpacing(8)
        top_bar.addWidget(QLabel("OCR并发"))
        self.ocr_concurrency_input = NoWheelSpinBox()
        self.ocr_concurrency_input.setRange(1, 16)
        self.ocr_concurrency_input.setFixedWidth(72)
        self.ocr_concurrency_input.setValue(int(self.config_manager.get("global", "ocr_max_concurrency", default=1) or 1))
        top_bar.addWidget(self.ocr_concurrency_input)
        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("设备 IP"))
        self.device_ip_input = QLineEdit()
        self.device_ip_input.setPlaceholderText("192.168.1.108")
        top_bar.addWidget(self.device_ip_input)
        self.add_device_button = self._button("添加设备", "primary")
        self.add_device_button.clicked.connect(self._add_device)
        top_bar.addWidget(self.add_device_button)
        self.scan_device_button = self._button("扫描设备")
        self.scan_device_button.clicked.connect(self._scan_devices)
        top_bar.addWidget(self.scan_device_button)
        top_bar.addSpacing(12)
        self.ocr_mode_label = QLabel(_ocr_mode_label(self.config_manager))
        self.ocr_mode_label.setObjectName("OcrModeLabel")
        top_bar.addWidget(self.ocr_mode_label)
        self.check_update_button = self._button("检查更新")
        self.check_update_button.clicked.connect(lambda: self._start_release_check(manual=True))
        top_bar.addWidget(self.check_update_button)
        top_bar.addStretch(1)
        layout.addWidget(top_panel)

        summary_panel = QFrame()
        summary_panel.setObjectName("SummaryPanel")
        summary = QHBoxLayout(summary_panel)
        summary.setContentsMargins(12, 10, 12, 10)
        summary.setSpacing(22)
        self.total_devices_value = self._metric(summary, "设备总数")
        self.running_devices_value = self._metric(summary, "运行中")
        self.alert_devices_value = self._metric(summary, "异常/熔断")
        summary.addStretch(1)
        layout.addWidget(summary_panel)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["\u9009\u62e9", "\u5f53\u524d\u5361\u5bc6", "\u5230\u671f\u65f6\u95f4", "\u8bbe\u5907IP", "\u8fd0\u884c\u72b6\u6001", "\u5e73\u5747\u7528\u65f6", "\u65e0\u5e72\u9884\u65f6\u957f", "\u8fd0\u884cApp", "\u4f59\u989d"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_device_context_menu)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_SELECT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_LICENSE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_EXPIRES, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DEVICE_IP, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_STATUS, STATUS_COLUMN_WIDTH)
        header.setSectionResizeMode(COL_AVG_REPLY_SECONDS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_UNATTENDED_SECONDS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_UNATTENDED_SECONDS, UNATTENDED_COLUMN_WIDTH)
        header.setSectionResizeMode(COL_RUNNING_APP, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_BALANCE, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_BALANCE, BALANCE_COLUMN_WIDTH)
        layout.addWidget(self.table)

        action_panel = QFrame()
        action_panel.setObjectName("ActionPanel")
        action_bar = QHBoxLayout(action_panel)
        action_bar.setContentsMargins(12, 10, 12, 10)
        action_bar.setSpacing(8)
        action_bar.addWidget(QLabel("\u65e5\u5fd7\u8bbe\u5907"))
        self.log_device_filter = QComboBox()
        self.log_device_filter.currentIndexChanged.connect(self._render_logs)
        action_bar.addWidget(self.log_device_filter)
        self.log_toggle = self._button("运行日志")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(False)
        self.log_toggle.clicked.connect(self._toggle_logs)
        action_bar.addWidget(self.log_toggle)
        self.select_all_button = self._button("全选")
        self.select_all_button.clicked.connect(self._select_all_devices)
        action_bar.addWidget(self.select_all_button)
        self.clear_selection_button = self._button("清空")
        self.clear_selection_button.clicked.connect(self._clear_device_selection)
        action_bar.addWidget(self.clear_selection_button)
        self.batch_start_button = self._button("启动选中", "primary")
        self.batch_start_button.clicked.connect(lambda: self._run_for_checked_rows(self._start_worker))
        action_bar.addWidget(self.batch_start_button)
        self.batch_pause_button = self._button("暂停选中")
        self.batch_pause_button.clicked.connect(lambda: self._run_for_checked_rows(self._pause_worker))
        action_bar.addWidget(self.batch_pause_button)
        self.batch_resume_button = self._button("恢复选中")
        self.batch_resume_button.clicked.connect(lambda: self._run_for_checked_rows(self._resume_worker))
        action_bar.addWidget(self.batch_resume_button)
        self.batch_stop_button = self._button("停止选中", "danger")
        self.batch_stop_button.clicked.connect(lambda: self._run_for_checked_rows(self._stop_worker))
        action_bar.addWidget(self.batch_stop_button)
        action_bar.addStretch(1)
        layout.addWidget(action_panel)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumHeight(180)
        self.logs.setVisible(False)
        layout.addWidget(self.logs)

        self.setCentralWidget(root)
        self._load_devices()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._drain_events)
        self.timer.start(100)
        QTimer.singleShot(800, lambda: self._start_release_check(manual=False))

    def _button(self, text: str, variant: str = "default") -> QPushButton:
        button = QPushButton(text)
        if variant == "primary":
            button.setObjectName("PrimaryButton")
        elif variant == "danger":
            button.setObjectName("DangerButton")
        elif variant == "quiet":
            button.setObjectName("QuietButton")
        return button

    def _metric(self, layout: QHBoxLayout, label: str) -> QLabel:
        group = QVBoxLayout()
        value = QLabel("0")
        value.setObjectName("MetricValue")
        caption = QLabel(label)
        caption.setObjectName("MetricLabel")
        group.addWidget(value)
        group.addWidget(caption)
        layout.addLayout(group)
        return value

    def _package_root(self) -> Path:
        return Path(getattr(self.config_manager, "config_path", Path("config.json"))).resolve().parent

    def _startup_update_check_enabled(self) -> bool:
        root = self._package_root()
        return (root / "runtime_manifest.json").is_file() and (root / "runtime" / "install_state.json").is_file()

    def _skip_update_version(self) -> str:
        return str(self.config_manager.get("update_skip_version", default="") or "").strip()

    def _set_skip_update_version(self, version: str) -> None:
        self.config_manager.set("update_skip_version", value=str(version or "").strip())

    def _start_release_check(self, *, manual: bool) -> None:
        if not manual and not self._startup_update_check_enabled():
            return
        if self._update_check_in_progress:
            if manual:
                QMessageBox.information(self, "检查更新", "正在检查更新，请稍后。")
            return
        self._update_check_in_progress = True
        self.check_update_button.setEnabled(False)
        self.check_update_button.setText("检查中...")
        threading.Thread(
            target=self._run_release_check,
            kwargs={"manual": manual},
            daemon=True,
        ).start()

    def _run_release_check(self, *, manual: bool) -> None:
        try:
            decision = resolve_release_decision(package_root=self._package_root())
        except ReleaseManifestError as exc:
            self.update_events.put(("check_error", {"manual": manual, "message": str(exc)}))
            return
        self.update_events.put(("check_complete", {"manual": manual, "decision": decision}))

    def _finish_release_check(self) -> None:
        self._update_check_in_progress = False
        self.check_update_button.setEnabled(True)
        self.check_update_button.setText("检查更新")

    def _show_release_decision_dialog(self, decision: ReleaseDecision) -> None:
        notes = decision.release_notes.strip()
        details = [
            f"当前版本：{decision.current_app_version}",
            f"目标版本：{decision.target_app_version}",
        ]
        if notes:
            details.append(f"更新说明：{notes}")
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Information)
        if decision.action == "incremental":
            message.setWindowTitle("发现更新")
            message.setText("发现新的更新包。")
            proceed_button = message.addButton("立即更新", QMessageBox.ButtonRole.AcceptRole)
        else:
            message.setWindowTitle("需要下载全量包")
            message.setText("当前版本需要下载新的全量包。")
            proceed_button = message.addButton("前往下载", QMessageBox.ButtonRole.AcceptRole)
        skip_button = message.addButton("本次更新不再提示", QMessageBox.ButtonRole.DestructiveRole)
        later_button = message.addButton("稍后再说", QMessageBox.ButtonRole.RejectRole)
        message.setInformativeText("\n".join(details))
        message.exec()
        clicked = message.clickedButton()
        if clicked is skip_button:
            self._set_skip_update_version(decision.target_app_version)
            return
        if clicked is later_button:
            return
        self._set_skip_update_version("")
        if clicked is proceed_button:
            if decision.action == "incremental":
                self._start_update_download(decision)
            else:
                QDesktopServices.openUrl(QUrl(decision.full_package_url))

    def _start_update_download(self, decision: ReleaseDecision) -> None:
        if self._download_cancel_event is not None:
            QMessageBox.information(self, "更新下载", "当前已有更新下载任务在进行。")
            return
        self._pending_update_decision = decision
        self._pending_update_package = None
        self._download_cancel_event = threading.Event()
        self._download_progress_dialog = QProgressDialog("正在解析更新包...", "取消", 0, 0, self)
        self._download_progress_dialog.setWindowTitle("下载更新")
        self._download_progress_dialog.setAutoClose(False)
        self._download_progress_dialog.setAutoReset(False)
        self._download_progress_dialog.setMinimumDuration(0)
        self._download_progress_dialog.canceled.connect(self._cancel_update_download)
        self._download_progress_dialog.show()
        threading.Thread(
            target=self._run_update_download,
            kwargs={"decision": decision, "cancel_event": self._download_cancel_event},
            daemon=True,
        ).start()

    def _cancel_update_download(self) -> None:
        if self._download_cancel_event is not None:
            self._download_cancel_event.set()

    def _run_update_download(self, *, decision: ReleaseDecision, cancel_event: threading.Event) -> None:
        target_path: Path | None = None
        try:
            with requests.Session() as http:
                resolved = resolve_lanzou_update_package(
                    share_url=decision.update_share_url,
                    file_name=decision.update_file_name,
                    session=http,
                )
                self.update_events.put(
                    (
                        "download_resolved",
                        {"file_name": resolved.file_name, "size": resolved.size, "package": resolved},
                    )
                )
                target_dir = self._package_root() / "runtime" / "tmp" / "updates"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / resolved.file_name
                with http.get(
                    resolved.download_url,
                    stream=True,
                    timeout=30,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": resolved.detail_url,
                    },
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length", "0") or "0")
                    self.update_events.put(
                        (
                            "download_started",
                            {"file_name": resolved.file_name, "total": total},
                        )
                    )
                    downloaded = 0
                    with target_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=UPDATE_DOWNLOAD_CHUNK_SIZE):
                            if cancel_event.is_set():
                                raise RuntimeError("cancelled")
                            if not chunk:
                                continue
                            handle.write(chunk)
                            downloaded += len(chunk)
                            self.update_events.put(
                                (
                                    "download_progress",
                                    {"downloaded": downloaded, "total": total},
                                )
                            )
            self.update_events.put(
                (
                    "download_complete",
                    {
                        "archive_path": str(target_path),
                        "package": resolved,
                    },
                )
            )
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                if target_path is not None:
                    target_path.unlink(missing_ok=True)
                self.update_events.put(("download_cancelled", {}))
                return
            self.update_events.put(("download_error", {"message": str(exc)}))
        except (UpdateSourceError, requests.RequestException, OSError) as exc:
            if target_path is not None:
                target_path.unlink(missing_ok=True)
            self.update_events.put(("download_error", {"message": str(exc)}))

    def _close_download_progress_dialog(self) -> None:
        if self._download_progress_dialog is not None:
            self._download_progress_dialog.close()
            self._download_progress_dialog.deleteLater()
            self._download_progress_dialog = None
        self._download_cancel_event = None

    def _launch_update_apply(self, archive_path: str) -> None:
        root = self._package_root()
        updater = root / "update_apply.py"
        if not updater.is_file():
            QMessageBox.critical(self, "更新失败", f"缺少更新脚本：{updater}")
            return
        command = [
            sys.executable,
            str(updater),
            "--package-root",
            str(root),
            "--archive",
            str(archive_path),
            "--pid",
            str(os.getpid()),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen(command, cwd=root, creationflags=creationflags)
        except OSError as exc:
            QMessageBox.critical(self, "更新失败", f"无法启动更新程序：{exc}")
            return
        QApplication.instance().quit()

    def _load_devices(self) -> None:
        devices = self.config_manager.devices()
        self.table.setRowCount(len(devices))
        app_names = get_registered_apps()
        for row, device in enumerate(devices):
            if not isinstance(device, dict):
                continue
            self.table.setItem(row, COL_SELECT, self._check_item())
            self.table.setItem(row, COL_LICENSE, QTableWidgetItem(mask_secret(str(device.get("license_key", "")))))
            self.table.setItem(row, COL_EXPIRES, QTableWidgetItem(str(device.get("license_expires", "") or "-")))
            self.table.setItem(row, COL_DEVICE_IP, QTableWidgetItem(str(device.get("ip", ""))))
            self.table.setItem(row, COL_STATUS, self._status_item("stopped"))
            self.table.setItem(row, COL_AVG_REPLY_SECONDS, self._avg_reply_item())
            self.table.setItem(row, COL_UNATTENDED_SECONDS, self._unattended_item(self._unattended_seconds_for_device(str(device.get("ip", "")))))
            app_select = NoWheelComboBox()
            for key, app_cls in app_names.items():
                app_select.addItem(app_cls.name, key)
            selected = resolve_app_name(device.get("selected_app", None))
            index = app_select.findData(selected)
            if index >= 0:
                app_select.setCurrentIndex(index)
            self.table.setCellWidget(row, COL_RUNNING_APP, app_select)
            self.table.setItem(row, COL_BALANCE, QTableWidgetItem("--"))
        self._refresh_log_device_filter(devices)
        self._refresh_summary()

    def _show_device_context_menu(self, position) -> None:
        row = self.table.rowAt(position.y())
        if row < 0:
            return
        self.table.selectRow(row)
        menu = QMenu(self)
        menu.addAction("启动", lambda: self._start_worker(row))
        menu.addAction("暂停", lambda: self._pause_worker(row))
        menu.addAction("恢复", lambda: self._resume_worker(row))
        menu.addAction("停止", lambda: self._stop_worker(row))
        menu.addSeparator()
        menu.addAction("卡密绑定", lambda: self._bind_license(row))
        menu.addAction("app设置", lambda: self._edit_app_config(row))
        menu.addAction("设备设置", lambda: self._edit_device_settings(row))
        menu.addSeparator()
        menu.addAction("删除当前设备", lambda: self._delete_device(row))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _device_for_row(self, row: int) -> dict[str, Any] | None:
        devices = self.config_manager.devices()
        if row < 0 or row >= len(devices):
            return None
        return devices[row]

    def _device_ip_for_row(self, row: int) -> str:
        item = self.table.item(row, COL_DEVICE_IP)
        return item.text() if item else ""

    def _set_row_status(self, device_ip: str, status: str, detail: str = "") -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                self.table.setItem(row, COL_STATUS, self._status_item(status, detail))
                self._refresh_summary()
                return

    def _set_row_balance(self, device_ip: str, balance: str) -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                item = QTableWidgetItem(str(balance or "--"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, COL_BALANCE, item)
                return

    def _row_status_for_device(self, device_ip: str) -> str:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                item = self.table.item(row, COL_STATUS)
                return str(item.data(256)) if item else ""
        return ""

    def _check_item(self, checked: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _status_item(self, status: str, detail: str = "") -> QTableWidgetItem:
        label, background, foreground = STATUS_COLORS.get(status, (status, "#f1f5f9", "#475569"))
        if status == "abnormal":
            label, background, foreground = "异常", "#fee2e2", "#991b1b"
        detail_text = strip_status_coordinates(detail)
        text = label
        if detail_text and status == "running":
            text = detail_text
        elif detail_text and status in {"paused", "abnormal"}:
            text = f"{label}-{detail_text}"
        item = QTableWidgetItem(text)
        item.setData(256, status)
        item.setData(257, detail_text)
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        if detail_text:
            item.setToolTip(detail_text)
        return item

    def _avg_reply_item(self, average_seconds: float | None = None, reply_round_count: int = 0) -> QTableWidgetItem:
        count = max(0, int(reply_round_count))
        value = max(0.0, float(average_seconds or 0.0))
        text = f"{value:.1f}\u79d2" if count > 0 else "--"
        item = QTableWidgetItem(text)
        item.setData(256, count)
        item.setData(257, value)
        if count > 0:
            item.setToolTip(f"\u5df2\u5b8c\u6210 {count} \u8f6e\u56de\u590d\uff0c\u5e73\u5747 {value:.1f} \u79d2")
        return item

    def _unattended_item(self, seconds: float | None = None) -> QTableWidgetItem:
        item = QTableWidgetItem(format_duration(seconds))
        item.setData(256, None if seconds is None else max(0.0, float(seconds)))
        if seconds is not None:
            item.setToolTip(f"从启动到停止：{format_duration(seconds)}")
        return item

    def _set_row_unattended_time(self, device_ip: str, seconds: float | None) -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                self.table.setItem(row, COL_UNATTENDED_SECONDS, self._unattended_item(seconds))
                return

    def _start_unattended_timer(self, device_ip: str) -> None:
        self.unattended_started_at[device_ip] = time.monotonic()
        self.unattended_elapsed_seconds[device_ip] = 0.0
        self._set_row_unattended_time(device_ip, 0.0)

    def _unattended_seconds_for_device(self, device_ip: str) -> float | None:
        started_at = self.unattended_started_at.get(device_ip)
        if started_at is not None:
            return time.monotonic() - started_at
        return self.unattended_elapsed_seconds.get(device_ip)

    def _finalize_unattended_timer(self, device_ip: str) -> None:
        seconds = self._unattended_seconds_for_device(device_ip)
        self.unattended_started_at.pop(device_ip, None)
        if seconds is not None:
            self.unattended_elapsed_seconds[device_ip] = max(0.0, float(seconds))
        self._set_row_unattended_time(device_ip, seconds)

    def _refresh_unattended_durations(self) -> None:
        for device_ip in list(self.unattended_started_at):
            self._set_row_unattended_time(device_ip, self._unattended_seconds_for_device(device_ip))

    def _set_row_average_reply_time(
        self,
        device_ip: str,
        average_seconds: float | None,
        reply_round_count: int = 0,
    ) -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                self.table.setItem(row, COL_AVG_REPLY_SECONDS, self._avg_reply_item(average_seconds, reply_round_count))
                return

    def _refresh_summary(self) -> None:
        total = self.table.rowCount()
        running = 0
        alerts = 0
        for row in range(total):
            item = self.table.item(row, COL_STATUS)
            status = str(item.data(256)) if item else ""
            if status == "running":
                running += 1
            if status in {"circuit_open", "abnormal", "license_error"}:
                alerts += 1
        self.total_devices_value.setText(str(total))
        self.running_devices_value.setText(str(running))
        self.alert_devices_value.setText(str(alerts))

    def _selected_app_for_row(self, row: int) -> str:
        widget = self.table.cellWidget(row, COL_RUNNING_APP)
        if isinstance(widget, QComboBox):
            return resolve_app_name(widget.currentData())
        return get_default_app_name()

    def _checked_rows(self) -> list[int]:
        rows: list[int] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                rows.append(row)
        return rows

    def _select_all_devices(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_device_selection(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _default_app_overrides(self) -> dict[str, dict[str, Any]]:
        return {app_name: {} for app_name in get_registered_apps()}

    def _run_for_checked_rows(self, action) -> None:
        for row in self._checked_rows():
            action(row)

    def _start_worker(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", ""))
        existing = self.workers.get(device_ip)
        if existing and existing.is_alive():
            status_item = self.table.item(row, COL_STATUS)
            current_status = str(status_item.data(256)) if status_item else ""
            if current_status == "paused":
                existing.resume()
                self._set_row_status(device_ip, "running")
            return
        selected_app = self._selected_app_for_row(row)
        device["selected_app"] = selected_app
        self.config_manager.update_device(device_ip, {"selected_app": selected_app})
        worker = DeviceWorker(device, events=self.events, config_manager=self.config_manager)
        self.workers[device_ip] = worker
        worker.start()
        self._start_unattended_timer(device_ip)
        self._set_row_status(device_ip, "running")

    def _pause_worker(self, row: int) -> None:
        device_ip = self._device_ip_for_row(row)
        worker = self.workers.get(device_ip)
        if worker:
            worker.pause()
            self._set_row_status(device_ip, "paused")

    def _resume_worker(self, row: int) -> None:
        device_ip = self._device_ip_for_row(row)
        worker = self.workers.get(device_ip)
        if worker:
            worker.resume()
            self._set_row_status(device_ip, "running")

    def _stop_worker(self, row: int) -> None:
        device_ip = self._device_ip_for_row(row)
        worker = self.workers.get(device_ip)
        if worker:
            worker.stop()
            self._set_row_status(device_ip, "stopping")

    def _delete_device(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", ""))
        answer = QMessageBox.question(
            self,
            "删除设备",
            f"确认删除设备 {device_ip} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        worker = self.workers.get(device_ip)
        if worker:
            worker.stop()
            self.workers.pop(device_ip, None)
        self.unattended_started_at.pop(device_ip, None)
        self.unattended_elapsed_seconds.pop(device_ip, None)
        if self.config_manager.remove_device(device_ip):
            self._load_devices()
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: device removed")

    def _save_apikey(self) -> None:
        value = self.apikey_input.text().strip()
        if value and not value.startswith("****"):
            self.config_manager.set("global", "apikey", value=value)
            self.apikey_input.setText(mask_secret(value))
        self.config_manager.set("global", "ocr_max_concurrency", value=self.ocr_concurrency_input.value())

    def _add_device(self) -> None:
        device_ip = self.device_ip_input.text().strip()
        if not device_ip:
            return
        created = self.config_manager.add_device(
            {
                "ip": device_ip,
                "license_key": "",
                "license_expires": "",
                "selected_app": get_default_app_name(),
                "persona": {},
                "contact_personas": {},
                "app_overrides": self._default_app_overrides(),
            }
        )
        if not created:
            QMessageBox.information(self, "设备", "该设备已存在")
            return
        self.device_ip_input.clear()
        self._load_devices()

    def _scan_devices(self) -> None:
        self.scan_device_button.setEnabled(False)
        QApplication.processEvents()
        try:
            found = scan_devices(timeout=0.3, max_workers=50)
        except Exception as exc:
            QMessageBox.warning(self, "扫描设备", str(exc))
            return
        finally:
            self.scan_device_button.setEnabled(True)
        added = 0
        for device_ip in found:
            if self.config_manager.add_device(
                {
                    "ip": device_ip,
                    "license_key": "",
                    "license_expires": "",
                    "selected_app": get_default_app_name(),
                    "persona": {},
                    "contact_personas": {},
                    "app_overrides": self._default_app_overrides(),
                }
            ):
                added += 1
        self._load_devices()
        QMessageBox.information(self, "扫描设备", f"发现 {len(found)} 台，新增 {added} 台")

    def _bind_license(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", ""))
        current_key = str(device.get("license_key", "") or "")
        license_key, accepted = QInputDialog.getText(
            self,
            "卡密绑定",
            f"为设备 {device_ip} 输入卡密：",
            text=current_key,
        )
        if not accepted:
            return
        license_key = license_key.strip()
        if not license_key:
            QMessageBox.warning(self, "卡密", "请先输入卡密")
            return

        for other in self.config_manager.devices():
            if other.get("ip") != device_ip and other.get("license_key") == license_key:
                QMessageBox.warning(self, "卡密", "卡密已被其他设备使用")
                return

        global_config = self.config_manager.get("global", default={})
        client = LicenseClient(
            api_url=str(global_config.get("license_api_url", "")),
            project_uuid=str(global_config.get("project_uuid", "")),
            project_secret=str(global_config.get("project_secret", "")),
        )
        result = client.verify_with_offline(device_ip, license_key, str(device.get("license_expires", "")))
        if not result.ok:
            self._set_row_status(device_ip, "license_error")
            QMessageBox.warning(self, "卡密", result.message)
            return
        self.config_manager.update_device(
            device_ip,
            {"license_key": license_key, "license_expires": result.expires_at},
        )
        self.table.setItem(row, COL_LICENSE, QTableWidgetItem(mask_secret(license_key)))
        self.table.setItem(row, COL_EXPIRES, QTableWidgetItem(result.expires_at or "-"))
        self._set_row_status(device_ip, "normal" if result.status == "ok" else result.status)

    def _edit_app_config(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", ""))
        app_name = self._selected_app_for_row(row)
        template_root = Path("apps") / app_name / "templates"
        dialog_config = _deep_merge(_app_default_config(app_name), self.config_manager.get_app_overrides(app_name))
        dialog_config = _deep_merge(dialog_config, self.config_manager.get_device_runtime_overrides(device_ip))
        dialog_config = _deep_merge(dialog_config, self.config_manager.get_device_app_overrides(device_ip, app_name))
        dialog = AppConfigDialog(
            dialog_config,
            device_ip=device_ip,
            app_name=app_name,
            template_root=template_root,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        app_updates = {
            key: copy.deepcopy(values[key])
            for key in (
                "ocr",
                "ocr_modes",
                "templates",
                "match_threshold",
                "popup_match_threshold",
                "template_match_regions",
                "template_match_thresholds",
            )
            if key in values
        }
        runtime_updates = {
            key: copy.deepcopy(values[key])
            for key in ("poll_interval", "step_timeout", "keep_awake_interval_seconds", "exclude_ocr_wait_from_step_timeout")
            if key in values
        }
        persona = values.get("persona", {})
        device_app_updates = {"persona": copy.deepcopy(persona)} if isinstance(persona, dict) else {}

        self.config_manager.update_app_config(app_name, app_updates)
        self.config_manager.update_device_runtime(device_ip, runtime_updates)
        self.config_manager.update_device_app_overrides(device_ip, app_name, device_app_updates)
        QMessageBox.information(self, "配置", f"{device_ip} / {app_name} 配置已保存")

    def _edit_device_settings(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", ""))
        app_name = self._selected_app_for_row(row)
        dialog_config = _deep_merge(_app_default_config(app_name), self.config_manager.get_app_overrides(app_name))
        dialog_config = _deep_merge(dialog_config, self.config_manager.get_device_app_overrides(device_ip, app_name))
        dialog = DeviceSettingsDialog(
            dialog_config,
            device_ip=device_ip,
            app_name=app_name,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.config_manager.update_device_app_overrides(device_ip, app_name, dialog.values())
        QMessageBox.information(self, "设备设置", f"{device_ip} / {app_name} 设备设置已保存")

    def _toggle_logs(self) -> None:
        self.logs.setVisible(self.log_toggle.isChecked())
        if self.log_toggle.isChecked():
            self._render_logs()

    def _refresh_log_device_filter(self, devices: list[dict[str, Any]]) -> None:
        if not hasattr(self, "log_device_filter"):
            return
        current = self.log_device_filter.currentData()
        self.log_device_filter.blockSignals(True)
        self.log_device_filter.clear()
        self.log_device_filter.addItem("全部设备", "")
        for device in devices:
            if not isinstance(device, dict):
                continue
            device_ip = str(device.get("ip", "") or "").strip()
            if device_ip:
                self.log_device_filter.addItem(device_ip, device_ip)
        index = self.log_device_filter.findData(current)
        self.log_device_filter.setCurrentIndex(index if index >= 0 else 0)
        self.log_device_filter.blockSignals(False)
        self._render_logs()

    def _render_logs(self, *_args: Any) -> None:
        if not hasattr(self, "logs"):
            return
        selected_ip = ""
        if hasattr(self, "log_device_filter"):
            selected_ip = str(self.log_device_filter.currentData() or "")
        self.logs.clear()
        for event in self.log_events:
            event_ip = str(getattr(event, "device_ip", "") or "")
            if selected_ip and event_ip != selected_ip:
                continue
            self._append_log_event(event)

    def _append_log_event(self, event: WorkerEvent) -> None:
        try:
            self.logs.append(f"{self._log_event_time(event)} {self._format_log_event(event)}")
        except Exception as exc:
            device_ip = str(getattr(event, "device_ip", "unknown") or "unknown")
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: UI事件日志格式化失败：{exc}")

    def _log_event_time(self, event: WorkerEvent) -> str:
        event_id = id(event)
        timestamp = self.log_event_times.get(event_id)
        if not timestamp:
            timestamp = time.strftime("%H:%M:%S")
            self.log_event_times[event_id] = timestamp
        return f"[{timestamp}]"

    def _format_log_event(self, event: WorkerEvent) -> str:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.level == "status" and event.message == "tick":
            return f"[{event.level}] {event.device_ip}: 心跳"
        if event.level == "status" and event.message.startswith("progress:"):
            detail = event.message.split(":", 1)[1].strip()
            return f"[{event.level}] {event.device_ip}: 进度：{detail}"
        if event.message == "runtime_step":
            detail = str(payload.get("detail", "") or payload.get("step", "") or "")
            return f"[{event.level}] {event.device_ip}: {detail}"
        if event.message == "dialogue_ocr_last_message":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            return f"[{event.level}] {event.device_ip}: OCR读取 [{contact}] {content}"
        if event.message == "ai_reply_text":
            contact = str(payload.get("contact", "") or "")
            reply_text = str(payload.get("reply_text", "") or "")
            if bool(payload.get("skip")):
                reason = str(payload.get("reason", "") or "")
                return f"[{event.level}] {event.device_ip}: AI跳过 [{contact}] {reason}"
            return f"[{event.level}] {event.device_ip}: AI回复 [{contact}] {reply_text}"
        if event.message == "target_selected":
            name = str(payload.get("name", "") or "")
            content = str(payload.get("content", "") or "")
            unread_count = payload.get("unread_count")
            favorability = payload.get("favorability")
            return f"[{event.level}] {event.device_ip}: 选中目标 [{name}] 未读 {unread_count} 相似度 {favorability} 内容：{content}"
        if event.message == "xinyu_message_input":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            source = str(payload.get("source", "") or "")
            return f"[{event.level}] {event.device_ip}: 心遇输入来源 {source} [{contact}] {content}"
        if event.message == "xinyu_reply_text":
            contact = str(payload.get("contact", "") or "")
            reply_text = str(payload.get("reply_text", "") or "")
            if bool(payload.get("skip")):
                reason = str(payload.get("reason", "") or "")
                return f"[{event.level}] {event.device_ip}: 心遇跳过 [{contact}] {reason}"
            return f"[{event.level}] {event.device_ip}: 心遇回复 [{contact}] {reply_text}"
        if event.message == "xinyu_send_clicks":
            return f"[{event.level}] {event.device_ip}: 心遇发送点击次数 {payload.get('clicks', '')}"
        if event.message == "xinyu_send_flow_summary":
            contact = str(payload.get("contact", "") or "")
            return (
                f"[{event.level}] {event.device_ip}: 心遇发送摘要 [{contact}] "
                f"输入框={payload.get('input_box_source', '') or '-'}"
                f"{('/' + str(payload.get('input_box_template', ''))) if payload.get('input_box_template') else ''}，"
                f"麦克风={payload.get('microphone_template', '') or '-'}，"
                f"TTS={'成功' if payload.get('tts_success') else '失败'}，"
                f"重试 {payload.get('tts_retry_count', 0)} 次，"
                f"发送点击 {payload.get('send_clicks', '')} 次，"
                f"补点 {payload.get('send_clear_extra_clicks', 0)} 次，"
                f"返回 {payload.get('return_clicks', '')} 次，"
                f"原因={payload.get('return_reason', '') or '-'}"
            )
        if event.message == "xinyu_round_metrics":
            return (
                f"[{event.level}] {event.device_ip}: 心遇本轮统计 "
                f"OCR {payload.get('ocr_count', '')} 次，截图 {payload.get('capture_count', '')} 次，"
                f"详情回退 {payload.get('dialogue_compact_fallback_count', 0)} 次，"
                f"耗时 {payload.get('seconds', '')} 秒"
            )
        if event.message == "xinyu_balance":
            return f"[{event.level}] {event.device_ip}: 心遇余额 {payload.get('balance', '')}"
        if event.message == "xinyu_balance_failed":
            return f"[{event.level}] {event.device_ip}: 心遇余额读取失败 {payload.get('error', '')}"
        if event.message == "检测到黑屏":
            return f"[{event.level}] {event.device_ip}: 检测到黑屏 {payload.get('error', '')}"
        if event.message == "检测到无手机":
            return f"[{event.level}] {event.device_ip}: 检测到无手机 {payload.get('error', '')}"
        if event.message == "防熄屏失败":
            return f"[{event.level}] {event.device_ip}: 防熄屏失败 {payload.get('error', '')}"
        if event.message == "心遇重复选中同一会话":
            name = str(payload.get("name", "") or "")
            content = str(payload.get("content", "") or "")
            unread_count = payload.get("unread_count", "")
            return f"[{event.level}] {event.device_ip}: 心遇重复选中 [{name}] 未读 {unread_count} 内容：{content}"
        if event.message == "心遇复用待发送回复":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            return f"[{event.level}] {event.device_ip}: 心遇复用待发送回复 [{contact}] {content}"
        if event.message == "心遇发送按钮被连续点击":
            contact = str(payload.get("contact", "") or "")
            clicks = payload.get("clicks", "")
            return f"[{event.level}] {event.device_ip}: 心遇发送按钮连续点击 [{contact}] {clicks} 次"
        if event.message == "心遇发送后未能返回列表":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            return f"[{event.level}] {event.device_ip}: 心遇发送后未能返回列表 [{contact}] {content}"
        if event.message == "huanyou_balance":
            return f"[{event.level}] {event.device_ip}: 欢友余额 {payload.get('balance', '')}"
        if event.message == "huanyou_balance_failed":
            return f"[{event.level}] {event.device_ip}: 欢友余额读取失败 {payload.get('error', '')}"
        if event.message == "huanyou_message_input":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            source = str(payload.get("source", "") or "")
            return f"[{event.level}] {event.device_ip}: 欢友输入来源 {source} [{contact}] {content}"
        if event.message == "huanyou_reply_text":
            contact = str(payload.get("contact", "") or "")
            reply_text = str(payload.get("reply_text", "") or "")
            if bool(payload.get("skip")):
                reason = str(payload.get("reason", "") or "")
                return f"[{event.level}] {event.device_ip}: 欢友跳过 [{contact}] {reason}"
            return f"[{event.level}] {event.device_ip}: 欢友回复 [{contact}] {reply_text}"
        if event.message == "huanyou_round_metrics":
            return (
                f"[{event.level}] {event.device_ip}: 欢友本轮统计 "
                f"OCR {payload.get('ocr_count', '')} 次，截图 {payload.get('capture_count', '')} 次，"
                f"耗时 {payload.get('seconds', '')} 秒"
            )
        if event.message == "huanyou_new_greetings":
            return f"[{event.level}] {event.device_ip}: 欢友新招呼处理 {payload.get('count', '')} 次"
        if event.message == "huanyou_chat_window_swipe":
            start = payload.get("start", {}) if isinstance(payload.get("start", {}), dict) else {}
            end = payload.get("end", {}) if isinstance(payload.get("end", {}), dict) else {}
            return (
                f"[{event.level}] {event.device_ip}: 欢友聊天上滑 "
                f"({start.get('x', '')}, {start.get('y', '')}) -> ({end.get('x', '')}, {end.get('y', '')})"
            )
        if event.message == "huanyou_send_clicks":
            return f"[{event.level}] {event.device_ip}: 欢友发送点击次数 {payload.get('clicks', '')}"
        if event.message == "huanyou_contact_send_count":
            contact = str(payload.get("contact", "") or "")
            count = payload.get("count", "")
            return f"[{event.level}] {event.device_ip}: 欢友本次启动已给 [{contact}] 发送 {count} 次"
        if event.message == "huanyou_next_chat_verified":
            previous = str(payload.get("previous_name", "") or "")
            current = str(payload.get("current_name", "") or "")
            clicks = payload.get("clicks", "")
            return f"[{event.level}] {event.device_ip}: 欢友下一条已切人 {previous} -> {current}，点击 {clicks} 次"
        if event.message == "huanyou_next_chat_missing":
            previous = str(payload.get("previous_name", "") or "")
            reason = str(payload.get("reason", "") or "")
            return f"[{event.level}] {event.device_ip}: 欢友未找到下一条 [{previous}] {reason}"
        if event.message == "huanyou_input_box_retracted":
            return f"[{event.level}] {event.device_ip}: 欢友已收回输入框 ({payload.get('x', '')}, {payload.get('y', '')})"
        if event.message == "huanyou_target_selected":
            name = str(payload.get("name", "") or "")
            content = str(payload.get("content", "") or "")
            unread_count = payload.get("unread_count")
            favorability = payload.get("favorability")
            return f"[{event.level}] {event.device_ip}: 欢友选中目标 [{name}] 未读 {unread_count} 好感度 {favorability} 内容：{content}"
        if event.message == "huanyou_startup_back_button_cached":
            return f"[{event.level}] {event.device_ip}: 欢友启动检查已缓存返回按钮"
        if event.message == "huanyou_startup_back_button_missing":
            return f"[{event.level}] {event.device_ip}: 欢友启动检查未找到返回按钮，停止进程 {payload.get('error', '')}"
        if event.message == "no_phone_detected":
            return f"[{event.level}] {event.device_ip}: 检测到无手机 {payload.get('error', '')}"
        if event.message == "black_screen_detected":
            return f"[{event.level}] {event.device_ip}: 检测到黑屏 {payload.get('error', '')}"
        if event.message == "keep_awake_failed":
            return f"[{event.level}] {event.device_ip}: 防熄屏失败 {payload.get('error', '')}"
        if event.message.startswith("欢友步骤不可用："):
            count = payload.get("error_count", "")
            threshold = payload.get("circuit_breaker_threshold", "")
            if count != "" and threshold != "":
                return f"[{event.level}] {event.device_ip}: {event.message}（错误计数 {count}/{threshold}）"
            return f"[{event.level}] {event.device_ip}: {event.message}"
        if event.message == "app_error":
            label = str(payload.get("label", "") or payload.get("step", "") or "")
            error = str(payload.get("error", "") or "")
            count = payload.get("error_count", "")
            suffix = f"（第 {count} 次）" if count != "" else ""
            return f"[{event.level}] {event.device_ip}: 运行异常{suffix} {label}：{error}"
        if event.message in {"memory hit", "memory miss", "memory updated", "memory lookup failed", "memory update failed"}:
            contact = str(payload.get("contact", "") or "")
            conversation_id = str(payload.get("conversation_id", "") or "")
            short_id = conversation_id[-8:] if conversation_id else "-"
            error = str(payload.get("error", "") or "")
            labels = {
                "memory hit": "记忆命中",
                "memory miss": "记忆未命中",
                "memory updated": "记忆已更新",
                "memory lookup failed": "记忆读取失败",
                "memory update failed": "记忆更新失败",
            }
            detail = f" [{contact}] 会话 {short_id}"
            if error:
                detail += f"：{error}"
            return f"[{event.level}] {event.device_ip}: {labels[event.message]}{detail}"
        if event.message == "using pending reply":
            contact = str(payload.get("contact", "") or "")
            content = str(payload.get("content", "") or "")
            return f"[{event.level}] {event.device_ip}: 复用待发送回复 [{contact}] {content}"
        generic_labels = {
            "started": "已启动",
            "stopping": "停止中",
            "stopped": "已停止",
            "resumed": "已恢复",
            "paused": "已暂停",
            "circuit open": "熔断已触发",
            "device removed": "设备已移除",
            "license expired": "许可证已过期",
            "license_api_url not configured; skipping online license check": "未配置许可证校验地址，已跳过在线校验",
        }
        if event.message in generic_labels:
            detail = str(payload.get("detail", "") or "")
            suffix = f"：{detail}" if detail else ""
            return f"[{event.level}] {event.device_ip}: {generic_labels[event.message]}{suffix}"
        return f"[{event.level}] {event.device_ip}: {event.message}"

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.log_event_times.setdefault(id(event), time.strftime("%H:%M:%S"))
            self.log_events.append(event)
            try:
                self._handle_worker_event(event)
            except Exception as exc:
                device_ip = str(getattr(event, "device_ip", "unknown") or "unknown")
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: UI事件处理失败：{exc}")
        try:
            self._refresh_unattended_durations()
        except Exception as exc:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] [error] UI: 无干预时长刷新失败：{exc}")

        self._drain_update_events()

    def _drain_update_events(self) -> None:
        while True:
            try:
                event_type, payload = self.update_events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_update_event(event_type, payload)
            except Exception as exc:
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] [error] UI: 更新事件处理失败：{exc}")

    def _handle_update_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type in {"check_complete", "check_error"}:
            self._finish_release_check()

        if event_type == "check_error":
            message = str(payload.get("message", "") or "检查更新失败")
            manual = bool(payload.get("manual"))
            if manual:
                QMessageBox.warning(self, "检查更新", message)
            else:
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] [warning] UI: {message}")
            return

        if event_type == "check_complete":
            decision = payload.get("decision")
            manual = bool(payload.get("manual"))
            if not isinstance(decision, ReleaseDecision):
                return
            if decision.action == "none":
                if manual:
                    QMessageBox.information(self, "检查更新", "当前已经是最新版本。")
                return
            if not manual and self._skip_update_version() == decision.target_app_version:
                return
            self._show_release_decision_dialog(decision)
            return

        if event_type == "download_resolved":
            package = payload.get("package")
            if isinstance(package, ResolvedUpdatePackage):
                self._pending_update_package = package
            if self._download_progress_dialog is not None:
                file_name = str(payload.get("file_name", "") or "")
                size = str(payload.get("size", "") or "")
                suffix = f" ({size})" if size else ""
                self._download_progress_dialog.setLabelText(f"正在下载：{file_name}{suffix}")
            return

        if event_type == "download_started":
            if self._download_progress_dialog is not None:
                total = int(payload.get("total") or 0)
                if total > 0:
                    self._download_progress_dialog.setRange(0, total)
                    self._download_progress_dialog.setValue(0)
                else:
                    self._download_progress_dialog.setRange(0, 0)
            return

        if event_type == "download_progress":
            if self._download_progress_dialog is not None:
                total = int(payload.get("total") or 0)
                downloaded = int(payload.get("downloaded") or 0)
                if total > 0:
                    self._download_progress_dialog.setRange(0, total)
                    self._download_progress_dialog.setValue(min(downloaded, total))
            return

        if event_type == "download_cancelled":
            self._close_download_progress_dialog()
            QMessageBox.information(self, "下载更新", "已取消本次更新下载。")
            return

        if event_type == "download_error":
            self._close_download_progress_dialog()
            QMessageBox.warning(self, "下载更新", str(payload.get("message", "") or "下载更新失败"))
            return

        if event_type == "download_complete":
            self._close_download_progress_dialog()
            archive_path = str(payload.get("archive_path", "") or "").strip()
            if not archive_path:
                QMessageBox.warning(self, "下载更新", "更新包已下载，但路径无效。")
                return
            answer = QMessageBox.question(
                self,
                "下载完成",
                "更新包已下载完成。\n点击“是”将关闭当前程序并开始应用更新。\n更新完成后会提示你重新打开主程序。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._launch_update_apply(archive_path)
            else:
                QMessageBox.information(self, "下载完成", f"更新包已保存在：{archive_path}")
            return

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        event_ip = str(event.device_ip or "")
        message = str(event.message or "")
        level = str(event.level or "")
        selected_ip = ""
        if hasattr(self, "log_device_filter"):
            selected_ip = str(self.log_device_filter.currentData() or "")
        if not selected_ip or event_ip == selected_ip:
            self._append_log_event(event)
        if level == "status":
            if self._row_status_for_device(event_ip) in {"stopping", "stopped"}:
                return
            is_open = bool(payload.get("is_circuit_open"))
            current_step = str(payload.get("current_step", "") or "")
            detail = strip_status_coordinates(str(payload.get("current_step_label", "") or ""))
            reply_round_count = int(payload.get("reply_round_count") or 0)
            average_reply_seconds = payload.get("average_reply_seconds")
            self._set_row_average_reply_time(event_ip, average_reply_seconds, reply_round_count)
            if is_open:
                row_status = "circuit_open"
            elif current_step.startswith(("black_screen", "no_phone")):
                row_status = "abnormal"
            else:
                row_status = "running"
            self._set_row_status(event_ip, row_status, detail)
        elif message in {"paused", "stopped", "stopping"}:
            if message == "stopped":
                self._finalize_unattended_timer(event_ip)
            detail = str(payload.get("detail", "") or "")
            self._set_row_status(event_ip, message, detail)
        elif message in {"xinyu_balance", "huanyou_balance"}:
            self._set_row_balance(event_ip, str(payload.get("balance", "") or ""))

class _LegacyAppConfigDialog(QDialog):
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        template_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = with_app_defaults(overrides)
        self.device_ip = device_ip
        self.app_name = app_name
        self.template_root = template_root
        self.setWindowTitle(f"App 配置 - {device_ip} / {app_name}")
        self.resize(840, 620)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备配置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("当前设置只作用于这台设备当前选中的 App，包括运行参数、模板和人设。")
        scope.setObjectName("HelperLabel")
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        root.addWidget(tabs)

        config_page = QWidget()
        config_root = QVBoxLayout(config_page)
        config_root.setContentsMargins(0, 0, 0, 0)
        config_root.setSpacing(10)
        runtime_panel = QFrame()
        runtime_panel.setObjectName("InsetPanel")
        runtime_layout = QVBoxLayout(runtime_panel)
        runtime_layout.setContentsMargins(14, 12, 14, 12)
        runtime_layout.setSpacing(8)
        runtime_title = QLabel("运行参数")
        runtime_title.setObjectName("SectionLabel")
        runtime_layout.addWidget(runtime_title)
        runtime_help = QLabel("这些参数只影响当前设备当前 App 的识别轮询和匹配策略。")
        runtime_help.setObjectName("HelperLabel")
        runtime_layout.addWidget(runtime_help)
        coordinate_help = QLabel("OCR、模板匹配、联动测试和实际点击统一使用竖屏归一化坐标，当前调试阶段不再额外转换点击坐标。")
        coordinate_help.setObjectName("HelperLabel")
        coordinate_help.setWordWrap(True)
        runtime_layout.addWidget(coordinate_help)
        layout = QFormLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)
        runtime_layout.addLayout(layout)
        config_root.addWidget(runtime_panel)

        self.poll_interval = QSpinBox()
        self.poll_interval.setRange(1, 3600)
        self.poll_interval.setValue(int(overrides.get("poll_interval", 1)))
        layout.addRow("轮询间隔（秒）", self.poll_interval)

        self.step_timeout = QSpinBox()
        self.step_timeout.setRange(1, 300)
        self.step_timeout.setValue(int(overrides.get("step_timeout", 10)))
        layout.addRow("单步超时（秒）", self.step_timeout)

        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.1, 1.0)
        self.match_threshold.setSingleStep(0.01)
        self.match_threshold.setValue(float(overrides.get("match_threshold", 0.7)))
        self.popup_match_threshold = QDoubleSpinBox()
        self.popup_match_threshold.setRange(0.1, 1.0)
        self.popup_match_threshold.setSingleStep(0.01)
        self.popup_match_threshold.setValue(float(overrides.get("popup_match_threshold", 0.8)))
        layout.addRow("\u5f39\u7a97\u9608\u503c", self.popup_match_threshold)
        layout.addRow("模板阈值", self.match_threshold)

        ocr = overrides.get("ocr", {}) if isinstance(overrides.get("ocr", {}), dict) else {}
        self.det_db_thresh = QDoubleSpinBox()
        self.det_db_thresh.setRange(0.0, 1.0)
        self.det_db_thresh.setSingleStep(0.01)
        self.det_db_thresh.setValue(float(ocr.get("det_db_thresh", 0.3)))
        layout.addRow("OCR det_db_thresh", self.det_db_thresh)

        self.rec_thresh = QDoubleSpinBox()
        self.rec_thresh.setRange(0.0, 1.0)
        self.rec_thresh.setSingleStep(0.01)
        self.rec_thresh.setValue(float(ocr.get("rec_thresh", 0.5)))
        layout.addRow("OCR rec_thresh", self.rec_thresh)

        ocr_modes = overrides.get("ocr_modes", {}) if isinstance(overrides.get("ocr_modes", {}), dict) else {}
        self.ocr_mode_fields: dict[str, dict[str, Any]] = {}
        for mode_key, mode_label in [("dialogue", "\u5bf9\u8bdd\u6a21\u5f0f"), ("list", "\u5217\u8868\u6a21\u5f0f"), ("points", "\u79ef\u5206\u6a21\u5f0f")]:
            mode_config = ocr_modes.get(mode_key, {}) if isinstance(ocr_modes.get(mode_key, {}), dict) else {}
            preprocess = mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {}
            rules = mode_config.get("rules", {}) if isinstance(mode_config.get("rules", {}), dict) else {}

            mode_panel = QFrame()
            mode_panel.setObjectName("InsetPanel")
            mode_panel_layout = QFormLayout(mode_panel)
            mode_panel_layout.setContentsMargins(12, 10, 12, 10)
            mode_panel_layout.setSpacing(8)

            preprocess_box = QComboBox()
            preprocess_box.addItems(["original"])
            preprocess_box.setCurrentText("original")
            mode_panel_layout.addRow(f"{mode_label} 预处理", preprocess_box)

            scale_box = QSpinBox()
            scale_box.setRange(1, 4)
            scale_box.setValue(int(preprocess.get("scale", 2)))
            mode_panel_layout.addRow(f"{mode_label} 放大倍数", scale_box)

            min_confidence = QDoubleSpinBox()
            min_confidence.setRange(0.0, 1.0)
            min_confidence.setSingleStep(0.01)
            min_confidence.setValue(float(rules.get("min_confidence", 0.3)))
            mode_panel_layout.addRow(f"{mode_label} 最低置信度", min_confidence)

            min_height = QDoubleSpinBox()
            min_height.setRange(0.0, 0.2)
            min_height.setSingleStep(0.005)
            min_height.setDecimals(3)
            min_height.setValue(float(rules.get("min_box_height_ratio", 0.02)))
            mode_panel_layout.addRow(f"{mode_label} 最小框高比例", min_height)

            fields: dict[str, Any] = {
                "preprocess": preprocess_box,
                "scale": scale_box,
                "min_confidence": min_confidence,
                "min_box_height_ratio": min_height,
            }
            self.ocr_mode_fields[mode_key] = fields
            runtime_layout.addWidget(mode_panel)

        config_root.addStretch(1)
        tabs.addTab(config_page, "运行设置")

        template_page = QWidget()
        template_layout = QVBoxLayout(template_page)
        template_layout.setContentsMargins(0, 0, 0, 0)
        self.template_manager = TemplateManagerWidget(
            template_root,
            self.device_ip,
            self.app_name,
            self.overrides,
            show_offset_controls=False,
            parent=template_page,
        )
        template_layout.addWidget(self.template_manager)
        tabs.addTab(template_page, "模板管理")

        persona_page = QWidget()
        persona_root = QVBoxLayout(persona_page)
        persona_root.setContentsMargins(0, 0, 0, 0)
        persona_root.setSpacing(10)
        persona_panel = QFrame()
        persona_panel.setObjectName("InsetPanel")
        persona_panel_layout = QVBoxLayout(persona_panel)
        persona_panel_layout.setContentsMargins(14, 12, 14, 12)
        persona_panel_layout.setSpacing(8)
        persona_title = QLabel("人设参数")
        persona_title.setObjectName("SectionLabel")
        persona_panel_layout.addWidget(persona_title)
        persona_help = QLabel("AI 回复时会优先使用这里的人设，覆盖全局默认值。")
        persona_help.setObjectName("HelperLabel")
        persona_panel_layout.addWidget(persona_help)
        persona_layout = QFormLayout()
        persona_layout.setContentsMargins(0, 4, 0, 0)
        persona_layout.setSpacing(10)
        persona = overrides.get("persona", {}) if isinstance(overrides.get("persona", {}), dict) else {}
        self.persona_inputs: dict[str, QLineEdit] = {}
        persona_selector, self.request_persona_combo, self.request_persona_custom_input = _create_request_persona_selector(
            str(persona.get("requestPersona", ""))
        )
        persona_layout.addRow("人设授权 ID", persona_selector)
        for key, label in [
            ("name", "名字"),
            ("age", "年龄"),
            ("work", "职业"),
            ("location", "住址"),
            ("renshe", "自定义人设"),
        ]:
            line = QLineEdit(str(persona.get(key, "")))
            self.persona_inputs[key] = line
            persona_layout.addRow(label, line)
        persona_panel_layout.addLayout(persona_layout)
        persona_root.addWidget(persona_panel)
        persona_root.addStretch(1)
        tabs.addTab(persona_page, "人设")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> dict[str, Any]:
        values = dict(self.overrides)
        ocr = dict(values.get("ocr", {}) if isinstance(values.get("ocr", {}), dict) else {})
        ocr["det_db_thresh"] = self.det_db_thresh.value()
        ocr["rec_thresh"] = self.rec_thresh.value()

        ocr_modes = dict(values.get("ocr_modes", {}) if isinstance(values.get("ocr_modes", {}), dict) else {})
        for mode_key, fields in self.ocr_mode_fields.items():
            mode_config = dict(ocr_modes.get(mode_key, {}) if isinstance(ocr_modes.get(mode_key, {}), dict) else {})
            preprocess = dict(mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {})
            rules = dict(mode_config.get("rules", {}) if isinstance(mode_config.get("rules", {}), dict) else {})
            preprocess["mode"] = "original"
            preprocess["scale"] = fields["scale"].value()
            preprocess.pop("auxiliary_mode", None)
            rules["min_confidence"] = fields["min_confidence"].value()
            rules["min_box_height_ratio"] = fields["min_box_height_ratio"].value()
            if mode_key == "dialogue":
                rules["ocr_result_excluded_keywords"] = self._keywords_from_editor(
                    self.dialogue_ocr_result_excluded_keywords
                )
                rules["line_excluded_keywords"] = self._keywords_from_editor(
                    self.dialogue_line_excluded_keywords
                )
            mode_config["preprocess"] = preprocess
            mode_config["rules"] = rules
            ocr_modes[mode_key] = mode_config

        values.update({
            "poll_interval": self.poll_interval.value(),
            "step_timeout": self.step_timeout.value(),
            "match_threshold": self.match_threshold.value(),
            "popup_match_threshold": self.popup_match_threshold.value(),
            "ocr": ocr,
            "ocr_modes": ocr_modes,
            "templates": {
                "buttons": str(self.template_manager.buttons_dir),
                "screens": str(self.template_manager.screens_dir),
            },
            "persona": {
                "requestPersona": _request_persona_value(self.request_persona_combo, self.request_persona_custom_input),
                **{key: field.text().strip() for key, field in self.persona_inputs.items()},
            },
        })
        if self.app_name == "huanyou":
            values["next_message_click_x_offset"] = self.template_manager.next_message_click_x_offset
            values["next_message_click_y_offset"] = self.template_manager.next_message_click_y_offset
        return values


class _LegacyTemplateManagerWidget(QWidget):
    def __init__(self, template_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.template_root = template_root
        self.buttons_dir = template_root / "buttons"
        self.screens_dir = template_root / "screens"
        self.buttons_dir.mkdir(parents=True, exist_ok=True)
        self.screens_dir.mkdir(parents=True, exist_ok=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        intro_panel = QFrame()
        intro_panel.setObjectName("InsetPanel")
        intro_layout = QVBoxLayout(intro_panel)
        intro_layout.setContentsMargins(14, 12, 14, 12)
        intro_layout.setSpacing(4)
        title = QLabel("模板资源")
        title.setObjectName("SectionLabel")
        intro_layout.addWidget(title)
        subtitle = QLabel("导入的模板只保存在当前设备当前 App 目录，不会影响其他设备。")
        subtitle.setObjectName("HelperLabel")
        intro_layout.addWidget(subtitle)
        coordinate_note = QLabel("模板匹配、弹窗关闭点位、OCR 结果坐标、联动测试输入和当前实际点击，都按竖屏归一化坐标理解。")
        coordinate_note.setObjectName("HelperLabel")
        coordinate_note.setWordWrap(True)
        intro_layout.addWidget(coordinate_note)
        location = QLabel(f"目录：{self.template_root}")
        location.setObjectName("HelperLabel")
        intro_layout.addWidget(location)
        layout.addWidget(intro_panel)

        import_panel = QFrame()
        import_panel.setObjectName("InsetPanel")
        import_layout = QVBoxLayout(import_panel)
        import_layout.setContentsMargins(14, 12, 14, 12)
        import_layout.setSpacing(8)
        import_title = QLabel("导入模板")
        import_title.setObjectName("SectionLabel")
        import_layout.addWidget(import_title)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        import_layout.addLayout(buttons)
        layout.addWidget(import_panel)

        list_panel = QFrame()
        list_panel.setObjectName("InsetPanel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(14, 12, 14, 12)
        list_layout.setSpacing(8)
        list_title = QLabel("当前模板清单")
        list_title.setObjectName("SectionLabel")
        list_layout.addWidget(list_title)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类型", "文件", "匹配范围", "操作"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        list_layout.addWidget(self.table)
        layout.addWidget(list_panel)

        for rule in import_template_rules():
            button = QPushButton(f"导入{rule.label}")
            button.clicked.connect(lambda checked=False, k=rule.key: self._import_template(k))
            buttons.addWidget(button)
        self.reload()

    def reload(self) -> None:
        files: list[tuple[str, Path]] = []
        for path in sorted(self.buttons_dir.glob("*.jpg")):
            kind = "弹窗" if is_popup_template_name(path.name) else "按钮"
            files.append((kind, path))
        for path in sorted(self.screens_dir.glob("*.jpg")):
            files.append(("界面", path))
        self.table.setRowCount(len(files))
        for row, (kind, path) in enumerate(files):
            self.table.setItem(row, 0, QTableWidgetItem(kind))
            self.table.setItem(row, 1, QTableWidgetItem(path.name))
            delete_button = QPushButton("删除")
            delete_button.clicked.connect(lambda checked=False, p=path: self._delete_template(p))
            self.table.setCellWidget(row, 2, delete_button)

    def _import_template(self, key: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模板图片",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )
        if not file_path:
            return
        source = Path(file_path)
        target = import_target_path(key, self.buttons_dir, self.screens_dir)
        Image.open(source).convert("RGB").save(target, format="JPEG", quality=95)
        self.reload()

    def _delete_template(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        self.reload()


class PopupCropLabel(QWidget):
    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.origin = QPoint()
        self.selection = QRect()
        self.on_point_selected: Any | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.source_pixmap = pixmap
        self.display_pixmap = pixmap.scaled(
            760,
            1200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.origin = QPoint()
        self.selection = QRect()
        self.setFixedSize(self.display_pixmap.size())
        self.update()

    def _display_point_to_source_ratio(self, point: QPoint) -> tuple[float, float]:
        width = max(1, self.display_pixmap.width())
        height = max(1, self.display_pixmap.height())
        x = max(0.0, min(1.0, point.x() / width))
        y = max(0.0, min(1.0, point.y() / height))
        return round(x, 4), round(y, 4)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.origin = self._clamp_point(event.position().toPoint())
        self.selection = QRect(self.origin, self.origin)
        if callable(self.on_point_selected):
            self.on_point_selected(*self._display_point_to_source_ratio(self.origin))

    def mouseMoveEvent(self, event: Any) -> None:
        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        self.selection = QRect(self.origin, self._clamp_point(event.position().toPoint())).normalized()
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.selection = QRect(self.origin, self._clamp_point(event.position().toPoint())).normalized()
        if callable(self.on_point_selected):
            self.on_point_selected(*self._display_point_to_source_ratio(self.selection.center()))
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.display_pixmap)
        if self.selection.isValid() and not self.selection.isNull():
            painter.setPen(QPen(QColor(249, 115, 22), 2))
            painter.setBrush(QColor(249, 115, 22, 40))
            painter.drawRect(self.selection)

    def _clamp_point(self, point: QPoint) -> QPoint:
        return QPoint(
            max(0, min(point.x(), self.display_pixmap.width())),
            max(0, min(point.y(), self.display_pixmap.height())),
        )

    def crop_box(self) -> tuple[int, int, int, int] | None:
        rect = self.selection.intersected(QRect(0, 0, self.display_pixmap.width(), self.display_pixmap.height()))
        if rect.width() < 4 or rect.height() < 4:
            return None
        scale_x = self.source_pixmap.width() / self.display_pixmap.width()
        scale_y = self.source_pixmap.height() / self.display_pixmap.height()
        left = int(rect.left() * scale_x)
        top = int(rect.top() * scale_y)
        right = int((rect.left() + rect.width()) * scale_x)
        bottom = int((rect.top() + rect.height()) * scale_y)
        return left, top, right, bottom


class PopupCropDialog(QDialog):
    def __init__(
        self,
        image: Image.Image,
        template_label: str = "模板",
        help_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_image = image.convert("RGB")
        self.cropped_image: Image.Image | None = None
        self.crop_region_norm: list[float] | None = None
        self.setWindowTitle(f"框选{template_label}")
        self.resize(880, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        help_label = QLabel(help_text or f"在截图中拖拽框选要保存的{template_label}区域。")
        help_label.setObjectName("HelperLabel")
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        self.crop_label = PopupCropLabel(_pixmap_from_pil_image(self.source_image))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.crop_label)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("确定")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self._accept_crop)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _accept_crop(self) -> None:
        box = self.crop_label.crop_box()
        if box is None:
            QMessageBox.warning(self, "未选择区域", "请先在图片上框选区域")
            return
        width, height = self.source_image.size
        left, top, right, bottom = box
        self.crop_region_norm = [
            round(left / width, 3),
            round(top / height, 3),
            round((right - left) / width, 3),
            round((bottom - top) / height, 3),
        ]
        self.cropped_image = self.source_image.crop(box)
        self.accept()


class PopupDualRegionLabel(QWidget):
    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.origin = QPoint()
        self.active_region = "template"
        self.selections: dict[str, QRect] = {}
        self.on_selection_changed: Any | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.source_pixmap = pixmap
        self.display_pixmap = pixmap.scaled(
            760,
            1200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.origin = QPoint()
        self.selections = {}
        self.setFixedSize(self.display_pixmap.size())
        self.update()

    def set_active_region(self, key: str) -> None:
        self.active_region = key
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.origin = self._clamp_point(event.position().toPoint())
        self.selections[self.active_region] = QRect(self.origin, self.origin)
        self._emit_selection_changed()

    def mouseMoveEvent(self, event: Any) -> None:
        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        self.selections[self.active_region] = QRect(
            self.origin,
            self._clamp_point(event.position().toPoint()),
        ).normalized()
        self._emit_selection_changed()
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.selections[self.active_region] = QRect(
            self.origin,
            self._clamp_point(event.position().toPoint()),
        ).normalized()
        self._emit_selection_changed()
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.display_pixmap)
        styles = {
            "template": (QColor(249, 115, 22), QColor(249, 115, 22, 40)),
            "match": (QColor(37, 99, 235), QColor(37, 99, 235, 36)),
        }
        for key in ("template", "match"):
            rect = self.selections.get(key)
            if rect is None or not rect.isValid() or rect.isNull():
                continue
            pen_color, brush_color = styles[key]
            painter.setPen(QPen(pen_color, 3 if key == self.active_region else 2))
            painter.setBrush(brush_color)
            painter.drawRect(rect)

    def crop_box(self, key: str) -> tuple[int, int, int, int] | None:
        rect = self.selections.get(key, QRect()).intersected(
            QRect(0, 0, self.display_pixmap.width(), self.display_pixmap.height())
        )
        if rect.width() < 4 or rect.height() < 4:
            return None
        scale_x = self.source_pixmap.width() / self.display_pixmap.width()
        scale_y = self.source_pixmap.height() / self.display_pixmap.height()
        left = int(rect.left() * scale_x)
        top = int(rect.top() * scale_y)
        right = int((rect.left() + rect.width()) * scale_x)
        bottom = int((rect.top() + rect.height()) * scale_y)
        return left, top, right, bottom

    def normalized_region(self, key: str) -> list[float] | None:
        box = self.crop_box(key)
        if box is None:
            return None
        width = max(1, self.source_pixmap.width())
        height = max(1, self.source_pixmap.height())
        left, top, right, bottom = box
        return [
            round(left / width, 3),
            round(top / height, 3),
            round((right - left) / width, 3),
            round((bottom - top) / height, 3),
        ]

    def _emit_selection_changed(self) -> None:
        if callable(self.on_selection_changed):
            self.on_selection_changed()

    def _clamp_point(self, point: QPoint) -> QPoint:
        return QPoint(
            max(0, min(point.x(), self.display_pixmap.width())),
            max(0, min(point.y(), self.display_pixmap.height())),
        )


class PopupDualRegionDialog(QDialog):
    def __init__(self, image: Image.Image, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_image = image.convert("RGB")
        self.cropped_image: Image.Image | None = None
        self.template_region_norm: list[float] | None = None
        self.match_region_norm: list[float] | None = None
        self.setWindowTitle("添加弹窗")
        self.resize(1080, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        self.crop_label = PopupDualRegionLabel(_pixmap_from_pil_image(self.source_image))
        self.crop_label.on_selection_changed = self._refresh_region_labels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.crop_label)
        body.addWidget(scroll, 1)

        side_panel = QFrame()
        side_panel.setObjectName("InsetPanel")
        side = QVBoxLayout(side_panel)
        side.setContentsMargins(14, 12, 14, 12)
        side.setSpacing(10)
        title = QLabel("弹窗位置")
        title.setObjectName("SectionLabel")
        side.addWidget(title)
        help_label = QLabel("在左侧同一张截图上分别画两个框：当前图片位置用于裁剪保存弹窗模板；匹配范围用于运行时限定这张弹窗的查找区域。")
        help_label.setObjectName("HelperLabel")
        help_label.setWordWrap(True)
        side.addWidget(help_label)

        self.template_button = QPushButton("设置当前图片位置")
        self.template_button.setObjectName("PrimaryButton")
        self.template_button.clicked.connect(lambda: self._set_active_region("template"))
        side.addWidget(self.template_button)
        self.template_region_label = QLabel("当前图片位置：未设置")
        self.template_region_label.setWordWrap(True)
        side.addWidget(self.template_region_label)

        self.match_button = QPushButton("设置匹配范围")
        self.match_button.clicked.connect(lambda: self._set_active_region("match"))
        side.addWidget(self.match_button)
        self.match_region_label = QLabel("匹配范围：未设置")
        self.match_region_label.setWordWrap(True)
        side.addWidget(self.match_region_label)
        side.addStretch(1)
        body.addWidget(side_panel)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self._accept_regions)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._set_active_region("template")
        self._refresh_region_labels()

    def _set_active_region(self, key: str) -> None:
        self.crop_label.set_active_region(key)
        self.template_button.setObjectName("PrimaryButton" if key == "template" else "")
        self.match_button.setObjectName("PrimaryButton" if key == "match" else "")
        self.template_button.style().unpolish(self.template_button)
        self.template_button.style().polish(self.template_button)
        self.match_button.style().unpolish(self.match_button)
        self.match_button.style().polish(self.match_button)

    def _refresh_region_labels(self) -> None:
        template_region = self.crop_label.normalized_region("template")
        match_region = self.crop_label.normalized_region("match")
        self.template_region_label.setText(f"当前图片位置：{self._region_text(template_region)}")
        self.match_region_label.setText(f"匹配范围：{self._region_text(match_region)}")

    @staticmethod
    def _region_text(region: list[float] | None) -> str:
        if region is None:
            return "未设置"
        return ", ".join(f"{float(value):.3f}" for value in region)

    def _accept_regions(self) -> None:
        template_box = self.crop_label.crop_box("template")
        match_region = self.crop_label.normalized_region("match")
        if template_box is None:
            QMessageBox.warning(self, "未选择当前图片位置", "请先框选当前弹窗图片位置")
            return
        if match_region is None:
            QMessageBox.warning(self, "未选择匹配范围", "请先框选这张弹窗的匹配范围")
            return
        self.cropped_image = self.source_image.crop(template_box)
        self.template_region_norm = self.crop_label.normalized_region("template")
        self.match_region_norm = match_region
        self.accept()


class TemplateManagerWidget(QWidget):
    def __init__(
        self,
        template_root: Path,
        device_ip: str,
        app_name: str,
        overrides: dict[str, Any] | None = None,
        show_offset_controls: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.template_root = template_root
        self.device_ip = device_ip
        self.app_name = app_name
        self.overrides = overrides
        self.show_offset_controls = show_offset_controls
        overrides = overrides if isinstance(overrides, dict) else {}
        self.buttons_dir = template_root / "buttons"
        self.screens_dir = template_root / "screens"
        self.buttons_dir.mkdir(parents=True, exist_ok=True)
        self.screens_dir.mkdir(parents=True, exist_ok=True)
        self.next_message_click_x_offset = float(overrides.get("next_message_click_x_offset", 0.0) or 0.0)
        self.next_message_click_y_offset = float(overrides.get("next_message_click_y_offset", 0.0) or 0.0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        intro_panel = QFrame()
        intro_panel.setObjectName("InsetPanel")
        intro_layout = QVBoxLayout(intro_panel)
        intro_layout.setContentsMargins(14, 12, 14, 12)
        intro_layout.setSpacing(6)
        title = QLabel("模板管理")
        title.setObjectName("SectionLabel")
        intro_layout.addWidget(title)
        subtitle = QLabel("这里只保留弹窗模板采集：从当前机械臂截图，手动框选弹窗区域后保存到当前设备当前 App 的模板目录。")
        subtitle.setObjectName("HelperLabel")
        subtitle.setWordWrap(True)
        intro_layout.addWidget(subtitle)
        location = QLabel(f"目录：{self.template_root}")
        location.setObjectName("HelperLabel")
        intro_layout.addWidget(location)
        layout.addWidget(intro_panel)

        action_panel = QFrame()
        action_panel.setObjectName("InsetPanel")
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(14, 12, 14, 12)
        action_layout.setSpacing(8)
        template_action_rows: list[list[tuple[str, str, str]]] = [
            [
                ("添加弹窗", CUSTOM_POPUP, "弹窗模板"),
                ("添加返回按钮", CUSTOM_BACK_BUTTON, "返回按钮模板"),
                ("添加发送按钮", SEND_BUTTON, "发送按钮模板"),
                ("添加麦克风", MICROPHONE_BUTTON, "麦克风模板"),
                ("添加输入框", INPUT_BOX_BUTTON, "输入框模板"),
            ],
            [
                ("添加下一条", NEXT_MESSAGE_BUTTON, "下一条模板"),
                ("添加新招呼", NEW_GREETING_BUTTON, "新招呼模板"),
                ("添加聊天界面", CHAT_WINDOW_SCREEN, "聊天界面模板"),
                ("添加聊天列表", CHAT_LIST_SCREEN, "聊天列表模板"),
                ("添加我的界面", MY_SCREEN, "我的界面模板"),
            ],
        ]
        for row_index, action_specs in enumerate(template_action_rows):
            template_actions = QHBoxLayout()
            template_actions.setSpacing(8)
            for text, target_key, label in action_specs:
                button = QPushButton(text)
                if row_index == 0 and target_key in {CUSTOM_POPUP, CUSTOM_BACK_BUTTON}:
                    button.setObjectName("PrimaryButton")
                button.clicked.connect(lambda checked=False, key=target_key, title=label: self._capture_template(key, title))
                template_actions.addWidget(button)
            template_actions.addStretch(1)
            action_layout.addLayout(template_actions)
        layout.addWidget(action_panel)

        list_panel = QFrame()
        list_panel.setObjectName("InsetPanel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(14, 12, 14, 12)
        list_layout.setSpacing(8)
        list_title = QLabel("当前纠错模板")
        list_title.setObjectName("SectionLabel")
        list_layout.addWidget(list_title)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类型", "文件", "匹配范围", "操作"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        list_layout.addWidget(self.table)
        layout.addWidget(list_panel)
        self.reload()

    def reload(self) -> None:
        files: list[tuple[str, Path]] = [
            ("弹窗", path)
            for path in sorted(self.buttons_dir.glob("popup_*.jpg"))
            if is_popup_template_name(path.name)
        ]
        files.extend(
            ("返回按钮", path)
            for path in back_button_template_paths({"templates": {"buttons": str(self.buttons_dir)}})
            if is_back_button_template_name(path.name)
        )
        for rule in fixed_template_rules():
            root = self.buttons_dir if rule.group == "buttons" else self.screens_dir
            for path in template_variant_paths(root, rule.filename):
                files.append((rule.label, path))
        seen: set[Path] = set()
        unique_files: list[tuple[str, Path]] = []
        for kind, path in files:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_files.append((kind, path))
        self.table.setRowCount(len(unique_files))
        for row, (kind, path) in enumerate(unique_files):
            self.table.setItem(row, 0, QTableWidgetItem(kind))
            self.table.setItem(row, 1, QTableWidgetItem(path.name))
            self.table.setItem(row, 2, QTableWidgetItem(self._template_region_text(path)))
            delete_button = QPushButton("删除")
            delete_button.clicked.connect(lambda checked=False, p=path: self._delete_template(p))
            self.table.setCellWidget(row, 3, delete_button)

    def _capture_popup_template(self) -> None:
        self._capture_template(CUSTOM_POPUP, "弹窗模板")

    def _capture_back_button_template(self) -> None:
        self._capture_template(CUSTOM_BACK_BUTTON, "返回按钮模板")

    def _capture_template(self, target_key: str, label: str) -> None:
        try:
            robot = RobotInterface(self.device_ip)
            image_bytes = capture_transform_image(robot, pre_move=True, label="template_capture")
            image = ensure_portrait_image(image_bytes)
        except Exception as exc:
            QMessageBox.critical(self, "截图失败", f"无法从机械臂获取截图：{exc}")
            return

        if target_key == CUSTOM_POPUP:
            dialog = PopupDualRegionDialog(image, self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.cropped_image is None:
                return
            match_region = dialog.match_region_norm
        else:
            dialog = PopupCropDialog(image, label, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.cropped_image is None:
                return
            match_region = None

        target = import_target_path(target_key, self.buttons_dir, self.screens_dir)
        try:
            dialog.cropped_image.convert("RGB").save(target, format="JPEG", quality=95)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存{label}：{exc}")
            return
        if target_key == CUSTOM_POPUP:
            self._update_template_match_region(target_key, match_region, target)
        self.reload()
        QMessageBox.information(self, "已保存", f"{label}已保存：{target.name}")

    def _update_template_match_region(self, target_key: str, region: list[float] | None, target: Path | None = None) -> None:
        if region is None:
            return
        if target_key != CUSTOM_POPUP or target is None:
            return
        region_key = popup_template_region_key(target)
        if not isinstance(self.overrides, dict):
            return
        regions = self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
        if not isinstance(regions, dict):
            regions = {}
        regions[region_key] = region
        self.overrides[TEMPLATE_MATCH_REGIONS_CONFIG_KEY] = regions

    def _template_region_text(self, path: Path) -> str:
        if not is_popup_template_name(path.name) or not isinstance(self.overrides, dict):
            return "-"
        regions = self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
        if not isinstance(regions, dict):
            return "未设置"
        region = regions.get(popup_template_region_key(path))
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return "未设置"
        try:
            return ", ".join(f"{float(value):.3f}" for value in region)
        except (TypeError, ValueError):
            return "未设置"

    def _delete_template(self, path: Path) -> None:
        if is_popup_template_name(path.name) and isinstance(self.overrides, dict):
            regions = self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
            if isinstance(regions, dict):
                regions.pop(popup_template_region_key(path), None)
        path.unlink(missing_ok=True)
        self.reload()


class DeviceSettingsDialog(QDialog):
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = copy.deepcopy(overrides if isinstance(overrides, dict) else {})
        self.device_ip = device_ip
        self.app_name = app_name
        self.setWindowTitle(f"设备设置 - {device_ip} / {app_name}")
        self.resize(520, 300)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备设置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("这里保存当前设备当前 App 的设备级补充配置。模板相关偏移会保存到 设备-App 级别。")
        scope.setObjectName("HelperLabel")
        scope.setWordWrap(True)
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        panel = QFrame()
        panel.setObjectName("InsetPanel")
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(14, 12, 14, 12)
        panel_root.setSpacing(8)
        section_title = QLabel("模板点击偏移")
        section_title.setObjectName("SectionLabel")
        panel_root.addWidget(section_title)
        section_help = QLabel("用于模板相关点击坐标的额外偏移。当前主要给欢友“下一条未读”点击使用。")
        section_help.setObjectName("HelperLabel")
        section_help.setWordWrap(True)
        panel_root.addWidget(section_help)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(10)
        panel_root.addLayout(form)

        self.next_message_click_x_offset = NoWheelDoubleSpinBox()
        self.next_message_click_x_offset.setRange(-0.2, 0.2)
        self.next_message_click_x_offset.setDecimals(3)
        self.next_message_click_x_offset.setSingleStep(0.001)
        self.next_message_click_x_offset.setValue(float(self.overrides.get("next_message_click_x_offset", 0.02) or 0.02))
        form.addRow("点击 X 偏移", self.next_message_click_x_offset)

        self.next_message_click_y_offset = NoWheelDoubleSpinBox()
        self.next_message_click_y_offset.setRange(-0.2, 0.2)
        self.next_message_click_y_offset.setDecimals(3)
        self.next_message_click_y_offset.setSingleStep(0.001)
        self.next_message_click_y_offset.setValue(float(self.overrides.get("next_message_click_y_offset", 0.0) or 0.0))
        form.addRow("点击 Y 偏移", self.next_message_click_y_offset)

        root.addWidget(panel)
        root.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> dict[str, Any]:
        return {
            "next_message_click_x_offset": self.next_message_click_x_offset.value(),
            "next_message_click_y_offset": self.next_message_click_y_offset.value(),
        }


class AppConfigDialog(QDialog):
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = copy.deepcopy(overrides if isinstance(overrides, dict) else {})
        self.device_ip = device_ip
        self.app_name = app_name
        self.setWindowTitle(f"设备设置 - {device_ip} / {app_name}")
        self.resize(620, 360)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备设置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("这里保存当前设备当前 App 的补充配置，不同 App 会显示不同内容。")
        scope.setObjectName("HelperLabel")
        scope.setWordWrap(True)
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        body = QFrame()
        body.setObjectName("InsetPanel")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(10)
        root.addWidget(body)

        if self.app_name == "huanyou":
            self._build_huanyou_offset_panel(body_layout)
        else:
            empty_label = QLabel("当前 App 暂无可配置的设备级补充项。")
            empty_label.setObjectName("HelperLabel")
            empty_label.setWordWrap(True)
            body_layout.addWidget(empty_label)

        root.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_huanyou_offset_panel(self, root: QVBoxLayout) -> None:
        title = QLabel("欢友偏移")
        title.setObjectName("SectionLabel")
        root.addWidget(title)
        help_label = QLabel("这里分别保存“新招呼”和“下一条”的点击偏移。")
        help_label.setObjectName("HelperLabel")
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        new_greeting_form = QFormLayout()
        new_greeting_form.setContentsMargins(0, 4, 0, 0)
        new_greeting_form.setSpacing(10)
        root.addLayout(new_greeting_form)
        self.new_greeting_click_x_offset = self._offset_spin(self.overrides.get("new_greeting_click_x_offset", 0.0))
        self.new_greeting_click_y_offset = self._offset_spin(self.overrides.get("new_greeting_click_y_offset", 0.0))
        new_greeting_form.addRow("新招呼 X 偏移", self.new_greeting_click_x_offset)
        new_greeting_form.addRow("新招呼 Y 偏移", self.new_greeting_click_y_offset)

        next_message_title = QLabel("下一条偏移")
        next_message_title.setObjectName("SectionLabel")
        root.addWidget(next_message_title)
        next_message_help = QLabel("用于点击“下一条”按钮时的额外偏移。")
        next_message_help.setObjectName("HelperLabel")
        next_message_help.setWordWrap(True)
        root.addWidget(next_message_help)

        next_message_form = QFormLayout()
        next_message_form.setContentsMargins(0, 4, 0, 0)
        next_message_form.setSpacing(10)
        root.addLayout(next_message_form)
        self.next_message_click_x_offset = self._offset_spin(self.overrides.get("next_message_click_x_offset", 0.0))
        self.next_message_click_y_offset = self._offset_spin(self.overrides.get("next_message_click_y_offset", 0.0))
        next_message_form.addRow("下一条 X 偏移", self.next_message_click_x_offset)
        next_message_form.addRow("下一条 Y 偏移", self.next_message_click_y_offset)

    def _offset_spin(self, value: Any) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-0.2, 0.2)
        spin.setDecimals(3)
        spin.setSingleStep(0.001)
        spin.setValue(float(value or 0.0))
        return spin

    def values(self) -> dict[str, Any]:
        if self.app_name != "huanyou":
            return {}
        return {
            "new_greeting_click_x_offset": self.new_greeting_click_x_offset.value(),
            "new_greeting_click_y_offset": self.new_greeting_click_y_offset.value(),
            "next_message_click_x_offset": self.next_message_click_x_offset.value(),
            "next_message_click_y_offset": self.next_message_click_y_offset.value(),
        }
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        template_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = with_app_defaults(overrides)
        self.device_ip = device_ip
        self.app_name = app_name
        self.template_root = template_root
        self.setWindowTitle(f"App 配置 - {device_ip} / {app_name}")
        self.resize(860, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备配置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("运行设置保存到设备级别；OCR设置和模板管理保存到 App 级别；人设保存到 设备-App 级别；模板偏移请在设备设置里改。")
        scope.setObjectName("HelperLabel")
        scope.setWordWrap(True)
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        self._build_runtime_tab(self.overrides)
        self._build_ocr_tab(self.overrides)
        self._build_template_tab(template_root)
        self._build_persona_tab(self.overrides)
        self.tabs.setTabVisible(self.ocr_tab_index, False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_runtime_tab(self, overrides: dict[str, Any]) -> None:
        page = QWidget()
        page_root = QVBoxLayout(page)
        page_root.setContentsMargins(0, 0, 0, 0)
        page_root.setSpacing(10)

        panel = QFrame()
        panel.setObjectName("InsetPanel")
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(14, 12, 14, 12)
        panel_root.setSpacing(8)
        title = QLabel("运行设置")
        title.setObjectName("SectionLabel")
        panel_root.addWidget(title)
        help_label = QLabel("配置主循环、单步超时和防熄屏跳转。防熄屏会通过列表 OCR 识别底部导航，点击“发现”后再点回“微信”。")
        help_label.setObjectName("HelperLabel")
        help_label.setWordWrap(True)
        panel_root.addWidget(help_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(10)
        panel_root.addLayout(form)

        self.poll_interval = QSpinBox()
        self.poll_interval.setRange(1, 3600)
        self.poll_interval.setValue(int(overrides.get("poll_interval", 1)))
        form.addRow("轮询间隔（秒）", self.poll_interval)

        self.step_timeout = QSpinBox()
        self.step_timeout.setRange(1, 300)
        self.step_timeout.setValue(int(overrides.get("step_timeout", 10)))
        form.addRow("单步超时（秒）", self.step_timeout)

        self.exclude_ocr_wait_from_step_timeout = QCheckBox("OCR等待不计入单步超时")
        self.exclude_ocr_wait_from_step_timeout.setChecked(
            bool(overrides.get("exclude_ocr_wait_from_step_timeout", True))
        )
        form.addRow("", self.exclude_ocr_wait_from_step_timeout)

        self.keep_awake_interval_seconds = QSpinBox()
        self.keep_awake_interval_seconds.setRange(0, 86400)
        self.keep_awake_interval_seconds.setValue(int(overrides.get("keep_awake_interval_seconds", DEFAULT_RUNTIME_CONFIG["keep_awake_interval_seconds"])))
        form.addRow("防熄屏跳转间隔（秒，0关闭）", self.keep_awake_interval_seconds)

        self.ocr_toggle_button = QPushButton("显示OCR设置")
        self.ocr_toggle_button.clicked.connect(self._toggle_ocr_tab)
        panel_root.addWidget(self.ocr_toggle_button, 0, Qt.AlignmentFlag.AlignLeft)

        page_root.addWidget(panel)
        page_root.addStretch(1)
        self.tabs.addTab(page, "运行设置")

    def _build_ocr_tab(self, overrides: dict[str, Any]) -> None:
        page = QWidget()
        page_root = QVBoxLayout(page)
        page_root.setContentsMargins(0, 0, 0, 0)
        page_root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QFrame()
        panel.setObjectName("InsetPanel")
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(14, 12, 14, 12)
        panel_root.setSpacing(8)
        title = QLabel("OCR设置")
        title.setObjectName("SectionLabel")
        panel_root.addWidget(title)
        help_label = QLabel("OCR、模板匹配、联动测试和实际点击统一使用竖屏归一化坐标，当前阶段不再额外转换点击坐标。")
        help_label.setObjectName("HelperLabel")
        help_label.setWordWrap(True)
        panel_root.addWidget(help_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(10)
        panel_root.addLayout(form)

        self.match_threshold = NoWheelDoubleSpinBox()
        self.match_threshold.setRange(0.1, 1.0)
        self.match_threshold.setSingleStep(0.01)
        self.match_threshold.setValue(float(overrides.get("match_threshold", 0.7)))
        self.popup_match_threshold = NoWheelDoubleSpinBox()
        self.popup_match_threshold.setRange(0.1, 1.0)
        self.popup_match_threshold.setSingleStep(0.01)
        self.popup_match_threshold.setValue(float(overrides.get("popup_match_threshold", 0.8)))
        form.addRow("\u5f39\u7a97\u9608\u503c", self.popup_match_threshold)
        form.addRow("模板阈值", self.match_threshold)

        ocr = overrides.get("ocr", {}) if isinstance(overrides.get("ocr", {}), dict) else {}
        self.det_db_thresh = NoWheelDoubleSpinBox()
        self.det_db_thresh.setRange(0.0, 1.0)
        self.det_db_thresh.setSingleStep(0.01)
        self.det_db_thresh.setValue(float(ocr.get("det_db_thresh", 0.3)))
        form.addRow("OCR det_db_thresh", self.det_db_thresh)

        self.rec_thresh = NoWheelDoubleSpinBox()
        self.rec_thresh.setRange(0.0, 1.0)
        self.rec_thresh.setSingleStep(0.01)
        self.rec_thresh.setValue(float(ocr.get("rec_thresh", 0.5)))
        form.addRow("OCR rec_thresh", self.rec_thresh)

        ocr_modes = overrides.get("ocr_modes", {}) if isinstance(overrides.get("ocr_modes", {}), dict) else {}
        dialogue_config = ocr_modes.get("dialogue", {}) if isinstance(ocr_modes.get("dialogue", {}), dict) else {}
        dialogue_rules = (
            dialogue_config.get("rules", {}) if isinstance(dialogue_config.get("rules", {}), dict) else {}
        )

        regions = ocr.get("regions", {}) if isinstance(ocr.get("regions", {}), dict) else {}
        chat_area = regions.get("chat_area", [0.0, 0.1, 1.0, 0.8])
        if not isinstance(chat_area, (list, tuple)) or len(chat_area) != 4:
            chat_area = [0.0, 0.1, 1.0, 0.8]
        chat_left = float(chat_area[0])
        chat_top = float(chat_area[1])
        chat_right = min(1.0, chat_left + float(chat_area[2]))
        chat_bottom = min(1.0, chat_top + float(chat_area[3]))

        region_title = QLabel("对话模式识别范围（竖屏归一化坐标）")
        region_title.setObjectName("SectionLabel")
        panel_root.addWidget(region_title)
        region_help = QLabel("这里控制聊天详情页 OCR 只识别哪块区域：左/上是起点，右/下是终点，范围都是 0 到 1。")
        region_help.setObjectName("HelperLabel")
        region_help.setWordWrap(True)
        panel_root.addWidget(region_help)

        region_form = QFormLayout()
        region_form.setContentsMargins(0, 0, 0, 0)
        region_form.setSpacing(8)
        panel_root.addLayout(region_form)

        self.dialogue_region_left = self._region_spin(chat_left)
        self.dialogue_region_top = self._region_spin(chat_top)
        self.dialogue_region_right = self._region_spin(chat_right)
        self.dialogue_region_bottom = self._region_spin(chat_bottom)
        region_form.addRow("对话模式左边界", self.dialogue_region_left)
        region_form.addRow("对话模式上边界", self.dialogue_region_top)
        region_form.addRow("对话模式右边界", self.dialogue_region_right)
        region_form.addRow("对话模式下边界", self.dialogue_region_bottom)

        keyword_title = QLabel("对话模式过滤词")
        keyword_title.setObjectName("SectionLabel")
        panel_root.addWidget(keyword_title)
        keyword_help = QLabel("每行一个关键词。第一组作用在单个 OCR 文字块，第二组作用在合并后的整条对话。只要包含关键词就会按规则过滤。")
        keyword_help.setObjectName("HelperLabel")
        keyword_help.setWordWrap(True)
        panel_root.addWidget(keyword_help)

        keyword_form = QFormLayout()
        keyword_form.setContentsMargins(0, 0, 0, 0)
        keyword_form.setSpacing(8)
        panel_root.addLayout(keyword_form)
        self.dialogue_ocr_result_excluded_keywords = self._keyword_editor(
            dialogue_rules.get("ocr_result_excluded_keywords", [])
        )
        self.dialogue_line_excluded_keywords = self._keyword_editor(
            dialogue_rules.get("line_excluded_keywords", [])
        )
        keyword_form.addRow("单个 OCR 块命中即丢弃", self.dialogue_ocr_result_excluded_keywords)
        keyword_form.addRow("整条对话命中即丢弃", self.dialogue_line_excluded_keywords)

        self.ocr_mode_fields: dict[str, dict[str, Any]] = {}
        for mode_key, mode_label in [("dialogue", "\u5bf9\u8bdd\u6a21\u5f0f"), ("list", "\u5217\u8868\u6a21\u5f0f"), ("points", "\u79ef\u5206\u6a21\u5f0f")]:
            mode_config = ocr_modes.get(mode_key, {}) if isinstance(ocr_modes.get(mode_key, {}), dict) else {}
            preprocess = mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {}
            rules = mode_config.get("rules", {}) if isinstance(mode_config.get("rules", {}), dict) else {}

            mode_panel = QFrame()
            mode_panel.setObjectName("InsetPanel")
            mode_panel_layout = QFormLayout(mode_panel)
            mode_panel_layout.setContentsMargins(12, 10, 12, 10)
            mode_panel_layout.setSpacing(8)

            preprocess_box = QComboBox()
            preprocess_box.addItems(["original"])
            preprocess_box.setCurrentText("original")
            mode_panel_layout.addRow(f"{mode_label} 预处理", preprocess_box)

            scale_box = NoWheelSpinBox()
            scale_box.setRange(1, 4)
            scale_box.setValue(int(preprocess.get("scale", 2)))
            mode_panel_layout.addRow(f"{mode_label} 放大倍数", scale_box)

            min_confidence = NoWheelDoubleSpinBox()
            min_confidence.setRange(0.0, 1.0)
            min_confidence.setSingleStep(0.01)
            min_confidence.setValue(float(rules.get("min_confidence", 0.3)))
            mode_panel_layout.addRow(f"{mode_label} 最低置信度", min_confidence)

            min_height = NoWheelDoubleSpinBox()
            min_height.setRange(0.0, 0.2)
            min_height.setSingleStep(0.005)
            min_height.setDecimals(3)
            min_height.setValue(float(rules.get("min_box_height_ratio", 0.02)))
            mode_panel_layout.addRow(f"{mode_label} 最小框高比例", min_height)

            self.ocr_mode_fields[mode_key] = {
                "preprocess": preprocess_box,
                "scale": scale_box,
                "min_confidence": min_confidence,
                "min_box_height_ratio": min_height,
            }
            panel_root.addWidget(mode_panel)

        panel_root.addStretch(1)
        scroll.setWidget(panel)
        page_root.addWidget(scroll, 1)
        self.ocr_tab_index = self.tabs.addTab(page, "OCR设置")

    def _region_spin(self, value: float) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setValue(max(0.0, min(1.0, float(value))))
        return spin

    def _keyword_editor(self, keywords: Any) -> QTextEdit:
        editor = QTextEdit()
        editor.setObjectName("KeywordEditor")
        editor.setAcceptRichText(False)
        editor.setMinimumHeight(92)
        editor.setMaximumHeight(130)
        editor.setPlainText("\n".join(self._keyword_lines(keywords)))
        return editor

    @staticmethod
    def _keyword_lines(keywords: Any) -> list[str]:
        if isinstance(keywords, str):
            raw_items = keywords.splitlines()
        elif isinstance(keywords, (list, tuple)):
            raw_items = [str(item) for item in keywords]
        else:
            raw_items = []
        return [item.strip() for item in raw_items if item.strip()]

    @classmethod
    def _keywords_from_editor(cls, editor: QTextEdit) -> list[str]:
        return cls._keyword_lines(editor.toPlainText())

    def _build_template_tab(self, template_root: Path) -> None:
        page = QWidget()
        page_root = QVBoxLayout(page)
        page_root.setContentsMargins(0, 0, 0, 0)
        self.template_manager = TemplateManagerWidget(
            template_root,
            self.device_ip,
            self.app_name,
            self.overrides,
            show_offset_controls=False,
            parent=page,
        )
        page_root.addWidget(self.template_manager)
        self.tabs.addTab(page, "模板管理")

    def _build_persona_tab(self, overrides: dict[str, Any]) -> None:
        page = QWidget()
        page_root = QVBoxLayout(page)
        page_root.setContentsMargins(0, 0, 0, 0)
        page_root.setSpacing(10)

        panel = QFrame()
        panel.setObjectName("InsetPanel")
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(14, 12, 14, 12)
        panel_root.setSpacing(8)
        title = QLabel("人设参数")
        title.setObjectName("SectionLabel")
        panel_root.addWidget(title)
        help_label = QLabel("AI 回复时会优先使用这里的人设，覆盖全局默认值。")
        help_label.setObjectName("HelperLabel")
        panel_root.addWidget(help_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(10)
        persona = overrides.get("persona", {}) if isinstance(overrides.get("persona", {}), dict) else {}
        self.persona_inputs: dict[str, QLineEdit] = {}
        persona_selector, self.request_persona_combo, self.request_persona_custom_input = _create_request_persona_selector(
            str(persona.get("requestPersona", ""))
        )
        form.addRow("人设授权 ID", persona_selector)
        for key, label in [
            ("name", "名字"),
            ("age", "年龄"),
            ("work", "职业"),
            ("location", "住址"),
            ("renshe", "自定义人设"),
        ]:
            line = QLineEdit(str(persona.get(key, "")))
            self.persona_inputs[key] = line
            form.addRow(label, line)
        panel_root.addLayout(form)
        page_root.addWidget(panel)
        page_root.addStretch(1)
        self.tabs.addTab(page, "人设")

    def _toggle_ocr_tab(self) -> None:
        is_visible = self.tabs.isTabVisible(self.ocr_tab_index)
        self.tabs.setTabVisible(self.ocr_tab_index, not is_visible)
        self.ocr_toggle_button.setText("隐藏OCR设置" if not is_visible else "显示OCR设置")
        if not is_visible:
            self.tabs.setCurrentIndex(self.ocr_tab_index)

    def values(self) -> dict[str, Any]:
        values = dict(self.overrides)
        values.pop("new_greeting_click_x_offset", None)
        values.pop("new_greeting_click_y_offset", None)
        values.pop("next_message_click_x_offset", None)
        values.pop("next_message_click_y_offset", None)
        ocr = dict(values.get("ocr", {}) if isinstance(values.get("ocr", {}), dict) else {})
        ocr["det_db_thresh"] = self.det_db_thresh.value()
        ocr["rec_thresh"] = self.rec_thresh.value()
        regions = dict(ocr.get("regions", {}) if isinstance(ocr.get("regions", {}), dict) else {})
        chat_left = self.dialogue_region_left.value()
        chat_top = self.dialogue_region_top.value()
        chat_right = max(chat_left, self.dialogue_region_right.value())
        chat_bottom = max(chat_top, self.dialogue_region_bottom.value())
        regions["chat_area"] = [
            round(chat_left, 3),
            round(chat_top, 3),
            round(chat_right - chat_left, 3),
            round(chat_bottom - chat_top, 3),
        ]
        ocr["regions"] = regions
        ocr_modes = dict(values.get("ocr_modes", {}) if isinstance(values.get("ocr_modes", {}), dict) else {})
        for mode_key, fields in self.ocr_mode_fields.items():
            mode_config = dict(ocr_modes.get(mode_key, {}) if isinstance(ocr_modes.get(mode_key, {}), dict) else {})
            preprocess = dict(mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {})
            rules = dict(mode_config.get("rules", {}) if isinstance(mode_config.get("rules", {}), dict) else {})
            preprocess["mode"] = "original"
            preprocess["scale"] = fields["scale"].value()
            preprocess.pop("auxiliary_mode", None)
            rules["min_confidence"] = fields["min_confidence"].value()
            rules["min_box_height_ratio"] = fields["min_box_height_ratio"].value()
            if mode_key == "dialogue":
                rules["ocr_result_excluded_keywords"] = self._keywords_from_editor(
                    self.dialogue_ocr_result_excluded_keywords
                )
                rules["line_excluded_keywords"] = self._keywords_from_editor(
                    self.dialogue_line_excluded_keywords
                )
            mode_config["preprocess"] = preprocess
            mode_config["rules"] = rules
            ocr_modes[mode_key] = mode_config

        values.update({
            "poll_interval": self.poll_interval.value(),
            "step_timeout": self.step_timeout.value(),
            "keep_awake_interval_seconds": self.keep_awake_interval_seconds.value(),
            "exclude_ocr_wait_from_step_timeout": self.exclude_ocr_wait_from_step_timeout.isChecked(),
            "match_threshold": self.match_threshold.value(),
            "popup_match_threshold": self.popup_match_threshold.value(),
            "ocr": ocr,
            "ocr_modes": ocr_modes,
            "templates": {
                "buttons": str(self.template_manager.buttons_dir),
                "screens": str(self.template_manager.screens_dir),
            },
            "persona": {
                "requestPersona": _request_persona_value(self.request_persona_combo, self.request_persona_custom_input),
                **{key: field.text().strip() for key, field in self.persona_inputs.items()},
            },
        })
        return values


class DeviceSettingsDialogFixed(QDialog):
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = copy.deepcopy(overrides if isinstance(overrides, dict) else {})
        self.device_ip = device_ip
        self.app_name = app_name
        self.setWindowTitle(f"设备设置 - {device_ip} / {app_name}")
        self.resize(620, 360)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备设置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("这里保存当前设备当前 App 的补充配置，不同 App 会显示不同内容。")
        scope.setObjectName("HelperLabel")
        scope.setWordWrap(True)
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        panel = QFrame()
        panel.setObjectName("InsetPanel")
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(14, 12, 14, 12)
        panel_root.setSpacing(10)
        root.addWidget(panel)

        if self.app_name == "huanyou":
            self._build_huanyou_offset_panel(panel_root)
        else:
            empty_label = QLabel("当前 App 暂无可配置的设备级补充项。")
            empty_label.setObjectName("HelperLabel")
            empty_label.setWordWrap(True)
            panel_root.addWidget(empty_label)

        root.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _offset_spin(self, value: Any) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-0.2, 0.2)
        spin.setDecimals(3)
        spin.setSingleStep(0.001)
        spin.setValue(float(value or 0.0))
        return spin

    def _build_huanyou_offset_panel(self, root: QVBoxLayout) -> None:
        new_greeting_title = QLabel("欢友：新招呼偏移")
        new_greeting_title.setObjectName("SectionLabel")
        root.addWidget(new_greeting_title)
        new_greeting_help = QLabel("用于点击“新招呼”按钮时的额外偏移。")
        new_greeting_help.setObjectName("HelperLabel")
        new_greeting_help.setWordWrap(True)
        root.addWidget(new_greeting_help)

        new_greeting_form = QFormLayout()
        new_greeting_form.setContentsMargins(0, 4, 0, 0)
        new_greeting_form.setSpacing(10)
        self.new_greeting_click_x_offset = self._offset_spin(self.overrides.get("new_greeting_click_x_offset", 0.0))
        self.new_greeting_click_y_offset = self._offset_spin(self.overrides.get("new_greeting_click_y_offset", 0.0))
        new_greeting_form.addRow("新招呼 X 偏移", self.new_greeting_click_x_offset)
        new_greeting_form.addRow("新招呼 Y 偏移", self.new_greeting_click_y_offset)
        root.addLayout(new_greeting_form)

        next_message_title = QLabel("欢友：下一条偏移")
        next_message_title.setObjectName("SectionLabel")
        root.addWidget(next_message_title)
        next_message_help = QLabel("用于点击“下一条”按钮时的额外偏移。")
        next_message_help.setObjectName("HelperLabel")
        next_message_help.setWordWrap(True)
        root.addWidget(next_message_help)

        next_message_form = QFormLayout()
        next_message_form.setContentsMargins(0, 4, 0, 0)
        next_message_form.setSpacing(10)
        self.next_message_click_x_offset = self._offset_spin(self.overrides.get("next_message_click_x_offset", 0.0))
        self.next_message_click_y_offset = self._offset_spin(self.overrides.get("next_message_click_y_offset", 0.0))
        next_message_form.addRow("下一条 X 偏移", self.next_message_click_x_offset)
        next_message_form.addRow("下一条 Y 偏移", self.next_message_click_y_offset)
        root.addLayout(next_message_form)

    def values(self) -> dict[str, Any]:
        if self.app_name != "huanyou":
            return {}
        return {
            "new_greeting_click_x_offset": self.new_greeting_click_x_offset.value(),
            "new_greeting_click_y_offset": self.new_greeting_click_y_offset.value(),
            "next_message_click_x_offset": self.next_message_click_x_offset.value(),
            "next_message_click_y_offset": self.next_message_click_y_offset.value(),
        }


class DeviceSettingsDialogInteractive(QDialog):
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = copy.deepcopy(overrides if isinstance(overrides, dict) else {})
        self.device_ip = device_ip
        self.app_name = app_name
        self.selected_point: tuple[float, float] | None = None
        self.current_screenshot: Image.Image | None = None
        self.offset_spin_boxes: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        self.test_target_combo: QComboBox | None = None
        self.setWindowTitle(f"设备设置 - {device_ip} / {app_name}")
        self.resize(1180, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header_panel = QFrame()
        header_panel.setObjectName("DialogPanel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"设备设置 / {device_ip} / {app_name}")
        title.setObjectName("DialogTitle")
        header_layout.addWidget(title)
        scope = QLabel("这里保存当前设备当前 App 的设备级补充配置。截图取点、偏移设置和测试点击都在这里完成，保存级别为设备-App。")
        scope.setObjectName("HelperLabel")
        scope.setWordWrap(True)
        header_layout.addWidget(scope)
        root.addWidget(header_panel)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        root.addLayout(content, 1)

        screenshot_panel = QFrame()
        screenshot_panel.setObjectName("InsetPanel")
        screenshot_layout = QVBoxLayout(screenshot_panel)
        screenshot_layout.setContentsMargins(14, 12, 14, 12)
        screenshot_layout.setSpacing(8)
        screenshot_title = QLabel("当前设备截图")
        screenshot_title.setObjectName("SectionLabel")
        screenshot_layout.addWidget(screenshot_title)
        screenshot_help = QLabel("点击“刷新截图”后，可在图片上点一个基点坐标。右侧会自动叠加当前偏移，方便直接测试点击。")
        screenshot_help.setObjectName("HelperLabel")
        screenshot_help.setWordWrap(True)
        screenshot_layout.addWidget(screenshot_help)
        self.refresh_screenshot_button = QPushButton("刷新截图")
        self.refresh_screenshot_button.setObjectName("PrimaryButton")
        self.refresh_screenshot_button.clicked.connect(self._refresh_screenshot)
        self.save_error_screenshot_button = QPushButton("保存当前图片到报错路径")
        self.save_error_screenshot_button.clicked.connect(self._save_current_screenshot_to_error_path)
        screenshot_button_row = QHBoxLayout()
        screenshot_button_row.setContentsMargins(0, 0, 0, 0)
        screenshot_button_row.setSpacing(8)
        screenshot_button_row.addWidget(self.refresh_screenshot_button)
        screenshot_button_row.addWidget(self.save_error_screenshot_button)
        screenshot_button_row.addStretch(1)
        screenshot_layout.addLayout(screenshot_button_row)

        placeholder = QPixmap(720, 1280)
        placeholder.fill(QColor("#f8fafc"))
        self.crop_label = PopupCropLabel(placeholder)
        self.crop_label.on_point_selected = self._update_selected_point

        self.screenshot_scroll = QScrollArea()
        self.screenshot_scroll.setWidgetResizable(False)
        self.screenshot_scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.screenshot_scroll.setWidget(self.crop_label)
        screenshot_layout.addWidget(self.screenshot_scroll, 1)
        content.addWidget(screenshot_panel, 1)

        control_panel = QFrame()
        control_panel.setObjectName("InsetPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(14, 12, 14, 12)
        control_layout.setSpacing(10)
        content.addWidget(control_panel)

        control_title = QLabel("偏移与测试")
        control_title.setObjectName("SectionLabel")
        control_layout.addWidget(control_title)
        control_help = QLabel("你可以先设置偏移，再选图上的基点直接测试点击。后续新增动态字段时，也按当前 App 的设备级字段继续扩展。")
        control_help.setObjectName("HelperLabel")
        control_help.setWordWrap(True)
        control_layout.addWidget(control_help)

        field_specs = self._field_specs()
        if field_specs:
            target_form = QFormLayout()
            target_form.setContentsMargins(0, 0, 0, 0)
            target_form.setSpacing(8)
            self.test_target_combo = QComboBox()
            for field_key, field_label in field_specs:
                self.test_target_combo.addItem(field_label, field_key)
            self.test_target_combo.currentIndexChanged.connect(self._update_click_labels)
            target_form.addRow("测试字段", self.test_target_combo)
            control_layout.addLayout(target_form)
        else:
            no_fields_label = QLabel("当前 App 暂无设备级偏移字段。你仍然可以截图取点并测试原始点击坐标。")
            no_fields_label.setObjectName("HelperLabel")
            no_fields_label.setWordWrap(True)
            control_layout.addWidget(no_fields_label)

        coord_form = QFormLayout()
        coord_form.setContentsMargins(0, 0, 0, 0)
        coord_form.setSpacing(8)
        self.point_x_label = QLabel("--")
        self.point_y_label = QLabel("--")
        self.click_x_label = QLabel("--")
        self.click_y_label = QLabel("--")
        coord_form.addRow("基点 X", self.point_x_label)
        coord_form.addRow("基点 Y", self.point_y_label)
        coord_form.addRow("点击 X", self.click_x_label)
        coord_form.addRow("点击 Y", self.click_y_label)
        control_layout.addLayout(coord_form)

        for field_key, field_label in field_specs:
            section_title = QLabel(f"{field_label} 偏移")
            section_title.setObjectName("SectionLabel")
            control_layout.addWidget(section_title)
            section_help = QLabel(f"这里设置“{field_label}”相关内容的点击偏移，保存后立即写入当前设备当前 App。")
            section_help.setObjectName("HelperLabel")
            section_help.setWordWrap(True)
            control_layout.addWidget(section_help)

            section_form = QFormLayout()
            section_form.setContentsMargins(0, 0, 0, 0)
            section_form.setSpacing(8)
            x_spin = self._offset_spin(self.overrides.get(f"{field_key}_click_x_offset", 0.0))
            y_spin = self._offset_spin(self.overrides.get(f"{field_key}_click_y_offset", 0.0))
            x_spin.valueChanged.connect(self._update_click_labels)
            y_spin.valueChanged.connect(self._update_click_labels)
            section_form.addRow("X 偏移", x_spin)
            section_form.addRow("Y 偏移", y_spin)
            control_layout.addLayout(section_form)
            self.offset_spin_boxes[field_key] = (x_spin, y_spin)

            if field_key == "new_greeting":
                self.new_greeting_click_x_offset = x_spin
                self.new_greeting_click_y_offset = y_spin
            elif field_key == "next_message":
                self.next_message_click_x_offset = x_spin
                self.next_message_click_y_offset = y_spin

        self.test_click_button = QPushButton("测试点击机械臂")
        self.test_click_button.clicked.connect(self._test_click_point)
        control_layout.addWidget(self.test_click_button, 0, Qt.AlignmentFlag.AlignLeft)
        control_layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button:
            ok_button.setText("保存")
            ok_button.setObjectName("PrimaryButton")
        if cancel_button:
            cancel_button.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _field_specs(self) -> list[tuple[str, str]]:
        if self.app_name == "huanyou":
            return [
                ("new_greeting", "新招呼"),
                ("next_message", "下一条"),
            ]
        return []

    def _offset_spin(self, value: Any) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-0.2, 0.2)
        spin.setDecimals(3)
        spin.setSingleStep(0.001)
        spin.setValue(float(value or 0.0))
        return spin

    def _refresh_screenshot(self) -> None:
        self.refresh_screenshot_button.setEnabled(False)
        original_text = self.refresh_screenshot_button.text()
        self.refresh_screenshot_button.setText("刷新中...")
        try:
            image_bytes = capture_transform_image(
                RobotInterface(self.device_ip),
                {"_runtime": {"device_ip": self.device_ip, "app_name": self.app_name}},
                pre_move=True,
                label="device_settings",
            )
            image = ensure_portrait_image(image_bytes)
            self._set_screenshot(image)
        except Exception as exc:
            QMessageBox.critical(self, "截图失败", f"无法从当前设备获取截图：{exc}")
            return
        finally:
            self.refresh_screenshot_button.setEnabled(True)
            self.refresh_screenshot_button.setText(original_text)

    def _set_screenshot(self, image: Image.Image) -> None:
        self.current_screenshot = image.copy()
        self.crop_label.set_pixmap(_pixmap_from_pil_image(image))
        self.selected_point = None
        self.point_x_label.setText("--")
        self.point_y_label.setText("--")
        self._update_click_labels()

    def _save_current_screenshot_to_error_path(self) -> None:
        if self.current_screenshot is None:
            QMessageBox.warning(self, "未刷新截图", "请先刷新截图，再保存当前图片。")
            return
        error_dir = Path("photo") / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        safe_ip = re.sub(r"[^0-9A-Za-z_.-]+", "_", self.device_ip).replace(".", "_")
        safe_app = re.sub(r"[^0-9A-Za-z_.-]+", "_", self.app_name)
        file_name = f"manual_{safe_ip}_{safe_app}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        target = error_dir / file_name
        image = self.current_screenshot
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        try:
            image.save(target, format="JPEG", quality=95)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存当前图片：{exc}")
            return
        QMessageBox.information(self, "保存完成", f"已保存到：{target}")

    def _selected_field_key(self) -> str | None:
        if self.test_target_combo is not None and self.test_target_combo.count() > 0:
            current = self.test_target_combo.currentData()
            return str(current) if current is not None else None
        if self.offset_spin_boxes:
            return next(iter(self.offset_spin_boxes))
        return None

    def _field_offsets(self, field_key: str | None) -> tuple[float, float]:
        if field_key is None:
            return 0.0, 0.0
        spins = self.offset_spin_boxes.get(field_key)
        if spins is None:
            return 0.0, 0.0
        return spins[0].value(), spins[1].value()

    def _apply_offset(self, field_key: str | None, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        offset_x, offset_y = self._field_offsets(field_key)
        if field_key == "new_greeting":
            return (
                round(min(0.98, max(0.02, x + offset_x)), 4),
                round(min(0.98, max(0.02, y + offset_y)), 4),
            )
        if field_key == "next_message":
            return (
                round(min(0.98, x + offset_x), 4),
                round(min(0.98, max(0.02, y + offset_y)), 4),
            )
        return round(x, 4), round(y, 4)

    def _update_selected_point(self, x: float, y: float) -> None:
        self.selected_point = (float(x), float(y))
        self.point_x_label.setText(f"{x:.4f}")
        self.point_y_label.setText(f"{y:.4f}")
        self._update_click_labels()

    def _current_click_point(self) -> tuple[float, float] | None:
        if self.selected_point is None:
            return None
        return self._apply_offset(self._selected_field_key(), self.selected_point)

    def _update_click_labels(self, *_args: Any) -> None:
        point = self._current_click_point()
        if point is None:
            self.click_x_label.setText("--")
            self.click_y_label.setText("--")
            return
        self.click_x_label.setText(f"{point[0]:.4f}")
        self.click_y_label.setText(f"{point[1]:.4f}")

    def _test_click_point(self) -> None:
        point = self._current_click_point()
        if point is None:
            QMessageBox.warning(self, "未选择坐标", "请先刷新截图，并在图片上点击一个基点位置。")
            return
        try:
            robot = RobotInterface(self.device_ip)
            try:
                setattr(robot, "_codex_click_jitter_enabled", True)
            except Exception:
                pass
            click_x, click_y = robot_point_with_jitter(robot, *point)
            robot.move_click(click_x, click_y)
        except Exception as exc:
            QMessageBox.critical(self, "测试点击失败", f"机械臂点击失败：{exc}")
            return
        QMessageBox.information(self, "测试点击完成", f"已点击坐标：({click_x:.4f}, {click_y:.4f})")

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field_key, (x_spin, y_spin) in self.offset_spin_boxes.items():
            values[f"{field_key}_click_x_offset"] = x_spin.value()
            values[f"{field_key}_click_y_offset"] = y_spin.value()
        return values


DeviceSettingsDialog = DeviceSettingsDialogInteractive


def run_app(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: Any) -> None:
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    window = MainWindow()
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        app.quit()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
