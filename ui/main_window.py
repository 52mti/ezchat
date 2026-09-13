from __future__ import annotations

import copy
import html
import importlib
import ipaddress
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
import zipfile
from collections import deque
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import requests
from PyQt6.QtCore import QPoint, QRect, Qt, QLockFile, QTimer, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap, QTextCursor
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
    QPlainTextEdit,
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
from PIL import Image, ImageDraw

from apps import get_default_app_name, get_registered_apps, resolve_app_name
from services.defaults import DEFAULT_RUNTIME_CONFIG, with_app_defaults
from apps.coordinates import robot_point_with_jitter
from services.photo_capture import capture_transform_image, ensure_photo_root
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
    match_template_with_region,
    match_template_candidates_with_region,
    next_template_variant_path,
    TemplateMatchRegionMissingError,
    popup_template_paths,
    template_match_region,
    popup_template_region_key,
    template_variant_paths,
    template_path,
)
from apps import run_ocr_mode_for
from vision import get_ocr_queue_status, match
from lgb.lgb import RobotInterface, scan_devices
from services import (
    ChatHistoryStore,
    ConfigManager,
    LicenseClient,
    LocalTtsServer,
    ReleaseDecision,
    ReleaseManifestError,
    ResolvedUpdatePackage,
    UpdateSourceError,
    resolve_lanzou_update_package,
    resolve_release_decision,
)
from services.persona import resolve_persona
from services.coordinate_cache import load_latest_into_runtime
from services.vision_usage import reset as reset_vision_usage
from ui.chat_history_window import ChatHistoryDialog
from vision.screen import ensure_portrait_image
from workers import DeviceWorker, WorkerEvent
from workers.device_worker import is_stale_worker_event



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


class DeviceScreenshotThread(QThread):
    """在后台获取设备截图，避免机械臂网络请求阻塞 Qt 主线程。"""

    image_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, device_ip: str, app_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device_ip = str(device_ip)
        self.app_name = str(app_name)

    def run(self) -> None:
        try:
            image_bytes = capture_transform_image(
                RobotInterface(self.device_ip),
                {"_runtime": {"device_ip": self.device_ip, "app_name": self.app_name}},
                pre_move=True,
                label="device_settings",
            )
            self.image_ready.emit(ensure_portrait_image(image_bytes))
        except Exception as exc:
            self.failed.emit(str(exc))


class VisionCalibrationThread(QThread):
    """后台执行一次视觉+OCR 标定；截图与配置由调用方预先准备好。"""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        app_config: dict[str, Any],
        image: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.app_config = app_config
        self.image = image

    def run(self) -> None:
        try:
            from apps.xinyu import steps as xinyu_steps

            device_ip = str(self.app_config.get("_runtime", {}).get("device_ip", "") or "")
            result = xinyu_steps.calibrate_dialogue_chat_area(
                RobotInterface(device_ip),
                self.app_config,
                image=self.image,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class VisionCalibrationDialog(QDialog):
    """实时标定窗口：先显示当前截图与“标定中...”，成功后把学到的范围画上去。"""

    def __init__(self, image: Image.Image, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("视觉+OCR标定")
        self.resize(560, 880)
        self._canvas = image.convert("RGB").copy()
        self._annotated = False
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        self.image_label = QLabel()
        self._refresh_pixmap()
        root.addWidget(self.image_label)

        self.status_label = QLabel("标定中...（视觉+OCR 识别约需十几秒，请稍候）")
        status_font = self.status_label.font()
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #b36b00;")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        root.addWidget(self.detail_label)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        root.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight)

    def _refresh_pixmap(self) -> None:
        pixmap = _pixmap_from_pil_image(self._canvas)
        self.image_label.setPixmap(pixmap.scaledToHeight(680, Qt.TransformationMode.SmoothTransformation))

    def set_result(self, result: dict[str, Any]) -> None:
        chat_area = result.get("chat_area_dynamic")
        dialogue_message = result.get("dialogue_message")
        lines: list[str] = []
        drew_any = False
        draw = ImageDraw.Draw(self._canvas)
        width, height = self._canvas.size
        for region, color, label in (
            (chat_area, (232, 32, 32), "红框 聊天识别区域 chat_area_dynamic"),
            (dialogue_message, (36, 96, 232), "蓝框 消息OCR条带 dialogue_message"),
        ):
            if isinstance(region, (list, tuple)) and len(region) >= 4:
                try:
                    x, y, w, h = (float(value) for value in region[:4])
                except (TypeError, ValueError):
                    lines.append(f"{label}：数值无效")
                    continue
                draw.rectangle(
                    [x * width, y * height, (x + w) * width, (y + h) * height],
                    outline=color,
                    width=3,
                )
                drew_any = True
                lines.append(f"{label}：[{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]")
            else:
                lines.append(f"{label}：未学到")
        self.detail_label.setText("\n".join(lines))
        if result.get("ok") and drew_any:
            self.status_label.setText("标定完成，聊天区域已写入坐标缓存")
            self.status_label.setStyleSheet("color: #1d7a2e;")
        else:
            reason = str(result.get("reason") or "未知原因")
            self.status_label.setText(f"标定未生效：{reason}")
            self.status_label.setStyleSheet("color: #c22f2f;")
        if drew_any:
            self._refresh_pixmap()

    def set_error(self, message: str) -> None:
        self.status_label.setText(f"标定失败：{message}")
        self.status_label.setStyleSheet("color: #c22f2f;")


class _PersonaNetworkThread(QThread):
    """人设 tab 的 DeepSeek 网络调用（连通性测试 / 模型列表）后台执行。

    result_ready 携带 list[str]（成功）或 str（错误信息）；key 为空时回退 .env。
    """

    result_ready = pyqtSignal(object)

    def __init__(self, api_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.api_key = api_key.strip()

    def run(self) -> None:
        from services.deepseek_client import DeepSeekClient

        try:
            client = DeepSeekClient(api_key=self.api_key) if self.api_key else DeepSeekClient()
            models = client.list_models()
        except Exception as exc:  # noqa: BLE001 - UI 需要把任何失败变成可读文案
            self.result_ready.emit(str(exc))
            return
        self.result_ready.emit([str(model) for model in models])


class TemplatePreviewThread(QThread):
    """模板匹配预览在后台执行，避免多模板检测卡住设置窗口。"""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image: Image.Image, callback: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = image.copy()
        self.callback = callback

    def run(self) -> None:
        try:
            self.completed.emit(self.callback(self.image))
        except Exception as exc:
            self.failed.emit(str(exc))


class OcrPreviewThread(QThread):
    """在后台运行设备设置中的 OCR 测试，避免识别阻塞设置窗口。"""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image: Image.Image, config: dict[str, Any], mode: str, region: Any, app_name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = image.copy()
        self.config = copy.deepcopy(config)
        self.mode = mode
        self.region = copy.deepcopy(region)
        self.engine = run_ocr_mode_for(app_name)

    def run(self) -> None:
        try:
            self.completed.emit(self.engine(self.image, self.config, self.mode, region=self.region))
        except Exception as exc:
            self.failed.emit(str(exc))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
QTextEdit, QPlainTextEdit {
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

DEVICE_HEALTH_STYLES = {
    "unknown": ("待检测", "#f1f5f9", "#475569"),
    "normal": ("在线 · 画面正常", "#dcfce7", "#166534"),
    "too_dark": ("在线 · 画面过暗", "#fef3c7", "#92400e"),
    "overexposed": ("在线 · 画面过曝", "#fef3c7", "#92400e"),
    "black_screen": ("在线 · 黑屏", "#fee2e2", "#991b1b"),
    "capture_error": ("截图异常", "#fee2e2", "#991b1b"),
}


COL_SELECT = 0
COL_LICENSE = 1
COL_EXPIRES = 2
COL_DEVICE_IP = 3
COL_STATUS = 4
COL_DEVICE_HEALTH = 5
COL_AVG_REPLY_SECONDS = 6
COL_UNATTENDED_SECONDS = 7
COL_RUNNING_APP = 8
COL_BALANCE = 9
STATUS_COLUMN_WIDTH = 150
DEVICE_HEALTH_COLUMN_WIDTH = 124
UNATTENDED_COLUMN_WIDTH = 112
BALANCE_COLUMN_WIDTH = 86

STATUS_COORDINATE_RE = re.compile(r"\s*[：:,-]?\s*\(?\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*\)?\s*$")
UPDATE_DOWNLOAD_CHUNK_SIZE = 1024 * 256
LOG_EVENT_LIMIT = 5000
LOG_RENDER_LIMIT = 1000
RENDERED_LOG_BLOCK_LIMIT = 1200
UI_EVENT_DRAIN_LIMIT = 120
UI_EVENT_DRAIN_SECONDS = 0.02
UPDATE_EVENT_DRAIN_LIMIT = 50
UNATTENDED_REFRESH_SECONDS = 1.0


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
        self.monitor_wall_window = None
        self._device_scan_running = False
        self._vision_calibration_thread: VisionCalibrationThread | None = None
        self._vision_calibration_dialog: VisionCalibrationDialog | None = None
        self.log_events: "deque[tuple[str, str]]" = deque(maxlen=LOG_EVENT_LIMIT)
        self._last_unattended_refresh_at = 0.0
        self.unattended_started_at: dict[str, float] = {}
        self.unattended_elapsed_seconds: dict[str, float] = {}
        self._update_check_in_progress = False
        self._download_cancel_event: threading.Event | None = None
        self._download_progress_dialog: QProgressDialog | None = None
        self._pending_update_decision: ReleaseDecision | None = None
        self._pending_update_package: ResolvedUpdatePackage | None = None
        self.local_tts_server = LocalTtsServer()
        self._tts_bind_lock = threading.Lock()
        self._tts_bind_in_progress: set[str] = set()
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
        top_bar.addWidget(QLabel("OCR并发"))
        self.ocr_concurrency_input = NoWheelSpinBox()
        self.ocr_concurrency_input.setRange(1, 16)
        self.ocr_concurrency_input.setFixedWidth(72)
        self.ocr_concurrency_input.setValue(int(self.config_manager.get("global", "ocr_max_concurrency", default=1) or 1))
        self.ocr_concurrency_input.valueChanged.connect(self._save_ocr_concurrency)
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
        self.ocr_queue_label = QLabel("OCR等待：0")
        self.ocr_queue_label.setObjectName("OcrQueueLabel")
        self.ocr_queue_label.setToolTip("当前等待 OCR 并发槽位的任务数量")
        top_bar.addWidget(self.ocr_queue_label)
        top_bar.addStretch(1)
        self.monitor_wall_button = self._button("打开监控墙")
        self.monitor_wall_button.setToolTip("实时查看每台设备 worker 最近一次截图请求的画面")
        self.monitor_wall_button.clicked.connect(self._open_monitor_wall)
        top_bar.addWidget(self.monitor_wall_button)
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

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["\u9009\u62e9", "\u5f53\u524d\u5361\u5bc6", "\u5230\u671f\u65f6\u95f4", "\u8bbe\u5907IP", "\u8fd0\u884c\u72b6\u6001", "\u8bbe\u5907\u72b6\u6001", "\u5e73\u5747\u7528\u65f6", "\u65e0\u5e72\u9884\u65f6\u957f", "\u8fd0\u884cApp", "\u4f59\u989d"])
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
        header.setSectionResizeMode(COL_DEVICE_HEALTH, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_DEVICE_HEALTH, DEVICE_HEALTH_COLUMN_WIDTH)
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
        self.batch_stop_button = self._button("停止选中", "danger")
        self.batch_stop_button.clicked.connect(lambda: self._run_for_checked_rows(self._stop_worker))
        action_bar.addWidget(self.batch_stop_button)
        action_bar.addStretch(1)
        layout.addWidget(action_panel)

        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(RENDERED_LOG_BLOCK_LIMIT)
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
        # 开发目录同样会有运行时清单，不能仅靠文件存在与否判断为客户安装包。
        if os.environ.get("JIXIEBI_PACKAGED_LAUNCH", "").strip() != "1":
            return False
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
                resolved = None
                if decision.update_url:
                    try:
                        response = http.head(decision.update_url, timeout=30, allow_redirects=True)
                        response.raise_for_status()
                        resolved = SimpleNamespace(
                            file_name=decision.update_file_name or Path(decision.update_url).name,
                            size=response.headers.get("Content-Length", ""),
                            download_url=decision.update_url,
                            detail_url=decision.update_url,
                        )
                    except (requests.RequestException, OSError):
                        resolved = None
                if resolved is None:
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
                if not zipfile.is_zipfile(target_path):
                    raise UpdateSourceError("下载到的更新包不是有效的 zip 文件。")
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
        updater = root / ("update_apply.pyc" if os.name == "nt" else "update_apply.py")
        if not updater.is_file():
            QMessageBox.critical(self, "更新失败", f"缺少更新脚本：{updater}")
            return
        command = self._update_apply_command(root=root, updater=updater, archive_path=archive_path)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen(command, cwd=root, creationflags=creationflags)
        except OSError as exc:
            QMessageBox.critical(self, "更新失败", f"无法启动更新程序：{exc}")
            return
        # quit() 不会触发 closeEvent；TTS 子进程必须在这里显式停掉，
        # 否则它会在端口上遗留旧版本孤儿进程，被更新后的新会话一直复用。
        self.local_tts_server.stop()
        QApplication.instance().quit()

    def _update_apply_command(self, *, root: Path, updater: Path, archive_path: str) -> list[str]:
        if os.name != "nt":
            return [
                sys.executable,
                "-s",
                str(updater),
                "--package-root",
                str(root),
                "--archive",
                str(archive_path),
                "--pid",
                str(os.getpid()),
            ]

        script_path = self._write_windows_update_script()
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-PackageRoot",
            str(root),
            "-ArchivePath",
            str(archive_path),
            "-PidToWait",
            str(os.getpid()),
        ]

    def _write_windows_update_script(self) -> Path:
        # 编排脚本唯一来源是 update_orchestration 模块；禁止在此内嵌第二份拷贝。
        from update_orchestration import write_orchestration_script

        script_dir = Path(tempfile.gettempdir()) / "jxb_update_apply"
        return write_orchestration_script(script_dir)


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
            self.table.setItem(row, COL_DEVICE_HEALTH, self._device_health_item("unknown"))
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
        menu.addAction("停止", lambda: self._stop_worker(row))
        menu.addSeparator()
        menu.addAction("卡密绑定", lambda: self._bind_license(row))
        menu.addAction("app设置", lambda: self._edit_app_config(row))
        menu.addAction("设备设置", lambda: self._edit_device_settings(row))
        menu.addAction("查看聊天记录", lambda: self._show_chat_history(row))
        menu.addAction("打开设备后台", lambda: self._open_device_backend(row))
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

    def _open_device_backend(self, row: int) -> None:
        device_ip = self._device_ip_for_row(row).strip()
        if not device_ip:
            return
        try:
            normalized_ip = str(ipaddress.ip_address(device_ip))
        except ValueError:
            QMessageBox.warning(self, "打开设备后台", "当前设备 IP 格式无效。")
            return

        # 设备后台使用表单会话登录，程序请求登录接口得到的 Cookie 无法交给默认
        # 浏览器。因此由浏览器打开一次性自动提交页，让 Cookie 在浏览器会话中创建。
        login_url = f"http://{normalized_ip}:8888/login"
        login_page = Path(tempfile.gettempdir()) / f"jixiebijiaoben_device_backend_{normalized_ip.replace(':', '_')}.html"
        page_html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Opening device backend</title></head>
<body><form id=\"login\" action=\"{html.escape(login_url, quote=True)}\" method=\"post\">
<input type=\"hidden\" name=\"password\" value=\"admin\">
</form><script>document.getElementById('login').submit();</script></body></html>
"""
        try:
            login_page.write_text(page_html, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "打开设备后台", f"无法创建后台登录页：{exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(login_page)))

    def _show_chat_history(self, row: int) -> None:
        device = self._device_for_row(row)
        if not device:
            return
        device_ip = str(device.get("ip", "") or "").strip()
        app_name = self._selected_app_for_row(row)
        if not device_ip or not app_name:
            return
        app_cls = get_registered_apps().get(app_name)
        app_label = getattr(app_cls, "name", app_name) if app_cls is not None else app_name
        persona = resolve_persona(
            self.config_manager,
            device_ip,
            app_name,
            device=device,
        )
        try:
            dialog = ChatHistoryDialog(
                store=ChatHistoryStore(),
                device_ip=device_ip,
                app_name=app_name,
                app_label=str(app_label),
                persona=persona,
                parent=self,
            )
        except Exception as exc:
            QMessageBox.warning(self, "聊天记录", f"聊天记录读取失败：{exc}")
            return
        dialog.exec()

    def _set_row_status(self, device_ip: str, status: str, detail: str = "") -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                detail_text = strip_status_coordinates(detail)
                current = self.table.item(row, COL_STATUS)
                if current and current.data(256) == status and current.data(257) == detail_text:
                    return
                self.table.setItem(row, COL_STATUS, self._status_item(status, detail))
                self._refresh_summary()
                return

    def _set_row_balance(self, device_ip: str, balance: str) -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) == device_ip:
                text = str(balance or "--")
                current = self.table.item(row, COL_BALANCE)
                if current and current.text() == text:
                    return
                item = QTableWidgetItem(str(balance or "--"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, COL_BALANCE, item)
                return

    def _device_health_item(self, health: str, metrics: dict[str, Any] | None = None) -> QTableWidgetItem:
        label, background, foreground = DEVICE_HEALTH_STYLES.get(health, DEVICE_HEALTH_STYLES["unknown"])
        item = QTableWidgetItem(label)
        item.setData(256, health)
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(metrics, dict) and metrics:
            item.setToolTip(
                f"平均亮度 {metrics.get('mean_luma', '--')}；"
                f"暗像素比例 {metrics.get('dark_ratio', '--')}；"
                f"亮像素比例 {metrics.get('bright_ratio', '--')}；"
                f"饱和通道 {metrics.get('clipped_channel_count', '--')}"
            )
        return item

    def _set_row_device_health(self, device_ip: str, health: str, metrics: dict[str, Any] | None = None) -> None:
        for row in range(self.table.rowCount()):
            if self._device_ip_for_row(row) != device_ip:
                continue
            current = self.table.item(row, COL_DEVICE_HEALTH)
            if current and current.data(256) == health:
                return
            self.table.setItem(row, COL_DEVICE_HEALTH, self._device_health_item(health, metrics))
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
                text = format_duration(seconds)
                current = self.table.item(row, COL_UNATTENDED_SECONDS)
                if current and current.text() == text:
                    return
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
                count = max(0, int(reply_round_count))
                value = max(0.0, float(average_seconds or 0.0))
                current = self.table.item(row, COL_AVG_REPLY_SECONDS)
                if current and current.data(256) == count and float(current.data(257) or 0.0) == value:
                    return
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
            if existing.stop_requested or current_status == "stopping":
                # 旧进程还在退出中（长步骤/网络挂起会拖几十秒），此刻启动会被
                # 静默忽略——明确告知用户，等状态变成"已停止"再点启动。
                self._append_log_line(
                    device_ip,
                    f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: 旧进程正在退出，请等状态变为已停止后再启动",
                )
            return
        selected_app = self._selected_app_for_row(row)
        device["selected_app"] = selected_app
        self.config_manager.update_device(device_ip, {"selected_app": selected_app})
        # 仅在实际启动当前设备时，重新写入该设备后台的本地 TTS 地址。
        self._start_and_bind_local_tts([device_ip])
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
            self._append_log_line(device_ip, f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: device removed")

    def _save_ocr_concurrency(self) -> None:
        self.config_manager.set("global", "ocr_max_concurrency", value=self.ocr_concurrency_input.value())

    def _open_monitor_wall(self) -> None:
        """监控墙：吸附在主窗口右侧的独立弹窗（不压缩主界面），按钮切换显示/隐藏。"""
        from ui.monitor_wall import MonitorWallWindow

        if self.monitor_wall_window is None:
            self.monitor_wall_window = MonitorWallWindow(self.config_manager.devices, self)
        visible = not self.monitor_wall_window.isVisible()
        self.monitor_wall_window.setVisible(visible)
        if visible:
            self.monitor_wall_window._sync_position()
            self.monitor_wall_window.raise_()
        self.monitor_wall_button.setText("收起监控墙" if visible else "打开监控墙")

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
                "app_overrides": self._default_app_overrides(),
            }
        )
        if not created:
            QMessageBox.information(self, "设备", "该设备已存在")
            return
        self.device_ip_input.clear()
        self._load_devices()

    def _scan_devices(self) -> None:
        """后台扫描全网段；扫描期间 UI 不阻塞（原先同步扫描会卡住界面数秒，Windows 直接标记无响应）。"""
        if self._device_scan_running:
            return
        self._device_scan_running = True
        self.scan_device_button.setEnabled(False)
        self.scan_device_button.setText("扫描中…")

        def scan() -> None:
            try:
                found = scan_devices(timeout=0.3, max_workers=50)
            except Exception as exc:
                self.events.put(WorkerEvent("", "warning", "scan_error", {"error": str(exc)}))
                return
            # 设备入库在 UI 线程做（scan_complete 分支），避免和界面保存配置抢写。
            self.events.put(WorkerEvent("", "info", "scan_complete", {"devices": list(found)}))

        threading.Thread(target=scan, name="device-scan", daemon=True).start()

    def _finish_device_scan(self, devices: list[str], *, error: str = "") -> None:
        self._device_scan_running = False
        self.scan_device_button.setEnabled(True)
        self.scan_device_button.setText("扫描设备")
        if error:
            QMessageBox.warning(self, "扫描设备", error)
            return
        added = 0
        for device_ip in devices:
            if self.config_manager.add_device(
                {
                    "ip": device_ip,
                    "license_key": "",
                    "license_expires": "",
                    "selected_app": get_default_app_name(),
                    "persona": {},
                    "app_overrides": self._default_app_overrides(),
                }
            ):
                added += 1
        self._load_devices()
        QMessageBox.information(self, "扫描设备", f"发现 {len(devices)} 台，新增 {added} 台")

    def _start_and_bind_local_tts(self, device_ips: list[str] | None = None) -> None:
        """后台启动本地 TTS；只有显式传入的设备 IP 会被重新绑定。"""
        start_only = device_ips is None
        targets = list(dict.fromkeys(str(ip).strip() for ip in (device_ips or []) if str(ip).strip()))
        with self._tts_bind_lock:
            new_targets = [ip for ip in targets if ip not in self._tts_bind_in_progress]
            self._tts_bind_in_progress.update(new_targets)
        if not new_targets and not start_only:
            return

        def bind() -> None:
            try:
                if start_only:
                    self.local_tts_server.start()
                    self.events.put(WorkerEvent("本地TTS", "info", "本地 TTS 服务已启动"))
                    return
                results = self.local_tts_server.bind_devices(new_targets)
                for device_ip, tts_url, error in results:
                    if error:
                        self.events.put(WorkerEvent(device_ip, "warning", f"本地 TTS 绑定失败：{error}"))
                    else:
                        self.events.put(WorkerEvent(device_ip, "info", f"本地 TTS 已绑定：{tts_url}"))
            except Exception as exc:
                self.events.put(WorkerEvent("本地TTS", "warning", f"本地 TTS 服务启动失败：{exc}"))
            finally:
                with self._tts_bind_lock:
                    self._tts_bind_in_progress.difference_update(new_targets)

        threading.Thread(target=bind, name="local-tts-bind", daemon=True).start()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # TTS 端口清理要跑 netstat/taskkill（秒级），放后台避免关窗卡死；
        # 清理未完成也无妨：下次启动会按 server_version 自检并清孤儿。
        def stop_tts() -> None:
            try:
                self.local_tts_server.stop()
            except Exception:
                pass

        threading.Thread(target=stop_tts, name="tts-stop", daemon=True).start()
        super().closeEvent(event)

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
        dialog_config["persona_chat_model"] = str(self.config_manager.get("global", "persona_chat_model", default="") or "")
        dialog = AppConfigDialog(
            dialog_config,
            device_ip=device_ip,
            app_name=app_name,
            template_root=template_root,
            on_reset_visual_recognition=lambda: self._reset_visual_recognition(device_ip, app_name),
            on_calibrate_vision=lambda finisher: self._calibrate_vision(device_ip, app_name, finisher),
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
        persona_keys = (
            "persona_profile_json",
            "greetings_txt",
            "judge_enabled",
            "greeting_ai_enabled",
            "history_turns",
            "summary_interval",
        )
        device_app_updates = {
            key: copy.deepcopy(values[key])
            for key in ("photo_sent_cooldown_seconds", "online_photo_send_count", *persona_keys)
            if key in values
        }

        self.config_manager.update_app_config(app_name, app_updates)
        self.config_manager.update_device_runtime(device_ip, runtime_updates)
        self.config_manager.update_device_app_overrides(device_ip, app_name, device_app_updates)
        # 人设 AI 新链路（docs/13 五节）：模型为机器级（global），key 只写 .env
        model = str(values.get("persona_chat_model", "") or "").strip()
        if model:
            self.config_manager.update_global("persona_chat_model", model)
        api_key = str(values.get("deepseek_api_key", "") or "").strip()
        if api_key:
            from services.deepseek_client import write_local_api_key

            write_local_api_key(None, api_key)
        QMessageBox.information(self, "配置", f"{device_ip} / {app_name} 配置已保存")

    def _reset_visual_recognition(self, device_ip: str, app_name: str) -> None:
        answer = QMessageBox.question(
            self,
            "重置视觉识别",
            f"确认重置 {device_ip} / {app_name} 的视觉识别调用次数和规则学习状态吗？\n"
            "重置后会重新允许视觉识别启动。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        reset_vision_usage({"_runtime": {"device_ip": device_ip, "app_name": app_name}})
        self._append_log_line(
            device_ip,
            f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: {app_name} 视觉识别已重置",
        )
        QMessageBox.information(self, "重置视觉识别", f"{device_ip} / {app_name} 已重置，可重新启动视觉识别。")

    def _calibrate_vision(self, device_ip: str, app_name: str, on_done: Any = None) -> None:
        def _release_button() -> None:
            if callable(on_done):
                try:
                    on_done()
                except RuntimeError:
                    pass  # 设置窗口可能已被用户关闭

        if self._vision_calibration_thread is not None and self._vision_calibration_thread.isRunning():
            QMessageBox.information(self, "视觉+OCR识别", "上一次标定还在执行，请稍候。")
            _release_button()
            return
        answer = QMessageBox.question(
            self,
            "视觉+OCR识别一次",
            "请先让设备停留在某个聊天窗口（对话界面）再继续。\n"
            "将立即执行一次视觉+OCR 识别并把聊天区域写入坐标缓存。\n"
            "手动标定不消耗视觉识别额度。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            _release_button()
            return
        # 用完整配置同步截图（约 1~2 秒）：既拿到弹窗要显示的画面，
        # 也在此完成透视签名同步与坐标缓存加载，标定结果才能正确落盘。
        from services.defaults import with_app_defaults

        app_config = with_app_defaults(self.config_manager.get_app_config(device_ip, app_name, {}))
        app_config["_runtime"] = {"device_ip": device_ip, "app_name": app_name}
        app_config["_runtime_cache"] = {}
        self._append_log_line(
            device_ip,
            f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: 开始视觉+OCR标定...",
        )
        try:
            image_bytes = capture_transform_image(
                RobotInterface(device_ip),
                app_config,
                pre_move=True,
                label="vision_calibration",
            )
            calibration_image = ensure_portrait_image(image_bytes)
        except Exception as exc:
            self._append_log_line(
                device_ip,
                f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: 视觉标定截图失败：{exc}",
            )
            QMessageBox.critical(self, "视觉+OCR识别", f"无法获取设备截图：{exc}")
            _release_button()
            return
        dialog = VisionCalibrationDialog(calibration_image, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda: setattr(self, "_vision_calibration_dialog", None))
        self._vision_calibration_dialog = dialog
        dialog.show()
        thread = VisionCalibrationThread(app_config, image=calibration_image, parent=self)
        self._vision_calibration_thread = thread
        thread.completed.connect(lambda result: self._on_vision_calibration_done(device_ip, result))
        thread.failed.connect(lambda message: self._on_vision_calibration_failed(device_ip, message))
        thread.finished.connect(lambda: self._finish_vision_calibration(thread))
        thread.start()

    def _deliver_calibration_result(self, payload: dict[str, Any]) -> None:
        dialog = self._vision_calibration_dialog
        if dialog is None:
            return
        try:
            if "reason" in payload and payload.get("is_error"):
                dialog.set_error(str(payload.get("reason")))
            else:
                dialog.set_result(payload)
        except RuntimeError:
            pass  # 用户已关闭标定窗口

    def _on_vision_calibration_done(self, device_ip: str, result: dict[str, Any]) -> None:
        self._deliver_calibration_result(result)
        if result.get("ok"):
            self._append_log_line(
                device_ip,
                f"[{time.strftime('%H:%M:%S')}] [info] {device_ip}: 视觉标定完成 "
                f"chat_area={result.get('chat_area_dynamic')} dialogue_message={result.get('dialogue_message')}",
            )
        else:
            self._append_log_line(
                device_ip,
                f"[{time.strftime('%H:%M:%S')}] [warning] {device_ip}: 视觉标定未生效：{result.get('reason')}",
            )

    def _on_vision_calibration_failed(self, device_ip: str, message: str) -> None:
        self._deliver_calibration_result({"is_error": True, "reason": message})
        self._append_log_line(
            device_ip,
            f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: 视觉标定失败：{message}",
        )

    def _finish_vision_calibration(self, thread: VisionCalibrationThread) -> None:
        if self._vision_calibration_thread is thread:
            self._vision_calibration_thread = None
        thread.deleteLater()

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
        else:
            self.logs.clear()

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
        if self._logs_visible():
            self._render_logs()

    def _render_logs(self, *_args: Any) -> None:
        if not hasattr(self, "logs"):
            return
        selected_ip = self._selected_log_device_ip()
        lines = [line for event_ip, line in self.log_events if not selected_ip or event_ip == selected_ip]
        if len(lines) > LOG_RENDER_LIMIT:
            lines = lines[-LOG_RENDER_LIMIT:]
        self.logs.setPlainText("\n".join(lines))
        self.logs.moveCursor(QTextCursor.MoveOperation.End)

    def _append_log_event(self, event: WorkerEvent) -> None:
        try:
            self._append_log_line(str(event.device_ip or ""), f"{self._log_event_time()} {self._format_log_event(event)}")
        except Exception as exc:
            device_ip = str(getattr(event, "device_ip", "unknown") or "unknown")
            self._append_log_line(device_ip, f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: UI事件日志格式化失败：{exc}")

    def _log_event_time(self) -> str:
        return f"[{time.strftime('%H:%M:%S')}]"

    def _logs_visible(self) -> bool:
        return bool(hasattr(self, "logs") and hasattr(self, "log_toggle") and self.log_toggle.isChecked())

    def _selected_log_device_ip(self) -> str:
        if hasattr(self, "log_device_filter"):
            return str(self.log_device_filter.currentData() or "")
        return ""

    def _append_log_line(self, device_ip: str, line: str) -> None:
        self.log_events.append((device_ip, line))
        if not self._logs_visible():
            return
        selected_ip = self._selected_log_device_ip()
        if selected_ip and device_ip != selected_ip:
            return
        self.logs.appendPlainText(line)

    def _should_store_log_event(self, event: WorkerEvent) -> bool:
        if str(event.level or "") == "status":
            return False
        return True

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
            reuse_text = ""
            if payload.get("ocr_cache_hit_count", 0) or payload.get("detection_cache_hit_count", 0):
                reuse_text = (
                    f"复用 OCR {payload.get('ocr_cache_hit_count', 0)} 次，"
                    f"复用检测 {payload.get('detection_cache_hit_count', 0)} 次，"
                )
            return (
                f"[{event.level}] {event.device_ip}: 心遇本轮统计 "
                f"OCR {payload.get('ocr_count', '')} 次，{reuse_text}截图 {payload.get('capture_count', '')} 次，"
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
        if event.message == "no_phone_detected":
            return f"[{event.level}] {event.device_ip}: 检测到无手机 {payload.get('error', '')}"
        if event.message == "black_screen_detected":
            return f"[{event.level}] {event.device_ip}: 检测到黑屏 {payload.get('error', '')}"
        if event.message == "keep_awake_failed":
            return f"[{event.level}] {event.device_ip}: 防熄屏失败 {payload.get('error', '')}"
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
        started_at = time.monotonic()
        processed = 0
        while processed < UI_EVENT_DRAIN_LIMIT:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            try:
                self._handle_worker_event(event)
            except Exception as exc:
                device_ip = str(getattr(event, "device_ip", "unknown") or "unknown")
                self._append_log_line(device_ip, f"[{time.strftime('%H:%M:%S')}] [error] {device_ip}: UI事件处理失败：{exc}")
            if time.monotonic() - started_at >= UI_EVENT_DRAIN_SECONDS:
                break
        now = time.monotonic()
        if now - self._last_unattended_refresh_at >= UNATTENDED_REFRESH_SECONDS:
            self._last_unattended_refresh_at = now
            try:
                self._refresh_unattended_durations()
            except Exception as exc:
                self._append_log_line("", f"[{time.strftime('%H:%M:%S')}] [error] UI: 无干预时长刷新失败：{exc}")

        self._drain_update_events()
        self._refresh_ocr_queue_count()

    def _refresh_ocr_queue_count(self) -> None:
        try:
            status = get_ocr_queue_status()
            waiting = max(0, int(status.get("waiting", 0) or 0)) if isinstance(status, dict) else 0
        except Exception:
            waiting = 0
        self.ocr_queue_label.setText(f"OCR等待：{waiting}")

    def _drain_update_events(self) -> None:
        processed = 0
        while processed < UPDATE_EVENT_DRAIN_LIMIT:
            try:
                event_type, payload = self.update_events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            try:
                self._handle_update_event(event_type, payload)
            except Exception as exc:
                self._append_log_line("", f"[{time.strftime('%H:%M:%S')}] [error] UI: 更新事件处理失败：{exc}")

    def _handle_update_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type in {"check_complete", "check_error"}:
            self._finish_release_check()

        if event_type == "check_error":
            message = str(payload.get("message", "") or "检查更新失败")
            manual = bool(payload.get("manual"))
            if manual:
                QMessageBox.warning(self, "检查更新", message)
            else:
                self._append_log_line("", f"[{time.strftime('%H:%M:%S')}] [warning] UI: {message}")
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
        if message == "scan_complete":
            devices = [str(ip) for ip in payload.get("devices", []) if str(ip).strip()]
            self._finish_device_scan(devices)
            return
        if message == "scan_error":
            self._finish_device_scan([], error=str(payload.get("error", "") or "扫描失败"))
            return
        if is_stale_worker_event(event, self.workers.get(event_ip)):
            # 旧 worker 迟到的退出/状态事件：新 worker 已接管或设备已删除，直接丢弃，
            # 否则会把正在运行的行状态打回"已停止"。
            return
        if self._should_store_log_event(event):
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
            # 流程进度事件通常不附带截图健康度；不能把已有结果覆盖回“待检测”。
            if "phone_screen_health" in payload:
                health = str(payload.get("phone_screen_health", "unknown") or "unknown")
                metrics = payload.get("phone_screen_health_metrics")
                self._set_row_device_health(event_ip, health, metrics if isinstance(metrics, dict) else None)
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
        elif message == "xinyu_balance":
            self._set_row_balance(event_ip, str(payload.get("balance", "") or ""))

class PopupCropLabel(QWidget):
    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.origin = QPoint()
        self.selection = QRect()
        self.match_points: list[tuple[float, float]] = []
        self.region_overlays: list[tuple[str, tuple[float, float, float, float], QColor]] = []
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
        self.match_points = []
        self.setFixedSize(self.display_pixmap.size())
        self.update()

    def set_match_points(self, points: list[tuple[float, float]]) -> None:
        self.match_points = [
            (max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y))))
            for x, y in points
        ]
        self.update()

    def set_region_overlays(self, regions: list[tuple[str, tuple[float, float, float, float], QColor]]) -> None:
        self.region_overlays = regions
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
        for label, (left, top, width, height), color in self.region_overlays:
            rect = QRect(
                int(round(left * self.display_pixmap.width())),
                int(round(top * self.display_pixmap.height())),
                int(round(width * self.display_pixmap.width())),
                int(round(height * self.display_pixmap.height())),
            )
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            painter.setPen(QPen(color, 1))
            painter.drawText(rect.adjusted(3, 15, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, label)
        painter.setPen(QPen(QColor("#dc2626"), 3))
        for x_ratio, y_ratio in self.match_points:
            x = int(round(x_ratio * self.display_pixmap.width()))
            y = int(round(y_ratio * self.display_pixmap.height()))
            painter.drawLine(x - 12, y, x + 12, y)
            painter.drawLine(x, y - 12, x, y + 12)
            painter.drawEllipse(QPoint(x, y), 4, 4)
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

        # 添加入口与下方表格同源：目录里存在什么族，上面就有什么「添加X」。
        # 弹窗是例外——它是用户自行框图、自选匹配范围的采集入口（范围存入
        # 用户本机配置），必须始终可用，不依赖目录里已有弹窗文件；返回按钮
        # 有专属采集流程，存在文件时出现；新增普通分类保存为该分类的下一
        # 个客户变体，template_variant_paths 运行时自动把新变体纳入匹配。
        dynamic_specs: list[tuple[str, str]] = [(CUSTOM_POPUP, "弹窗")]
        if back_button_template_paths({"templates": {"buttons": str(self.buttons_dir)}}):
            dynamic_specs.append((CUSTOM_BACK_BUTTON, "返回按钮"))
        dynamic_specs.extend(
            (f"variant::{group}::{base_name}", self._template_display_name(base_name))
            for group, base_name in self._scan_template_bases()
        )
        per_row = 5
        for start in range(0, len(dynamic_specs), per_row):
            template_actions = QHBoxLayout()
            template_actions.setSpacing(8)
            for target_key, label in dynamic_specs[start : start + per_row]:
                button = QPushButton(f"添加{label}")
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

    def _scan_template_bases(self) -> list[tuple[str, str]]:
        """扫描模板目录，返回 (分组, 基础文件名) 列表。

        同一模板的变体（btn_x.jpg / btn_x_2.jpg / btn_x__customer_1.jpg /
        btn_x__customer_{机器MAC}_{序号}.jpg，新客户模板带 MAC 命名空间）
        归并到基础文件名；弹窗走专属流程，跳过。
        """
        bases: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for group, root in (("buttons", self.buttons_dir), ("screens", self.screens_dir)):
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.jpg")):
                if path.name.startswith("popup_"):
                    continue
                if group == "buttons" and is_back_button_template_name(path.name):
                    # 返回按钮有专属入口与规则，不重复生成动态按钮。
                    continue
                stem = re.sub(r"(?:__(?:customer|developer)(?:_[0-9a-f]{12})?)?_\d+$", "", path.stem)
                base_name = f"{stem}{path.suffix}"
                if (group, base_name) in seen:
                    continue
                seen.add((group, base_name))
                bases.append((group, base_name))
        return bases

    @staticmethod
    def _template_display_name(base_name: str) -> str:
        """分类显示名：去掉 btn_/screen_ 前缀与扩展名。"""
        stem = Path(base_name).stem
        for prefix in ("btn_", "screen_"):
            if stem.startswith(prefix):
                return stem[len(prefix) :]
        return stem

    @staticmethod
    def _parse_variant_key(target_key: str) -> tuple[str, str] | None:
        """解析 variant::分组::基础文件名 形式的动态模板键。"""
        if not target_key.startswith("variant::"):
            return None
        parts = target_key.split("::", 2)
        if len(parts) != 3 or parts[1] not in {"buttons", "screens"} or not parts[2].endswith(".jpg"):
            return None
        return parts[1], parts[2]

    def reload(self) -> None:
        # 完全由模板目录驱动：目录里有什么文件就显示什么族（弹窗每文件
        # 独立、返回按钮走专属族），显示名从文件名派生；范围列来自 config
        # 的 template_match_regions（button.custom.* 反查 + 少数固定键兜底）。
        # 不依赖任何写死的分类清单。
        families: list[tuple[str, list[Path]]] = []
        existing_paths: set[Path] = set()

        def _add_family(kind: str, paths: list[Path]) -> None:
            merged = [path for path in paths if path.resolve() not in existing_paths]
            if not merged:
                return
            existing_paths.update(path.resolve() for path in merged)
            families.append((kind, merged))

        for path in sorted(self.buttons_dir.glob("popup_*.jpg")):
            if is_popup_template_name(path.name):
                _add_family("弹窗", [path])
        _add_family("返回按钮", back_button_template_paths({"templates": {"buttons": str(self.buttons_dir)}}))
        for group, base_name in self._scan_template_bases():
            root = self.buttons_dir if group == "buttons" else self.screens_dir
            _add_family(self._template_display_name(base_name), template_variant_paths(root, base_name))

        self.table.setRowCount(len(families))
        for row, (kind, paths) in enumerate(families):
            primary = paths[0]
            file_text = primary.name if len(paths) == 1 else f"{primary.name}（共 {len(paths)} 个变体）"
            self.table.setItem(row, 0, QTableWidgetItem(kind))
            self.table.setItem(row, 1, QTableWidgetItem(file_text))
            self.table.setItem(row, 2, QTableWidgetItem(self._template_region_text(primary)))
            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(6)
            test_button = QPushButton("测试当前图")
            test_button.clicked.connect(lambda checked=False, ps=list(paths): self._test_saved_template(ps))
            actions_layout.addWidget(test_button)
            delete_button = QPushButton("删除")
            delete_button.clicked.connect(lambda checked=False, ps=list(paths): self._delete_template_family(ps))
            actions_layout.addWidget(delete_button)
            self.table.setCellWidget(row, 3, actions)

    def _delete_template_family(self, paths: list[Path]) -> None:
        """删除模板：单变体直接走原确认流程；多变体弹出菜单选具体文件。"""
        if len(paths) == 1:
            self._delete_template(paths[0])
            return
        menu = QMenu(self)
        for path in paths:
            menu.addAction(path.name, lambda checked=False, p=path: self._delete_template(p))
        button = self.sender()
        if isinstance(button, QPushButton):
            menu.exec(button.mapToGlobal(QPoint(0, button.height())))
        else:
            menu.exec()

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

        # 主 UI 中新增的模板明确标记为客户模板，更新包不会覆盖这些文件。
        variant_spec = self._parse_variant_key(target_key)
        if variant_spec is not None:
            group, base_name = variant_spec
            root = self.buttons_dir if group == "buttons" else self.screens_dir
            target = next_template_variant_path(root, base_name, owner="customer")
        else:
            target = import_target_path(
                target_key,
                self.buttons_dir,
                self.screens_dir,
                owner="customer",
            )
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
        if not isinstance(self.overrides, dict):
            return "-"
        regions = self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
        if not isinstance(regions, dict):
            return "未设置"
        key = self._template_region_key(path)
        if not key:
            return "-"
        region = regions.get(key)
        if region is None and key == BACK_BUTTON:
            region = regions.get(CUSTOM_BACK_BUTTON)
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return "未设置"
        try:
            return ", ".join(f"{float(value):.3f}" for value in region)
        except (TypeError, ValueError):
            return "未设置"

    def _template_region_key(self, path: Path) -> str:
        if is_popup_template_name(path.name):
            return popup_template_region_key(path)
        if is_back_button_template_name(path.name) or path.name == "btn_back.jpg":
            return BACK_BUTTON
        for rule in fixed_template_rules():
            base_stem = Path(rule.filename).stem
            suffix = Path(rule.filename).suffix or ".jpg"
            if path.name == rule.filename or re.fullmatch(
                rf"{re.escape(base_stem)}_\d+{re.escape(suffix)}",
                path.name,
            ) or re.fullmatch(
                rf"{re.escape(base_stem)}__(?:customer|developer)(?:_[0-9a-f]{{12}})?_\d+{re.escape(suffix)}",
                path.name,
            ):
                return rule.key
        # 固定规则之外的按钮模板（如心遇的主界面心遇按钮等自定义按钮）：
        # 按 template_match_regions 的 button.custom.* 配置反查，主文件与
        # 变体都归到同一键，范围列才能显示出来。
        regions = (
            self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
            if isinstance(self.overrides, dict)
            else {}
        )
        if isinstance(regions, dict):
            suffix = path.suffix or ".jpg"
            for config_key in regions:
                key = str(config_key)
                if not key.startswith("button.custom."):
                    continue
                name = key[len("button.custom."):].strip()
                if not name:
                    continue
                stems = [f"btn_{name}"]
                if name == "相册":
                    stems.append("btn_照片按钮")
                for stem in stems:
                    if path.name == f"{stem}{suffix}" or re.fullmatch(
                        rf"{re.escape(stem)}_\d+{re.escape(suffix)}",
                        path.name,
                    ) or re.fullmatch(
                        rf"{re.escape(stem)}__(?:customer|developer)(?:_[0-9a-f]{{12}})?_\d+{re.escape(suffix)}",
                        path.name,
                    ):
                        return key
        return ""

    def _test_saved_template(self, paths: list[Path]) -> None:
        primary = paths[0]
        try:
            robot = RobotInterface(self.device_ip)
            image = ensure_portrait_image(capture_transform_image(robot, pre_move=True, label="template_test"))
        except Exception as exc:
            QMessageBox.critical(self, "模板测试失败", f"无法获取截图：{exc}")
            return
        key = self._template_region_key(primary)
        if not key:
            QMessageBox.warning(self, "模板测试", f"无法确定模板类别范围：{primary.name}")
            return
        config = self.overrides if isinstance(self.overrides, dict) else {}
        thresholds = config.get("template_match_thresholds", {})
        threshold = float(
            config.get("popup_match_threshold", 0.8)
            if is_popup_template_name(primary.name)
            else config.get("match_threshold", 0.7)
        )
        if isinstance(thresholds, dict) and key in thresholds:
            try:
                threshold = float(thresholds[key])
            except (TypeError, ValueError):
                pass
        try:
            # 整族多变体择优（与业务运行时一致）；用最低阈值取回最高候选，
            # 未达到正式阈值时也必须把相似度展示给用户。
            result = match_template_candidates_with_region(
                image, list(paths), config, key,
                threshold=-1.0,
                match_func=match,
                single_match_func=match_template_with_region,
            )
        except Exception as exc:
            QMessageBox.critical(self, "模板测试失败", f"匹配 {primary.name} 时出错：{exc}")
            return
        name = primary.name if len(paths) == 1 else f"{primary.name}（共 {len(paths)} 个变体）"
        self._show_template_test_result(image, name, key, result, threshold)

    def _show_template_test_result(
        self,
        image: Image.Image,
        name: str,
        key: str,
        result: dict[str, Any] | None,
        threshold: float,
    ) -> None:
        pixmap = _pixmap_from_pil_image(image)
        painter = QPainter(pixmap)
        regions = self.overrides.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {}) if isinstance(self.overrides, dict) else {}
        region = regions.get(key) if isinstance(regions, dict) else None
        if isinstance(region, (list, tuple)) and len(region) == 4:
            x, y, w, h = (float(value) for value in region)
            painter.setPen(QPen(QColor(249, 115, 22), 4))
            painter.drawRect(QRect(int(x * pixmap.width()), int(y * pixmap.height()), int(w * pixmap.width()), int(h * pixmap.height())))
        if result:
            painter.setPen(QPen(QColor(34, 197, 94), 4))
            painter.drawRect(QRect(int(result["x"] * pixmap.width()), int(result["y"] * pixmap.height()), int(result["w"] * pixmap.width()), int(result["h"] * pixmap.height())))
        painter.end()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"模板测试：{name}")
        dialog.resize(600, 820)
        layout = QVBoxLayout(dialog)
        score = float(result.get("score", -1.0)) if result else None
        matched = score is not None and score >= threshold
        if matched:
            status = f"匹配成功：相似度 {score:.3f} / 阈值 {threshold:.3f}。绿色框为命中位置；橙色框为配置匹配范围。"
        elif score is not None:
            status = f"未达到阈值：相似度 {score:.3f} / 阈值 {threshold:.3f}。绿色框为最高候选；橙色框为配置匹配范围。"
        else:
            status = f"未找到有效候选：阈值 {threshold:.3f}。橙色框为配置匹配范围。"
        layout.addWidget(QLabel(status))
        label = QLabel()
        label.setPixmap(pixmap.scaled(560, 720, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(label, 1)
        dialog.exec()

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

    def _offset_spin(self, value: Any) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-0.2, 0.2)
        spin.setDecimals(3)
        spin.setSingleStep(0.001)
        spin.setValue(float(value or 0.0))
        return spin

    def values(self) -> dict[str, Any]:
        return {}
    def __init__(
        self,
        overrides: dict[str, Any],
        device_ip: str,
        app_name: str,
        template_root: Path,
        on_reset_visual_recognition: Callable[[], None] | None = None,
        on_calibrate_vision: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = with_app_defaults(overrides)
        self.device_ip = device_ip
        self.app_name = app_name
        self.template_root = template_root
        self._on_reset_visual_recognition = on_reset_visual_recognition
        self._on_calibrate_vision = on_calibrate_vision
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

        self.poll_interval = NoWheelSpinBox()
        self.poll_interval.setRange(1, 3600)
        self.poll_interval.setValue(int(overrides.get("poll_interval", 1)))
        form.addRow("轮询间隔（秒）", self.poll_interval)

        self.step_timeout = NoWheelSpinBox()
        self.step_timeout.setRange(1, 300)
        self.step_timeout.setValue(int(overrides.get("step_timeout", 10)))
        form.addRow("单步超时（秒）", self.step_timeout)

        self.exclude_ocr_wait_from_step_timeout = QCheckBox("OCR等待不计入单步超时")
        self.exclude_ocr_wait_from_step_timeout.setChecked(
            bool(overrides.get("exclude_ocr_wait_from_step_timeout", True))
        )
        form.addRow("", self.exclude_ocr_wait_from_step_timeout)

        self.keep_awake_interval_seconds = NoWheelSpinBox()
        self.keep_awake_interval_seconds.setRange(0, 86400)
        self.keep_awake_interval_seconds.setValue(int(overrides.get("keep_awake_interval_seconds", DEFAULT_RUNTIME_CONFIG["keep_awake_interval_seconds"])))
        form.addRow("防熄屏跳转间隔（秒，0关闭）", self.keep_awake_interval_seconds)

        self.photo_sent_cooldown_seconds = NoWheelSpinBox()
        self.photo_sent_cooldown_seconds.setRange(0, 2592000)
        self.photo_sent_cooldown_seconds.setValue(int(overrides.get("photo_sent_cooldown_seconds", 86400) or 0))
        self.photo_sent_cooldown_seconds.setToolTip("主动照片发送成功后，同一用户在此时间内不会再次收到主动照片。")
        form.addRow("同一用户发图冷却（秒，0不限制）", self.photo_sent_cooldown_seconds)

        self.online_photo_send_count = NoWheelSpinBox()
        self.online_photo_send_count.setRange(0, 50)
        self.online_photo_send_count.setValue(int(overrides.get("online_photo_send_count", 3) or 0))
        self.online_photo_send_count.setToolTip("没有消息进入空闲状态时，主动给多少位在线用户发送照片。0 表示关闭主动发图。")
        form.addRow("每次空闲主动发图人数（0关闭）", self.online_photo_send_count)

        self.reset_visual_recognition_button = QPushButton("重置视觉识别")
        self.reset_visual_recognition_button.setToolTip("清空当前设备当前 App 的视觉调用额度和连续无新增规则计数。")
        self.reset_visual_recognition_button.clicked.connect(self._reset_visual_recognition)
        panel_root.addWidget(self.reset_visual_recognition_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.calibrate_vision_button = QPushButton("视觉+OCR识别一次")
        self.calibrate_vision_button.setToolTip(
            "让设备停留在某个聊天窗口，点击后立即执行一次视觉+OCR 识别，"
            "把聊天区域范围学进坐标缓存。手动标定独立于运行额度，不消耗视觉识别次数。"
        )
        self.calibrate_vision_button.clicked.connect(self._calibrate_vision)
        panel_root.addWidget(self.calibrate_vision_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.ocr_toggle_button = QPushButton("显示OCR设置")
        self.ocr_toggle_button.clicked.connect(self._toggle_ocr_tab)
        panel_root.addWidget(self.ocr_toggle_button, 0, Qt.AlignmentFlag.AlignLeft)

        page_root.addWidget(panel)
        page_root.addStretch(1)
        self.tabs.addTab(page, "运行设置")

    def _reset_visual_recognition(self) -> None:
        if self._on_reset_visual_recognition is not None:
            self._on_reset_visual_recognition()

    def _calibrate_vision(self) -> None:
        if self._on_calibrate_vision is None:
            return
        self.calibrate_vision_button.setEnabled(False)
        self.calibrate_vision_button.setText("标定中...")
        self._on_calibrate_vision(self._finish_calibration_ui)

    def _finish_calibration_ui(self) -> None:
        self.calibrate_vision_button.setEnabled(True)
        self.calibrate_vision_button.setText("视觉+OCR识别一次")

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
        """人设 tab 四组（docs/13 第六节 + persona_ui_mockup.svg）：AI 接入 / 人设文件 / 对话记忆 / 招呼语。"""
        from services.deepseek_client import FALLBACK_MODELS, resolve_api_key

        page = QWidget()
        page_root = QVBoxLayout(page)
        page_root.setContentsMargins(0, 0, 0, 0)
        page_root.setSpacing(10)

        def _panel(title_text: str, helper_text: str = "") -> tuple[QFrame, QVBoxLayout]:
            frame = QFrame()
            frame.setObjectName("InsetPanel")
            root = QVBoxLayout(frame)
            root.setContentsMargins(14, 12, 14, 12)
            root.setSpacing(8)
            title = QLabel(title_text)
            title.setObjectName("SectionLabel")
            root.addWidget(title)
            if helper_text:
                helper = QLabel(helper_text)
                helper.setObjectName("HelperLabel")
                helper.setWordWrap(True)
                root.addWidget(helper)
            return frame, root

        # ---------- 组1：AI 接入 ----------
        ai_panel, ai_root = _panel(
            "AI 接入（DeepSeek 官方直连）",
            "key 由客户在 platform.deepseek.com 自行注册充值；保存到 .env/deepseek_api_key.txt，绝不写入 config.json。",
        )
        ai_form = QFormLayout()
        ai_form.setContentsMargins(0, 4, 0, 0)
        ai_form.setSpacing(10)
        key_row = QHBoxLayout()
        self.persona_key_input = QLineEdit()
        self.persona_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        if resolve_api_key():
            self.persona_key_input.setPlaceholderText("已配置（留空保持不变）")
        else:
            self.persona_key_input.setPlaceholderText("sk-…（DeepSeek API Key）")
        self.persona_test_button = QPushButton("连通性测试")
        self.persona_test_button.clicked.connect(self._test_persona_connectivity)
        key_row.addWidget(self.persona_key_input, 1)
        key_row.addWidget(self.persona_test_button)
        ai_form.addRow("API Key", key_row)
        self.persona_test_result = QLabel("")
        self.persona_test_result.setObjectName("HelperLabel")
        ai_form.addRow("", self.persona_test_result)
        model_row = QHBoxLayout()
        self.persona_model_combo = QComboBox()
        current_model = str(overrides.get("persona_chat_model", "") or "").strip() or "deepseek-flash"
        for model_id in FALLBACK_MODELS:
            self.persona_model_combo.addItem(model_id)
        if current_model not in FALLBACK_MODELS:
            self.persona_model_combo.addItem(current_model)
        self.persona_model_combo.setCurrentText(current_model)
        model_row.addWidget(self.persona_model_combo, 0)
        model_hint = QLabel("列表来自 GET /models，失败回退内置清单")
        model_hint.setObjectName("HelperLabel")
        model_row.addWidget(model_hint, 1)
        ai_form.addRow("模型", model_row)
        ai_root.addLayout(ai_form)
        page_root.addWidget(ai_panel)
        self._persona_model_threads: list[_PersonaNetworkThread] = []
        self._refresh_persona_models_async()

        # ---------- 组2：人设文件（方案包 JSON） ----------
        profile_panel, profile_root = _panel(
            "人设文件（方案包 JSON）",
            "界面内容直接写回该 JSON 文件（唯一事实源）；未配置或留空回退随包默认 apps/<App>/persona_default.json（四字段）。人设改动即时生效，对话记忆连续不重开场。",
        )
        path_row = QHBoxLayout()
        self.persona_profile_path = QLineEdit(str(overrides.get("persona_profile_json", "") or "").strip())
        browse_button = QPushButton("浏览…")
        browse_button.clicked.connect(self._browse_persona_profile)
        save_as_button = QPushButton("另存为…")
        save_as_button.clicked.connect(self._save_as_persona_profile)
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self._reset_persona_profile)
        for widget in (self.persona_profile_path, browse_button, save_as_button, reset_button):
            path_row.addWidget(widget, 1 if widget is self.persona_profile_path else 0)
        profile_form = QFormLayout()
        profile_form.setContentsMargins(0, 4, 0, 0)
        profile_form.setSpacing(10)
        profile_form.addRow("JSON 文件", path_row)
        profile_root.addLayout(profile_form)

        self.persona_text_edits: dict[str, QPlainTextEdit] = {}
        persona_text_meta = [
            ("基础设定", "稳定前缀 · 命中缓存 · 改动即时生效"),
            ("人物信息", "稳定前缀 · 与基础设定拼为同一 System 段"),
            ("边界条件", "尾部注入，不占缓存前缀，每轮按新内容重复计费，建议 ≤ 200 字"),
            ("审核标准", "独立请求窗口，建议 ≥ 512 字，稳定长提示词可跨轮命中缓存"),
        ]
        for key, hint in persona_text_meta:
            edit = QPlainTextEdit()
            edit.setPlaceholderText(f"{key}（留空使用随包默认）" if key != "人物信息" else "人物信息（自由书写，可留空）")
            edit.setMaximumHeight(84)
            self.persona_text_edits[key] = edit
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            hint_label = QLabel(hint)
            hint_label.setObjectName("HelperLabel")
            hint_label.setWordWrap(True)
            row.addWidget(hint_label, 0)
            profile_form.addRow(QLabel(key), row)
        judge_row = QHBoxLayout()
        self.persona_judge_checkbox = QCheckBox("启用 AI 审核（默认开；关闭则跳过审核，标准留空时用内置默认）")
        self.persona_judge_checkbox.setChecked(bool(overrides.get("judge_enabled", True)))
        judge_row.addWidget(self.persona_judge_checkbox)
        judge_row.addStretch(1)
        profile_root.addLayout(profile_form)
        profile_root.addLayout(judge_row)
        page_root.addWidget(profile_panel)
        self._load_persona_texts_into_editors()

        # ---------- 组3：对话记忆 ----------
        memory_panel, memory_root = _panel("对话记忆")
        memory_form = QFormLayout()
        memory_form.setContentsMargins(0, 4, 0, 0)
        memory_form.setSpacing(10)
        self.persona_history_turns = NoWheelSpinBox()
        self.persona_history_turns.setRange(0, 30)
        self.persona_history_turns.setValue(int(overrides.get("history_turns", 6) or 6))
        memory_form.addRow("保留对话轮数", self.persona_history_turns)
        history_hint = QLabel("每轮注入最近 N 轮原始对话；设为 0 = 只靠前情摘要记忆；窗口越大上下文越足、费用越高")
        history_hint.setObjectName("HelperLabel")
        memory_form.addRow("", history_hint)
        self.persona_summary_interval = NoWheelSpinBox()
        self.persona_summary_interval.setRange(1, 50)
        self.persona_summary_interval.setValue(int(overrides.get("summary_interval", 10) or 10))
        memory_form.addRow("总结间隔", self.persona_summary_interval)
        interval_hint = QLabel("每积累 K 轮触发一次记忆提炼（前情摘要）；建议 ≥ 10，过小会频繁打断缓存前缀、成本上升")
        interval_hint.setObjectName("HelperLabel")
        memory_form.addRow("", interval_hint)
        memory_root.addLayout(memory_form)
        page_root.addWidget(memory_panel)

        # ---------- 组4：招呼语 ----------
        greeting_panel, greeting_root = _panel(
            "招呼语",
            "每行一条；内容不进界面，运行时直接读取。未配置时用 runtime/greetings/pingtai.txt，缺失回退随包默认。",
        )
        greeting_path_row = QHBoxLayout()
        self.persona_greetings_path = QLineEdit(str(overrides.get("greetings_txt", "") or "").strip())
        greeting_browse = QPushButton("浏览…")
        greeting_browse.clicked.connect(self._browse_greetings_file)
        self.persona_greetings_count = QLabel("")
        self.persona_greetings_count.setObjectName("HelperLabel")
        greeting_path_row.addWidget(self.persona_greetings_path, 1)
        greeting_path_row.addWidget(greeting_browse)
        greeting_path_row.addWidget(self.persona_greetings_count)
        greeting_form = QFormLayout()
        greeting_form.setContentsMargins(0, 4, 0, 0)
        greeting_form.addRow("招呼语文件", greeting_path_row)
        greeting_root.addLayout(greeting_form)
        greeting_switch_row = QHBoxLayout()
        self.persona_greeting_ai_checkbox = QCheckBox("启用 AI 优化招呼语（默认开：按当前人设改写并自动避开历史开场白；关闭则随机直发）")
        self.persona_greeting_ai_checkbox.setChecked(bool(overrides.get("greeting_ai_enabled", True)))
        greeting_switch_row.addWidget(self.persona_greeting_ai_checkbox)
        greeting_switch_row.addStretch(1)
        greeting_root.addLayout(greeting_switch_row)
        page_root.addWidget(greeting_panel)
        self._refresh_greetings_count()

        page_root.addStretch(1)
        self.tabs.addTab(page, "人设")

    # ---------- 人设 tab 辅助 ----------
    def _persona_apps_dir(self) -> Path:
        return Path("apps") / self.app_name

    def _read_persona_default(self) -> dict[str, str]:
        """随包默认人设：apps/<app>/persona_default.json（四字段）。"""
        try:
            data = json.loads((self._persona_apps_dir() / "persona_default.json").read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: str(data.get(key) or "").strip() for key in ("基础设定", "人物信息", "边界条件", "审核标准")}

    def _effective_persona_profile(self) -> dict[str, str]:
        path_text = self.persona_profile_path.text().strip()
        if path_text:
            try:
                data = json.loads(Path(path_text).read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    return {key: str(data.get(key) or "").strip() for key in ("基础设定", "人物信息", "边界条件", "审核标准")}
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _load_persona_texts_into_editors(self) -> None:
        profile = self._effective_persona_profile()
        defaults = self._read_persona_default()
        for key, edit in self.persona_text_edits.items():
            edit.setPlainText(profile.get(key, "") or defaults.get(key, ""))

    def _collect_persona_texts(self) -> dict[str, str]:
        return {key: edit.toPlainText().strip() for key, edit in self.persona_text_edits.items()}

    def _browse_persona_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择人设方案包 JSON", "", "JSON (*.json)")
        if path:
            self.persona_profile_path.setText(path)
            self._load_persona_texts_into_editors()

    def _save_as_persona_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "另存人设方案包", "persona.json", "JSON (*.json)")
        if not path:
            return
        payload = self._collect_persona_texts()
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "人设", f"写入失败：{exc}")
            return
        self.persona_profile_path.setText(path)
        QMessageBox.information(self, "人设", "方案包已保存。拷到其它设备后用「浏览…」选中即可复用。")

    def _reset_persona_profile(self) -> None:
        self.persona_profile_path.clear()
        self._load_persona_texts_into_editors()

    def _effective_greetings_path(self) -> Path | None:
        configured = self.persona_greetings_path.text().strip()
        if configured:
            return Path(configured)
        default = Path("runtime/greetings/pingtai.txt")
        if default.is_file():
            return default
        bundled = self._persona_apps_dir() / "greetings_default.txt"
        return bundled if bundled.is_file() else None

    def _refresh_greetings_count(self) -> None:
        path = self._effective_greetings_path()
        count = 0
        if path is not None:
            try:
                count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#"))
            except OSError:
                count = 0
        self.persona_greetings_count.setText(f"当前 {count} 条" if path is not None else "未找到招呼语库")

    def _browse_greetings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择招呼语文件（每行一条）", "", "文本 (*.txt)")
        if path:
            self.persona_greetings_path.setText(path)
            self._refresh_greetings_count()

    def _start_persona_network_thread(self, api_key: str, on_result: Callable[[object], None]) -> None:
        thread = _PersonaNetworkThread(api_key=api_key)
        thread.result_ready.connect(on_result)
        self._persona_model_threads.append(thread)
        thread.finished.connect(lambda t=thread: self._persona_model_threads.remove(t) if t in self._persona_model_threads else None)
        thread.start()

    def _refresh_persona_models_async(self) -> None:
        self._start_persona_network_thread(self.persona_key_input.text().strip(), self._on_persona_models_ready)

    def _on_persona_models_ready(self, result: object) -> None:
        if isinstance(result, list) and result:
            current = self.persona_model_combo.currentText()
            self.persona_model_combo.clear()
            self.persona_model_combo.addItems([str(item) for item in result])
            if current in result:
                self.persona_model_combo.setCurrentText(current)
            self.persona_test_result.setText(f"✓ 模型列表已刷新（{len(result)} 个）")

    def _test_persona_connectivity(self) -> None:
        self.persona_test_button.setEnabled(False)
        self.persona_test_result.setText("测试中…")

        def _on_result(result: object) -> None:
            self.persona_test_button.setEnabled(True)
            if isinstance(result, list):
                self._on_persona_models_ready(result)
                self.persona_test_result.setText(f"✓ 连接正常 · 可用模型 {len(result)} 个（{' / '.join(str(m) for m in result)}）")
            else:
                self.persona_test_result.setText(f"✗ 连接失败：{result}")

        self._start_persona_network_thread(self.persona_key_input.text().strip(), _on_result)

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
        chat_left = self.dialogue_region_left.value()
        chat_top = self.dialogue_region_top.value()
        chat_right = max(chat_left, self.dialogue_region_right.value())
        chat_bottom = max(chat_top, self.dialogue_region_bottom.value())
        regions = dict(ocr.get("regions", {}) if isinstance(ocr.get("regions", {}), dict) else {})
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
            "photo_sent_cooldown_seconds": self.photo_sent_cooldown_seconds.value(),
            "online_photo_send_count": self.online_photo_send_count.value(),
            "exclude_ocr_wait_from_step_timeout": self.exclude_ocr_wait_from_step_timeout.isChecked(),
            "match_threshold": self.match_threshold.value(),
            "popup_match_threshold": self.popup_match_threshold.value(),
            "ocr": ocr,
            "ocr_modes": ocr_modes,
            "templates": {
                "buttons": str(self.template_manager.buttons_dir),
                "screens": str(self.template_manager.screens_dir),
            },
            # 人设新链路客户设置（docs/13 五/六节）：设备 IP+App 级 + 机器级 key/model
            "persona_profile_json": self._persist_persona_texts(),
            "greetings_txt": self.persona_greetings_path.text().strip(),
            "judge_enabled": self.persona_judge_checkbox.isChecked(),
            "greeting_ai_enabled": self.persona_greeting_ai_checkbox.isChecked(),
            "history_turns": self.persona_history_turns.value(),
            "summary_interval": self.persona_summary_interval.value(),
            "persona_chat_model": self.persona_model_combo.currentText().strip(),
            "deepseek_api_key": self.persona_key_input.text().strip(),
        })
        return values

    def _persist_persona_texts(self) -> str:
        """四个文本编辑框的内容写回方案包 JSON（唯一事实源）。

        无路径但内容相对随包默认有修改时，自动落到引导位
        runtime/configs/<app>/persona_profiles/，避免编辑静默丢失。
        """
        path_text = self.persona_profile_path.text().strip()
        texts = self._collect_persona_texts()
        defaults = self._read_persona_default()
        changed = any(texts.get(key, "") and texts.get(key, "") != defaults.get(key, "") for key in texts)
        if not path_text:
            if not changed:
                return ""
            guided = Path("runtime/configs") / self.app_name / "persona_profiles"
            try:
                guided.mkdir(parents=True, exist_ok=True)
                path_text = str(guided / f"persona_{time.strftime('%Y%m%d_%H%M%S')}.json")
            except OSError:
                return ""
        try:
            Path(path_text).write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return path_text
        return path_text


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

    def values(self) -> dict[str, Any]:
        return {}
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
        self._screenshot_thread: DeviceScreenshotThread | None = None
        self._template_preview_thread: TemplatePreviewThread | None = None
        self._ocr_preview_thread: OcrPreviewThread | None = None
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

        self.template_match_summary = QLabel("当前截图模板匹配：请先刷新截图。")
        self.template_match_summary.setObjectName("HelperLabel")
        self.template_match_summary.setWordWrap(True)
        control_layout.addWidget(self.template_match_summary)
        self.template_match_table = QTableWidget(0, 4)
        self.template_match_table.setHorizontalHeaderLabels(["类别", "状态", "命中模板", "相似度"])
        self.template_match_table.setMinimumHeight(170)
        self.template_match_table.setMaximumHeight(280)
        self.template_match_table.verticalHeader().setVisible(False)
        self.template_match_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.template_match_table.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents if column != 2 else QHeaderView.ResizeMode.Stretch)
        control_layout.addWidget(self.template_match_table)

        ocr_test_row = QHBoxLayout()
        ocr_test_row.setContentsMargins(0, 0, 0, 0)
        ocr_test_row.setSpacing(8)
        self.test_list_ocr_button = QPushButton("测试小范围list")
        self.test_list_ocr_button.clicked.connect(lambda: self._run_ocr_preview("list"))
        self.test_dialogue_ocr_button = QPushButton("测试dialogue")
        self.test_dialogue_ocr_button.clicked.connect(lambda: self._run_ocr_preview("dialogue"))
        ocr_test_row.addWidget(self.test_list_ocr_button)
        ocr_test_row.addWidget(self.test_dialogue_ocr_button)
        control_layout.addLayout(ocr_test_row)
        self.ocr_preview_summary = QLabel("OCR 测试：请先刷新截图。虚线框显示当前 App 已配置和运行缓存中的范围。")
        self.ocr_preview_summary.setObjectName("HelperLabel")
        self.ocr_preview_summary.setWordWrap(True)
        control_layout.addWidget(self.ocr_preview_summary)
        self.ocr_preview_output = QPlainTextEdit()
        self.ocr_preview_output.setReadOnly(True)
        self.ocr_preview_output.setMaximumHeight(180)
        self.ocr_preview_output.setPlaceholderText("识别结果会显示在这里。")
        control_layout.addWidget(self.ocr_preview_output)

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
        return []
    def _offset_spin(self, value: Any) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(-0.2, 0.2)
        spin.setDecimals(3)
        spin.setSingleStep(0.001)
        spin.setValue(float(value or 0.0))
        return spin

    def _refresh_screenshot(self) -> None:
        if self._screenshot_thread is not None and self._screenshot_thread.isRunning():
            return
        self.refresh_screenshot_button.setEnabled(False)
        self.refresh_screenshot_button.setText("刷新中...")
        self.template_match_summary.setText("正在获取设备截图，请稍候...")
        thread = DeviceScreenshotThread(self.device_ip, self.app_name, self)
        self._screenshot_thread = thread
        thread.image_ready.connect(self._on_screenshot_ready)
        thread.failed.connect(self._on_screenshot_failed)
        thread.finished.connect(self._finish_screenshot_refresh)
        thread.start()

    def _on_screenshot_ready(self, image: Image.Image) -> None:
        if self._screenshot_thread is None:
            return
        try:
            self._set_screenshot(image)
            self._detect_saved_templates(image)
        except Exception as exc:
            QMessageBox.critical(self, "截图失败", f"无法从当前设备获取截图：{exc}")

    def _on_screenshot_failed(self, message: str) -> None:
        QMessageBox.critical(self, "截图失败", f"无法从当前设备获取截图：{message}")

    def _finish_screenshot_refresh(self) -> None:
        thread = self._screenshot_thread
        self._screenshot_thread = None
        self.refresh_screenshot_button.setEnabled(True)
        self.refresh_screenshot_button.setText("刷新截图")
        if thread is not None:
            thread.deleteLater()

    def _set_screenshot(self, image: Image.Image) -> None:
        self.current_screenshot = image.copy()
        self.crop_label.set_pixmap(_pixmap_from_pil_image(image))
        self.selected_point = None
        self.point_x_label.setText("--")
        self.point_y_label.setText("--")
        self._show_ocr_region_overlays()
        self._update_click_labels()

    @staticmethod
    def _valid_region(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            left, top, width, height = (float(part) for part in value)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return (
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, top)),
            max(0.0, min(1.0 - left, width)),
            max(0.0, min(1.0 - top, height)),
        )

    def _ocr_preview_config(self) -> dict[str, Any]:
        config = _deep_merge(_app_default_config(self.app_name), self.overrides)
        config["_runtime"] = {"device_ip": self.device_ip, "app_name": self.app_name}
        load_latest_into_runtime(config)
        return config

    def _ocr_test_region(self, mode: str, config: dict[str, Any]) -> Any:
        regions = config.get("ocr", {}).get("regions", {}) if isinstance(config.get("ocr"), dict) else {}
        runtime = config.get("_runtime_cache", {}) if isinstance(config.get("_runtime_cache"), dict) else {}
        cached = runtime.get("ocr_regions", {}) if isinstance(runtime.get("ocr_regions"), dict) else {}
        if not isinstance(regions, dict):
            return None
        if mode == "list":
            return cached.get("list_compact") or regions.get("message_list")
        return cached.get("dialogue_message") or cached.get("chat_area_dynamic") or regions.get("chat_area")

    def _show_ocr_region_overlays(self) -> None:
        config = self._ocr_preview_config()
        regions = config.get("ocr", {}).get("regions", {}) if isinstance(config.get("ocr"), dict) else {}
        runtime = config.get("_runtime_cache", {}) if isinstance(config.get("_runtime_cache"), dict) else {}
        cached = runtime.get("ocr_regions", {}) if isinstance(runtime.get("ocr_regions"), dict) else {}
        candidates = [
            ("list 配置范围", regions.get("message_list") if isinstance(regions, dict) else None, QColor("#2563eb")),
            ("dialogue 配置范围", regions.get("chat_area") if isinstance(regions, dict) else None, QColor("#16a34a")),
            ("list 缓存范围", cached.get("list_compact"), QColor("#9333ea")),
            ("dialogue 缓存范围", cached.get("dialogue_message") or cached.get("chat_area_dynamic"), QColor("#ea580c")),
        ]
        overlays = [
            (label, region, color)
            for label, raw_region, color in candidates
            if (region := self._valid_region(raw_region)) is not None
        ]
        self.crop_label.set_region_overlays(overlays)

    def _run_ocr_preview(self, mode: str) -> None:
        if self.current_screenshot is None:
            QMessageBox.warning(self, "未刷新截图", "请先刷新截图，再进行 OCR 测试。")
            return
        if self._ocr_preview_thread is not None and self._ocr_preview_thread.isRunning():
            return
        config = self._ocr_preview_config()
        region = self._ocr_test_region(mode, config)
        region_label = "消息列表范围" if mode == "list" else "聊天内容范围"
        if self._valid_region(region) is None:
            QMessageBox.warning(self, "未设置范围", f"当前 App 未设置{region_label}，无法进行小范围 OCR 测试。")
            return
        self.test_list_ocr_button.setEnabled(False)
        self.test_dialogue_ocr_button.setEnabled(False)
        source = "运行缓存" if (
            (mode == "list" and isinstance(config.get("_runtime_cache"), dict) and isinstance(config["_runtime_cache"].get("ocr_regions"), dict) and config["_runtime_cache"]["ocr_regions"].get("list_compact"))
            or (mode == "dialogue" and isinstance(config.get("_runtime_cache"), dict) and isinstance(config["_runtime_cache"].get("ocr_regions"), dict) and (config["_runtime_cache"]["ocr_regions"].get("dialogue_message") or config["_runtime_cache"]["ocr_regions"].get("chat_area_dynamic")))
        ) else "App配置"
        self._ocr_preview_region_text = ", ".join(f"{float(value):.3f}" for value in region)
        self.ocr_preview_summary.setText(
            f"正在执行 {mode} OCR（范围来源：{source}，区域=({self._ocr_preview_region_text})），请稍候..."
        )
        self.ocr_preview_output.clear()
        thread = OcrPreviewThread(self.current_screenshot, config, mode, region, app_name=self.app_name, parent=self)
        self._ocr_preview_thread = thread
        thread.completed.connect(self._apply_ocr_preview)
        thread.failed.connect(self._on_ocr_preview_failed)
        thread.finished.connect(self._finish_ocr_preview)
        thread.start()

    def _apply_ocr_preview(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_ocr_preview_failed("OCR 返回结果无效")
            return
        mode = str(payload.get("mode", "OCR"))
        result = payload.get("result", {})
        item_count = len(payload.get("ocr_items", [])) if isinstance(payload.get("ocr_items"), list) else 0
        region_text = getattr(self, "_ocr_preview_region_text", "-")
        self.ocr_preview_summary.setText(
            f"{mode} 小范围 OCR 完成：识别到 {item_count} 个文字块，实际区域=({region_text})。"
        )
        self.ocr_preview_output.setPlainText(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    def _on_ocr_preview_failed(self, message: str) -> None:
        self.ocr_preview_summary.setText(f"OCR 测试失败：{message}")

    def _finish_ocr_preview(self) -> None:
        thread = self._ocr_preview_thread
        self._ocr_preview_thread = None
        self.test_list_ocr_button.setEnabled(True)
        self.test_dialogue_ocr_button.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def _collect_saved_template_matches(self, image: Image.Image) -> tuple[list[tuple[str, str, str, str]], list[tuple[float, float]]]:
        """检测当前设备当前 App 的模板，只展示结果，不执行点击。"""
        rows: list[tuple[str, str, str, str]] = []
        points: list[tuple[float, float]] = []
        config = _deep_merge(_app_default_config(self.app_name), self.overrides)
        groups: list[tuple[str, str, list[Path], bool]] = []
        for rule in fixed_template_rules():
            root = Path(config.get("templates", {}).get(rule.group, ""))
            paths = back_button_template_paths(config) if rule.key == BACK_BUTTON else template_variant_paths(root, rule.filename)
            if paths:
                groups.append((rule.key, rule.label, paths, False))
        screens_root = Path(config.get("templates", {}).get("screens", ""))
        for key, label, filename in (("screen.no_phone", "无手机界面", "screen_无手机界面.jpg"), ("screen.black_screen", "黑屏界面", "screen_黑屏异常.jpg"), ("screen.call_window", "通话界面", "screen_通话界面.jpg")):
            paths = template_variant_paths(screens_root, filename)
            if paths:
                groups.append((key, label, paths, False))
        regions = config.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
        buttons_root = Path(config.get("templates", {}).get("buttons", ""))
        if isinstance(regions, dict):
            for key in sorted(str(item) for item in regions if str(item).startswith("button.custom.")):
                name = key[len("button.custom."):].strip()
                filenames = [f"btn_{name}.jpg"]
                if name == "相册":
                    filenames.append("btn_照片按钮.jpg")
                paths = [p for filename in filenames for p in template_variant_paths(buttons_root, filename)]
                if paths:
                    groups.append((key, f"自定义按钮 / {name}", paths, False))
        for path in popup_template_paths(config):
            groups.append((popup_template_region_key(path), f"弹窗 / {path.name}", [path], True))

        for key, label, paths, is_popup in groups:
            thresholds = config.get("template_match_thresholds", {})
            threshold = float(config.get("popup_match_threshold", 0.8) if is_popup else config.get("match_threshold", 0.7))
            if isinstance(thresholds, dict) and key in thresholds:
                threshold = float(thresholds[key])
            global_key = key in {"screen.no_phone", "screen.black_screen"}
            region = None if global_key else template_match_region(config, key)
            if region is None and not global_key:
                rows.append((label, "未设置范围", "-", f"- / {threshold:.2f}"))
                continue
            try:
                result = match_template_candidates_with_region(image, paths, config, key, threshold=threshold, match_func=match, single_match_func=match_template_with_region)
            except TemplateMatchRegionMissingError:
                result = None
            if result:
                x = float(result["x"]) + float(result["w"]) / 2
                y = float(result["y"]) + float(result["h"]) / 2
                points.append((x, y))
                rows.append((label, "命中", str(result.get("_template_name") or paths[0].name), f"{float(result.get('score', 0.0)):.3f} / {threshold:.2f}"))
            else:
                diagnostic = config.get("_runtime_cache", {}).get("template_match_diagnostics", {}).get(key)
                candidate = str(getattr(diagnostic, "template_name", "") or "-")
                score = getattr(diagnostic, "score", None)
                score_text = f"{float(score):.3f} / {threshold:.2f}" if score is not None else f"- / {threshold:.2f}"
                rows.append((label, "未命中", candidate, score_text))
        return rows, points

    def _detect_saved_templates(self, image: Image.Image) -> None:
        if self._template_preview_thread is not None and self._template_preview_thread.isRunning():
            return
        self.template_match_summary.setText("正在检测已保存模板，请稍候...")
        thread = TemplatePreviewThread(image, self._collect_saved_template_matches, self)
        self._template_preview_thread = thread
        thread.completed.connect(self._apply_template_preview)
        thread.failed.connect(self._on_template_preview_failed)
        thread.finished.connect(self._finish_template_preview)
        thread.start()

    def _apply_template_preview(self, payload: object) -> None:
        try:
            rows, points = payload
        except (TypeError, ValueError):
            self._on_template_preview_failed("模板检测结果无效")
            return
        self.template_match_table.setRowCount(len(rows))
        hit_count = 0
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(QColor("#16a34a") if value == "命中" else QColor("#ea580c"))
                self.template_match_table.setItem(row, column, item)
            hit_count += values[1] == "命中"
        self.template_match_summary.setText(f"模板检测完成：{hit_count}/{len(rows)} 类已命中。这里只显示结果，不会执行点击。")
        self.crop_label.set_match_points(points)

    def _on_template_preview_failed(self, message: str) -> None:
        self.template_match_summary.setText(f"模板检测失败：{message}")

    def _finish_template_preview(self) -> None:
        thread = self._template_preview_thread
        self._template_preview_thread = None
        if thread is not None:
            thread.deleteLater()

    def _save_current_screenshot_to_error_path(self) -> None:
        if self.current_screenshot is None:
            QMessageBox.warning(self, "未刷新截图", "请先刷新截图，再保存当前图片。")
            return
        error_dir = ensure_photo_root() / "errors"
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


def run_app(argv: list[str] | None = None, on_ready: Callable[[], Any] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    # 单实例锁（移植自 douyin）：客户机双开会让两组 worker 抢同一批机械臂。
    lock_path = str((Path("runtime") / "main_ui.lock").resolve())
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(lock_path)
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        return 0
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: Any) -> None:
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    window = MainWindow()
    window.show()
    if on_ready is not None:
        # 让窗口先完成首帧绘制，再执行可能加载大型 OCR runtime 的初始化。
        QTimer.singleShot(0, on_ready)
    try:
        return app.exec()
    except KeyboardInterrupt:
        app.quit()
        return 130
    finally:
        instance_lock.unlock()
        signal.signal(signal.SIGINT, previous_handler)
