from __future__ import annotations

"""监控墙：设备网格 + 每设备最近一帧 worker 截图（docs/13 主UI 扩展）。

- 照片来源：worker 每次截图请求（photo_capture → monitor_feed，限频 1s/设备）
- 只显示设备列表中存在的设备，设备增删即时反映
- 形态：独立弹窗（Qt.Tool，无任务栏图标），吸附在主窗口右侧外侧——
  主界面保持原尺寸不被压缩；主窗口移动/缩放时监控墙跟随
- 渲染结构：JPEG 解码与缩放在后台线程完成（每设备只保留最新帧，旧帧
  直接丢弃防积压），经 Qt 信号把缩放后的 QImage 送回主线程——主线程
  只做 QPixmap 转换与 setPixmap（毫秒级）。旧实现在 UI 线程逐帧"全尺寸
  解码 + 平滑缩放"，每台每秒占用主线程 30~80ms，多台时是 UI 卡顿源。
"""

import threading
from typing import Any, Callable

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

TILE_IMAGE_WIDTH = 360
# 画面更新已改为"截图即达"（monitor_feed 订阅推送，见 _CaptureSignals）；
# 定时器只负责发现设备列表的增删，低频即可。
DEVICE_SYNC_INTERVAL_MS = 5000
TILE_COLUMN_STRIDE = TILE_IMAGE_WIDTH + 40
DEFAULT_WALL_WIDTH = TILE_COLUMN_STRIDE * 2  # 默认两列，一行行排布


class _CaptureSignals(QObject):
    # worker 线程 emit，Qt 自动排队投递到 UI 线程。
    frame = pyqtSignal(str)


class _DecodedImageSignals(QObject):
    # QImage 可跨线程传递；QPixmap 只能在主线程构造，因此回程只带 QImage。
    ready = pyqtSignal(str, QImage)


class _FrameDecodeWorker(threading.Thread):
    """后台帧解码：按设备覆盖式取最新帧，缩放解码后信号回主线程。"""

    def __init__(self, signals: _DecodedImageSignals) -> None:
        super().__init__(daemon=True, name="monitor-wall-decode")
        self._signals = signals
        self._latest: dict[str, bytes] = {}
        self._wake = threading.Event()

    def submit(self, device_ip: str, data: bytes) -> None:
        if not data:
            return
        # 覆盖旧帧：解码速度跟不上发布速度时丢中间帧，只保证显示最新。
        self._latest[device_ip] = data
        self._wake.set()

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            while self._latest:
                ip, data = self._latest.popitem()
                image = self._decode_scaled(data)
                if image is not None and not image.isNull():
                    self._signals.ready.emit(ip, image)

    @staticmethod
    def _decode_scaled(data: bytes) -> QImage | None:
        # QImage 可在非 GUI 线程构造/操作（QPixmap 不行，因此回程只带
        # QImage）。fromData + scaledToWidth 是旧主线程实现的原组合，
        # 行为一致，仅换线程执行。
        image = QImage.fromData(data)
        if image.isNull():
            return None
        if image.width() != TILE_IMAGE_WIDTH:
            image = image.scaledToWidth(TILE_IMAGE_WIDTH, Qt.TransformationMode.SmoothTransformation)
        return image


class _DeviceTile(QFrame):
    def __init__(self, device_ip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InsetPanel")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)
        self.header_label = QLabel(device_ip)
        self.header_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.header_label)
        self.image_label = QLabel("等待画面…")
        self.image_label.setFixedWidth(TILE_IMAGE_WIDTH)
        self.image_label.setMinimumHeight(int(TILE_IMAGE_WIDTH * 1.9))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #2B2F36; color: #8A8A8A;")
        root.addWidget(self.image_label)
        self._last_ts = 0.0

    def update_snapshot(
        self,
        device_ip: str,
        data: dict[str, Any] | None,
        decode_submit: Callable[[str, bytes], None],
    ) -> None:
        """主线程只更新头部文本；图像字节交后台解码。"""
        if not data:
            return
        ts = float(data.get("ts", 0.0) or 0.0)
        if ts <= self._last_ts:
            return
        self._last_ts = ts
        from datetime import datetime

        updated_at = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        app = str(data.get("app", "") or "")
        label = str(data.get("label", "") or "")
        header = f"{device_ip}" + (f" · {app}" if app else "") + f" · {updated_at}" + (f" · {label}" if label else "")
        self.header_label.setText(header)
        decode_submit(device_ip, bytes(data.get("image") or b""))

    def apply_decoded_image(self, image: QImage) -> None:
        # 主线程执行（经信号排队）：QPixmap 构造 + setPixmap，毫秒级。
        self.image_label.setPixmap(QPixmap.fromImage(image))


class MonitorWallWidget(QWidget):
    """停靠面板内容：由 MainWindow 装进 QDockWidget 挂在右侧。"""

    def __init__(
        self,
        devices_provider: Callable[[], list[dict[str, Any]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.devices_provider = devices_provider
        self._tiles: dict[str, _DeviceTile] = {}
        self._wanted: dict[str, dict[str, Any]] = {}

        self._decode_signals = _DecodedImageSignals()
        self._decode_signals.ready.connect(self._on_image_ready)
        self._decoder = _FrameDecodeWorker(self._decode_signals)
        self._decoder.start()

        # 画面更新：worker 截图发布即推送（截图才更新，不再定时轮询图像）。
        self._capture_signals = _CaptureSignals()
        self._capture_signals.frame.connect(self._on_capture_published)
        from services import monitor_feed as _feed_init

        _feed_init.add_capture_listener(self._capture_signals.frame.emit)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        hint = QLabel("画面来源：worker 每次截图请求（每设备限频 1 秒）· 仅显示设备列表中的设备")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8A8A8A;")
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(DEVICE_SYNC_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)

    def _on_capture_published(self, device_ip: str) -> None:
        """worker 截图已发布：立即把该设备的最新帧交给后台解码（UI 线程执行，
        经 Qt 跨线程信号排队到达）。墙隐藏时跳过，避免无视觉意义的解码。"""
        if not self.isVisible():
            return
        from services import monitor_feed

        tile = self._tiles.get(device_ip)
        if tile is None:
            return
        snap = monitor_feed.snapshots().get(device_ip)
        if snap:
            tile.update_snapshot(device_ip, snap, self._decoder.submit)

    def _on_image_ready(self, device_ip: str, image: QImage) -> None:
        tile = self._tiles.get(device_ip)
        if tile is not None:
            tile.apply_decoded_image(image)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().showEvent(event)
        self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        # 列数随窗口宽变化：旧实现只在设备集合变化时重排，窗口拖宽列数
        # 也不会变（默认宽度 420 只有一列，被用户看到"竖着排一列"）。
        columns = self._columns()
        if columns != getattr(self, "_layout_columns", None) and self._tiles:
            self._relayout(columns)

    def _columns(self) -> int:
        return max(1, self.width() // TILE_COLUMN_STRIDE)

    def _relayout(self, columns: int) -> None:
        for tile in self._tiles.values():
            self.grid.removeWidget(tile)
        for index, tile in enumerate(self._tiles.values()):
            self.grid.addWidget(tile, index // columns, index % columns)
        self._layout_columns = columns

    def _sync_devices(self) -> None:
        devices = self.devices_provider() or []
        wanted: dict[str, dict[str, Any]] = {}
        for device in devices:
            ip = str(device.get("ip", "") or "").strip() if isinstance(device, dict) else ""
            if ip:
                wanted[ip] = device
        self._wanted = wanted

    def refresh(self) -> None:
        from services import monitor_feed

        self._sync_devices()
        wanted = self._wanted
        monitor_feed.retain_devices(wanted.keys())
        snaps = monitor_feed.snapshots()

        for ip in list(self._tiles):
            if ip not in wanted:
                tile = self._tiles.pop(ip)
                self.grid.removeWidget(tile)
                tile.deleteLater()

        layout_changed = False
        ordered = list(wanted.keys())
        for index, ip in enumerate(ordered):
            tile = self._tiles.get(ip)
            if tile is None:
                tile = _DeviceTile(ip)
                self._tiles[ip] = tile
                layout_changed = True
            tile.update_snapshot(ip, snaps.get(ip), self._decoder.submit)
            if layout_changed or index >= self.grid.count():
                self.grid.addWidget(tile, index // self._columns(), index % self._columns())
        if layout_changed or len(self._tiles) != self.grid.count():
            self._relayout(self._columns())


class MonitorWallWindow(QWidget):
    """吸附式独立弹窗：贴在主窗口右侧外侧，跟随主窗口移动/缩放，不压缩主界面。"""

    def __init__(self, devices_provider: Callable[[], list[dict[str, Any]]], main_window: QWidget) -> None:
        super().__init__(main_window)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.Window)
        self.setWindowTitle("监控墙")
        self._main = main_window
        main_window.installEventFilter(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        hint = QLabel("画面来源：worker 每次截图请求（每设备限频 1 秒）· 仅显示设备列表中的设备")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8A8A8A;")
        root.addWidget(hint)

        self.wall = MonitorWallWidget(devices_provider)
        root.addWidget(self.wall, 1)
        self.resize(DEFAULT_WALL_WIDTH, max(480, main_window.height()))
        self._sync_position()

    def _sync_position(self) -> None:
        frame = self._main.frameGeometry()
        geo = self._main.geometry()
        title_bar = max(0, geo.top() - frame.top())
        self.setGeometry(frame.right() + 1, geo.top() - title_bar, self.width(), geo.height())

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt 命名
        if obj is self._main and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            if self.isVisible():
                self._sync_position()
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self.wall._timer.stop()
        super().closeEvent(event)
