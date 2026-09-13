from __future__ import annotations

"""监控墙：设备网格 + 每设备最近一帧 worker 截图（docs/13 主UI 扩展）。

- 照片来源：worker 每次截图请求（photo_capture → monitor_feed，限频 1s/设备）
- 只显示设备列表中存在的设备，设备增删即时反映
- 形态：独立弹窗（Qt.Tool，无任务栏图标），吸附在主窗口右侧外侧——
  主界面保持原尺寸不被压缩；主窗口移动/缩放时监控墙跟随
"""

from typing import Any, Callable

from PyQt6.QtCore import QEvent, Qt, QTimer
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
REFRESH_INTERVAL_MS = 1000
DEFAULT_WALL_WIDTH = 420


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

    def update_snapshot(self, device_ip: str, data: dict[str, Any] | None) -> None:
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
        image = QImage.fromData(data.get("image") or b"")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaledToWidth(TILE_IMAGE_WIDTH, Qt.TransformationMode.SmoothTransformation)
        )


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
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().showEvent(event)
        self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._timer.stop()
        super().hideEvent(event)

    def _columns(self) -> int:
        return max(1, self.width() // (TILE_IMAGE_WIDTH + 40))

    def refresh(self) -> None:
        from services import monitor_feed

        devices = self.devices_provider() or []
        wanted: dict[str, dict[str, Any]] = {}
        for device in devices:
            ip = str(device.get("ip", "") or "").strip() if isinstance(device, dict) else ""
            if ip:
                wanted[ip] = device
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
            tile.update_snapshot(ip, snaps.get(ip))
            if layout_changed or index >= self.grid.count():
                self.grid.addWidget(tile, index // self._columns(), index % self._columns())
        if layout_changed or len(self._tiles) != self.grid.count():
            # 设备集合变化后整体重排（先全部摘下再按序放回）
            for tile in self._tiles.values():
                self.grid.removeWidget(tile)
            for index, tile in enumerate(self._tiles.values()):
                self.grid.addWidget(tile, index // self._columns(), index % self._columns())


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
