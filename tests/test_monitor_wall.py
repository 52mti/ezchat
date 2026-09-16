from __future__ import annotations

"""监控墙（docs/13 主UI 扩展）：数据源限频/设备过滤 + 窗口只显示列表内设备。"""

import time

from services import monitor_feed


def _png_bytes() -> bytes:
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    image = QImage(120, 240, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert QPainter(image).end() or True
    return bytes(image_to_png(image, app))


def image_to_png(image, app):  # pragma: no cover - 辅助
    from PyQt6.QtCore import QBuffer, QIODevice

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return buffer.data().data()


def test_monitor_feed_throttles_per_device():
    monitor_feed.reset_for_tests()
    image = b"\x89PNG-fake-bytes"
    monitor_feed.publish_capture("1.1.1.1", "xinyu", image, label="a")
    monitor_feed.publish_capture("1.1.1.1", "xinyu", b"different", label="b")  # 1 秒内被限频
    monitor_feed.publish_capture("2.2.2.2", "xinyu", image)
    snaps = monitor_feed.snapshots()
    assert set(snaps) == {"1.1.1.1", "2.2.2.2"}
    assert snaps["1.1.1.1"]["label"] == "a" and snaps["1.1.1.1"]["image"] == image


def test_monitor_feed_throttle_expires(monkeypatch):
    monitor_feed.reset_for_tests()
    base = time.time()
    clock = [base]
    monkeypatch.setattr(monitor_feed.time, "time", lambda: clock[0])
    monitor_feed.publish_capture("1.1.1.1", "xinyu", b"one")
    clock[0] = base + 1.5
    monitor_feed.publish_capture("1.1.1.1", "xinyu", b"two")
    assert monitor_feed.snapshots()["1.1.1.1"]["image"] == b"two"


def test_monitor_feed_retain_devices_drops_removed():
    monitor_feed.reset_for_tests()
    monitor_feed.publish_capture("1.1.1.1", "xinyu", b"img")
    monitor_feed.publish_capture("2.2.2.2", "xinyu", b"img")
    monitor_feed.retain_devices(["2.2.2.2"])
    assert set(monitor_feed.snapshots()) == {"2.2.2.2"}


def test_photo_capture_publishes_to_monitor_feed(tmp_path, monkeypatch):
    """capture_transform_image 成功路径会发布最近一帧（worker 请求即数据源）。"""
    monitor_feed.reset_for_tests()

    class FakeRobot:
        host = "9.9.9.9"

        def get_screenshot(self):
            class Shot:
                def __init__(self):
                    from PyQt6.QtGui import QImage
                    from PyQt6.QtWidgets import QApplication

                    QApplication.instance() or QApplication([])
                    img = QImage(60, 120, QImage.Format.Format_RGB32)
                    img.fill(0x112233)

                    from PyQt6.QtCore import QBuffer, QIODevice

                    buf = QBuffer()
                    buf.open(QIODevice.OpenModeFlag.WriteOnly)
                    img.save(buf, "PNG")
                    self._png = buf.data().data()

                def as_png(self):
                    return self._png

                def as_array(self):
                    import numpy as np

                    return np.zeros((120, 60, 3), dtype="uint8")

            return Shot()

    from services import photo_capture

    config = {"_runtime": {"device_ip": "9.9.9.9", "app_name": "xinyu"}, "capture": {}}
    monkeypatch.setattr(photo_capture, "_sync_coordinate_cache", lambda *a, **k: None)
    monkeypatch.setattr(photo_capture, "_sync_photo_result_cache", lambda *a, **k: None)
    monkeypatch.setattr(photo_capture, "_update_phone_screen_health", lambda *a, **k: None)
    monkeypatch.setattr(
        photo_capture,
        "_capture_local_perspective_image",
        lambda robot, **kwargs: b"\x89PNG-real-frame",
    )
    photo_capture.capture_transform_image(FakeRobot(), config, label="unit")
    snaps = monitor_feed.snapshots()
    assert "9.9.9.9" in snaps and snaps["9.9.9.9"]["image"] == b"\x89PNG-real-frame"
    assert snaps["9.9.9.9"]["app"] == "xinyu" and snaps["9.9.9.9"]["label"] == "unit"


def test_monitor_wall_shows_only_listed_devices(qtbot, tmp_path):
    from ui.monitor_wall import MonitorWallWidget

    monitor_feed.reset_for_tests()
    image = _png_bytes()
    monitor_feed.publish_capture("1.1.1.1", "xinyu", image)
    monitor_feed.publish_capture("2.2.2.2", "xinyu", image)
    monitor_feed.publish_capture("3.3.3.3", "xinyu", image)

    devices = [{"ip": "1.1.1.1"}, {"ip": "3.3.3.3"}]  # 列表中没有 2.2.2.2
    dialog = MonitorWallWidget(lambda: devices)
    qtbot.addWidget(dialog)
    dialog.refresh()

    assert set(dialog._tiles) == {"1.1.1.1", "3.3.3.3"}
    assert "2.2.2.2" not in monitor_feed.snapshots()  # 已被 retain_devices 清掉
    tile = dialog._tiles["1.1.1.1"]
    # 解码已移至后台线程（经 Qt 信号回主线程），等待信号送达再断言。
    from PyQt6.QtCore import QTimer

    for _ in range(100):  # 最多 ~2s
        app = QTimer.__module__ and None
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()
        pixmap = tile.image_label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            break
        time.sleep(0.02)
    pixmap = tile.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()

    devices.remove({"ip": "3.3.3.3"})
    dialog.refresh()
    assert set(dialog._tiles) == {"1.1.1.1"}
