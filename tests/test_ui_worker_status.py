"""行状态回归：活 worker 不得被 UI 误标成"已停止"后锁死。

现场症状（客户长跑）：worker 线程正常跑（日志照常刷、机器照常动），
但设备行显示"已停止"。根因是表格重建（扫描/增删设备）把所有行硬置
stopped，且 tick 守卫会把活 worker 的状态事件全部丢弃，错误状态永远
无法自愈。
"""

from __future__ import annotations

import json
import threading

import pytest

from services.config_manager import ConfigManager
from ui.main_window import MainWindow
from workers.device_worker import WorkerEvent


class _LiveWorker(threading.Thread):
    """行为对齐 DeviceWorker 最小面的替身：存活、可停、可暂停。"""

    def __init__(self, worker_id: str = "w-test") -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self._release = threading.Event()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def run(self) -> None:
        self._release.wait(10)

    def release(self) -> None:
        self._release.set()


def _window(qtbot, tmp_path, devices):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"devices": devices}, ensure_ascii=False), encoding="utf-8")
    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    window = MainWindow(manager)
    qtbot.addWidget(window)
    return window


@pytest.fixture()
def live_worker():
    worker = _LiveWorker()
    worker.start()
    yield worker
    worker.release()


def test_table_rebuild_keeps_running_worker_status(qtbot, tmp_path, live_worker):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    window.workers["192.168.1.108"] = live_worker
    window._set_row_status("192.168.1.108", "running", "扫描列表")

    window._load_devices()

    assert window._row_status_for_device("192.168.1.108") == "running"


def test_table_rebuild_marks_unregistered_device_stopped(qtbot, tmp_path):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])

    window._load_devices()

    assert window._row_status_for_device("192.168.1.108") == "stopped"


def test_mislabeled_stopped_row_self_heals_on_tick(qtbot, tmp_path, live_worker):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    window.workers["192.168.1.108"] = live_worker
    window._set_row_status("192.168.1.108", "stopped")

    window._handle_worker_event(
        WorkerEvent("192.168.1.108", "status", "progress: 检查弹窗", {"current_step": "check_popup"}, live_worker.worker_id)
    )

    assert window._row_status_for_device("192.168.1.108") == "running"


def test_dying_worker_late_tick_still_dropped(qtbot, tmp_path, live_worker):
    window = _window(qtbot, tmp_path, [{"ip": "192.168.1.108", "selected_app": "wechat"}])
    live_worker._stop.set()
    window.workers["192.168.1.108"] = live_worker
    window._set_row_status("192.168.1.108", "stopped")

    window._handle_worker_event(
        WorkerEvent("192.168.1.108", "status", "progress: 检查弹窗", {"current_step": "check_popup"}, live_worker.worker_id)
    )

    assert window._row_status_for_device("192.168.1.108") == "stopped"
