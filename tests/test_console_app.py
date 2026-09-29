"""控制台行为：原先由主窗口保证的启停、行状态和配置分层。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
import time

from services.config_manager import ConfigManager
from services.console_app import ConsoleApp, InstanceLock
from workers.device_worker import WorkerEvent


class _LiveWorker(threading.Thread):
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


def _console(tmp_path, devices, **kwargs):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"devices": devices}, ensure_ascii=False), encoding="utf-8")
    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    return ConsoleApp(manager, package_root=tmp_path, template_root=tmp_path / "templates", **kwargs)


def test_devices_are_sorted_by_ip(tmp_path):
    console = _console(
        tmp_path,
        [
            {"ip": "192.168.1.20", "selected_app": "xinyu"},
            {"ip": "192.168.1.3", "selected_app": "xinyu"},
        ],
    )
    assert [item["ip"] for item in console.list_devices()] == ["192.168.1.3", "192.168.1.20"]


def test_delete_device_removes_it(tmp_path):
    console = _console(
        tmp_path,
        [
            {"ip": "192.168.1.108", "selected_app": "xinyu"},
            {"ip": "192.168.1.109", "selected_app": "xinyu"},
        ],
    )
    assert console.delete_device("192.168.1.108") is True
    assert [item["ip"] for item in console.list_devices()] == ["192.168.1.109"]


def test_refresh_keeps_running_worker_status(tmp_path):
    worker = _LiveWorker()
    worker.start()
    try:
        console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
        console.workers["192.168.1.108"] = worker
        console.views["192.168.1.108"].status = "running"
        console.refresh_rows()
        assert console.list_devices()[0]["status"] == "running"
    finally:
        worker.release()


def test_refresh_marks_unregistered_device_stopped(tmp_path):
    console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
    console.refresh_rows()
    assert console.list_devices()[0]["status"] == "stopped"


def test_mislabeled_stopped_row_self_heals_on_tick(tmp_path):
    worker = _LiveWorker()
    worker.start()
    try:
        console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
        console.workers["192.168.1.108"] = worker
        console.views["192.168.1.108"].status = "stopped"
        console.handle_worker_event(
            WorkerEvent(
                "192.168.1.108",
                "status",
                "progress: 检查弹窗",
                {"current_step": "check_popup", "current_step_label": "检查弹窗"},
                worker.worker_id,
            )
        )
        assert console.list_devices()[0]["status"] == "running"
    finally:
        worker.release()


def test_dying_worker_late_tick_is_dropped(tmp_path):
    worker = _LiveWorker()
    worker.start()
    worker._stop.set()
    try:
        console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
        console.workers["192.168.1.108"] = worker
        console.views["192.168.1.108"].status = "stopped"
        console.handle_worker_event(
            WorkerEvent(
                "192.168.1.108",
                "status",
                "progress: 检查弹窗",
                {"current_step": "check_popup"},
                worker.worker_id,
            )
        )
        assert console.list_devices()[0]["status"] == "stopped"
    finally:
        worker.release()


def test_stale_stopped_event_does_not_overwrite_new_worker(tmp_path):
    current = _LiveWorker("new")
    current.start()
    try:
        console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
        console.workers["192.168.1.108"] = current
        console.views["192.168.1.108"].status = "running"
        console.handle_worker_event(WorkerEvent("192.168.1.108", "info", "stopped", {}, "old"))
        assert console.list_devices()[0]["status"] == "running"
    finally:
        current.release()


def test_start_while_stopping_is_rejected(tmp_path):
    worker = _LiveWorker()
    worker.start()
    worker._stop.set()
    try:
        console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
        console.workers["192.168.1.108"] = worker
        console.views["192.168.1.108"].status = "stopping"
        result = console.start_device("192.168.1.108")
        assert result["ok"] is False
        assert "已停止" in result["error"]
    finally:
        worker.release()


def test_duplicate_license_is_rejected(tmp_path):
    console = _console(
        tmp_path,
        [
            {"ip": "192.168.1.108", "license_key": "same-key", "selected_app": "xinyu"},
            {"ip": "192.168.1.109", "selected_app": "xinyu"},
        ],
    )
    result = console.bind_license("192.168.1.109", "same-key")
    assert result["ok"] is False


def test_settings_split_runtime_and_device_app(tmp_path):
    console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
    console.save_settings(
        "192.168.1.108",
        "xinyu",
        {
            "poll_interval": 9,
            "photo_sent_cooldown_seconds": 30,
            "persona_texts": {"基础设定": "只在测试里出现的人设"},
        },
    )
    runtime = console.config_manager.get_device_runtime_overrides("192.168.1.108")
    device_app = console.config_manager.get_device_app_overrides("192.168.1.108", "xinyu")
    assert runtime["poll_interval"] == 9
    assert device_app["photo_sent_cooldown_seconds"] == 30
    profile = Path(device_app["persona_profile_json"]).read_text(encoding="utf-8")
    assert "只在测试里出现的人设" in profile


def test_template_crop_is_saved_with_region(tmp_path):
    console = _console(tmp_path, [{"ip": "192.168.1.108", "selected_app": "xinyu"}])
    saved = console.save_template("xinyu", "buttons", "popup_custom", b"\xff\xd8jpeg", [0.1, 0.2, 0.3, 0.4])
    assert (tmp_path / "templates" / "xinyu" / "templates" / "buttons" / "popup_custom.jpg").is_file()
    regions = console.config_manager.get_app_overrides("xinyu").get("template_match_regions", {})
    assert regions["popup.popup_custom"] == [0.1, 0.2, 0.3, 0.4]
    assert saved["name"] == "popup_custom.jpg"


def test_scan_adds_new_devices(tmp_path):
    console = _console(tmp_path, [], scan_func=lambda timeout, max_workers: ["10.0.0.2", "10.0.0.8"])
    console.scan_devices()
    deadline = time.time() + 2
    while time.time() < deadline and len(console.list_devices()) < 2:
        console.drain_once()
        time.sleep(0.01)
    assert [item["ip"] for item in console.list_devices()] == ["10.0.0.2", "10.0.0.8"]
    assert "新增 2 台" in console.status_snapshot()["scan_message"]


def test_ocr_mode_label_follows_install_state(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "install_state.json").write_text(
        json.dumps({"backend": "cpu", "ocr_device": "cpu"}),
        encoding="utf-8",
    )
    console = _console(tmp_path, [])
    assert console.status_snapshot()["ocr_mode"] == "OCR: CPU"
    (runtime / "install_state.json").write_text(
        json.dumps({"backend": "paddle_cuda", "ocr_device": "gpu:0"}),
        encoding="utf-8",
    )
    assert console.status_snapshot()["ocr_mode"] == "OCR: GPU"


def test_instance_lock_rejects_a_second_holder(tmp_path):
    first = InstanceLock(tmp_path / "main_ui.lock")
    second = InstanceLock(tmp_path / "main_ui.lock")
    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
    assert second.acquire() is True
    second.release()
