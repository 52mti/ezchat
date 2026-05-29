from __future__ import annotations

import queue
import time
from types import SimpleNamespace

from services.config_manager import ConfigManager
from workers import device_worker as device_worker_module
from workers.device_worker import DeviceWorker


DEFAULT_PERSONA_ID = "REMOVED-BEARER-TOKEN"


class DummyApp:
    name = "dummy"

    def __init__(self, robot, config, reply_provider=None, sent_callback=None):
        self.robot = robot
        self.config = config
        self.status = SimpleNamespace(is_circuit_open=False, error_count=0)

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_log_callback(self, callback):
        self.log_callback = callback

    def get_status(self):
        return {"is_circuit_open": False}

    def run_once(self):
        return None


class PauseRequestedApp(DummyApp):
    def __init__(self, robot, config, reply_provider=None, sent_callback=None):
        super().__init__(robot, config, reply_provider, sent_callback)
        self.status = SimpleNamespace(is_circuit_open=False, error_count=0, pause_reason="")

    def get_status(self):
        return {"is_circuit_open": False, "pause_reason": self.status.pause_reason}

    def run_once(self):
        self.status.pause_reason = "重复消息保护"


def test_worker_suppresses_app_status_after_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": DummyApp})
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    robot = object()
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={"poll_interval": 1},
        events=events,
        config_manager=manager,
        robot=robot,
    )

    assert worker.app_config["_runtime"]["should_stop"]() is False
    assert worker.app_config["poll_interval"] == 1
    assert worker.app_config["keep_awake_interval_seconds"] == 200
    assert worker.app_config["persona"]["requestPersona"] == DEFAULT_PERSONA_ID
    worker.stop()
    assert worker.app_config["_runtime"]["should_stop"]() is True
    worker._emit_app_status({"current_step_label": "列表OCR扫描"})

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())

    assert [event.message for event in emitted] == ["stopping"]


def test_worker_emits_runtime_step_log_with_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": DummyApp})
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={"poll_interval": 1},
        events=events,
        config_manager=manager,
        robot=object(),
    )

    worker._emit_app_status(
        {
            "current_step": "enter_chat",
            "current_step_label": "进入聊天",
            "current_step_log_label": "进入聊天：无敌阿强 (0.123, 0.456)",
        }
    )

    first = events.get_nowait()
    second = events.get_nowait()
    assert first.message == "runtime_step"
    assert first.payload["detail"] == "进入聊天：无敌阿强 (0.123, 0.456)"
    assert second.level == "status"
    assert second.payload["current_step_label"] == "进入聊天"


def test_worker_uses_idle_poll_interval_for_empty_list_target(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": DummyApp})
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={"poll_interval": 1, "idle_poll_interval_seconds": 5},
        events=queue.Queue(),
        config_manager=manager,
        robot=object(),
    )

    assert worker._poll_interval_for_status({"current_step": "empty_list_target"}, 1) == 5
    assert worker._poll_interval_for_status({"current_step": "enter_chat"}, 1) == 1


def test_worker_pause_can_emit_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": DummyApp})
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={"poll_interval": 1},
        events=events,
        config_manager=manager,
        robot=object(),
    )

    worker.pause("黑屏熔断")
    event = events.get_nowait()

    assert event.message == "paused"
    assert event.payload == {"detail": "黑屏熔断"}


def test_worker_pauses_when_app_requests_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": PauseRequestedApp})
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={"poll_interval": 30},
        events=events,
        config_manager=manager,
        robot=object(),
    )

    worker.start()
    emitted = []
    for _ in range(50):
        while not events.empty():
            emitted.append(events.get_nowait())
        if any(event.message == "paused" for event in emitted):
            break
        time.sleep(0.02)
    worker.stop()
    worker.join(timeout=2)

    while not events.empty():
        emitted.append(events.get_nowait())
    paused = [event for event in emitted if event.message == "paused"]
    assert paused
    assert paused[0].payload == {"detail": "重复消息保护"}
