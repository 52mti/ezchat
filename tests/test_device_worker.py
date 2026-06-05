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


class StartupCheckApp(DummyApp):
    def set_runtime_step(self, step, label, log_label=None):
        if hasattr(self, "status_callback"):
            payload = {"current_step": step, "current_step_label": label}
            if log_label is not None:
                payload["current_step_log_label"] = log_label
            self.status_callback(payload)


class RaisingApp(DummyApp):
    def __init__(self, robot, config, reply_provider=None, sent_callback=None):
        super().__init__(robot, config, reply_provider=reply_provider, sent_callback=sent_callback)
        self.circuit_breaker_threshold = 1

    def run_once(self):
        raise RuntimeError("boom")

    def get_status(self):
        return {
            "current_step": "raising_step",
            "current_step_label": "Raising Step",
            "error_count": self.status.error_count,
            "is_circuit_open": self.status.is_circuit_open,
            "last_error": getattr(self.status, "last_error", ""),
        }


class CountingApp(DummyApp):
    def __init__(self, robot, config, reply_provider=None, sent_callback=None):
        super().__init__(robot, config, reply_provider=reply_provider, sent_callback=sent_callback)
        self.count = 0

    def run_once(self):
        self.count += 1

    def get_status(self):
        return {"is_circuit_open": False, "current_step": "counting", "current_step_label": "Counting"}


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


def test_worker_stops_on_startup_when_back_button_template_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": StartupCheckApp})
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={
            "poll_interval": 1,
            "robot_monitor": {"enabled": False},
            "templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)},
        },
        events=events,
        config_manager=manager,
        robot=object(),
    )

    worker.run()

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())

    assert any(event.message == "没有返回按钮" and event.level == "error" for event in emitted)
    assert any(event.message == "progress: 没有返回按钮" for event in emitted)
    assert emitted[-1].message == "stopped"


def test_worker_startup_back_button_check_accepts_app_template(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"dummy": DummyApp})
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    buttons_dir.mkdir()
    screens_dir.mkdir()
    (buttons_dir / "btn_返回按钮.jpg").write_bytes(b"template")
    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "dummy"},
        app_config={
            "poll_interval": 1,
            "robot_monitor": {"enabled": False},
            "templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)},
        },
        events=queue.Queue(),
        config_manager=manager,
        robot=object(),
    )

    assert worker._required_back_button_templates_available() is True


def test_worker_exception_is_isolated_to_that_device(tmp_path, monkeypatch):
    monkeypatch.setattr(device_worker_module, "get_registered_apps", lambda: {"bad": RaisingApp, "good": CountingApp})
    monkeypatch.setattr(device_worker_module, "back_button_template_paths", lambda _config: ["template"])
    events: "queue.Queue" = queue.Queue()
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    bad_worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "bad"},
        app_config={"poll_interval": 0.01, "robot_monitor": {"enabled": False}},
        events=events,
        config_manager=manager,
        robot=object(),
    )
    good_worker = DeviceWorker(
        {"ip": "192.168.1.109", "selected_app": "good"},
        app_config={"poll_interval": 0.01, "robot_monitor": {"enabled": False}},
        events=events,
        config_manager=manager,
        robot=object(),
    )

    bad_worker.start()
    good_worker.start()
    deadline = time.monotonic() + 2.0
    emitted = []
    try:
        while time.monotonic() < deadline:
            while not events.empty():
                emitted.append(events.get_nowait())
            bad_error = any(event.device_ip == "192.168.1.108" and event.message == "app_error" for event in emitted)
            good_tick = any(event.device_ip == "192.168.1.109" and event.message == "tick" for event in emitted)
            if bad_error and good_tick:
                break
            time.sleep(0.01)
    finally:
        bad_worker.stop()
        good_worker.stop()
        bad_worker.join(timeout=1)
        good_worker.join(timeout=1)

    assert any(event.device_ip == "192.168.1.108" and event.message == "app_error" for event in emitted)
    assert any(event.device_ip == "192.168.1.109" and event.message == "tick" for event in emitted)
    assert good_worker.app.count > 0
