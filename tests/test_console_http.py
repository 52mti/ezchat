"""本机控制台 HTTP 接口。"""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from services.config_manager import ConfigManager
from services.console_app import ConsoleApp
from server.console_server import create_app


class _FakeWorker:
    def __init__(self, device, events=None, config_manager=None) -> None:
        self.device = device
        self.events = events
        self.worker_id = "http-worker"
        self._stop = threading.Event()
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self.started and not self._stop.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()


def _client(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"devices": [{"ip": "192.168.1.108", "selected_app": "xinyu"}]}),
        encoding="utf-8",
    )
    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    console = ConsoleApp(
        manager,
        package_root=tmp_path,
        template_root=tmp_path / "templates",
        worker_factory=_FakeWorker,
        local_tts_server=_SilentTts(),
    )
    return TestClient(create_app(console)), console


class _SilentTts:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def bind_devices(self, device_ips):
        return [(ip, "http://127.0.0.1:9999/tts", "") for ip in device_ips]


def test_device_list_and_lifecycle(tmp_path):
    client, console = _client(tmp_path)
    listed = client.get("/api/devices")
    assert listed.status_code == 200
    assert listed.json()["devices"][0]["ip"] == "192.168.1.108"
    assert listed.json()["devices"][0]["status"] == "stopped"

    added = client.post("/api/devices", json={"ip": "192.168.1.20"})
    assert added.status_code == 200
    duplicate = client.post("/api/devices", json={"ip": "192.168.1.20"})
    assert duplicate.status_code == 409

    started = client.post("/api/devices/192.168.1.108/start")
    assert started.status_code == 200
    assert console.workers["192.168.1.108"].started is True
    rows = {item["ip"]: item for item in client.get("/api/devices").json()["devices"]}
    assert rows["192.168.1.108"]["status"] == "running"

    stopped = client.post("/api/devices/192.168.1.108/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopping"

    removed = client.delete("/api/devices/192.168.1.20")
    assert removed.status_code == 200
    ips = [item["ip"] for item in removed.json()["devices"]]
    assert "192.168.1.20" not in ips


def test_settings_and_template_endpoints(tmp_path):
    client, _console = _client(tmp_path)
    saved = client.put(
        "/api/settings",
        json={
            "device_ip": "192.168.1.108",
            "app_name": "xinyu",
            "values": {"poll_interval": 6, "judge_enabled": True},
        },
    )
    assert saved.status_code == 200
    assert saved.json()["runtime"]["poll_interval"] == 6

    created = client.post(
        "/api/templates",
        json={
            "app_name": "xinyu",
            "group": "buttons",
            "name": "返回按钮",
            "image_base64": "aGVsbG8=",
            "region": [0.1, 0.2, 0.3, 0.4],
        },
    )
    assert created.status_code == 200
    listed = client.get("/api/templates", params={"app_name": "xinyu"})
    names = [item["name"] for item in listed.json()["templates"]]
    assert "返回按钮.jpg" in names


def test_events_websocket_sends_current_devices(tmp_path):
    client, _console = _client(tmp_path)
    with client.websocket_connect("/api/events") as websocket:
        payload = websocket.receive_json()
    assert payload["type"] == "devices"
    assert payload["devices"][0]["ip"] == "192.168.1.108"


def test_monitor_image_uses_published_frame(tmp_path):
    client, _console = _client(tmp_path)
    from services.monitor_feed import publish_capture, reset_for_tests

    reset_for_tests()
    publish_capture("192.168.1.108", "xinyu", b"\xff\xd8\xffframe", label="列表")
    meta = client.get("/api/monitor")
    assert meta.status_code == 200
    assert meta.json()["frames"][0]["has_image"] is True
    image = client.get("/api/monitor/192.168.1.108.jpg")
    assert image.status_code == 200
    assert image.content.startswith(b"\xff\xd8")
    reset_for_tests()
