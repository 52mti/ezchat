from __future__ import annotations

from services import local_tts


class _Response:
    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], float]] = []

    def post(self, url: str, *, data: dict[str, str], timeout: float) -> _Response:
        self.calls.append(("POST", url, data, timeout))
        return _Response()


def test_bind_device_tts_logs_in_and_saves_local_url(monkeypatch) -> None:
    session = _Session()
    monkeypatch.setattr(local_tts.requests, "Session", lambda: session)
    monkeypatch.setattr(local_tts, "tts_url_for_device", lambda _ip, port: f"http://192.168.1.96:{port}/tts")

    result = local_tts.bind_device_tts("192.168.1.210", port=9999, timeout=3.0)

    assert result == "http://192.168.1.96:9999/tts"
    assert session.calls == [
        ("POST", "http://192.168.1.210:8888/login", {"password": "admin"}, 3.0),
        ("POST", "http://192.168.1.210:8888/save_sys_config", {"tts_url": result}, 3.0),
    ]


def test_bind_device_tts_uses_30_second_default_timeout(monkeypatch) -> None:
    session = _Session()
    monkeypatch.setattr(local_tts.requests, "Session", lambda: session)
    monkeypatch.setattr(local_tts, "tts_url_for_device", lambda _ip, port: f"http://192.168.1.96:{port}/tts")

    local_tts.bind_device_tts("192.168.1.210")

    assert [call[-1] for call in session.calls] == [30.0, 30.0]


def test_tts_url_uses_device_route_local_ip(monkeypatch) -> None:
    monkeypatch.setattr(local_tts, "local_ip_for_device", lambda _ip: "192.168.1.96")

    assert local_tts.tts_url_for_device("192.168.1.210", port=9988) == "http://192.168.1.96:9988/tts"


def test_expected_server_version_matches_tts_server() -> None:
    from tts.server import SERVER_VERSION

    assert local_tts.EXPECTED_SERVER_VERSION == SERVER_VERSION


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True


def _server_with_health(monkeypatch, tmp_path, payload: dict) -> tuple[local_tts.LocalTtsServer, list[str]]:
    server = local_tts.LocalTtsServer(port=9999, root=tmp_path)
    events: list[str] = []

    monkeypatch.setattr(server, "_health_payload", lambda: payload)
    monkeypatch.setattr(server, "_stop_port_orphans", lambda: events.append("swept"))
    monkeypatch.setattr(
        local_tts.subprocess,
        "Popen",
        lambda *args, **kwargs: events.append("spawned") or _FakeProcess(),
    )
    return server, events


def test_start_reuses_ready_server_with_matching_version(monkeypatch, tmp_path) -> None:
    _server, events = _server_with_health(
        monkeypatch, tmp_path, {"status": "ok", "server_version": local_tts.EXPECTED_SERVER_VERSION}
    )
    _server.start()
    assert events == []


def test_start_replaces_ready_server_from_old_version(monkeypatch, tmp_path) -> None:
    _server, events = _server_with_health(monkeypatch, tmp_path, {"status": "ok"})
    _server.start()
    assert events == ["swept", "spawned"]


def test_stop_sweeps_port_orphans(monkeypatch) -> None:
    server = local_tts.LocalTtsServer(port=9999)
    swept: list[str] = []
    monkeypatch.setattr(server, "_stop_port_orphans", lambda: swept.append("swept"))
    server.stop()
    assert swept == ["swept"]
