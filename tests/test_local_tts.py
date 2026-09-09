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
