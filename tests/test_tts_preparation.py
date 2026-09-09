from __future__ import annotations

from apps.utils import text_input
from fastapi.testclient import TestClient

from tts.server import _audio_filename, create_app


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"url": "http://192.168.1.9:9999/static/audio.mp3"}


class _Robot:
    host = "192.168.1.210"

    def __init__(self) -> None:
        self.played: list[tuple[str, bool]] = []

    def play_audio(self, text: str, wait: bool = True) -> None:
        self.played.append((text, wait))


def test_prepare_tts_generates_each_chunk_before_microphone_click(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str], float]] = []
    monkeypatch.setattr(text_input, "tts_url_for_device", lambda host: "http://192.168.1.9:9999/tts")
    monkeypatch.setattr(
        text_input.requests,
        "post",
        lambda url, *, json, timeout: calls.append((url, json, timeout)) or _Response(),
    )

    chunks = text_input.prepare_tts_input(_Robot(), "第一句。第二句。", max_chars=4)

    assert chunks == ["第一句。", "第二句。"]
    assert [call[1]["text"] for call in calls] == chunks


def test_play_prepared_tts_only_plays_prepared_chunks(monkeypatch) -> None:
    robot = _Robot()
    monkeypatch.setattr(text_input, "cancellable_sleep", lambda seconds, robot: None)

    text_input.play_prepared_tts_input(robot, ["第一句。", "第二句。"])

    assert robot.played == [("第一句。", True), ("第二句。", True)]


def test_tts_cache_filename_is_stable_and_voice_specific() -> None:
    assert _audio_filename("你好", "zh-CN-XiaoxiaoNeural") == _audio_filename("你好", "zh-CN-XiaoxiaoNeural")
    assert _audio_filename("你好", "zh-CN-XiaoxiaoNeural") != _audio_filename("你好", "zh-CN-YunxiNeural")
    assert _audio_filename("你好", "zh-CN-XiaoxiaoNeural", speaker_rate=1.0) != _audio_filename(
        "你好", "zh-CN-XiaoxiaoNeural", speaker_rate=1.2
    )


def test_tts_server_returns_cached_audio_after_preparation(monkeypatch, tmp_path) -> None:
    saved: list[str] = []

    def synthesize(text: str, path, rate: float = 1.0) -> None:
        saved.append(text)
        path.write_bytes(b"wav")

    monkeypatch.setattr("tts.server._synthesize_windows_sapi", synthesize)
    client = TestClient(create_app(tmp_path))

    first = client.post("/tts", json={"text": "缓存测试"})
    second = client.post("/tts", json={"text": "缓存测试"})

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert first.json()["engine"] == "windows_sapi"
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["engine"] == "windows_sapi"
    assert len(saved) == 1


def test_tts_server_falls_back_to_piper_when_sapi_fails(monkeypatch, tmp_path) -> None:
    def fail_sapi(text: str, path, rate: float = 1.0) -> None:
        raise RuntimeError("SAPI5不可用")

    def synthesize_piper(text: str, path, rate: float = 1.0) -> None:
        path.write_bytes(b"wav")

    monkeypatch.setattr("tts.server._synthesize_windows_sapi", fail_sapi)
    monkeypatch.setattr("tts.server._synthesize_piper", synthesize_piper)
    client = TestClient(create_app(tmp_path))

    response = client.post("/tts", json={"text": "回退测试"})

    assert response.status_code == 200
    assert response.json()["engine"] == "piper"
    assert "SAPI5不可用" in response.json()["fallback_reason"]


def test_tts_server_returns_error_when_all_offline_engines_fail(monkeypatch, tmp_path) -> None:
    def fail_piper(*args) -> None:
        raise RuntimeError("Piper失败")

    def fail_sapi(*args) -> None:
        raise RuntimeError("SAPI失败")

    monkeypatch.setattr("tts.server._synthesize_piper", fail_piper)
    monkeypatch.setattr("tts.server._synthesize_windows_sapi", fail_sapi)
    client = TestClient(create_app(tmp_path))

    response = client.post("/tts", json={"text": "失败测试"})

    assert response.status_code == 500
    assert "Piper失败" in response.json()["detail"]
