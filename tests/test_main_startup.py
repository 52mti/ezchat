from __future__ import annotations

import json

import main


def test_preload_ocr_runtime_uses_install_state(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "cpu",
                "model_tier": "lite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeEngine:
        def preload(self) -> str:
            captured["preload"] = True
            return "cpu"

    monkeypatch.setattr(main, "_runtime_state_path", lambda: state_path)

    def fake_create_ocr_runtime_engine(config):
        captured["config"] = dict(config)
        return FakeEngine()

    monkeypatch.setattr(main, "create_ocr_runtime_engine", fake_create_ocr_runtime_engine)

    assert main._preload_ocr_runtime() == "cpu"
    assert captured["config"] == {"backend": "cpu", "model_tier": "lite"}
    assert captured["preload"] is True


def test_preload_ocr_runtime_retries_auto_after_gpu_fallback(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "cpu",
                "fallback_from": "paddle_cuda",
                "model_tier": "lite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeEngine:
        def preload(self) -> str:
            return "gpu:0"

    monkeypatch.setattr(main, "_runtime_state_path", lambda: state_path)

    def fake_create_ocr_runtime_engine(config):
        captured["config"] = dict(config)
        return FakeEngine()

    monkeypatch.setattr(main, "create_ocr_runtime_engine", fake_create_ocr_runtime_engine)

    assert main._preload_ocr_runtime() == "gpu:0"
    assert captured["config"] == {"backend": "auto", "model_tier": "lite"}


def test_main_preloads_runtime_before_run_app(monkeypatch):
    calls: list[str] = []

    class FakeConfigManager:
        def get(self, *keys, default=None):
            return 7

    monkeypatch.setattr(main, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(main, "setup_runtime_logging", lambda: None)
    monkeypatch.setattr(main, "cleanup_old_logs", lambda **kwargs: None)
    monkeypatch.setattr(main, "cleanup_old_error_captures", lambda **kwargs: None)
    monkeypatch.setattr(main, "_preload_ocr_runtime", lambda: calls.append("preload") or "cpu")
    monkeypatch.setattr(main, "run_app", lambda argv: calls.append("run_app") or 0)

    assert main.main() == 0
    assert calls == ["preload", "run_app"]


def test_main_continues_when_ocr_preload_fails(monkeypatch):
    calls: list[str] = []

    class FakeConfigManager:
        def get(self, *keys, default=None):
            return 7

    monkeypatch.setattr(main, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(main, "setup_runtime_logging", lambda: None)
    monkeypatch.setattr(main, "cleanup_old_logs", lambda **kwargs: None)
    monkeypatch.setattr(main, "cleanup_old_error_captures", lambda **kwargs: None)

    def fail_preload():
        calls.append("preload")
        raise RuntimeError("broken ocr")

    monkeypatch.setattr(main, "_preload_ocr_runtime", fail_preload)
    monkeypatch.setattr(main, "run_app", lambda argv: calls.append("run_app") or 0)

    assert main.main() == 0
    assert calls == ["preload", "run_app"]
