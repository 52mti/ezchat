from __future__ import annotations

import json

import vision.ocr_runtime as runtime_module


def test_effective_backend_config_prefers_installed_runtime_state(tmp_path, monkeypatch):
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

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "select_ocr_backend", lambda preferred_backend, hardware=None: "paddle_cuda")

    backend, model_tier = runtime_module._effective_backend_config({"backend": "auto", "model_tier": "auto"})

    assert backend == "cpu"
    assert model_tier == "lite"


def test_load_installed_runtime_state_rejects_invalid_status(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "invalid",
                "validation": "failed",
                "backend": "cpu",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)

    assert runtime_module.load_installed_runtime_state() == {}
