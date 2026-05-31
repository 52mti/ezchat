from __future__ import annotations

import json

import numpy as np

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


def test_create_ocr_runtime_engine_falls_back_to_cpu_and_persists_state(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)

    class GpuEngine:
        def preload(self) -> str:
            raise RuntimeError("gpu failed")

        def predict(self, image_array, **kwargs):
            raise RuntimeError("gpu failed")

    class CpuEngine:
        def preload(self) -> str:
            return "cpu"

        def predict(self, image_array, **kwargs):
            return [{"rec_texts": ["cpu"], "rec_scores": [0.9], "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}]

    def fake_create_engine(backend: str, model_tier: str, manifest, config):
        return GpuEngine() if backend == "paddle_cuda" else CpuEngine()

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "_effective_backend_config", lambda config: ("paddle_cuda", "lite"))
    monkeypatch.setattr(runtime_module, "_create_engine_for_backend", fake_create_engine)

    engine = runtime_module.create_ocr_runtime_engine({"backend": "auto"})

    assert engine.preload() == "cpu"
    assert engine.predict(np.zeros((4, 4, 3), dtype=np.uint8))[0]["rec_texts"] == ["cpu"]

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "cpu"
    assert payload["fallback_from"] == "paddle_cuda"
    assert "gpu failed" in payload["fallback_reason"]
