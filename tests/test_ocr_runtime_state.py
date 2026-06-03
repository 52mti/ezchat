from __future__ import annotations

import json

import numpy as np
import pytest

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


def test_serialize_prediction_payload_handles_non_json_objects():
    class Font:
        def __str__(self) -> str:
            return "Font(debug)"

    payload = {
        "rec_texts": ["你好"],
        "rec_scores": [np.float32(0.96)],
        "font": Font(),
        "flag": np.bool_(True),
        "indexes": {np.int64(3)},
    }

    serialized = runtime_module._serialize_prediction_payload(payload)

    json.dumps(serialized, ensure_ascii=False)
    assert serialized["font"] == "Font(debug)"
    assert serialized["rec_scores"][0] == pytest.approx(0.96)
    assert serialized["flag"] is True
    assert serialized["indexes"] == [3]


def test_choose_cuda_lane_prefers_cu118_for_pascal_windows(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")

    lane, reason = runtime_module.choose_cuda_lane(
        None,
        "12.6",
        compute_capability="6.1",
        gpu_name="NVIDIA GeForce GTX 1060 5GB",
    )

    assert lane == "cu118"
    assert "compute capability 6.1" in reason
    assert "cu118" in reason


def test_resolve_bundled_model_dir_mirrors_non_ascii_windows_path(tmp_path, monkeypatch):
    package_root = tmp_path / "机械臂脚本"
    model_dir = package_root / "models" / "ocr" / "paddlex_named_models" / "PP-OCRv5_mobile_det"
    model_dir.mkdir(parents=True)
    for file_name in ("config.json", "inference.json", "inference.pdiparams", "inference.yml"):
        (model_dir / file_name).write_text(file_name, encoding="utf-8")
    mirror_root = tmp_path / "ascii_cache"

    monkeypatch.setattr(runtime_module.os, "name", "nt")
    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: package_root)
    monkeypatch.setattr(runtime_module, "_bundled_model_mirror_root", lambda project_root=None: mirror_root)

    resolved = runtime_module._resolve_bundled_model_dir("PP-OCRv5_mobile_det")

    resolved_path = runtime_module.Path(resolved)
    assert resolved_path.parent == mirror_root
    assert (resolved_path / "inference.json").is_file()
    assert resolved_path != model_dir


def test_build_kwargs_mirrors_custom_non_ascii_model_dir_on_windows(tmp_path, monkeypatch):
    model_dir = tmp_path / "自定义模型" / "PP-OCRv5_mobile_det"
    model_dir.mkdir(parents=True)
    for file_name in ("config.json", "inference.json", "inference.pdiparams", "inference.yml"):
        (model_dir / file_name).write_text(file_name, encoding="utf-8")
    mirror_root = tmp_path / "ascii_cache"

    monkeypatch.setattr(runtime_module.os, "name", "nt")
    monkeypatch.setattr(runtime_module, "_bundled_model_mirror_root", lambda project_root=None: mirror_root)
    monkeypatch.setattr(runtime_module, "_resolve_bundled_model_dir", lambda model_name, project_root=None: "")

    manifest = runtime_module.load_runtime_manifest()
    engine = runtime_module.PaddleOcrRuntimeEngine(
        backend="cpu",
        model_tier="lite",
        manifest=manifest,
        config={
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_model_dir": str(model_dir),
        },
    )

    kwargs = engine._build_kwargs("cpu")

    resolved_path = runtime_module.Path(kwargs["text_detection_model_dir"])
    assert resolved_path.parent == mirror_root
    assert (resolved_path / "inference.json").is_file()
