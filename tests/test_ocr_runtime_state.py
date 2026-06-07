from __future__ import annotations

import json
import io

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


def test_effective_backend_config_retries_gpu_after_cpu_fallback_state(tmp_path, monkeypatch):
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

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "select_ocr_backend", lambda preferred_backend, hardware=None: "paddle_cuda")

    backend, model_tier = runtime_module._effective_backend_config({"backend": "auto", "model_tier": "auto"})

    assert backend == "paddle_cuda"
    assert model_tier == "lite"


def test_subprocess_runtime_uses_legacy_state_python_when_backend_runtime_missing(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    legacy_python = tmp_path / ".runtime_env" / "Scripts" / "python.exe"
    legacy_python.parent.mkdir(parents=True)
    legacy_python.write_text("python", encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "paddle_cuda",
                "runtime_name": "legacy",
                "python_executable": str(legacy_python),
                "runtime_path": str(legacy_python.parents[1]),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: tmp_path)
    monkeypatch.setattr(runtime_module, "runtime_python", lambda root, runtime_name, gui=False: tmp_path / "runtime" / runtime_name / "python.exe")

    engine = runtime_module.SubprocessOcrRuntimeEngine(
        backend="paddle_cuda",
        model_tier="lite",
        manifest=runtime_module.load_runtime_manifest(),
        config={},
    )

    assert engine._runtime_python_path() == legacy_python


def test_subprocess_runtime_start_uses_backend_env_and_cache_home(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    cache_home = tmp_path / "runtime" / ".paddlex"
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "paddle_cuda",
                "cache_home": str(cache_home),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime_python = tmp_path / "runtime" / "ocr_nvidia" / "python.exe"
    scripts_dir = runtime_python.parent / "Scripts"
    scripts_dir.mkdir(parents=True)
    runtime_python.write_text("python", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeProcess:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = dict(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: tmp_path)
    monkeypatch.setattr(runtime_module, "runtime_python", lambda root, runtime_name, gui=False: runtime_python)
    monkeypatch.setattr(runtime_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime_module.subprocess, "STARTUPINFO", lambda: type("StartupInfo", (), {"dwFlags": 0})())
    monkeypatch.setattr(runtime_module.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)

    engine = runtime_module.SubprocessOcrRuntimeEngine(
        backend="paddle_cuda",
        model_tier="lite",
        manifest=runtime_module.load_runtime_manifest(),
        config={},
    )

    process = engine._start_process()
    env = captured["env"]

    assert process is not None
    assert captured["command"][0] == str(runtime_python)
    assert env["PADDLE_PDX_CACHE_HOME"] == str(cache_home)
    assert env["PADDLEX_HOME"] == str(cache_home)
    assert env["FLAGS_allocator_strategy"] == "auto_growth"
    assert env["FLAGS_fraction_of_gpu_memory_to_use"] == "0.6"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert str(runtime_python.parent) in env["PATH"]
    assert str(scripts_dir) in env["PATH"]


def test_subprocess_runtime_ignores_packaging_staging_cache_home(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "paddle_cuda",
                "cache_home": "D:/Program Files (x86)/appkaifa/jixiebijiaoben/dist/_staging/build_001/jixiebijiaoben/runtime/.paddlex",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime_python = tmp_path / "runtime" / "ocr_nvidia" / "python.exe"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("python", encoding="utf-8")
    public_root = tmp_path / "Public"
    captured: dict[str, object] = {}

    class FakeProcess:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProcess()

    monkeypatch.setenv("PUBLIC", str(public_root))
    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: tmp_path)
    monkeypatch.setattr(runtime_module, "runtime_python", lambda root, runtime_name, gui=False: runtime_python)
    monkeypatch.setattr(runtime_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime_module.subprocess, "STARTUPINFO", lambda: type("StartupInfo", (), {"dwFlags": 0})())
    monkeypatch.setattr(runtime_module.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)

    engine = runtime_module.SubprocessOcrRuntimeEngine(
        backend="paddle_cuda",
        model_tier="lite",
        manifest=runtime_module.load_runtime_manifest(),
        config={},
    )

    engine._start_process()

    expected = public_root / "jixiebijiaoben_runtime" / ".paddlex"
    assert captured["env"]["PADDLE_PDX_CACHE_HOME"] == str(expected)
    assert captured["env"]["PADDLEX_HOME"] == str(expected)


def test_subprocess_runtime_closes_failed_child_process(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"status": "installed", "validation": "ok", "backend": "paddle_cuda"}, ensure_ascii=False),
        encoding="utf-8",
    )
    runtime_python = tmp_path / "runtime" / "ocr_nvidia" / "python.exe"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("python", encoding="utf-8")

    class FakeStdin:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write(self, value: str) -> None:
            self.lines.append(value)

        def flush(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.stdout = io.StringIO('{"ok": false, "error": "bad paddlex state"}\n')
            self.stderr = io.StringIO("")
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

    process = FakeProcess()

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: tmp_path)
    monkeypatch.setattr(runtime_module, "runtime_python", lambda root, runtime_name, gui=False: runtime_python)
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime_module.subprocess, "STARTUPINFO", lambda: type("StartupInfo", (), {"dwFlags": 0})())
    monkeypatch.setattr(runtime_module.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)

    engine = runtime_module.SubprocessOcrRuntimeEngine(
        backend="paddle_cuda",
        model_tier="lite",
        manifest=runtime_module.load_runtime_manifest(),
        config={},
    )

    with pytest.raises(RuntimeError, match="bad paddlex state"):
        engine.preload()

    assert engine._process is None
    assert process.terminated is True
    assert any('"action": "shutdown"' in line for line in process.stdin.lines)


def test_manifest_subprocess_mode_uses_ocr_runtime_without_app_runtime(tmp_path, monkeypatch):
    (tmp_path / "runtime_manifest.json").write_text(
        json.dumps({"executionMode": "subprocess"}, ensure_ascii=False),
        encoding="utf-8",
    )
    ocr_python = tmp_path / "runtime" / "ocr_nvidia" / "python.exe"
    ocr_python.parent.mkdir(parents=True)
    ocr_python.write_text("python", encoding="utf-8")

    monkeypatch.setattr(runtime_module, "_package_root", lambda project_root=None: tmp_path)

    assert runtime_module._effective_execution_mode({}) == "subprocess"


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


def test_gpu_preload_success_clears_previous_cpu_fallback_state(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "install_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "installed",
                "validation": "ok",
                "backend": "cpu",
                "ocr_device": "cpu",
                "fallback_from": "paddle_cuda",
                "fallback_reason": "old failure",
                "model_tier": "lite",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class GpuEngine:
        def preload(self) -> str:
            return "gpu:0"

    class CpuEngine:
        def preload(self) -> str:
            return "cpu"

    def fake_create_engine(backend: str, model_tier: str, manifest, config):
        return GpuEngine() if backend == "paddle_cuda" else CpuEngine()

    monkeypatch.setattr(runtime_module, "_runtime_state_path", lambda project_root=None: state_path)
    monkeypatch.setattr(runtime_module, "select_ocr_backend", lambda preferred_backend, hardware=None: "paddle_cuda")
    monkeypatch.setattr(runtime_module, "_create_engine_for_backend", fake_create_engine)

    engine = runtime_module.create_ocr_runtime_engine({"backend": "auto", "model_tier": "auto"})

    assert engine.preload() == "gpu:0"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "paddle_cuda"
    assert payload["ocr_device"] == "gpu:0"
    assert "fallback_from" not in payload
    assert "fallback_reason" not in payload


def test_create_local_runtime_engine_can_disable_same_process_fallback(monkeypatch):
    class GpuEngine:
        def preload(self) -> str:
            raise RuntimeError("gpu failed")

    class CpuEngine:
        def preload(self) -> str:
            return "cpu"

    def fake_create_engine(backend: str, model_tier: str, manifest, config):
        return GpuEngine() if backend == "paddle_cuda" else CpuEngine()

    monkeypatch.setattr(runtime_module, "_effective_backend_config", lambda config: ("paddle_cuda", "lite"))
    monkeypatch.setattr(runtime_module, "_create_engine_for_backend", fake_create_engine)

    engine = runtime_module.create_local_runtime_engine({"backend": "paddle_cuda"}, allow_fallback=False)

    with pytest.raises(RuntimeError, match="gpu failed"):
        engine.preload()


def test_paddle_runtime_import_error_reports_backend_and_dependency(monkeypatch):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "paddleocr":
            raise ImportError("No module named 'paddleocr'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    engine = runtime_module.PaddleOcrRuntimeEngine(
        backend="cpu",
        model_tier="lite",
        manifest=runtime_module.load_runtime_manifest(),
        config={},
    )

    with pytest.raises(RuntimeError, match="Missing packaged PaddleOCR runtime for backend cpu: No module named 'paddleocr'"):
        engine.preload()


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
