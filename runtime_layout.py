from __future__ import annotations

import os
from pathlib import Path


ENGINE_DIR_NAME = "engine"
DATA_DIR_NAME = "runtime"


RUNTIME_NAMES = ("app", "ocr_cpu", "ocr_nvidia", "ocr_directml")
MODEL_PROVIDER_PADDLEX = "paddlex_named_models"
REQUIRED_MODEL_FILES = ("config.json", "inference.json", "inference.pdiparams", "inference.yml")


def package_root_from_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    return resolved if resolved.is_dir() else resolved.parent


def runtime_root(root: str | Path) -> Path:
    """引擎根目录：Python 解释器与各 OCR 运行时（engine/）。

    基础服务不属于客户数据，放在 engine/ 下由包清单管理，
    runtime/ 只保留日志、缓存等运行数据。
    """
    return package_root_from_path(root) / ENGINE_DIR_NAME


def runtime_dir(root: str | Path, runtime_name: str) -> Path:
    normalized = str(runtime_name or "").strip()
    if not normalized:
        raise ValueError("runtime_name is required")
    return runtime_root(root) / normalized


def backend_runtime_name(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    mapping = {
        "cpu": "ocr_cpu",
        "paddle_cuda": "ocr_nvidia",
        "onnx_directml": "ocr_directml",
        "app": "app",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported runtime backend: {backend}")
    return mapping[normalized]

def _python_candidates(runtime_dir_path: Path, *, gui: bool) -> tuple[Path, ...]:
    if os.name == "nt":
        gui_names = ("pythonw.exe", "python.exe") if gui else ("python.exe", "pythonw.exe")
        candidates: list[Path] = []
        for name in gui_names:
            candidates.append(runtime_dir_path / name)
        for name in gui_names:
            candidates.append(runtime_dir_path / "Scripts" / name)
        return tuple(candidates)
    executable = runtime_dir_path / "bin" / "python"
    return (executable,)


def runtime_python(root: str | Path, runtime_name: str, *, gui: bool = False) -> Path:
    for candidate in _python_candidates(runtime_dir(root, runtime_name), gui=gui):
        if candidate.is_file():
            return candidate
    runtime_dir_path = runtime_dir(root, runtime_name)
    return _python_candidates(runtime_dir_path, gui=gui)[0]


def runtime_state_path(root: str | Path) -> Path:
    # install_state 属于安装状态数据，固定在 runtime/ 下，不随引擎目录迁移。
    return package_root_from_path(root) / DATA_DIR_NAME / "install_state.json"


def runtime_reports_dir(root: str | Path) -> Path:
    return package_root_from_path(root) / DATA_DIR_NAME / "reports"


def models_root(root: str | Path) -> Path:
    return package_root_from_path(root) / "models" / "ocr"


def bundled_paddlex_models_root(root: str | Path) -> Path:
    return models_root(root) / "paddlex_named_models"


def bundled_paddlex_model_dir(root: str | Path, model_name: str) -> Path:
    return bundled_paddlex_models_root(root) / str(model_name or "").strip()


def model_dir_is_complete(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / file_name).is_file() for file_name in REQUIRED_MODEL_FILES)


def has_packaged_runtime_layout(root: str | Path) -> bool:
    base = package_root_from_path(root)
    return runtime_dir(base, "app").is_dir() and any(runtime_dir(base, name).is_dir() for name in RUNTIME_NAMES if name != "app")
