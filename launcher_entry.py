from __future__ import annotations

import ctypes
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime_layout import (
    bundled_paddlex_model_dir,
    has_packaged_runtime_layout,
    model_dir_is_complete,
    runtime_dir,
    runtime_python,
    runtime_state_path,
)


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _runtime_dir(root: Path) -> Path:
    return runtime_dir(root, "app")


def _runtime_python(root: Path) -> Path:
    return runtime_python(root, "app", gui=True)


def _runtime_python_fallback(root: Path) -> Path:
    return runtime_python(root, "app", gui=False)


def _is_valid_runtime_python(path: Path) -> bool:
    """Reject malformed layouts such as engine/app/pythonw/.exe."""
    return path.is_file() and path.name.lower() in {"python.exe", "pythonw.exe"}


def _runtime_state_path(root: Path) -> Path:
    return runtime_state_path(root)


def _load_runtime_state(root: Path) -> dict[str, object] | None:
    path = _runtime_state_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_home_from_state(root: Path, state: dict[str, object]) -> Path:
    configured = str(state.get("cache_home", "") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            return (root / candidate).resolve()
        normalized = str(candidate).replace("\\", "/").lower()
        if "/dist/_staging/" not in normalized and _path_anchor_exists(candidate):
            return candidate.resolve()
    if os.name == "nt":
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")).resolve()
        return public_root / "jixiebijiaoben_runtime" / ".paddlex"
    return root / "runtime" / ".paddlex"


def _path_anchor_exists(path: Path) -> bool:
    anchor = path.anchor
    if not anchor:
        return True
    try:
        return Path(anchor).exists()
    except OSError:
        return False


def _bundled_models_ready(root: Path, state: dict[str, object]) -> bool:
    models = state.get("models", {})
    if not isinstance(models, dict):
        return True
    if str(models.get("provider", "")).strip().lower() != "paddlex_named_models":
        return True
    for model_name in (
        str(models.get("det", "") or "").strip(),
        str(models.get("rec", "") or "").strip(),
    ):
        if model_name and not model_dir_is_complete(bundled_paddlex_model_dir(root, model_name)):
            return False
    return True


def _runtime_is_ready(root: Path) -> bool:
    state = _load_runtime_state(root)
    if not state:
        return False
    if str(state.get("status", "")).strip().lower() != "installed":
        return False
    if str(state.get("validation", "")).strip().lower() != "ok":
        return False
    if not _runtime_dir(root).is_dir():
        return False
    if not (_runtime_python(root).is_file() or _runtime_python_fallback(root).is_file()):
        return False

    if has_packaged_runtime_layout(root):
        if not any(
            runtime_dir(root, runtime_name).is_dir()
            and runtime_python(root, runtime_name, gui=False).is_file()
            for runtime_name in ("ocr_cpu", "ocr_nvidia", "ocr_directml")
        ):
            return False

    models = state.get("models", {})
    if not isinstance(models, dict):
        return True
    if str(models.get("provider", "")).strip().lower() != "paddlex_named_models":
        return True
    if _bundled_models_ready(root, state):
        return True

    cache_home = _cache_home_from_state(root, state)
    official_models_root = cache_home / "official_models"
    for model_name in (
        str(models.get("det", "") or "").strip(),
        str(models.get("rec", "") or "").strip(),
    ):
        if model_name and not model_dir_is_complete(official_models_root / model_name):
            return False
    return True


def _show_missing_runtime_message() -> None:
    message = "OCR runtime or bundled models are not ready. Rebuild the full package."
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, "机械臂脚本", 0x10)
        return
    print(message, file=sys.stderr)


def _write_launcher_error(root: Path, name: str, details: str) -> None:
    error_dir = root / "runtime" / "errors"
    try:
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / name).write_text(details, encoding="utf-8")
    except OSError:
        pass


def _spawn_packaged_app(root: Path) -> int:
    state = _load_runtime_state(root)
    if not state:
        _write_launcher_error(root, "launcher_start_error.txt", "missing runtime install state")
        return 1
    app_entry = root / "app_core_entry.pyc"
    app_modules = root / "app_modules.zip"
    if not app_entry.is_file() or not app_modules.is_file():
        _write_launcher_error(root, "launcher_start_error.txt", "missing protected application bytecode")
        return 1

    cache_home = _cache_home_from_state(root, state)
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_USER"] = "0"
    env["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
    env["PADDLEX_HOME"] = str(cache_home)
    env.pop("PYTHONPATH", None)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    env["JIXIEBI_PACKAGE_ROOT"] = str(root)
    # 仅发布启动器设置此标识。开发时直接运行 main.py 不应触发客户更新检查。
    env["JIXIEBI_PACKAGED_LAUNCH"] = "1"
    gui_python = _runtime_python(root)
    console_python = _runtime_python_fallback(root)
    # Some customer machines lose pythonw.exe during an interrupted update.
    # Always use a real file and fall back to python.exe instead of spawning a
    # path that only exists as the preferred candidate name.
    target_python = gui_python if _is_valid_runtime_python(gui_python) else console_python
    if not _is_valid_runtime_python(target_python):
        _write_launcher_error(root, "launcher_start_error.txt", "missing runtime python")
        return 1
    command = [str(target_python), "-s", "-u", str(app_entry)]
    error_log = root / "runtime" / "errors" / "application_stderr.log"
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8", errors="replace") as error_handle:
        error_handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        error_handle.write(f"python={target_python}\ncommand={command!r}\n")
        error_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                creationflags=creationflags,
                stdout=error_handle,
                stderr=error_handle,
            )
        except TypeError:
            # Keep compatibility with lightweight test doubles and old hosts.
            process = subprocess.Popen(command, cwd=root, env=env, creationflags=creationflags)
    exit_code = int(process.wait())
    if exit_code != 0:
        _write_launcher_error(
            root,
            "launcher_exit_status.txt",
            f"exit_code={exit_code}\npython={target_python}\ncommand={command!r}\n"
            f"stderr_log={error_log}\n",
        )
    return exit_code


def main() -> int:
    root = _app_root()
    os.chdir(root)
    if getattr(sys, "frozen", False):
        if not _runtime_is_ready(root):
            _show_missing_runtime_message()
            return 1
        return _spawn_packaged_app(root)
    return int(importlib.import_module("main").main())


if __name__ == "__main__":
    raise SystemExit(main())
