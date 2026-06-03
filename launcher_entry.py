from __future__ import annotations

import ctypes
import importlib
import json
import os
import subprocess
import sys
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
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")).resolve()
        return public_root / "jixiebijiaoben_runtime" / ".paddlex"
    return root / "runtime" / ".paddlex"


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


def _run_source_main() -> int:
    module = importlib.import_module("main")
    return int(module.main())


def _bootstrap_script(root: Path) -> str:
    return (
        "import sys, traceback\n"
        "from pathlib import Path\n"
        f"root = Path({str(root)!r})\n"
        "sys.path.insert(0, str(root))\n"
        "try:\n"
        "    from main import main as _entry\n"
        "    raise SystemExit(int(_entry()))\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception:\n"
        "    error_dir = root / 'runtime' / 'errors'\n"
        "    error_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (error_dir / 'launcher_bootstrap_error.txt').write_text(traceback.format_exc(), encoding='utf-8')\n"
        "    raise\n"
    )


def _spawn_packaged_app(root: Path) -> int:
    state = _load_runtime_state(root)
    if not state:
        _write_launcher_error(root, "launcher_start_error.txt", "missing runtime install state")
        return 1
    target_python = _runtime_python(root) if _runtime_python(root).is_file() else _runtime_python_fallback(root)
    if not target_python.is_file():
        _write_launcher_error(root, "launcher_start_error.txt", "missing runtime python")
        return 1

    cache_home = _cache_home_from_state(root, state)
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_USER"] = "0"
    env["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
    env["PADDLEX_HOME"] = str(cache_home)
    env.pop("PYTHONPATH", None)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    command = [str(target_python), "-s", "-u", "-c", _bootstrap_script(root)]
    process = subprocess.Popen(command, cwd=root, env=env, creationflags=creationflags)
    exit_code = int(process.wait())
    if exit_code != 0:
        _write_launcher_error(
            root,
            "launcher_exit_status.txt",
            f"exit_code={exit_code}\npython={target_python}\ncommand={command!r}\n",
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
    return _run_source_main()


if __name__ == "__main__":
    raise SystemExit(main())
