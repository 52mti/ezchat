from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

from services import (
    ConfigManager,
    cleanup_old_error_captures,
    cleanup_old_logs,
    setup_runtime_logging,
)
from ui.main_window import run_app
from vision.ocr_runtime import create_ocr_runtime_engine


PACKAGE_ROOT = Path(__file__).resolve().parent


def _log_retention_days(config_manager: ConfigManager | None = None) -> int:
    manager = config_manager or ConfigManager()
    value = manager.get("log_retention_days", default=7)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 7


def _runtime_state_path() -> Path:
    return PACKAGE_ROOT / "runtime" / "install_state.json"


def _load_runtime_state() -> dict[str, object] | None:
    path = _runtime_state_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _show_startup_error(message: str) -> None:
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, "机械臂脚本", 0x10)
        return
    print(message, file=sys.stderr)


def _preload_ocr_runtime() -> str:
    state = _load_runtime_state()
    if not state:
        return ""
    if str(state.get("status", "")).strip().lower() != "installed":
        return ""
    if str(state.get("validation", "")).strip().lower() != "ok":
        return ""
    config = {
        "backend": state.get("backend", "auto"),
        "model_tier": state.get("model_tier", "auto"),
    }
    execution_mode = str(state.get("execution_mode", "") or "").strip()
    if execution_mode:
        config["execution_mode"] = execution_mode
    engine = create_ocr_runtime_engine(config)
    return str(engine.preload() or "")


def main() -> int:
    config_manager = ConfigManager()
    setup_runtime_logging()
    cleanup_old_logs(retention_days=_log_retention_days(config_manager))
    cleanup_old_error_captures(retention_days=_log_retention_days(config_manager), max_files=200)
    try:
        _preload_ocr_runtime()
        return run_app(sys.argv)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _show_startup_error(f"OCR runtime preload failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
