from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from services import (
    ConfigManager,
    cleanup_old_error_captures,
    cleanup_old_logs,
    setup_runtime_logging,
)
from ui.main_window import run_app
from vision.ocr_runtime import create_ocr_runtime_engine


PACKAGE_ROOT = Path(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or Path(__file__).resolve().parent).resolve()


def _configure_quiet_native_runtime() -> None:
    """屏蔽 Paddle 初始化阶段的无害原生诊断，保留 Python 异常和业务日志。"""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("FLAGS_log_level", "3")
    os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")


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


@contextmanager
def _silence_native_preload_output():
    """OCR 预加载会调用第三方命令探测工具；隐藏其无害的控制台提示。"""
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        try:
            os.dup2(null_stream.fileno(), stdout_fd)
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _preload_ocr_runtime() -> str:
    state = _load_runtime_state()
    if not state:
        return ""
    if str(state.get("status", "")).strip().lower() != "installed":
        return ""
    if str(state.get("validation", "")).strip().lower() != "ok":
        return ""
    backend = state.get("backend", "auto")
    if str(backend).strip().lower() == "cpu" and str(state.get("fallback_from", "") or "").strip().lower() == "paddle_cuda":
        backend = "auto"
    config = {
        "backend": backend,
        "model_tier": state.get("model_tier", "auto"),
    }
    execution_mode = str(state.get("execution_mode", "") or "").strip()
    if execution_mode:
        config["execution_mode"] = execution_mode
    with _silence_native_preload_output():
        engine = create_ocr_runtime_engine(config)
        return str(engine.preload() or "")


def _start_ocr_preload_in_background() -> threading.Thread:
    """在主窗口显示后预热 OCR，避免第三方运行时阻塞启动界面。"""

    def preload() -> None:
        try:
            device = _preload_ocr_runtime()
            if device:
                logging.getLogger(__name__).info("OCR runtime preloaded: %s", device)
        except Exception:
            # 预热失败不应阻止主程序启动；首次业务调用仍会按运行时策略处理。
            logging.getLogger(__name__).exception("OCR runtime preload failed; continuing startup")

    thread = threading.Thread(target=preload, name="ocr-preload", daemon=True)
    thread.start()
    return thread


def main() -> int:
    _configure_quiet_native_runtime()
    config_manager = ConfigManager()
    setup_runtime_logging()
    cleanup_old_logs(retention_days=_log_retention_days(config_manager))
    cleanup_old_error_captures(retention_days=_log_retention_days(config_manager), max_files=200)
    try:
        # run_app 会先 show 主窗口，再通过 Qt 事件循环调用这个回调。
        return run_app(sys.argv, on_ready=_start_ocr_preload_in_background)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).exception("Main UI startup failed")
        _show_startup_error(f"Main UI startup failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
