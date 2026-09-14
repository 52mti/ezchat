"""Frozen application entry point used by release builds."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_root() -> Path:
    # Mutable data and external assets belong beside the executable, never in
    # PyInstaller's temporary extraction directory.
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    return (
        Path(configured).resolve()
        if configured
        else (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
    )


def _write_crash_log(root: Path, exc: BaseException) -> Path | None:
    """pythonw 无控制台：import 阶段的异常必须落盘，否则闪退无从诊断。"""
    try:
        import traceback
        from datetime import datetime

        log_dir = root / "runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "startup_crash.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"--- {datetime.now().isoformat(timespec='seconds')} ---\n")
            handle.write("".join(traceback.format_exception(exc)))
            handle.write("\n")
        return path
    except Exception:
        return None


def _show_crash(exc: BaseException, log_path: Path | None) -> None:
    try:
        if os.name == "nt" and not os.environ.get("JIXIEBI_NO_STARTUP_DIALOG"):
            import ctypes

            detail = f"\n\n崩溃详情已写入：{log_path}" if log_path is not None else ""
            ctypes.windll.user32.MessageBoxW(
                None, f"程序启动失败：{exc}\n{detail}", "机械臂脚本", 0x10
            )
            return
        print(f"startup failed: {exc}", file=sys.stderr)
    except Exception:
        pass


def main() -> int:
    root = _resolve_root()
    os.environ["JIXIEBI_PACKAGE_ROOT"] = str(root)
    os.chdir(root)
    modules = root / "app_modules.zip"
    if modules.is_file() and str(modules) not in sys.path:
        sys.path.insert(0, str(modules))
    try:
        from main import main as run_main

        return int(run_main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 启动期任何异常都要留痕
        log_path = _write_crash_log(root, exc)
        _show_crash(exc, log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
