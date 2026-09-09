"""Frozen application entry point used by release builds."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    # Mutable data and external assets belong beside the executable, never in
    # PyInstaller's temporary extraction directory.
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    root = Path(configured).resolve() if configured else (
        Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    )
    os.environ["JIXIEBI_PACKAGE_ROOT"] = str(root)
    os.chdir(root)
    modules = root / "app_modules.zip"
    if modules.is_file() and str(modules) not in sys.path:
        sys.path.insert(0, str(modules))
    from main import main as run_main
    return int(run_main())


if __name__ == "__main__":
    raise SystemExit(main())
