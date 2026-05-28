from __future__ import annotations

import sys

from services import (
    ConfigManager,
    cleanup_old_error_captures,
    cleanup_old_logs,
    setup_runtime_logging,
)
from ui.main_window import run_app


def _log_retention_days(config_manager: ConfigManager | None = None) -> int:
    manager = config_manager or ConfigManager()
    value = manager.get("log_retention_days", default=7)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 7


def main() -> int:
    config_manager = ConfigManager()
    setup_runtime_logging()
    cleanup_old_logs(retention_days=_log_retention_days(config_manager))
    cleanup_old_error_captures(retention_days=_log_retention_days(config_manager), max_files=200)
    try:
        return run_app(sys.argv)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
