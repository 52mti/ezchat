from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from services.retention import cleanup_old_error_captures
from services.runtime_logging import cleanup_old_logs, setup_runtime_logging


def test_cleanup_old_logs_removes_expired_files(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    old_log = logs / "app_20200101.log"
    old_log.write_text("old", encoding="utf-8")
    old_rotated = logs / "app_20200101.log.1"
    old_rotated.write_text("old rotated", encoding="utf-8")
    old_stamp = (datetime.now() - timedelta(days=30)).timestamp()
    os.utime(old_log, (old_stamp, old_stamp))
    os.utime(old_rotated, (old_stamp, old_stamp))
    cleanup_old_logs(logs, retention_days=7)

    assert not old_log.exists()
    assert not old_rotated.exists()


def test_setup_runtime_logging_uses_rotating_file_handler(tmp_path):
    logger = logging.getLogger("jixiebijiaoben")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger = setup_runtime_logging(tmp_path / "logs", max_bytes=128, backup_count=2)

    handler_types = {type(handler).__name__ for handler in logger.handlers}
    assert "RotatingFileHandler" in handler_types


def test_cleanup_old_error_captures_keeps_latest_files(tmp_path):
    error_dir = tmp_path / "errors"
    error_dir.mkdir()
    for index in range(5):
        path = error_dir / f"sample_{index}.jpg"
        path.write_bytes(b"x")
        path.touch()

    removed = cleanup_old_error_captures(error_dir, retention_days=7, max_files=3)

    remaining = list(error_dir.glob("*.jpg"))
    assert removed == 2
    assert len(remaining) == 3
