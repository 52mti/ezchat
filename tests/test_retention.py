from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from services.retention import cleanup_old_error_captures
from services.runtime_logging import cleanup_old_logs, setup_runtime_logging, write_error_incident


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


def test_write_error_incident_writes_json_and_log(tmp_path):
    json_path, log_path = write_error_incident(
        {
            "incident_id": "demo_001",
            "device_ip": "192.168.1.108",
            "app_name": "xinyu",
            "started_at": "2026-06-09T12:00:00",
            "ended_at": "2026-06-09T12:00:10",
            "status": "recovered",
            "error_count_peak": 1,
            "circuit_open": False,
            "ocr_backend": "GPU",
            "error_screenshot": "photo/errors/demo.jpg",
            "current_capture_path": "photo/captures/demo.jpg",
            "related_capture_paths": ["photo/captures/demo.jpg"],
            "template_paths": ["apps/xinyu/templates/screens/screen_聊天列表.jpg"],
            "pre_error_logs": [{"time": "2026-06-09 12:00:00", "level": "info", "message": "before", "payload": {}}],
            "timeline_logs": [{"time": "2026-06-09 12:00:01", "level": "warning", "message": "during", "payload": {}}],
            "errors": [{"time": "2026-06-09 12:00:01", "error_count": 1, "threshold": 5, "step_label": "列表 OCR 扫描", "error": "boom"}],
        },
        tmp_path / "incidents",
    )

    assert json_path.is_file()
    assert log_path.is_file()
    assert "demo_001" in log_path.read_text(encoding="utf-8")
