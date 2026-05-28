from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from pathlib import Path


def setup_runtime_logging(
    log_dir: str | Path = "runtime/logs",
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("jixiebijiaoben")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_file = path / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    handler = RotatingFileHandler(
        log_file,
        encoding="utf-8",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def cleanup_old_logs(log_dir: str | Path = "runtime/logs", retention_days: int = 7) -> int:
    path = Path(log_dir)
    if retention_days <= 0 or not path.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for item in path.glob("app_*.log*"):
        try:
            modified = datetime.fromtimestamp(item.stat().st_mtime)
        except OSError:
            continue
        if modified < cutoff:
            item.unlink(missing_ok=True)
            removed += 1
    return removed
