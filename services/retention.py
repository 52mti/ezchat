from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


def cleanup_old_error_captures(
    error_dir: str | Path = "photo/errors",
    *,
    retention_days: int = 7,
    max_files: int = 200,
) -> int:
    path = Path(error_dir)
    if not path.exists():
        return 0

    removed = 0
    files = [item for item in path.glob("*.jpg") if item.is_file()]
    cutoff = datetime.now() - timedelta(days=max(1, retention_days))
    for item in files:
        try:
            modified = datetime.fromtimestamp(item.stat().st_mtime)
        except OSError:
            continue
        if modified < cutoff:
            item.unlink(missing_ok=True)
            removed += 1

    remaining = [item for item in path.glob("*.jpg") if item.is_file()]
    if max_files > 0 and len(remaining) > max_files:
        remaining.sort(key=lambda item: item.stat().st_mtime)
        for item in remaining[:-max_files]:
            item.unlink(missing_ok=True)
            removed += 1
    return removed

