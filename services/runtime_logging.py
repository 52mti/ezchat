from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _incident_text(incident: dict[str, Any]) -> str:
    lines = [
        f"incident_id: {incident.get('incident_id', '')}",
        f"status: {incident.get('status', '')}",
        f"device_ip: {incident.get('device_ip', '')}",
        f"app_name: {incident.get('app_name', '')}",
        f"started_at: {incident.get('started_at', '')}",
        f"ended_at: {incident.get('ended_at', '')}",
        f"error_count_peak: {incident.get('error_count_peak', '')}",
        f"circuit_open: {incident.get('circuit_open', False)}",
        f"ocr_backend: {incident.get('ocr_backend', '')}",
        f"error_screenshot: {incident.get('error_screenshot', '')}",
        f"current_capture_path: {incident.get('current_capture_path', '')}",
    ]
    related_paths = incident.get("related_capture_paths", [])
    if related_paths:
        lines.append("related_capture_paths:")
        lines.extend(f"  - {item}" for item in related_paths)
    template_paths = incident.get("template_paths", [])
    if template_paths:
        lines.append("template_paths:")
        lines.extend(f"  - {item}" for item in template_paths)

    pre_logs = incident.get("pre_error_logs", [])
    if pre_logs:
        lines.append("")
        lines.append("pre_error_logs:")
        for item in pre_logs:
            lines.append(f"  {item.get('time', '')} [{item.get('level', '')}] {item.get('message', '')} {item.get('payload', {})}")

    errors = incident.get("errors", [])
    if errors:
        lines.append("")
        lines.append("errors:")
        for item in errors:
            lines.append(
                "  "
                f"{item.get('time', '')} count={item.get('error_count', '')}/{item.get('threshold', '')} "
                f"step={item.get('step_label', '') or item.get('step', '')} error={item.get('error', '')}"
            )

    timeline = incident.get("timeline_logs", [])
    if timeline:
        lines.append("")
        lines.append("timeline_logs:")
        for item in timeline:
            lines.append(f"  {item.get('time', '')} [{item.get('level', '')}] {item.get('message', '')} {item.get('payload', {})}")
    return "\n".join(lines) + "\n"


def write_error_incident(
    incident: dict[str, Any],
    incident_dir: str | Path = "runtime/error_incidents",
) -> tuple[Path, Path]:
    safe_incident = _json_safe(incident)
    if not isinstance(safe_incident, dict):
        raise TypeError("incident must be a dict")
    started_at = str(safe_incident.get("started_at", "") or datetime.now().isoformat(timespec="seconds"))
    day_key = started_at[:10].replace("-", "") if len(started_at) >= 10 else datetime.now().strftime("%Y%m%d")
    device_ip = str(safe_incident.get("device_ip", "") or "unknown").replace(".", "_").replace(":", "_")
    app_name = str(safe_incident.get("app_name", "") or "app").replace(".", "_").replace(":", "_")
    incident_id = str(safe_incident.get("incident_id", "") or datetime.now().strftime("%H%M%S"))
    root = Path(incident_dir) / day_key
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{device_ip}_{app_name}_{incident_id}"
    json_path = root / f"{base_name}.json"
    log_path = root / f"{base_name}.log"
    json_path.write_text(json.dumps(safe_incident, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(_incident_text(safe_incident), encoding="utf-8")
    return json_path, log_path
