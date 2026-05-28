from __future__ import annotations

import copy
import ipaddress
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "global": {
        "apikey": "",
        "license_api_url": "",
        "project_uuid": "",
        "project_secret": "",
        "chat_api_base": "http://150.158.173.140",
        "chat_bearer": "",
        "credits_bearer": "",
        "greetings_dir": "runtime/greetings",
        "persona": {
            "requestPersona": "",
            "name": "",
            "age": "",
            "work": "",
            "location": "",
            "renshe": "",
        },
    },
    "devices": [],
    "log_retention_days": 7,
}

DEFAULT_OUTBOX: dict[str, Any] = {"schema_version": 1, "messages": []}
CURRENT_SCHEMA_VERSION = 1


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _device_sort_key(device: dict[str, Any]) -> tuple[int, tuple[int, ...] | tuple[str]]:
    ip = str(device.get("ip", "")).strip()
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return (1, (ip,))
    return (0, tuple(int(part) for part in parsed.packed))


def _normalize_devices(devices: list[Any]) -> list[dict[str, Any]]:
    merged_by_ip: dict[str, dict[str, Any]] = {}
    ordered_ips: list[str] = []
    for raw_device in devices:
        if not isinstance(raw_device, dict):
            continue
        ip = str(raw_device.get("ip", "")).strip()
        if not ip:
            continue
        if ip not in merged_by_ip:
            merged_by_ip[ip] = {"ip": ip}
            ordered_ips.append(ip)
        merged_by_ip[ip].update(copy.deepcopy(raw_device))
        merged_by_ip[ip]["ip"] = ip
    normalized = [merged_by_ip[ip] for ip in ordered_ips]
    normalized.sort(key=_device_sort_key)
    return normalized


def _normalize_template_paths(device: dict[str, Any]) -> None:
    app_overrides = device.get("app_overrides", {})
    if not isinstance(app_overrides, dict):
        return
    for app_name, override in app_overrides.items():
        if not isinstance(override, dict):
            continue
        templates = override.get("templates", {})
        if not isinstance(templates, dict):
            continue
        for group in ("buttons", "screens"):
            value = templates.get(group)
            if not isinstance(value, str) or not value:
                continue
            normalized = value.replace("/", "\\")
            marker = "\\runtime\\templates\\"
            if normalized.startswith("runtime\\templates\\"):
                parts = normalized.split("\\")
                if len(parts) >= 5 and parts[2] != str(app_name):
                    templates[group] = str(Path("runtime") / "templates" / str(app_name) / group)
            elif marker in normalized:
                before, after = normalized.split(marker, 1)
                parts = after.split("\\")
                if len(parts) >= 3 and parts[0] != str(app_name):
                    templates[group] = str(Path(before) / "runtime" / "templates" / str(app_name) / group)


class ConfigManager:
    _lock = threading.Lock()

    def __init__(
        self,
        config_path: str | Path = "config.json",
        outbox_path: str | Path = "outbox.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.outbox_path = Path(outbox_path)
        self._config = self._load_json(self.config_path, DEFAULT_CONFIG)
        self._outbox = self._load_json(self.outbox_path, DEFAULT_OUTBOX)
        self._migrate_if_needed()
        self._normalize_loaded_template_paths()
        self._apply_environment_overrides()

    def _load_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return copy.deepcopy(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return copy.deepcopy(default)
        if not isinstance(data, dict):
            return copy.deepcopy(default)
        return _deep_merge(default, data)

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            tmp_name = handle.name
        os.replace(tmp_name, path)

    def _backup_file(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        return backup

    def _migrate_if_needed(self) -> None:
        config_version = int(self._config.get("schema_version", 0) or 0)
        outbox_version = int(self._outbox.get("schema_version", 0) or 0)
        if config_version < CURRENT_SCHEMA_VERSION:
            self._backup_file(self.config_path)
            self._config = self._migrate_config(self._config, config_version, CURRENT_SCHEMA_VERSION)
            self._atomic_write(self.config_path, self._config)
        if outbox_version < CURRENT_SCHEMA_VERSION:
            self._backup_file(self.outbox_path)
            self._outbox = self._migrate_outbox(self._outbox, outbox_version, CURRENT_SCHEMA_VERSION)
            self._atomic_write(self.outbox_path, self._outbox)

    def _normalize_loaded_template_paths(self) -> None:
        devices = self._config.get("devices", [])
        if not isinstance(devices, list):
            return
        changed = False
        for device in devices:
            if not isinstance(device, dict):
                continue
            before = copy.deepcopy(device.get("app_overrides", {}))
            _normalize_template_paths(device)
            if before != device.get("app_overrides", {}):
                changed = True
        if changed:
            self._atomic_write(self.config_path, self._config)

    def _migrate_config(
        self,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        migrated = _deep_merge(DEFAULT_CONFIG, data)
        migrated["schema_version"] = to_version
        devices = migrated.get("devices", [])
        if isinstance(devices, list):
            migrated["devices"] = _normalize_devices(devices)
            for device in migrated["devices"]:
                if not isinstance(device, dict):
                    continue
                device.setdefault("selected_app", "wechat")
                device.setdefault("persona", {})
                device.setdefault("contact_personas", {})
                device.setdefault("app_overrides", {"wechat": {}})
                _normalize_template_paths(device)
        return migrated

    def _migrate_outbox(
        self,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        migrated = _deep_merge(DEFAULT_OUTBOX, data)
        migrated["schema_version"] = to_version
        messages = migrated.get("messages", [])
        if not isinstance(messages, list):
            migrated["messages"] = []
        return migrated

    def _apply_environment_overrides(self) -> None:
        apikey = os.getenv("CHATBOT_APIKEY")
        if apikey:
            self._config.setdefault("global", {})["apikey"] = apikey

        devices = self._config.get("devices", [])
        if isinstance(devices, list):
            for device in devices:
                if not isinstance(device, dict):
                    continue
                ip = str(device.get("ip", ""))
                env_key = "CHATBOT_LICENSE_" + ip.replace(".", "_")
                license_key = os.getenv(env_key)
                if license_key:
                    device["license_key"] = license_key

    def data(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._config)

    def devices(self) -> list[dict[str, Any]]:
        with self._lock:
            devices = self._config.get("devices", [])
            normalized = _normalize_devices(devices if isinstance(devices, list) else [])
            return copy.deepcopy(normalized)

    def find_device(self, device_ip: str) -> dict[str, Any] | None:
        with self._lock:
            devices = self._config.get("devices", [])
            if not isinstance(devices, list):
                return None
            for device in devices:
                if isinstance(device, dict) and str(device.get("ip", "")) == device_ip:
                    return copy.deepcopy(device)
            return None

    def update_device(self, device_ip: str, updates: dict[str, Any]) -> None:
        with self._lock:
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            for device in devices:
                if isinstance(device, dict) and str(device.get("ip", "")) == device_ip:
                    device.update(copy.deepcopy(updates))
                    self._config["devices"] = _normalize_devices(devices)
                    self._atomic_write(self.config_path, self._config)
                    return
            new_device = {"ip": device_ip, **copy.deepcopy(updates)}
            devices.append(new_device)
            self._config["devices"] = _normalize_devices(devices)
            self._atomic_write(self.config_path, self._config)

    def add_device(self, device: dict[str, Any]) -> bool:
        device_ip = str(device.get("ip", "")).strip()
        if not device_ip:
            raise ValueError("device ip is required")
        with self._lock:
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            if any(isinstance(existing, dict) and str(existing.get("ip", "")).strip() == device_ip for existing in devices):
                return False
            devices.append(copy.deepcopy(device))
            self._config["devices"] = _normalize_devices(devices)
            self._atomic_write(self.config_path, self._config)
            return True

    def remove_device(self, device_ip: str) -> bool:
        device_ip = str(device_ip).strip()
        if not device_ip:
            return False
        with self._lock:
            devices = self._config.get("devices", [])
            if not isinstance(devices, list):
                return False
            remaining = [
                copy.deepcopy(device)
                for device in devices
                if not (isinstance(device, dict) and str(device.get("ip", "")).strip() == device_ip)
            ]
            if len(remaining) == len(devices):
                return False
            self._config["devices"] = _normalize_devices(remaining)
            self._atomic_write(self.config_path, self._config)
            return True

    def get_app_config(
        self,
        device_ip: str,
        app_name: str,
        default_config: dict[str, Any],
    ) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        overrides = (
            device.get("app_overrides", {}).get(app_name, {})
            if isinstance(device.get("app_overrides", {}), dict)
            else {}
        )
        return _deep_merge(default_config, overrides if isinstance(overrides, dict) else {})

    def get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            current: Any = self._config
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            return copy.deepcopy(current)

    def set(self, *keys: str, value: Any) -> None:
        if not keys:
            raise ValueError("at least one key is required")
        with self._lock:
            current = self._config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
                if not isinstance(current, dict):
                    raise ValueError(f"cannot set nested key under non-dict: {key}")
            current[keys[-1]] = value
            self._atomic_write(self.config_path, self._config)

    def save_outbox_entry(self, entry: dict[str, Any]) -> None:
        fingerprint = entry.get("message_fingerprint")
        if not fingerprint:
            raise ValueError("outbox entry requires message_fingerprint")
        with self._lock:
            messages = self._outbox.setdefault("messages", [])
            for index, existing in enumerate(messages):
                if existing.get("message_fingerprint") == fingerprint:
                    merged = copy.deepcopy(existing)
                    merged.update(copy.deepcopy(entry))
                    messages[index] = merged
                    break
            else:
                messages.append(copy.deepcopy(entry))
            self._atomic_write(self.outbox_path, self._outbox)

    def load_outbox(self) -> list[dict[str, Any]]:
        with self._lock:
            messages = self._outbox.get("messages", [])
            return copy.deepcopy(messages if isinstance(messages, list) else [])

    def pending_outbox(self) -> list[dict[str, Any]]:
        return [entry for entry in self.load_outbox() if entry.get("pending_send")]
