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

from apps import resolve_app_name


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "global": {
        "license_api_url": "",
        "project_uuid": "",
        "project_secret": "",
        "chat_api_base": "http://liaotianbao.icu",
        "chat_bearer": "",
        "credits_bearer": "",
        "greetings_dir": "runtime/greetings",
        "ocr_max_concurrency": 1,
        "rule_learning": {
            "upload_url": "",
            "rules_url": "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/rule-learning/{app}/rules.json",
            "bearer_token": "",
        },
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
CURRENT_SCHEMA_VERSION = 3
DEFAULT_MATCH_THRESHOLD = 0.7

APP_LEVEL_KEYS = {
    "ocr",
    "ocr_modes",
    "dialogue_engine",
    "dialogue_llm",
    "templates",
    "match_threshold",
    "popup_match_threshold",
    "enable_enhanced_template_match",
    "template_match_region_expand_ratio",
    "template_match_region_expand_ratios",
    "template_match_scales",
    "template_match_scales_by_key",
    "template_match_channels",
    "template_match_channels_by_key",
    "template_match_regions",
    "template_match_thresholds",
    "screen_detection_rules",
}
DEVICE_LEVEL_KEYS = {
    "poll_interval",
    "step_timeout",
    "keep_awake_interval_seconds",
    "exclude_ocr_wait_from_step_timeout",
    "idle_poll_interval_seconds",
}
DEVICE_APP_LEVEL_KEYS = {
    "persona",
    "photo_sent_cooldown_seconds",
    "online_photo_send_count",
    "new_greeting_click_x_offset",
    "new_greeting_click_y_offset",
    "next_message_click_x_offset",
    "next_message_click_y_offset",
}
APP_CONFIG_FILE_NAME = "config.json"
APP_DIALOGUE_OCR_CONFIG_FILE_NAME = "dialogue_ocr_config.json"
APP_LIST_OCR_CONFIG_FILE_NAME = "list_ocr_config.json"


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


def _split_override_scopes(override: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    app_level: dict[str, Any] = {}
    device_level: dict[str, Any] = {}
    device_app_level: dict[str, Any] = {}
    for key, value in override.items():
        if key in APP_LEVEL_KEYS:
            app_level[key] = copy.deepcopy(value)
        elif key in DEVICE_LEVEL_KEYS:
            device_level[key] = copy.deepcopy(value)
        elif key in DEVICE_APP_LEVEL_KEYS:
            device_app_level[key] = copy.deepcopy(value)
    return app_level, device_level, device_app_level


def _normalize_app_templates(app_name: str, app_config: dict[str, Any]) -> None:
    templates = app_config.get("templates", {})
    if not isinstance(templates, dict):
        return
    for group in ("buttons", "screens"):
        value = templates.get(group)
        if not isinstance(value, str) or not value:
            continue
        normalized = value.replace("/", "\\")
        app_template_path = str(Path("apps") / str(app_name) / "templates" / group)
        if normalized.startswith("runtime\\templates\\") or "\\runtime\\templates\\" in normalized:
            templates[group] = app_template_path
            continue
        if normalized.startswith("apps\\") and f"\\{str(app_name)}\\templates\\{group}" in normalized:
            templates[group] = app_template_path


def _normalize_match_threshold(app_config: dict[str, Any]) -> None:
    if not isinstance(app_config, dict):
        return
    try:
        app_config["match_threshold"] = float(app_config.get("match_threshold", DEFAULT_MATCH_THRESHOLD))
    except (TypeError, ValueError):
        app_config["match_threshold"] = DEFAULT_MATCH_THRESHOLD


def _normalize_huanyou_dialogue_rules(app_name: str, app_config: dict[str, Any]) -> None:
    if str(app_name) != "huanyou" or not isinstance(app_config, dict):
        return
    ocr_modes = app_config.get("ocr_modes", {})
    if not isinstance(ocr_modes, dict):
        return
    dialogue = ocr_modes.get("dialogue", {})
    if not isinstance(dialogue, dict):
        return
    rules = dialogue.get("rules", {})
    if not isinstance(rules, dict):
        return
    sender_detection = rules.get("sender_detection", {})
    if not isinstance(sender_detection, dict):
        return
    has_bubble_keys = any(str(key).startswith("bubble_color_") for key in sender_detection.keys())
    if str(sender_detection.get("method", "")).strip() == "center_position" and not has_bubble_keys:
        sender_detection["method"] = "bubble_color"
        sender_detection["bubble_color_min_saturation"] = 0.12
        sender_detection["bubble_color_min_value"] = 0.20
        sender_detection["bubble_color_sender2_hue_min"] = 110
        sender_detection["bubble_color_sender2_hue_max"] = 150
        sender_detection["bubble_color_sender2_blue_green_min"] = 65
        sender_detection["bubble_color_sender2_blue_red_ratio_min"] = 1.60
        sender_detection["bubble_color_sender2_blue_min_saturation"] = 0.30
    if "name_gap_ratio" not in rules and "name_gap" in rules:
        try:
            name_gap_value = float(rules["name_gap"])
        except (TypeError, ValueError):
            name_gap_value = 0.0
        if name_gap_value >= 1:
            rules["name_gap_ratio"] = 0.037
            rules.pop("name_gap", None)


def _normalize_app_configs(app_configs: dict[str, Any]) -> None:
    if not isinstance(app_configs, dict):
        return
    for app_name, app_config in app_configs.items():
        if isinstance(app_config, dict):
            _normalize_app_templates(str(app_name), app_config)
            _normalize_match_threshold(app_config)
            _normalize_huanyou_dialogue_rules(str(app_name), app_config)


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return copy.deepcopy(data) if isinstance(data, dict) else {}


def _normalize_secret_value(value: Any) -> str:
    text = str(value or "")
    for marker in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(marker, "")
    return text.strip()


def _read_local_apikey(config_path: Path) -> str:
    env_dir = config_path.parent / ".env"
    for path in (env_dir / "apikey.txt", env_dir / "apikey"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = _normalize_secret_value(line)
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                if _normalize_secret_value(key).lower() in {"apikey", "chatbot_apikey", "chatbot_api_key"}:
                    stripped = _normalize_secret_value(value).strip('"').strip("'")
                else:
                    continue
            return stripped
    return ""


def _write_local_apikey(config_path: Path, apikey: str) -> None:
    env_dir = config_path.parent / ".env"
    path = env_dir / "apikey.txt"
    value = _normalize_secret_value(apikey)
    env_dir.mkdir(parents=True, exist_ok=True)
    if not value:
        path.unlink(missing_ok=True)
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(f"CHATBOT_APIKEY={value}\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
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


def _app_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_CONFIG_FILE_NAME


def _app_dialogue_ocr_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_DIALOGUE_OCR_CONFIG_FILE_NAME


def _app_list_ocr_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_LIST_OCR_CONFIG_FILE_NAME


def _load_app_ocr_mode_override_files(app_config_root: Path, app_name: str) -> dict[str, Any]:
    ocr_modes: dict[str, Any] = {}
    for mode_name, path in (
        ("dialogue", _app_dialogue_ocr_config_path(app_config_root, app_name)),
        ("list", _app_list_ocr_config_path(app_config_root, app_name)),
    ):
        payload = _load_json_dict(path)
        if payload:
            ocr_modes[mode_name] = payload
    return {"ocr_modes": ocr_modes} if ocr_modes else {}


def _load_app_override_file(app_config_root: Path, app_name: str) -> dict[str, Any]:
    payload = _load_json_dict(_app_config_path(app_config_root, app_name))
    payload = _deep_merge(payload, _load_app_ocr_mode_override_files(app_config_root, app_name))
    app_level, _, _ = _split_override_scopes(payload)
    _normalize_app_templates(app_name, app_level)
    _normalize_match_threshold(app_level)
    _normalize_huanyou_dialogue_rules(app_name, app_level)
    return app_level


def _extract_legacy_app_level_offsets(app_config: dict[str, Any]) -> dict[str, Any]:
    offsets: dict[str, Any] = {}
    for key in (
        "new_greeting_click_x_offset",
        "new_greeting_click_y_offset",
        "next_message_click_x_offset",
        "next_message_click_y_offset",
    ):
        if key in app_config:
            offsets[key] = copy.deepcopy(app_config.pop(key))
    return offsets


def _normalize_device_scopes(device: dict[str, Any]) -> None:
    runtime_overrides = device.get("runtime_overrides", {})
    if not isinstance(runtime_overrides, dict):
        runtime_overrides = {}
    device["runtime_overrides"] = {
        key: copy.deepcopy(value)
        for key, value in runtime_overrides.items()
        if key in DEVICE_LEVEL_KEYS
    }

    app_overrides = device.get("app_overrides", {})
    if not isinstance(app_overrides, dict):
        device["app_overrides"] = {}
        return
    normalized: dict[str, dict[str, Any]] = {}
    for app_name, override in app_overrides.items():
        if not isinstance(override, dict):
            continue
        _, _, device_app_level = _split_override_scopes(override)
        if device_app_level:
            normalized[str(app_name)] = device_app_level
    device["app_overrides"] = normalized


def _merge_app_level_config(app_config_root: Path, app_name: str, updates: dict[str, Any]) -> None:
    if not updates:
        return
    existing = _load_app_override_file(app_config_root, app_name)
    merged = _deep_merge(existing, updates)
    _normalize_app_templates(app_name, merged)
    _normalize_match_threshold(merged)
    ocr_modes = merged.get("ocr_modes", {})
    ocr_modes = copy.deepcopy(ocr_modes) if isinstance(ocr_modes, dict) else {}
    dialogue_mode = ocr_modes.pop("dialogue", None)
    list_mode = ocr_modes.pop("list", None)
    if ocr_modes:
        merged["ocr_modes"] = ocr_modes
    else:
        merged.pop("ocr_modes", None)
    _atomic_write_json(_app_config_path(app_config_root, app_name), merged)
    for mode_payload, path in (
        (dialogue_mode, _app_dialogue_ocr_config_path(app_config_root, app_name)),
        (list_mode, _app_list_ocr_config_path(app_config_root, app_name)),
    ):
        if isinstance(mode_payload, dict) and mode_payload:
            _atomic_write_json(path, mode_payload)
        else:
            path.unlink(missing_ok=True)


def _merge_runtime_overrides(device: dict[str, Any], updates: dict[str, Any]) -> None:
    if not updates:
        return
    runtime_overrides = device.get("runtime_overrides", {})
    runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else {}
    runtime_overrides = _deep_merge(runtime_overrides, updates)
    device["runtime_overrides"] = runtime_overrides


def _merge_device_app_overrides(device: dict[str, Any], app_name: str, updates: dict[str, Any]) -> None:
    if not updates:
        return
    app_overrides = device.get("app_overrides", {})
    app_overrides = app_overrides if isinstance(app_overrides, dict) else {}
    existing = app_overrides.get(app_name, {})
    existing = existing if isinstance(existing, dict) else {}
    app_overrides[str(app_name)] = _deep_merge(existing, updates)
    device["app_overrides"] = app_overrides


def _apply_scoped_app_overrides(
    app_config_root: Path,
    device: dict[str, Any],
    app_overrides: dict[str, Any],
) -> None:
    if not isinstance(app_overrides, dict):
        return
    for raw_app_name, raw_override in app_overrides.items():
        if not isinstance(raw_override, dict):
            continue
        app_name = str(raw_app_name)
        app_level, device_level, device_app_level = _split_override_scopes(raw_override)
        _merge_app_level_config(app_config_root, app_name, app_level)
        _merge_runtime_overrides(device, device_level)
        _merge_device_app_overrides(device, app_name, device_app_level)
    _normalize_device_scopes(device)


class ConfigManager:
    _lock = threading.Lock()

    def __init__(
        self,
        config_path: str | Path = "config.json",
        outbox_path: str | Path = "outbox.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.outbox_path = Path(outbox_path)
        self.app_config_root = self.config_path.parent / "apps"
        self._config = self._load_json(self.config_path, DEFAULT_CONFIG)
        self._outbox = self._load_json(self.outbox_path, DEFAULT_OUTBOX)
        self._migrate_if_needed()
        self._normalize_loaded_device_scopes()
        self._move_config_apikey_to_env()
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
        _atomic_write_json(path, data)

    def _config_for_disk(self) -> dict[str, Any]:
        data = copy.deepcopy(self._config)
        global_config = data.get("global", {})
        if isinstance(global_config, dict):
            global_config.pop("apikey", None)
        return data

    def _write_config(self) -> None:
        self._atomic_write(self.config_path, self._config_for_disk())

    def _reload_config_for_write(self) -> None:
        self._config = self._load_json(self.config_path, DEFAULT_CONFIG)
        self._normalize_loaded_device_scopes()
        self._move_config_apikey_to_env()
        self._apply_environment_overrides()

    def _reload_outbox_for_write(self) -> None:
        self._outbox = self._load_json(self.outbox_path, DEFAULT_OUTBOX)

    def _backup_file(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        return backup

    def _migrate_if_needed(self) -> None:
        config_version = int(self._config.get("schema_version", 0) or 0)
        outbox_version = int(self._outbox.get("schema_version", 0) or 0)
        legacy_apps_present = "apps" in self._config
        if config_version < CURRENT_SCHEMA_VERSION or legacy_apps_present:
            self._backup_file(self.config_path)
            self._config = self._migrate_config(self._config, config_version, CURRENT_SCHEMA_VERSION)
            self._write_config()
        if outbox_version < CURRENT_SCHEMA_VERSION:
            self._backup_file(self.outbox_path)
            self._outbox = self._migrate_outbox(self._outbox, outbox_version, CURRENT_SCHEMA_VERSION)
            self._atomic_write(self.outbox_path, self._outbox)

    def _normalize_loaded_device_scopes(self) -> None:
        devices = self._config.get("devices", [])
        if not isinstance(devices, list):
            return
        changed = False
        for device in devices:
            if not isinstance(device, dict):
                continue
            before = copy.deepcopy(
                {
                    "runtime_overrides": device.get("runtime_overrides", {}),
                    "app_overrides": device.get("app_overrides", {}),
                }
            )
            _normalize_device_scopes(device)
            if before != {
                "runtime_overrides": device.get("runtime_overrides", {}),
                "app_overrides": device.get("app_overrides", {}),
            }:
                changed = True
        if changed:
            self._write_config()

    def _move_config_apikey_to_env(self) -> None:
        global_config = self._config.get("global", {})
        if not isinstance(global_config, dict) or "apikey" not in global_config:
            return
        file_apikey = _normalize_secret_value(global_config.pop("apikey", ""))
        if file_apikey and not os.getenv("CHATBOT_APIKEY") and not _read_local_apikey(self.config_path):
            _write_local_apikey(self.config_path, file_apikey)
        self._write_config()

    def _migrate_config(
        self,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        migrated = _deep_merge(DEFAULT_CONFIG, data)
        migrated["schema_version"] = to_version
        legacy_app_configs = migrated.pop("apps", {})
        if not isinstance(legacy_app_configs, dict):
            legacy_app_configs = {}
        devices = migrated.get("devices", [])
        legacy_app_offsets: dict[str, dict[str, Any]] = {}
        for app_name, app_config in list(legacy_app_configs.items()):
            if not isinstance(app_config, dict):
                continue
            normalized_app_config = copy.deepcopy(app_config)
            offsets = _extract_legacy_app_level_offsets(normalized_app_config)
            if offsets:
                legacy_app_offsets[str(app_name)] = offsets
            app_level, _, _ = _split_override_scopes(normalized_app_config)
            _merge_app_level_config(self.app_config_root, str(app_name), app_level)
        if isinstance(devices, list):
            migrated["devices"] = _normalize_devices(devices)
            for device in migrated["devices"]:
                if not isinstance(device, dict):
                    continue
                device.setdefault("selected_app", resolve_app_name(None))
                device.setdefault("persona", {})
                device.setdefault("contact_personas", {})
                runtime_overrides = device.get("runtime_overrides", {})
                runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else {}
                old_app_overrides = device.get("app_overrides", {})
                old_app_overrides = old_app_overrides if isinstance(old_app_overrides, dict) else {}
                device["runtime_overrides"] = runtime_overrides
                device["app_overrides"] = {}
                _apply_scoped_app_overrides(self.app_config_root, device, old_app_overrides)
                for app_name, offsets in legacy_app_offsets.items():
                    if str(device.get("selected_app", "")) == app_name or (
                        isinstance(old_app_overrides, dict) and app_name in old_app_overrides
                    ):
                        _merge_device_app_overrides(device, app_name, offsets)
                _normalize_device_scopes(device)
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
        apikey = _normalize_secret_value(os.getenv("CHATBOT_APIKEY")) or _read_local_apikey(self.config_path)
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
            self._reload_config_for_write()
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            raw_updates = copy.deepcopy(updates)
            scoped_app_overrides = raw_updates.pop("app_overrides", None)
            runtime_updates = raw_updates.pop("runtime_overrides", None)
            for device in devices:
                if isinstance(device, dict) and str(device.get("ip", "")) == device_ip:
                    device.update(raw_updates)
                    if isinstance(runtime_updates, dict):
                        _merge_runtime_overrides(device, runtime_updates)
                    if isinstance(scoped_app_overrides, dict):
                        _apply_scoped_app_overrides(self.app_config_root, device, scoped_app_overrides)
                    _normalize_device_scopes(device)
                    self._config["devices"] = _normalize_devices(devices)
                    self._write_config()
                    return
            new_device = {"ip": device_ip, **raw_updates}
            if isinstance(runtime_updates, dict):
                _merge_runtime_overrides(new_device, runtime_updates)
            if isinstance(scoped_app_overrides, dict):
                _apply_scoped_app_overrides(self.app_config_root, new_device, scoped_app_overrides)
            _normalize_device_scopes(new_device)
            devices.append(new_device)
            self._config["devices"] = _normalize_devices(devices)
            self._write_config()

    def add_device(self, device: dict[str, Any]) -> bool:
        device_ip = str(device.get("ip", "")).strip()
        if not device_ip:
            raise ValueError("device ip is required")
        with self._lock:
            self._reload_config_for_write()
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            if any(isinstance(existing, dict) and str(existing.get("ip", "")).strip() == device_ip for existing in devices):
                return False
            new_device = copy.deepcopy(device)
            _apply_scoped_app_overrides(self.app_config_root, new_device, new_device.get("app_overrides", {}))
            _normalize_device_scopes(new_device)
            devices.append(new_device)
            self._config["devices"] = _normalize_devices(devices)
            self._write_config()
            return True

    def remove_device(self, device_ip: str) -> bool:
        device_ip = str(device_ip).strip()
        if not device_ip:
            return False
        with self._lock:
            self._reload_config_for_write()
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
            self._write_config()
            return True

    def get_app_config(
        self,
        device_ip: str,
        app_name: str,
        default_config: dict[str, Any],
    ) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        app_level = self.get_app_overrides(app_name)
        runtime_overrides = (
            copy.deepcopy(device.get("runtime_overrides", {}))
            if isinstance(device.get("runtime_overrides", {}), dict)
            else {}
        )
        device_app_overrides = (
            copy.deepcopy(device.get("app_overrides", {}).get(app_name, {}))
            if isinstance(device.get("app_overrides", {}), dict)
            and isinstance(device.get("app_overrides", {}).get(app_name, {}), dict)
            else {}
        )
        merged = _deep_merge(default_config, app_level)
        merged = _deep_merge(merged, runtime_overrides)
        merged = _deep_merge(merged, device_app_overrides)
        _normalize_match_threshold(merged)
        return merged

    def get_app_overrides(self, app_name: str) -> dict[str, Any]:
        with self._lock:
            return _load_app_override_file(self.app_config_root, str(app_name))

    def update_app_config(self, app_name: str, updates: dict[str, Any]) -> None:
        scoped_updates, _, _ = _split_override_scopes(copy.deepcopy(updates))
        if not scoped_updates:
            return
        with self._lock:
            _merge_app_level_config(self.app_config_root, str(app_name), scoped_updates)

    def get_device_runtime_overrides(self, device_ip: str) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        runtime_overrides = device.get("runtime_overrides", {})
        return copy.deepcopy(runtime_overrides if isinstance(runtime_overrides, dict) else {})

    def update_device_runtime(self, device_ip: str, updates: dict[str, Any]) -> None:
        runtime_updates = {
            key: copy.deepcopy(value)
            for key, value in updates.items()
            if key in DEVICE_LEVEL_KEYS
        }
        if runtime_updates:
            self.update_device(device_ip, {"runtime_overrides": runtime_updates})

    def get_device_app_overrides(self, device_ip: str, app_name: str) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        app_overrides = device.get("app_overrides", {})
        if not isinstance(app_overrides, dict):
            return {}
        override = app_overrides.get(app_name, {})
        return copy.deepcopy(override if isinstance(override, dict) else {})

    def update_device_app_overrides(self, device_ip: str, app_name: str, updates: dict[str, Any]) -> None:
        _, _, device_app_updates = _split_override_scopes(copy.deepcopy(updates))
        if device_app_updates:
            self.update_device(device_ip, {"app_overrides": {app_name: device_app_updates}})

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
            self._reload_config_for_write()
            if keys == ("global", "apikey"):
                apikey = str(value or "").strip()
                _write_local_apikey(self.config_path, apikey)
                self._config.setdefault("global", {}).pop("apikey", None)
                self._write_config()
                if apikey:
                    self._config.setdefault("global", {})["apikey"] = apikey
                return
            current = self._config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
                if not isinstance(current, dict):
                    raise ValueError(f"cannot set nested key under non-dict: {key}")
            current[keys[-1]] = value
            self._write_config()

    def save_outbox_entry(self, entry: dict[str, Any]) -> None:
        fingerprint = entry.get("message_fingerprint")
        if not fingerprint:
            raise ValueError("outbox entry requires message_fingerprint")
        with self._lock:
            self._reload_outbox_for_write()
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
