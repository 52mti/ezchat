from __future__ import annotations

from typing import Any

from services.defaults import with_app_defaults

from .config_manager import ConfigManager, _deep_merge


def resolve_persona(
    config_manager: ConfigManager,
    device_ip: str,
    app_name: str,
    contact_name: str = "",
    *,
    device: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global_persona = config_manager.get("global", "persona", default={})
    current_device = config_manager.find_device(device_ip) or device or {}
    if not isinstance(current_device, dict):
        current_device = {}

    app_overrides = current_device.get("app_overrides", {})
    app_override = app_overrides.get(app_name, {}) if isinstance(app_overrides, dict) else {}
    app_persona = app_override.get("persona", {}) if isinstance(app_override, dict) else {}
    app_persona = with_app_defaults({"persona": app_persona if isinstance(app_persona, dict) else {}})["persona"]

    device_persona = current_device.get("persona", {})
    device_persona = device_persona if isinstance(device_persona, dict) else {}

    contact_personas = current_device.get("contact_personas", {})
    contact_persona = {}
    if contact_name and isinstance(contact_personas, dict):
        contact_persona = contact_personas.get(contact_name, {}) or {}

    persona = _deep_merge(global_persona if isinstance(global_persona, dict) else {}, device_persona)
    persona = _deep_merge(persona, app_persona)
    return _deep_merge(persona, contact_persona if isinstance(contact_persona, dict) else {})
