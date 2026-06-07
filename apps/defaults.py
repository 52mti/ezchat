from __future__ import annotations

import copy
from typing import Any


DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "poll_interval": 1,
    "keep_awake_interval_seconds": 200,
    "exclude_ocr_wait_from_step_timeout": True,
    "ocr_max_concurrency": 1,
}

DEFAULT_REQUEST_PERSONA_ID = "REMOVED-BEARER-TOKEN"

DEFAULT_PERSONA_CONFIG: dict[str, Any] = {
    "requestPersona": DEFAULT_REQUEST_PERSONA_ID,
    "name": "",
    "age": "",
    "work": "",
    "location": "",
    "renshe": "",
}

DEFAULT_APP_CONFIG: dict[str, Any] = {
    **DEFAULT_RUNTIME_CONFIG,
    "match_threshold": 0.7,
    "popup_match_threshold": 0.8,
    "persona": DEFAULT_PERSONA_CONFIG,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def with_runtime_defaults(config: dict[str, Any] | None) -> dict[str, Any]:
    return _deep_merge(DEFAULT_RUNTIME_CONFIG, config or {})


def with_app_defaults(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = _deep_merge(DEFAULT_APP_CONFIG, config or {})
    persona = merged.get("persona", {})
    if isinstance(persona, dict) and not str(persona.get("requestPersona", "") or "").strip():
        persona["requestPersona"] = DEFAULT_REQUEST_PERSONA_ID
    return merged
