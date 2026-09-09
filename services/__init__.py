from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AIClient",
    "ChatHistoryStore",
    "ConfigManager",
    "DEFAULT_FULL_PACKAGE_PAGE_URL",
    "DEFAULT_RELEASE_MANIFEST_URL",
    "DEFAULT_UPDATE_SHARE_URL",
    "LicenseClient",
    "LocalTtsServer",
    "LocalReleaseState",
    "MemoryStore",
    "MessageProcessor",
    "ReleaseDecision",
    "ReleaseManifest",
    "ReleaseManifestError",
    "ResolvedUpdatePackage",
    "UpdateSourceError",
    "VisionDialogueClient",
    "VisionDialogueError",
    "RuleLearningStore",
    "build_message_fingerprint",
    "cleanup_old_error_captures",
    "cleanup_old_logs",
    "contact_key",
    "evaluate_release_manifest",
    "fetch_release_manifest",
    "full_package_download_page_url",
    "load_local_release_state",
    "resolve_lanzou_update_package",
    "resolve_release_decision",
    "setup_runtime_logging",
    "update_release",
    "update_sources",
    "write_error_incident",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AIClient": (".ai_client", "AIClient"),
    "ChatHistoryStore": (".chat_history_store", "ChatHistoryStore"),
    "ConfigManager": (".config_manager", "ConfigManager"),
    "LicenseClient": (".license_client", "LicenseClient"),
    "LocalTtsServer": (".local_tts", "LocalTtsServer"),
    "MemoryStore": (".memory_store", "MemoryStore"),
    "MessageProcessor": (".message_processor", "MessageProcessor"),
    "VisionDialogueClient": (".vision_dialogue_client", "VisionDialogueClient"),
    "VisionDialogueError": (".vision_dialogue_client", "VisionDialogueError"),
    "RuleLearningStore": (".rule_learning", "RuleLearningStore"),
    "build_message_fingerprint": (".message_state", "build_message_fingerprint"),
    "contact_key": (".message_state", "contact_key"),
    "cleanup_old_error_captures": (".retention", "cleanup_old_error_captures"),
    "cleanup_old_logs": (".runtime_logging", "cleanup_old_logs"),
    "setup_runtime_logging": (".runtime_logging", "setup_runtime_logging"),
    "write_error_incident": (".runtime_logging", "write_error_incident"),
    "DEFAULT_RELEASE_MANIFEST_URL": (".update_release", "DEFAULT_RELEASE_MANIFEST_URL"),
    "LocalReleaseState": (".update_release", "LocalReleaseState"),
    "ReleaseDecision": (".update_release", "ReleaseDecision"),
    "ReleaseManifest": (".update_release", "ReleaseManifest"),
    "ReleaseManifestError": (".update_release", "ReleaseManifestError"),
    "evaluate_release_manifest": (".update_release", "evaluate_release_manifest"),
    "fetch_release_manifest": (".update_release", "fetch_release_manifest"),
    "load_local_release_state": (".update_release", "load_local_release_state"),
    "resolve_release_decision": (".update_release", "resolve_release_decision"),
    "DEFAULT_FULL_PACKAGE_PAGE_URL": (".update_sources", "DEFAULT_FULL_PACKAGE_PAGE_URL"),
    "DEFAULT_UPDATE_SHARE_URL": (".update_sources", "DEFAULT_UPDATE_SHARE_URL"),
    "ResolvedUpdatePackage": (".update_sources", "ResolvedUpdatePackage"),
    "UpdateSourceError": (".update_sources", "UpdateSourceError"),
    "full_package_download_page_url": (".update_sources", "full_package_download_page_url"),
    "resolve_lanzou_update_package": (".update_sources", "resolve_lanzou_update_package"),
    "update_release": (".update_release", None),
    "update_sources": (".update_sources", None),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
