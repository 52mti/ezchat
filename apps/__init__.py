from __future__ import annotations

from typing import TYPE_CHECKING

from apps.xinyu.app import XinyuApp
from apps.huanyou.app import HuanyouApp

if TYPE_CHECKING:
    from .base import BaseApp

REGISTERED_APPS: dict[str, type["BaseApp"]] = {
    "xinyu": XinyuApp,
    "huanyou": HuanyouApp,
}
LEGACY_APP_ALIASES: dict[str, str] = {
    "wechat": "xinyu",
}

def get_registered_apps() -> dict[str, type["BaseApp"]]:
    return dict(REGISTERED_APPS)


def get_default_app_name() -> str:
    return next(iter(REGISTERED_APPS))


def resolve_app_name(app_name: str | None) -> str:
    normalized = str(app_name or "").strip()
    if not normalized:
        return get_default_app_name()
    if normalized in REGISTERED_APPS:
        return normalized
    return LEGACY_APP_ALIASES.get(normalized, get_default_app_name())


__all__ = [
    "XinyuApp",
    "HuanyouApp",
    "REGISTERED_APPS",
    "LEGACY_APP_ALIASES",
    "get_registered_apps",
    "get_default_app_name",
    "resolve_app_name",
]
