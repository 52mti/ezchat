from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from apps.xinyu.app import XinyuApp

if TYPE_CHECKING:
    from .base import BaseApp

REGISTERED_APPS: dict[str, type["BaseApp"]] = {
    "xinyu": XinyuApp,
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


def run_ocr_mode_for(app_name: str | None):
    """返回指定 App 自己的 OCR 规则引擎入口（每个 App 独立维护）。

    跨 App 的通用消费方（visual_steps、UI 的 OCR 预览）用它按 App 路由；
    App 自己的 steps 应直接 import 本 App 的 ocr_rules，不经此路由。
    """
    from importlib import import_module

    resolved = resolve_app_name(app_name)
    try:
        module = import_module(f"apps.{resolved}.ocr_rules")
    except ImportError:
        module = import_module(f"apps.{get_default_app_name()}.ocr_rules")
    return module.run_ocr_mode


__all__ = [
    "XinyuApp",
    "REGISTERED_APPS",
    "LEGACY_APP_ALIASES",
    "get_registered_apps",
    "get_default_app_name",
    "resolve_app_name",
    "run_ocr_mode_for",
]
