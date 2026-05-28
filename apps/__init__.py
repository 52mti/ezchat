from __future__ import annotations

from typing import TYPE_CHECKING

from apps.huanyou.app import HuanyouApp
from apps.wechat.app import WechatApp
from apps.xinyu.app import XinyuApp

if TYPE_CHECKING:
    from .base import BaseApp


def get_registered_apps() -> dict[str, type["BaseApp"]]:
    return {
        "wechat": WechatApp,
        "xinyu": XinyuApp,
        "huanyou": HuanyouApp,
    }


__all__ = ["WechatApp", "XinyuApp", "HuanyouApp", "get_registered_apps"]
