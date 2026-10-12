from __future__ import annotations

from apps import get_registered_apps


def test_registered_apps_include_supported_runtime_choices():
    apps = get_registered_apps()

    assert list(apps) == ["xinyu", "xinyu_graph"]
    assert apps["xinyu"].name == "心遇"
    assert apps["xinyu_graph"].name == "心遇图"


def test_legacy_app_aliases_resolve_to_xinyu():
    from apps import resolve_app_name

    assert resolve_app_name("wechat") == "xinyu"
    assert resolve_app_name("huanyou") == "xinyu"  # 欢友已下线，老配置归一到心遇
