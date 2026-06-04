from __future__ import annotations

from apps import get_registered_apps


def test_registered_apps_include_supported_runtime_choices():
    apps = get_registered_apps()

    assert list(apps) == ["xinyu", "huanyou"]
    assert apps["xinyu"].name == "心遇"
    assert apps["huanyou"].name == "欢友"
