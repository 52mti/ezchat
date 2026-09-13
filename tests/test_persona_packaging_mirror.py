from __future__ import annotations

"""打包/更新镜像验收（docs/13 九节 C1）：

- apps/<app>/ 下四个随包人设默认文件存在（开发端，入库随包）；
- build.py 的收录范围覆盖 apps/（无需新增条目）；
- 更新包排除与更新器保护互为镜像：runtime/、.env 在两侧都不进包/不可触碰；
- 根 config.json 在更新器保护清单中（设备级人设设置所在）。
"""

import importlib.util
import json
from pathlib import Path

import update_apply

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_module", PROJECT_ROOT / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persona_bundled_default_files_exist():
    assert (PROJECT_ROOT / "apps" / "xinyu" / "persona_default.json").is_file()
    assert (PROJECT_ROOT / "apps" / "xinyu" / "greetings_default.txt").is_file()
    payload = json.loads((PROJECT_ROOT / "apps" / "xinyu" / "persona_default.json").read_text(encoding="utf-8"))
    assert set(payload) == {"基础设定", "人物信息", "边界条件", "审核标准"}


def test_build_include_paths_cover_apps_directory():
    build = _load_build_module()
    assert "apps" in build.INCLUDE_PATHS


def test_build_staging_excludes_env_directory():
    build = _load_build_module()
    assert ".env" in build._IGNORE_NAMES


def test_update_payload_excludes_runtime_files():
    build = _load_build_module()
    for relative in ("runtime/persona_sessions/1_1_1_1/xinyu/sessions.sqlite", "runtime/greetings/pingtai.txt"):
        assert build._archive_excludes(relative, update_payload=True), relative
    # apps 随包默认不受更新排除
    assert not build._archive_excludes("apps/xinyu/persona_default.json", update_payload=True)


def test_update_apply_never_touches_runtime_and_config_json():
    assert "runtime/" in update_apply.UNTOUCHABLE_PREFIXES
    assert any(prefix.rstrip("/").endswith("config.json") for prefix in update_apply.PRESERVE_PREFIXES)


def test_update_apply_allows_dev_default_overwrite_in_apps():
    # 开发端随包默认（apps/ 根级 txt）不在不可触碰/保留前缀内，也不属于
    # apps/<app>/templates/**/__customer_* 客户模板：更新包正常覆盖下发
    relative = "apps/xinyu/persona_default.json"
    assert not relative.startswith(tuple(update_apply.UNTOUCHABLE_PREFIXES))
    assert not relative.rstrip("/").startswith(tuple(p.rstrip("/") for p in update_apply.PRESERVE_PREFIXES))
    parts = relative.split("/")
    assert not ("templates" in parts and any("__customer_" in part for part in parts))
