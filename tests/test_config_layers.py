"""配置三层层级测试：代码默认 < apps/<app>/json（随包开发默认） < runtime/configs/<app>（客户层）。"""

from __future__ import annotations

import json

import services.config_manager as config_manager_module
from services.config_manager import ConfigManager, register_app_config_normalizer


def _make_manager(tmp_path):
    return ConfigManager(config_path=tmp_path / "config.json", outbox_path=tmp_path / "outbox.json")


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_packaged_app_json_overrides_code_defaults(tmp_path):
    defaults_root = tmp_path / "apps" / "xinyu"
    _write(defaults_root / "config.json", {"match_threshold": 0.9})
    (tmp_path / "runtime" / "configs").mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "runtime" / "configs" / ".layout_v2", {"note": "stamp"})

    manager = _make_manager(tmp_path)

    assert manager.get_app_overrides("xinyu")["match_threshold"] == 0.9


def test_customer_layer_visual_keys_are_stripped(tmp_path):
    """客户层的开发域视觉键读取时被剥离：随包默认不被遮蔽，popup.* 保留。"""
    defaults_root = tmp_path / "apps" / "xinyu"
    _write(defaults_root / "config.json", {"match_threshold": 0.9})
    customer = tmp_path / "runtime" / "configs" / "xinyu"
    _write(
        customer / "config.json",
        {
            "match_threshold": 0.6,
            "template_match_regions": {
                "button.back": [0, 0.9, 1.0, 0.1],
                "popup.custom.popup_custom_1": [0.1, 0.3, 0.8, 0.4],
            },
            "ocr_modes": {"points": {"rules": {}}},
            "photo_sent_cooldown_seconds": 3600,
        },
    )
    _write(tmp_path / "runtime" / "configs" / ".layout_v2", {"note": "stamp"})

    manager = _make_manager(tmp_path)
    overrides = manager.get_app_overrides("xinyu")

    assert overrides["match_threshold"] == 0.9  # 客户层的 0.6 被剥掉，随包默认生效
    regions = overrides["template_match_regions"]
    assert "popup.custom.popup_custom_1" in regions
    assert "button.back" not in regions
    assert "points" not in overrides.get("ocr_modes", {})
    assert overrides["photo_sent_cooldown_seconds"] == 3600  # 业务参数是客户资产，保留


def test_update_app_config_never_persists_developer_scoped_keys(tmp_path):
    defaults_root = tmp_path / "apps" / "xinyu"
    _write(defaults_root / "config.json", {"match_threshold": 0.9})
    _write(defaults_root / "dialogue_ocr_config.json", {"rules": {"min_confidence": 0.55}})
    _write(tmp_path / "runtime" / "configs" / ".layout_v2", {"note": "stamp"})

    manager = _make_manager(tmp_path)
    manager.update_app_config(
        "xinyu",
        {
            "match_threshold": 0.99,
            "template_match_regions": {"popup.custom.popup_custom_2": [0.2, 0.3, 0.5, 0.2]},
            "ocr_modes": {"dialogue": {"rules": {"min_confidence": 0.61}}},
        },
    )

    saved = json.loads(
        (tmp_path / "runtime" / "configs" / "xinyu" / "config.json").read_text(encoding="utf-8")
    )
    assert "match_threshold" not in saved
    assert list(saved["template_match_regions"]) == ["popup.custom.popup_custom_2"]
    assert "ocr_modes" not in saved
    # dialogue 规则写专属文件；随包默认的 0.55 不被复制进客户层。
    mode = json.loads(
        (tmp_path / "runtime" / "configs" / "xinyu" / "dialogue_ocr_config.json").read_text(encoding="utf-8")
    )
    assert mode["rules"]["min_confidence"] == 0.61
    overrides = manager.get_app_overrides("xinyu")
    assert overrides["match_threshold"] == 0.9


def test_packaged_ocr_mode_files_feed_ocr_modes(tmp_path):
    defaults_root = tmp_path / "apps" / "xinyu"
    _write(defaults_root / "dialogue_ocr_config.json", {"rules": {"min_confidence": 0.55}})
    _write(tmp_path / "runtime" / "configs" / ".layout_v2", {"note": "stamp"})

    manager = _make_manager(tmp_path)
    overrides = manager.get_app_overrides("xinyu")

    assert overrides["ocr_modes"]["dialogue"]["rules"]["min_confidence"] == 0.55


def test_app_config_normalizer_hook_is_invoked(tmp_path):
    _write(tmp_path / "apps" / "demo" / "config.json", {"match_threshold": 0.7})
    _write(tmp_path / "runtime" / "configs" / ".layout_v2", {"note": "stamp"})
    seen: list[dict] = []
    register_app_config_normalizer("demo", seen.append)
    try:
        manager = _make_manager(tmp_path)
        manager.get_app_overrides("demo")
    finally:
        config_manager_module._APP_CONFIG_NORMALIZERS.pop("demo", None)

    assert len(seen) == 1
    assert seen[0]["match_threshold"] == 0.7


def test_repo_packaged_defaults_exist_for_registered_apps():
    import apps

    for app_name in apps.get_registered_apps():
        for file_name in ("config.json", "dialogue_ocr_config.json", "list_ocr_config.json"):
            path = __import__("pathlib").Path("apps") / app_name / file_name
            assert path.is_file(), f"缺少随包默认文件: {path}"


def test_black_screen_detection_toggle_defaults_off():
    """临时关闭黑屏亮度判定（防夜间深色画面误判）；复查后应改回 True。"""
    from apps.xinyu.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["enable_black_screen_detection"] is False


def test_main_screen_detection_rules_ship_with_packaged_defaults():
    """主界面五 Tab 判定规则只住 apps/xinyu/config.json（数据不进 py）；曾被客户层剥离清空后丢失，自坐标缓存锚点重建。"""
    import json as _json

    payload = _json.loads(
        (__import__("pathlib").Path("apps") / "xinyu" / "config.json").read_text(encoding="utf-8")
    )
    rules = payload.get("screen_detection_rules") or []
    screens = {rule.get("screen") for rule in rules}
    assert screens == {"xinyu-缘分", "xinyu-动态", "xinyu-娱乐交友", "xinyu-消息", "xinyu-我的"}
    from apps.xinyu.config import DEFAULT_CONFIG

    assert "screen_detection_rules" not in DEFAULT_CONFIG  # 唯一来源是 json
    for rule in rules:
        condition = rule.get("condition") or {}
        assert condition.get("type") == "ocr_raw"
        assert condition.get("text")
        assert condition.get("region") or rule.get("match_region")


def test_packaged_regions_cover_actually_matched_custom_keys():
    """随包默认的区域全集必须覆盖代码实际匹配的每个类别（含全屏兜底键）。

    客户层剥离后区域只来自开发端默认；缺一个键，运行期就会抛
    TemplateMatchRegionMissingError（曾因 button.custom.主界面心遇按钮 缺失炸过）。
    """
    import apps.xinyu.steps as xinyu_steps

    matched_keys = {
        xinyu_steps.XINYU_APP_ENTRY_BUTTON,
        xinyu_steps.BACK_BUTTON,
        xinyu_steps.BACK_BUTTON_2,
        xinyu_steps.INPUT_BOX_BUTTON,
        xinyu_steps.MICROPHONE_BUTTON,
        xinyu_steps.SEND_BUTTON_KEY,
        xinyu_steps.NEW_GREETING_BUTTON,
        xinyu_steps.PHOTO_BUTTON,
        xinyu_steps.COMPLETE_BUTTON,
        xinyu_steps.CHAT_LIST_SCREEN,
        xinyu_steps.CHAT_WINDOW_SCREEN,
        xinyu_steps.MY_SCREEN,
        xinyu_steps.CALL_SCREEN,
        xinyu_steps.NO_PHONE_SCREEN,
        xinyu_steps.BLACK_SCREEN,
    }
    from services.config_manager import ConfigManager

    manager = ConfigManager()
    regions = manager.get_app_overrides("xinyu").get("template_match_regions", {})
    for key in sorted(matched_keys):
        # button.back.secondary 未配置时回退 button.back 的区域。
        effective_key = "button.back" if key == "button.back.secondary" else key
        assert effective_key in regions, f"xinyu 随包默认缺少区域: {key}"
    assert regions["screen.no_phone"] == [0.0, 0.0, 1.0, 1.0]
    assert regions["screen.black_screen"] == [0.0, 0.0, 1.0, 1.0]
