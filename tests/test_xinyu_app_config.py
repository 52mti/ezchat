from __future__ import annotations

from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG


def test_app_owns_dialogue_and_list_ocr_modes():
    config = XINYU_CONFIG
    assert "dialogue" in config["ocr_modes"]
    assert "list" in config["ocr_modes"]
    assert config["ocr_modes"]["dialogue"]["preprocess"]["mode"] == "original"
    assert config["ocr_modes"]["list"]["preprocess"]["mode"] == "original"
    assert "auxiliary_mode" not in config["ocr_modes"]["list"]["preprocess"]


def test_app_uses_shared_runtime_defaults():
    config = XINYU_CONFIG
    assert config["poll_interval"] == 1
    assert config["keep_awake_interval_seconds"] == 200


def test_xinyu_owns_xinyu_list_layout_rules():
    rules = XINYU_CONFIG["ocr_modes"]["list"]["rules"]

    assert rules["badge_position"] == "left"
    assert rules["badge_default_unread_count"] == 1
    assert "全部消息" in rules["bottom_nav_keywords"]
    assert "全部消息" in rules["top_nav_keywords"]
    assert "全部消息" in rules["system_keywords"]


    assert rules["name_max_x_ratio"] == 0.8


def test_app_owns_unread_fallback_rules():
    required_keys = {
        "badge_crop_padding",
        "badge_ocr_scale",
        "row_badge_ocr_scale",
        "row_badge_min_confidence",
        "row_strip_left_min_ratio",
        "row_strip_extra_left_ratio",
        "row_strip_width_ratio",
        "row_strip_height_ratio",
        "row_strip_max_height_ratio",
        "row_strip_min_height_px",
        "row_strip_bottom_nav_margin",
    }
    rules = XINYU_CONFIG["ocr_modes"]["list"]["rules"]
    assert required_keys.issubset(rules.keys())


def test_xinyu_bottom_nav_keeps_bottom_tabs():
    rules = XINYU_CONFIG["ocr_modes"]["list"]["rules"]

    assert "喜欢" in rules["bottom_nav_keywords"]
    assert "心动" in rules["bottom_nav_keywords"]
    assert "家族广场" in rules["bottom_nav_keywords"]
    assert "消息" in rules["bottom_nav_keywords"]
    assert "我的" in rules["bottom_nav_keywords"]
    assert rules["bottom_nav_accept_any_text"] is True
    assert "bottom_nav_fixed_names" not in rules
