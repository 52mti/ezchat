from __future__ import annotations

from apps.huanyou.config import DEFAULT_CONFIG as HUANYOU_CONFIG
from apps.wechat.config import DEFAULT_CONFIG as WECHAT_CONFIG
from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG


DEFAULT_PERSONA_ID = "REMOVED-BEARER-TOKEN"


def test_each_app_owns_dialogue_and_list_ocr_modes():
    for config in [WECHAT_CONFIG, XINYU_CONFIG, HUANYOU_CONFIG]:
        assert "dialogue" in config["ocr_modes"]
        assert "list" in config["ocr_modes"]
        assert config["ocr_modes"]["dialogue"]["preprocess"]["mode"] == "original"
        assert config["ocr_modes"]["list"]["preprocess"]["mode"] == "original"
        assert "auxiliary_mode" not in config["ocr_modes"]["list"]["preprocess"]


def test_each_app_uses_shared_runtime_defaults():
    for config in [WECHAT_CONFIG, XINYU_CONFIG]:
        assert config["poll_interval"] == 1
        assert config["keep_awake_interval_seconds"] == 200
        assert config["persona"]["requestPersona"] == DEFAULT_PERSONA_ID
    assert HUANYOU_CONFIG["poll_interval"] == 1
    assert HUANYOU_CONFIG["persona"]["requestPersona"] == DEFAULT_PERSONA_ID


def test_huanyou_uses_optimized_idle_intervals():
    assert HUANYOU_CONFIG["idle_poll_interval_seconds"] == 20
    assert HUANYOU_CONFIG["keep_awake_interval_seconds"] == 20


def test_huanyou_runtime_ocr_is_enabled_by_default():
    assert HUANYOU_CONFIG["ocr"]["runtime_enabled"] is True


def test_app_ocr_modes_are_not_shared_between_apps():
    assert WECHAT_CONFIG["ocr_modes"] is not XINYU_CONFIG["ocr_modes"]
    assert WECHAT_CONFIG["ocr_modes"] is not HUANYOU_CONFIG["ocr_modes"]
    assert XINYU_CONFIG["ocr_modes"] is not HUANYOU_CONFIG["ocr_modes"]


def test_huanyou_dialogue_sender_anchor_is_app_specific():
    huanyou_sender = HUANYOU_CONFIG["ocr_modes"]["dialogue"]["rules"]["sender_detection"]
    wechat_sender = WECHAT_CONFIG["ocr_modes"]["dialogue"]["rules"]["sender_detection"]
    xinyu_sender = XINYU_CONFIG["ocr_modes"]["dialogue"]["rules"]["sender_detection"]

    assert huanyou_sender["method"] == "bubble_color"
    assert huanyou_sender["bubble_color_sender2_hue_min"] == 110
    assert huanyou_sender["bubble_color_sender2_hue_max"] == 150
    assert huanyou_sender["bubble_color_sender2_blue_green_min"] == 65
    assert huanyou_sender["bubble_color_sender2_blue_red_ratio_min"] == 1.60
    assert huanyou_sender["bubble_color_sender2_blue_min_saturation"] == 0.30
    assert wechat_sender["method"] == "position"
    assert xinyu_sender["method"] == "bubble_color"
    assert xinyu_sender["bubble_color_sender2_hue_min"] == 0
    assert xinyu_sender["bubble_color_sender2_hue_max"] == 12
    assert xinyu_sender["bubble_color_sender2_blue_red_ratio_min"] == 0.80
    assert xinyu_sender["bubble_color_sender2_blue_min_saturation"] == 0.18


def test_huanyou_list_rules_enable_left_badge_unread_and_filter_marketing_rows():
    rules = HUANYOU_CONFIG["ocr_modes"]["list"]["rules"]

    assert rules["badge_default_unread_count"] == 1
    assert rules["enable_red_badge_detection"] is True
    assert rules["left_red_badge_force_default_unread_count"] is True
    assert rules["left_badge_enable_ocr_fallback"] is False
    assert rules["left_badge_read_red_badge_number"] is False
    assert rules["row_top_extra_top_ratio"] == 0.012
    assert rules["red_badge_enable_white_digit_bonus"] is True
    assert rules["red_badge_require_white_digit"] is True
    assert rules["red_badge_enable_relative_white_digit"] is True
    assert rules["red_badge_relative_white_value_delta"] == 20
    assert rules["red_badge_relative_white_saturation_delta"] == 20
    assert rules["red_badge_min_circularity"] == 0.65
    assert rules["left_red_badge_search_height_ratio"] == 0.05
    assert "在线匹配" in rules["excluded_keywords"]
    assert "在线配对" in rules["excluded_keywords"]
    assert "小助手" in rules["excluded_keywords"]
    assert "福利多多" in rules["excluded_keywords"]
    assert "对方回复5条消息" not in rules["content_excluded_keywords"]
    assert "对方回复5条消息" not in rules["unread_count_excluded_keywords"]
    assert "温馨提示" in rules["unread_count_excluded_keywords"]


def test_wechat_owns_wechat_list_layout_rules():
    rules = WECHAT_CONFIG["ocr_modes"]["list"]["rules"]

    assert rules["text_left_min_ratio"] == 0.16
    assert rules["text_right_max_ratio"] == 0.78
    assert rules["content_gap_ratio"] == 0.022
    assert rules["max_unread_count"] == 99
    assert len(rules["bottom_nav_keywords"]) == 5


def test_xinyu_owns_xinyu_list_layout_rules():
    rules = XINYU_CONFIG["ocr_modes"]["list"]["rules"]

    assert rules["badge_position"] == "left"
    assert rules["badge_default_unread_count"] == 1
    assert "全部消息" in rules["bottom_nav_keywords"]
    assert "全部消息" in rules["top_nav_keywords"]
    assert "全部消息" in rules["system_keywords"]


    assert rules["name_max_x_ratio"] == 0.8


def test_huanyou_owns_points_ocr_mode():
    points_mode = HUANYOU_CONFIG["ocr_modes"]["points"]

    assert points_mode["preprocess"]["mode"] == "original"
    assert "我的收益" in points_mode["rules"]["keywords"]
    assert "收益" in points_mode["rules"]["keywords"]
    assert "金币" in points_mode["rules"]["keywords"]


def test_huanyou_moves_next_message_click_offset_config_out_of_app_scope():
    assert "next_message_click_x_offset" not in HUANYOU_CONFIG
    assert "next_message_click_y_offset" not in HUANYOU_CONFIG


def test_each_app_owns_unread_fallback_rules():
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
    for config in [WECHAT_CONFIG, XINYU_CONFIG, HUANYOU_CONFIG]:
        rules = config["ocr_modes"]["list"]["rules"]
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
