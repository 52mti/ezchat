from __future__ import annotations

from apps.huanyou.config import DEFAULT_CONFIG as HUANYOU_CONFIG
from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG
from vision.ocr_rules import build_points_result


HUANYOU_POINTS_RULES = HUANYOU_CONFIG["ocr_modes"]["points"]["rules"]
XINYU_POINTS_RULES = XINYU_CONFIG["ocr_modes"]["points"]["rules"]


def test_points_rules_extract_my_points_value():
    items = [
        {
            "text": "\u6211\u7684\u79ef\u5206",
            "raw_text": "\u6211\u7684\u79ef\u5206",
            "confidence": 0.99,
            "min_x": 80,
            "max_x": 170,
            "min_y": 200,
            "max_y": 232,
            "center_x": 125,
            "center_y": 216,
            "width": 90,
            "height": 32,
            "background_saturation": 0.0,
        },
        {
            "text": "1288",
            "raw_text": "1288",
            "confidence": 0.99,
            "min_x": 190,
            "max_x": 260,
            "min_y": 202,
            "max_y": 232,
            "center_x": 225,
            "center_y": 217,
            "width": 70,
            "height": 30,
            "background_saturation": 0.0,
        },
    ]

    result = build_points_result(items, (681, 1498), XINYU_POINTS_RULES)

    assert result["points"] == "1288"
    assert result["balance"] == "1288"


def test_points_rules_picks_nearby_numeric_value_when_keyword_and_number_are_split():
    items = [
        {
            "text": "1013.25",
            "raw_text": "1013.25",
            "confidence": 0.9969,
            "min_x": 52,
            "max_x": 150,
            "min_y": 333,
            "max_y": 361,
            "center_x": 101.0,
            "center_y": 347.0,
            "width": 98,
            "height": 28,
            "background_saturation": 0.7362,
        },
        {
            "text": "\u6211\u7684\u79ef\u5206",
            "raw_text": "\u6211\u7684\u79ef\u5206",
            "confidence": 0.9985,
            "min_x": 49,
            "max_x": 117,
            "min_y": 370,
            "max_y": 393,
            "center_x": 83.0,
            "center_y": 381.5,
            "width": 68,
            "height": 23,
            "background_saturation": 0.7373,
        },
    ]

    result = build_points_result(items, (681, 1498), XINYU_POINTS_RULES)

    assert result["points"] == "1013.25"
    assert result["balance"] == "1013.25"


def test_points_rules_extract_huanyou_income_value_with_unit_near_keyword():
    items = [
        {
            "text": "124.15金币",
            "raw_text": "124.15金币",
            "confidence": 0.996,
            "min_x": 39,
            "max_x": 170,
            "min_y": 318,
            "max_y": 346,
            "center_x": 104.5,
            "center_y": 332.0,
            "width": 131,
            "height": 28,
            "background_saturation": 0.0,
        },
        {
            "text": "我的收益",
            "raw_text": "我的收益",
            "confidence": 0.998,
            "min_x": 36,
            "max_x": 106,
            "min_y": 360,
            "max_y": 384,
            "center_x": 71.0,
            "center_y": 372.0,
            "width": 70,
            "height": 24,
            "background_saturation": 0.0,
        },
    ]

    result = build_points_result(items, (681, 1498), HUANYOU_POINTS_RULES)

    assert result["points"] == "124.15"
    assert result["balance"] == "124.15"


def test_points_rules_extract_huanyou_coin_value_without_income_keyword():
    items = [
        {
            "text": "125.29金币",
            "raw_text": "125.29金币",
            "confidence": 0.996,
            "min_x": 39,
            "max_x": 170,
            "min_y": 318,
            "max_y": 346,
            "center_x": 104.5,
            "center_y": 332.0,
            "width": 131,
            "height": 28,
            "background_saturation": 0.0,
        },
    ]

    result = build_points_result(items, (681, 1498), HUANYOU_POINTS_RULES)

    assert result["points"] == "125.29"
    assert result["balance"] == "125.29"


def test_points_rules_prefers_number_before_coin_keyword():
    items = [
        {
            "text": "125.29金币56钻石加入家族关注粉丝好友编辑",
            "raw_text": "125.29金币56钻石加入家族关注粉丝好友编辑",
            "confidence": 0.996,
            "min_x": 39,
            "max_x": 170,
            "min_y": 318,
            "max_y": 346,
            "center_x": 104.5,
            "center_y": 332.0,
            "width": 131,
            "height": 28,
            "background_saturation": 0.0,
        },
    ]

    result = build_points_result(items, (681, 1498), HUANYOU_POINTS_RULES)

    assert result["points"] == "125.29"
    assert result["balance"] == "125.29"


def test_points_rules_ignore_trailing_numeric_noise_after_coin_keyword():
    items = [
        {
            "text": "125.29金币56钻石加入家族关注粉丝好友编辑",
            "raw_text": "125.29金币56钻石加入家族关注粉丝好友编辑",
            "confidence": 0.996,
            "min_x": 39,
            "max_x": 260,
            "min_y": 318,
            "max_y": 346,
            "center_x": 149.5,
            "center_y": 332.0,
            "width": 221,
            "height": 28,
            "background_saturation": 0.0,
        },
        {
            "text": "56",
            "raw_text": "56",
            "confidence": 0.995,
            "min_x": 180,
            "max_x": 220,
            "min_y": 320,
            "max_y": 344,
            "center_x": 200.0,
            "center_y": 332.0,
            "width": 40,
            "height": 24,
            "background_saturation": 0.0,
        },
    ]

    result = build_points_result(items, (681, 1498), HUANYOU_POINTS_RULES)

    assert result["points"] == "125.29"
    assert result["balance"] == "125.29"
