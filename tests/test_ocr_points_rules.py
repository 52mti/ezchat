from __future__ import annotations

from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG
from apps.xinyu.ocr_rules import build_points_result


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


