from __future__ import annotations

import copy

from apps.wechat.ocr_config import DIALOGUE_OCR_MODE as WECHAT_DIALOGUE_OCR_MODE
from vision.ocr_rules import build_dialogue_result


def _dialogue_rules(**overrides):
    rules = copy.deepcopy(WECHAT_DIALOGUE_OCR_MODE["rules"])
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(rules.get(key), dict):
            rules[key].update(value)
        else:
            rules[key] = value
    return rules


def test_dialogue_sender_defaults_to_position_not_color_saturation():
    items = [
        {
            "text": "left message",
            "raw_text": "left message",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 120,
            "min_y": 100,
            "max_y": 140,
            "center_x": 70,
            "center_y": 120,
            "width": 100,
            "height": 40,
            "background_saturation": 0.70,
        },
        {
            "text": "right message",
            "raw_text": "right message",
            "confidence": 0.99,
            "min_x": 140,
            "max_x": 190,
            "min_y": 220,
            "max_y": 260,
            "center_x": 165,
            "center_y": 240,
            "width": 50,
            "height": 40,
            "background_saturation": 0.05,
        },
    ]

    result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            name_gap=40,
            basic_filter={
                "enable_max_x_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
            },
        ),
    )

    assert result["dialogues"] == [
        {"sender": "sender1", "content": "left message"},
        {"sender": "sender2", "content": "right message"},
    ]


def test_dialogue_position_method_is_configurable():
    items = [
        {
            "text": "wide message",
            "raw_text": "wide message",
            "confidence": 0.99,
            "min_x": 80,
            "max_x": 160,
            "min_y": 100,
            "max_y": 140,
            "center_x": 120,
            "center_y": 120,
            "width": 80,
            "height": 40,
            "background_saturation": 0.05,
        },
    ]
    common_rules = _dialogue_rules(
        sender_detection={
            "method": "position",
            "sender2_min_x_ratio": 0.52,
        },
        basic_filter={"enable_non_cjk_non_emoji_filter": False},
    )

    default_result = build_dialogue_result(items, (200, 300), common_rules)
    center_result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            sender_detection={
                **common_rules["sender_detection"],
                "method": "center_position",
            },
            basic_filter={"enable_non_cjk_non_emoji_filter": False},
        ),
    )

    assert default_result["dialogues"] == [{"sender": "sender1", "content": "wide message"}]
    assert center_result["dialogues"] == [{"sender": "sender2", "content": "wide message"}]
