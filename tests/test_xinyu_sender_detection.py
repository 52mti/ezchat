from __future__ import annotations

import copy

from apps.xinyu.ocr_config import DIALOGUE_OCR_MODE as XINYU_DIALOGUE_OCR_MODE
import apps.xinyu.ocr_rules as xinyu_ocr_rules
from apps.xinyu.ocr_rules import build_dialogue_result


def _dialogue_rules(**overrides):
    rules = copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"])
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(rules.get(key), dict):
            rules[key].update(value)
        else:
            rules[key] = value
    return rules



def test_dialogue_position_method_ignores_raw_saturation():
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
            name_gap_ratio=40 / 300,
            sender_detection={"method": "position", "sender2_min_x_ratio": 0.52},
            basic_filter={
                "enable_max_x_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
                "enable_edge_spanning_filter": False,
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



def test_dialogue_bubble_color_method_prefers_purple_bubble_over_position():
    def item(text: str, min_x: int, max_x: int, min_y: int, max_y: int, hue: float) -> dict:
        return {
            "text": text,
            "raw_text": text,
            "confidence": 0.99,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "center_x": (min_x + max_x) / 2,
            "center_y": (min_y + max_y) / 2,
            "width": max_x - min_x,
            "height": max_y - min_y,
            "background_saturation": 0.45,
            "background_hue": hue,
            "background_value": 0.70,
            "background_blue_green_delta": 0.0,
            "background_blue_red_ratio": 1.0,
        }

    items = [
        item("non purple message", 30, 110, 100, 140, 90),
        item("purple message", 20, 100, 210, 250, 122),
    ]

    result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            name_gap_ratio=20 / 300,
            sender_detection={
                "method": "bubble_color",
                "sender2_min_x_ratio": 0.52,
                "bubble_color_min_saturation": 0.12,
                "bubble_color_min_value": 0.20,
                "bubble_color_sender2_hue_min": 110,
                "bubble_color_sender2_hue_max": 150,
            },
            basic_filter={
                "enable_max_x_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
                "enable_edge_spanning_filter": False,
            },
        ),
    )

    assert result["dialogues"] == [
        {"sender": "sender1", "content": "non purple message"},
        {"sender": "sender2", "content": "purple message"},
    ]



def test_dialogue_bubble_color_method_accepts_blue_purple_camera_shift():
    items = [
        {
            "text": "blue shifted purple",
            "raw_text": "blue shifted purple",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 150,
            "min_y": 100,
            "max_y": 140,
            "center_x": 85,
            "center_y": 120,
            "width": 130,
            "height": 40,
            "background_saturation": 0.30,
            "background_hue": 94,
            "background_value": 0.70,
            "background_blue_green_delta": 100,
            "background_blue_red_ratio": 2.2,
        }
    ]

    result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            name_gap_ratio=20 / 300,
            sender_detection={
                "method": "bubble_color",
                "sender2_min_x_ratio": 0.52,
                "bubble_color_min_saturation": 0.12,
                "bubble_color_min_value": 0.20,
                "bubble_color_sender2_hue_min": 110,
                "bubble_color_sender2_hue_max": 150,
                "bubble_color_enable_blue_detection": True,
                "bubble_color_sender2_blue_green_min": 65,
                "bubble_color_sender2_blue_red_ratio_min": 1.60,
                "bubble_color_sender2_blue_min_saturation": 0.30,
            },
            basic_filter={
                "enable_max_x_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
            },
        ),
    )

    assert result["dialogues"] == [{"sender": "sender2", "content": "blue shifted purple"}]



def test_dialogue_bubble_color_method_keeps_low_saturation_blue_card_as_sender1():
    items = [
        {
            "text": "blue image card",
            "raw_text": "blue image card",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 150,
            "min_y": 100,
            "max_y": 140,
            "center_x": 85,
            "center_y": 120,
            "width": 130,
            "height": 40,
            "background_saturation": 0.36,
            "background_hue": 93,
            "background_value": 0.67,
            "background_blue_green_delta": 72,
            "background_blue_red_ratio": 1.48,
        }
    ]

    result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            name_gap_ratio=20 / 300,
            sender_detection={
                "method": "bubble_color",
                "sender2_min_x_ratio": 0.52,
                "bubble_color_min_saturation": 0.12,
                "bubble_color_min_value": 0.20,
                "bubble_color_sender2_hue_min": 110,
                "bubble_color_sender2_hue_max": 150,
                "bubble_color_enable_blue_detection": True,
                "bubble_color_sender2_blue_green_min": 65,
                "bubble_color_sender2_blue_red_ratio_min": 1.60,
                "bubble_color_sender2_blue_min_saturation": 0.30,
            },
            basic_filter={
                "enable_max_x_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
            },
        ),
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "blue image card"}]



def test_xinyu_dialogue_bubble_color_method_marks_red_bubble_as_sender2():
    item = {
        "text": "红色气泡",
        "raw_text": "红色气泡",
        "confidence": 0.99,
        "min_x": 20,
        "max_x": 150,
        "min_y": 100,
        "max_y": 140,
        "center_x": 85,
        "center_y": 120,
        "width": 130,
        "height": 40,
        "background_saturation": 0.42,
        "background_hue": 4.0,
        "background_value": 0.78,
        "background_blue_green_delta": -8.0,
        "background_blue_red_ratio": 0.92,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender2", "content": "红色气泡"}]



def test_xinyu_dialogue_bubble_color_method_marks_wraparound_red_bubble_as_sender2():
    item = {
        "text": "深红气泡",
        "raw_text": "深红气泡",
        "confidence": 0.99,
        "min_x": 20,
        "max_x": 150,
        "min_y": 100,
        "max_y": 140,
        "center_x": 85,
        "center_y": 120,
        "width": 130,
        "height": 40,
        "background_saturation": 0.66,
        "background_hue": 172.0,
        "background_value": 0.53,
        "background_blue_green_delta": 21.0,
        "background_blue_red_ratio": 0.49,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender2", "content": "深红气泡"}]



def test_xinyu_dialogue_bubble_color_method_keeps_low_saturation_pink_bubble_as_sender1():
    item = {
        "text": "浅粉气泡",
        "raw_text": "浅粉气泡",
        "confidence": 0.99,
        "min_x": 20,
        "max_x": 150,
        "min_y": 100,
        "max_y": 140,
        "center_x": 85,
        "center_y": 120,
        "width": 130,
        "height": 40,
        "background_saturation": 0.40,
        "background_hue": 177.0,
        "background_value": 0.78,
        "background_blue_green_delta": -8.0,
        "background_blue_red_ratio": 0.92,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "浅粉气泡"}]



def test_xinyu_dialogue_sender_fusion_records_color_position_conflict():
    item = {
        "text": "测试内容",
        "raw_text": "测试内容",
        "confidence": 0.99,
        "min_x": 50,
        "max_x": 120,
        "min_y": 100,
        "max_y": 140,
        "center_x": 85,
        "center_y": 120,
        "width": 70,
        "height": 40,
        "background_saturation": 0.60,
        "background_hue": 5.0,
        "background_value": 0.78,
        "background_blue_green_delta": -8.0,
        "background_blue_red_ratio": 0.92,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender2", "content": "测试内容"}]
    assert item["sender_color"] == "sender2"
    assert item["sender_position"] == "sender1"
    assert item["sender_conflict"] is True
    assert item["sender_method"] == "color_position_fusion"
    assert 0.0 < item["sender_confidence"] < 1.0



def test_xinyu_dialogue_bubble_color_method_keeps_white_blue_bubble_as_sender1():
    item = {
        "text": "浅色气泡",
        "raw_text": "浅色气泡",
        "confidence": 0.99,
        "min_x": 24,
        "max_x": 104,
        "min_y": 170,
        "max_y": 210,
        "center_x": 64.0,
        "center_y": 190,
        "width": 80,
        "height": 40,
        "background_saturation": 0.09,
        "background_hue": 102.0,
        "background_value": 0.93,
        "background_blue_green_delta": 34.0,
        "background_blue_red_ratio": 1.34,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "浅色气泡"}]
    assert "stats" not in result["debug"]



def test_xinyu_dialogue_bubble_color_method_does_not_treat_dark_blue_header_as_sender2():
    item = {
        "text": "顶部标题",
        "raw_text": "顶部标题",
        "confidence": 0.99,
        "min_x": 20,
        "max_x": 150,
        "min_y": 100,
        "max_y": 140,
        "center_x": 85,
        "center_y": 120,
        "width": 130,
        "height": 40,
        "background_saturation": 0.42,
        "background_hue": 103.0,
        "background_value": 0.48,
        "background_blue_green_delta": 23.0,
        "background_blue_red_ratio": 9.5,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [item],
        (200, 300),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "顶部标题"}]


def test_xinyu_dialogue_wrap_line_inherits_first_line_sender():
    """换行续行按几何合并，sender 继承首行——续行自身判定不作门槛。

    回归背景：「哪？」这类我方消息的换行短续行，位置中心落在对方领地
    （sender1），旧逻辑要求续行 sender 判对才肯合并，颜色采样一旦失灵
    续行就会带着错的 sender 脱落成独立消息。
    """
    first_line = {
        "text": "我还没能去过呢，你最喜欢",
        "raw_text": "我还没能去过呢，你最喜欢",
        "confidence": 0.99,
        "min_x": 154,
        "max_x": 446,
        "min_y": 268,
        "max_y": 298,
        "center_x": 300,
        "center_y": 283,
        "width": 292,
        "height": 30,
        # 红气泡 → 首行可靠判为 sender2
        "background_saturation": 0.42,
        "background_hue": 4.0,
        "background_value": 0.78,
        "background_blue_green_delta": -8.0,
        "background_blue_red_ratio": 0.92,
    }
    wrap_line = {
        "text": "哪？",
        "raw_text": "哪？",
        "confidence": 0.99,
        "min_x": 154,
        "max_x": 198,
        "min_y": 318,
        "max_y": 346,
        "center_x": 176,
        "center_y": 332,
        "width": 44,
        "height": 28,
        # 无色采样（hue=None）→ 位置回退：中心 0.294 屏宽 → 会被独立判成 sender1
        "background_saturation": 0.02,
        "background_hue": None,
        "background_value": 0.9,
        "background_blue_green_delta": 0.0,
        "background_blue_red_ratio": 1.0,
    }

    result = xinyu_ocr_rules.build_dialogue_result(
        [first_line, wrap_line],
        (598, 1290),
        copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"]),
    )

    assert result["dialogues"] == [
        {"sender": "sender2", "content": "我还没能去过呢，你最喜欢哪？"}
    ]
