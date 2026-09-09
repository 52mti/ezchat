from __future__ import annotations

import copy

import numpy as np
from PIL import Image

import vision.ocr_rules as ocr_rules
from apps.wechat.ocr_config import LIST_OCR_MODE as WECHAT_LIST_OCR_MODE
from apps.xinyu.ocr_config import LIST_OCR_MODE as XINYU_LIST_OCR_MODE
from apps.huanyou.ocr_config import LIST_OCR_MODE as HUANYOU_LIST_OCR_MODE
from vision.ocr_rules import build_list_result as _build_list_result


def _wechat_rules(**overrides):
    rules = copy.deepcopy(WECHAT_LIST_OCR_MODE["rules"])
    rules.update(overrides)
    return rules


def _xinyu_rules(**overrides):
    rules = copy.deepcopy(XINYU_LIST_OCR_MODE["rules"])
    rules.update(overrides)
    return rules


def _huanyou_rules(**overrides):
    rules = copy.deepcopy(HUANYOU_LIST_OCR_MODE["rules"])
    rules.update(overrides)
    return rules


def _base_mode_for_rules(rules):
    if not isinstance(rules, dict):
        return WECHAT_LIST_OCR_MODE
    if rules.get("bottom_nav_accept_any_text") is True:
        return XINYU_LIST_OCR_MODE
    if "row_top_extra_top_ratio" in rules:
        return HUANYOU_LIST_OCR_MODE
    if rules.get("badge_position") == "left":
        return XINYU_LIST_OCR_MODE
    return WECHAT_LIST_OCR_MODE


def build_list_result(rule_items, image_size, rules=None, **kwargs):
    complete_rules = copy.deepcopy(_base_mode_for_rules(rules)["rules"])
    complete_rules.update(rules or {})
    return _build_list_result(rule_items, image_size, complete_rules, **kwargs)


def _item(text: str, min_x: int, max_x: int, min_y: int, max_y: int, confidence: float = 0.99) -> dict:
    return {
        "text": text,
        "raw_text": text,
        "confidence": confidence,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center_x": (min_x + max_x) / 2,
        "center_y": (min_y + max_y) / 2,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def test_list_rules_merge_name_content_and_ignore_avatar_time():
    items = [
        _item("绝世侠客", 140, 270, 360, 405),
        _item("09:30", 580, 650, 370, 402),
        _item("绝", 68, 94, 391, 430),
        _item("最近工作有点忙，可能要过几天..", 141, 542, 418, 456, 0.96),
        _item("最近工作有点忙，可能要过几天", 142, 541, 419, 456, 0.95),
        _item("无敌欧巴", 140, 269, 510, 550),
        _item("太棒了！", 141, 238, 564, 598),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.01))

    assert [row["name"] for row in result["message_list"]] == ["绝世侠客", "无敌欧巴"]
    assert [row["content"] for row in result["message_list"]] == ["最近工作有点忙，可能要过几天..", "太棒了！"]


def test_list_rules_output_bottom_nav_with_normalized_bounds():
    items = [
        _item("通讯录", 158, 234, 1358, 1388),
        _item("发现", 306, 360, 1355, 1388),
        _item("视频号", 431, 509, 1355, 1388),
        _item("我", 593, 625, 1356, 1387),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "bottom_nav_keywords": ["消息", "通讯录", "发现", "视频号", "我"],
        },
    )

    assert [item["name"] for item in result["bottom_nav"]] == ["通讯录", "发现", "视频号", "我"]
    assert "text" not in result["bottom_nav"][0]
    assert result["bottom_nav"][0]["bounds"] == [0.232, 0.907, 0.344, 0.927]


def test_list_rules_keep_right_side_unread_count_above_two():
    items = [
        _item("\u7edd\u4e16\u4fa0\u5ba2", 140, 270, 360, 405),
        _item("09:30", 580, 650, 370, 402),
        _item("3", 619, 640, 420, 444),
        _item("\u6700\u8fd1\u5de5\u4f5c\u6709\u70b9\u5fd9", 141, 542, 418, 456),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.01))

    assert result["message_list"][0]["name"] == "\u7edd\u4e16\u4fa0\u5ba2"
    assert result["message_list"][0]["favorability"] is None
    assert result["message_list"][0]["unread_count"] == 3
    assert result["message_list"][0]["content"] == "\u6700\u8fd1\u5de5\u4f5c\u6709\u70b9\u5fd9"
    assert len(result["message_list"][0]["bounds"]) == 4
    assert result["message_list"][0]["bounds"] == [
        round(value, 3) for value in result["message_list"][0]["bounds"]
    ]


def test_list_rules_keep_short_right_side_unread_badge():
    items = [
        _item("\u65e0\u654c\u963f\u5f3a", 141, 266, 797, 834),
        _item("09:30", 577, 645, 804, 830),
        _item("4", 620, 633, 854, 870),
        _item("\u4f60\u770b\u90a3\u90e8\u65b0\u51fa\u7684\u7535\u5f71\u4e86\u5417", 141, 539, 846, 887),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.02))

    assert result["message_list"][0]["name"] == "\u65e0\u654c\u963f\u5f3a"
    assert result["message_list"][0]["unread_count"] == 4


def test_list_rules_caps_right_side_99_plus_badge():
    items = [
        _item("\u7528\u6237\u901a\u77e5", 140, 270, 210, 247),
        _item("14:57", 579, 648, 226, 255),
        _item("99+", 598, 640, 276, 299),
        _item("\u60a8\u6536\u5230\u4e00\u6761\u65b0\u7684\u8bed\u97f3\u901a\u8bdd\u9080\u8bf7", 141, 514, 278, 316),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.01))

    assert result["message_list"][0]["name"] == "\u7528\u6237\u901a\u77e5"
    assert result["message_list"][0]["unread_count"] == 99


def test_list_rules_accept_noisy_right_side_unread_badge_text():
    items = [
        _item("\u7528\u6237\u901a\u77e5", 140, 270, 210, 247),
        _item("14:57", 579, 648, 226, 255),
        _item("99F", 598, 640, 276, 299),
        _item("\u60a8\u6536\u5230\u4e00\u6761\u65b0\u7684\u8bed\u97f3\u901a\u8bdd\u9080\u8bf7", 141, 514, 278, 316),
        _item("\u5fe7\u90c1\u8001\u738b", 140, 270, 925, 962),
        _item("09:30", 579, 648, 947, 974),
        _item("3.", 616, 634, 996, 1015),
        _item("\u7a0d\u7b49\u4e00\u4e0b", 141, 514, 979, 1016),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.01))

    assert result["message_list"][0]["unread_count"] == 99
    assert result["message_list"][1]["unread_count"] == 3


def test_list_rules_fill_missing_unread_count_from_row_strip_fallback(monkeypatch):
    items = [
        _item("\u65e0\u654c\u963f\u5f3a", 141, 266, 797, 834),
        _item("09:30", 577, 645, 804, 830),
        _item("\u4f60\u770b\u90a3\u90e8\u65b0\u51fa\u7684\u7535\u5f71\u4e86\u5417", 141, 539, 846, 887),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: None)
    monkeypatch.setattr(ocr_rules, "_read_unread_number_from_row_strip", lambda *args, **kwargs: 4)

    result = build_list_result(
        items,
        (681, 1498),
        {"min_box_height_ratio": 0.02},
        source_image=Image.new("RGB", (681, 1498), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "\u65e0\u654c\u963f\u5f3a"
    assert result["message_list"][0]["unread_count"] == 4


def test_list_rules_keep_row_bounds_tight_to_content():
    items = [
        _item("\u7528\u6237\u901a\u77e5", 140, 270, 210, 247),
        _item("14:57", 579, 648, 226, 255),
        _item("99+", 598, 640, 276, 299),
        _item("\u60a8\u6536\u5230\u4e00\u6761\u65b0\u7684\u8bed\u97f3\u901a\u8bdd\u9080\u8bf7", 141, 514, 278, 316),
        _item("\u7edd\u4e16\u4fa0\u5ba2", 140, 270, 360, 405),
        _item("09:30", 580, 650, 370, 402),
        _item("3", 619, 640, 420, 444),
        _item("\u6700\u8fd1\u5de5\u4f5c\u6709\u70b9\u5fd9", 141, 542, 418, 456),
    ]

    result = build_list_result(items, (681, 1498), _wechat_rules(min_box_height_ratio=0.01))

    assert result["message_list"][0]["name"] == "\u7528\u6237\u901a\u77e5"
    assert result["message_list"][0]["bounds"] == [0.182, 0.132, 0.975, 0.219]


def test_list_rules_skip_bottom_nav_rows_from_message_list():
    items = [
        _item("\u9633\u5149\u5c11\u5e74", 140, 270, 1069, 1106),
        _item("09:30", 579, 648, 1089, 1116),
        _item("3", 616, 634, 1140, 1159),
        _item("\u65e9\u4e0a\u597d", 141, 514, 1127, 1164),
        _item("\u901a\u8baf\u5f55", 158, 234, 1358, 1388),
        _item("\u53d1\u73b0", 306, 360, 1355, 1388),
        _item("\u89c6\u9891\u53f7", 431, 509, 1355, 1388),
        _item("\u6211", 593, 625, 1356, 1387),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "bottom_nav_keywords": ["\u6d88\u606f", "\u901a\u8baf\u5f55", "\u53d1\u73b0", "\u89c6\u9891\u53f7", "\u6211"],
        },
    )

    assert [row["name"] for row in result["message_list"]] == ["\u9633\u5149\u5c11\u5e74"]


def test_list_rules_treat_right_number_as_favorability_for_xinyu_layout():
    items = [
        _item("白嫖", 146, 260, 720, 760),
        _item("44", 582, 636, 726, 754),
        _item("你好", 146, 320, 770, 802),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "badge_position": "left",
        },
    )

    assert result["message_list"][0]["name"] == "白嫖"
    assert result["message_list"][0]["favorability"] == 44
    assert result["message_list"][0]["unread_count"] is None


def test_list_rules_support_xinyu_top_nav_and_left_badge_default_unread(monkeypatch):
    items = [
        _item("全部信息", 110, 190, 168, 198),
        _item("未回复", 230, 310, 168, 198),
        _item("在线", 360, 420, 168, 198),
        _item("距离", 492, 552, 168, 198),
        _item("心动对象", 146, 260, 360, 402),
        _item("刚刚", 582, 636, 366, 398),
        _item("你好呀", 146, 320, 420, 454),
    ]

    monkeypatch.setattr(
        ocr_rules,
        "_detect_red_badge_bounds",
        lambda *args, **kwargs: (32, 386, 52, 406),
    )
    monkeypatch.setattr(
        ocr_rules,
        "_read_badge_number_from_crop",
        lambda *args, **kwargs: None,
    )

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "badge_position": "left",
            "badge_default_unread_count": 1,
            "bottom_nav_keywords": ["全部信息", "未回复", "在线", "距离"],
            "top_nav_keywords": ["全部信息", "未回复", "在线", "距离"],
            "top_nav_top_ratio": 0.10,
            "top_nav_bottom_ratio": 0.20,
            "system_keywords": ["全部信息", "未回复", "在线", "距离"],
        },
        source_image=Image.new("RGB", (681, 1498), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert [item["name"] for item in result["bottom_nav"]] == ["全部信息", "未回复", "在线", "距离"]
    assert [row["name"] for row in result["message_list"]] == ["心动对象"]
    assert result["message_list"][0]["unread_count"] == 1


def test_list_rules_left_badge_does_not_default_unread_without_red_detection(monkeypatch):
    items = [
        _item("心动对象", 146, 260, 360, 402),
        _item("你好呀", 146, 320, 420, 454),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: None)

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "badge_position": "left",
            "badge_default_unread_count": 1,
        },
        source_image=Image.new("RGB", (681, 1498), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["unread_count"] is None


def test_list_rules_zero_badge_default_does_not_create_unread(monkeypatch):
    items = [
        _item("微风", 128, 210, 210, 248),
        _item("心情不好，能陪我聊会吗？", 128, 420, 268, 304),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: (10, 10, 24, 24))
    monkeypatch.setattr(ocr_rules, "_read_badge_number_from_crop", lambda *args, **kwargs: None)

    result = build_list_result(
        items,
        (560, 997),
        _huanyou_rules(
            min_box_height_ratio=0.01,
            badge_default_unread_count=0,
            enable_red_badge_detection=True,
            left_red_badge_force_default_unread_count=True,
        ),
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "微风"
    assert result["message_list"][0]["unread_count"] is None


def test_huanyou_list_rules_use_red_badge_block_as_unread_one(monkeypatch):
    items = [
        _item("梦幻之旅", 128, 260, 265, 297),
        _item("丘比特牵线礼物", 128, 360, 305, 338),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: (88, 240, 114, 266))
    monkeypatch.setattr(
        ocr_rules,
        "_read_left_badge_number_from_row_crop",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call left badge OCR")),
    )

    result = build_list_result(
        items,
        (560, 997),
        _huanyou_rules(min_box_height_ratio=0.01),
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "梦幻之旅"
    assert result["message_list"][0]["unread_count"] == 1


def test_huanyou_list_rules_force_cupid_gift_content_as_unread(monkeypatch):
    items = [
        _item("珍惜当下", 128, 260, 265, 297),
        _item("丘比特牵线礼物", 128, 360, 305, 338),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: None)

    result = build_list_result(
        items,
        (560, 997),
        _huanyou_rules(min_box_height_ratio=0.01),
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "珍惜当下"
    assert result["message_list"][0]["content"] == "丘比特牵线礼物"
    assert result["message_list"][0]["unread_count"] == 1


def test_huanyou_list_rules_force_fate_assistant_content_as_unread(monkeypatch):
    items = [
        _item("珍惜当下", 128, 260, 265, 297),
        _item("缘分小助手", 128, 360, 305, 338),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: None)

    result = build_list_result(
        items,
        (560, 997),
        _huanyou_rules(min_box_height_ratio=0.01),
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "珍惜当下"
    assert result["message_list"][0]["content"] == "缘分小助手"
    assert result["message_list"][0]["unread_count"] == 1


def test_detect_red_badge_bounds_ignore_small_red_patch_for_huanyou():
    rules = _huanyou_rules()
    image = Image.new("RGB", (560, 997), "white")
    pixels = image.load()
    for x in range(18, 27):
        for y in range(146, 155):
            pixels[x, y] = (220, 40, 40)
    for x in range(90, 115):
        for y in range(135, 160):
            if (x - 102) ** 2 + (y - 147) ** 2 <= 12 ** 2:
                pixels[x, y] = (220, 60, 45)
    for y in range(141, 154):
        pixels[102, y] = (245, 245, 245)
        pixels[103, y] = (245, 245, 245)

    badge_bounds = ocr_rules._detect_red_badge_bounds(
        np.array(image),
        (0, 140, 540, 220),
        560,
        997,
        rules,
    )

    assert badge_bounds is not None
    assert badge_bounds[0] >= 80


def test_detect_red_badge_bounds_requires_white_digit_for_huanyou():
    rules = _huanyou_rules()
    image = Image.new("RGB", (560, 997), "white")
    pixels = image.load()
    for x in range(90, 115):
        for y in range(135, 160):
            if (x - 102) ** 2 + (y - 147) ** 2 <= 12 ** 2:
                pixels[x, y] = (220, 60, 45)

    badge_bounds = ocr_rules._detect_red_badge_bounds(
        np.array(image),
        (0, 140, 540, 220),
        560,
        997,
        rules,
    )

    assert badge_bounds is None


def test_detect_red_badge_bounds_accepts_relative_white_digit_for_huanyou():
    rules = _huanyou_rules(
        red_badge_enable_relative_white_digit=True,
        red_badge_relative_white_value_delta=20,
        red_badge_relative_white_saturation_delta=20,
        red_badge_white_digit_value_min=160,
        red_badge_white_digit_saturation_max=80,
    )
    image = Image.new("RGB", (560, 997), "white")
    pixels = image.load()
    for x in range(90, 115):
        for y in range(135, 160):
            if (x - 102) ** 2 + (y - 147) ** 2 <= 12 ** 2:
                pixels[x, y] = (75, 45, 38)
    for y in range(141, 154):
        pixels[102, y] = (105, 115, 126)
        pixels[103, y] = (105, 115, 126)

    badge_bounds = ocr_rules._detect_red_badge_bounds(
        np.array(image),
        (0, 140, 540, 220),
        560,
        997,
        rules,
    )

    assert badge_bounds is not None
    assert 90 <= badge_bounds[0] <= 115


def test_detect_red_badge_bounds_prefers_white_digit_candidate_for_huanyou():
    rules = _huanyou_rules(
        red_badge_enable_white_digit_bonus=True,
        red_badge_require_white_digit=True,
        red_badge_enable_relative_white_digit=False,
        red_badge_white_digit_min_pixels=4,
        red_badge_white_digit_bonus_multiplier=2.0,
        left_badge_max_ratio=0.35,
    )
    image = Image.new("RGB", (560, 997), "white")
    pixels = image.load()
    for x in range(88, 105):
        for y in range(140, 157):
            if (x - 96) ** 2 + (y - 148) ** 2 <= 8 ** 2:
                pixels[x, y] = (220, 40, 40)
    for x in range(146, 170):
        for y in range(136, 160):
            if (x - 158) ** 2 + (y - 148) ** 2 <= 12 ** 2:
                pixels[x, y] = (220, 40, 40)
    for y in range(141, 155):
        pixels[158, y] = (245, 245, 245)
        pixels[159, y] = (245, 245, 245)

    badge_bounds = ocr_rules._detect_red_badge_bounds(
        np.array(image),
        (0, 140, 540, 220),
        560,
        997,
        rules,
    )

    assert badge_bounds is not None
    assert 140 <= badge_bounds[0] <= 170


def test_list_rules_keep_numeric_name_for_xinyu_layout():
    items = [
        _item("12345", 146, 260, 720, 760),
        _item("44", 582, 636, 726, 754),
        _item("浣犲ソ", 146, 320, 770, 802),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "badge_position": "left",
        },
    )

    assert result["message_list"][0]["name"] == "12345"
    assert result["message_list"][0]["favorability"] == 44
    assert result["message_list"][0]["content"] == "浣犲ソ"


def test_xinyu_list_rules_keep_thin_content_text():
    rules = XINYU_LIST_OCR_MODE["rules"]
    items = [
        _item("薄荷 [25岁·石家庄]", 134, 308, 492, 565, confidence=0.96),
        _item("下午0435", 448, 536, 508, 552, confidence=0.92),
        _item("看看你的诚意", 135, 260, 583, 633, confidence=0.99),
        _item("24G", 486, 536, 583, 633, confidence=0.80),
    ]

    result = build_list_result(items, (566, 2518), rules)

    assert result["message_list"][0]["name"] == "薄荷"
    assert result["message_list"][0]["content"] == "看看你的诚意"


def test_list_rules_strip_configured_name_suffix_keywords():
    items = [
        _item("微笑的绿巨人", 128, 270, 740, 780),
        _item("附近", 282, 330, 742, 778),
        _item("[对方发来搭讪礼物]", 128, 360, 790, 824),
    ]

    result = build_list_result(
        items,
        (560, 997),
        {
            "min_box_height_ratio": 0.01,
            "name_suffix_keywords": ["附近"],
        },
    )

    assert result["message_list"][0]["name"] == "微笑的绿巨人"
    assert result["message_list"][0]["content"] == "对方发来搭讪礼物"


def test_huanyou_list_rules_filter_marketing_entries_and_keep_real_chats():
    rules = HUANYOU_LIST_OCR_MODE["rules"]
    items = [
        _item("消息", 24, 75, 78, 108),
        _item("关系", 152, 205, 78, 108),
        _item("优质用户回归奖励", 92, 402, 140, 186),
        _item("谁偷偷看过你", 128, 260, 266, 302),
        _item("Ta距离你只有70m，快去看看吧~", 128, 432, 312, 342),
        _item("群聊派对", 128, 235, 388, 424),
        _item("已有1278人在这里嗨", 128, 360, 434, 468),
        _item("家族狂欢", 128, 245, 510, 548),
        _item("5201621人在玩", 250, 390, 516, 544),
        _item("火龙果", 128, 185, 650, 687),
        _item("[丘比特牵线礼物]", 128, 340, 700, 731),
        _item("梦想之中的钢笔", 128, 305, 760, 796),
        _item("[丘比特牵线礼物]", 128, 340, 810, 842),
        _item("交友", 40, 80, 920, 948),
        _item("广场", 154, 196, 920, 948),
        _item("娱乐", 265, 305, 920, 948),
        _item("消息", 382, 424, 920, 948),
        _item("我的", 496, 536, 920, 948),
    ]

    result = build_list_result(items, (560, 997), rules)

    assert [row["name"] for row in result["message_list"]] == ["火龙果", "梦想之中的钢笔"]
    assert result["message_list"][1]["content"] == "丘比特牵线礼物"
    assert [item["name"] for item in result["bottom_nav"]] == ["交友", "广场", "娱乐", "消息", "我的"]


def test_huanyou_list_rules_filter_online_match_and_reward_rows():
    rules = HUANYOU_LIST_OCR_MODE["rules"]
    items = [
        _item("地面上的绿茶", 128, 300, 150, 188),
        _item("对方回复5条消息，将获得额外奖励", 128, 460, 205, 238),
        _item("在线配对", 128, 260, 260, 298),
        _item("风中行者", 128, 260, 315, 348),
        _item("小时在线交友", 128, 320, 330, 368),
        _item("福利多多活动", 128, 320, 335, 372),
        _item("微风", 128, 210, 470, 508),
        _item("心情不好，能陪我聊会吗？", 128, 420, 525, 558),
    ]

    result = build_list_result(items, (560, 997), rules)

    assert [row["name"] for row in result["message_list"]] == ["地面上的绿茶", "微风"]
    assert result["message_list"][0]["content"] == "对方回复5条消息，将获得额外奖励"
    assert result["message_list"][0]["unread_count"] is None
    assert result["message_list"][1]["content"] == "心情不好，能陪我聊会吗？"


def test_huanyou_list_rules_filter_helper_rows_and_disable_visual_unread(monkeypatch):
    rules = HUANYOU_LIST_OCR_MODE["rules"]
    items = [
        _item("微笑的绿巨人", 128, 300, 302, 340),
        _item("你来了呀，我正想着你呢", 128, 445, 357, 390),
        _item("小助手", 128, 260, 397, 435),
        _item("温馨提示：单身找个单身的关注了你", 128, 520, 452, 486),
    ]

    monkeypatch.setattr(ocr_rules, "_detect_red_badge_bounds", lambda *args, **kwargs: (10, 10, 24, 24))
    monkeypatch.setattr(ocr_rules, "_read_badge_number_from_crop", lambda *args, **kwargs: 1)

    result = build_list_result(
        items,
        (560, 997),
        rules,
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert [row["name"] for row in result["message_list"]] == ["微笑的绿巨人"]
    assert result["message_list"][0]["unread_count"] == 1


def test_huanyou_list_rules_accept_left_ocr_unread_badge():
    rules = HUANYOU_LIST_OCR_MODE["rules"]
    items = [
        _item("1", 96, 108, 246, 270),
        _item("爱听歌的猕猴桃", 128, 330, 250, 288),
        _item("丘比特牵线礼物", 128, 360, 305, 338),
    ]

    result = build_list_result(items, (560, 997), rules)

    assert result["message_list"][0]["name"] == "爱听歌的猕猴桃"
    assert result["message_list"][0]["unread_count"] == 1


def test_huanyou_list_rules_do_not_split_numeric_username_as_unread():
    rules = HUANYOU_LIST_OCR_MODE["rules"]
    items = [
        _item("12z", 128, 190, 580, 615),
        _item("附近", 205, 250, 582, 612),
        _item("Hello，我是那个想和你一起创造记忆的人", 128, 520, 630, 660),
        _item("08:13", 500, 545, 635, 655),
    ]

    result = build_list_result(items, (560, 997), rules)

    assert result["message_list"][0]["name"] == "12z"
    assert result["message_list"][0]["unread_count"] is None
    assert result["message_list"][0]["content"].startswith("Hello")


def test_huanyou_list_rules_read_left_badge_from_zoomed_row_crop(monkeypatch):
    rules = _huanyou_rules(left_badge_enable_ocr_fallback=True)
    items = [
        _item("梦幻之旅", 128, 260, 265, 297),
        _item("丘比特牵线礼物", 128, 360, 305, 338),
    ]

    def fake_ocr(_image, config):
        assert config["scale"] == rules["left_badge_ocr_scale"]
        assert config["region"][0] == rules["left_badge_ocr_left_ratio"]
        return [{"text": "1", "confidence": 0.98}]

    monkeypatch.setattr(ocr_rules, "ocr", fake_ocr)

    result = build_list_result(
        items,
        (560, 997),
        rules,
        source_image=Image.new("RGB", (560, 997), "white"),
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["name"] == "梦幻之旅"
    assert result["message_list"][0]["unread_count"] == 1


def test_huanyou_list_rules_use_ratio_row_top_offset_for_left_badge():
    rules = _huanyou_rules(
        row_top_extra_top_ratio=0.03,
        left_red_badge_search_min_height_px=1,
        left_red_badge_search_height_ratio=0.04,
    )
    image = Image.new("RGB", (560, 1000), "white")
    pixels = image.load()
    for y in range(208, 232):
        for x in range(82, 106):
            if (x - 94) ** 2 + (y - 220) ** 2 <= 12 ** 2:
                pixels[x, y] = (210, 30, 30)
    for y in range(214, 227):
        pixels[94, y] = (245, 245, 245)
        pixels[95, y] = (245, 245, 245)
    items = [
        _item("target", 128, 260, 250, 282),
        _item("content", 128, 360, 288, 320),
    ]

    result = build_list_result(
        items,
        (560, 1000),
        rules,
        source_image=image,
        ocr_config={"runtime_enabled": True, "preprocess": "original"},
    )

    assert result["message_list"][0]["unread_count"] == 1


def test_list_rules_keep_xinyu_bottom_tabs_in_bottom_nav():
    items = [
        _item("喜欢", 86, 128, 1238, 1268),
        _item("心动", 210, 252, 1238, 1268),
        _item("家族广场", 324, 406, 1238, 1268),
        _item("消息", 480, 522, 1238, 1268),
        _item("我的", 602, 644, 1238, 1268),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "bottom_nav_top_ratio": 0.80,
            "bottom_nav_keywords": ["喜欢", "心动", "家族广场", "消息", "我的", "全部消息", "全部信息", "未回复", "在线", "距离"],
            "top_nav_keywords": ["全部消息", "全部信息", "未回复", "在线", "距离"],
            "top_nav_top_ratio": 0.10,
            "top_nav_bottom_ratio": 0.20,
        },
    )

    assert [item["name"] for item in result["bottom_nav"]] == ["喜欢", "心动", "家族广场", "消息", "我的"]


def test_list_rules_do_not_infer_xinyu_bottom_tabs_when_ocr_text_is_missing():
    result = build_list_result(
        [],
        (681, 1498),
        {
            "bottom_nav_top_ratio": 0.80,
            "bottom_nav_accept_any_text": True,
        },
    )

    assert result["bottom_nav"] == []


def test_list_rules_accept_any_xinyu_bottom_tab_text():
    items = [
        _item("邂逅", 86, 128, 1238, 1268),
        _item("动态", 210, 252, 1238, 1268),
        _item("广场", 346, 388, 1238, 1268),
        _item("消息", 480, 522, 1238, 1268),
        _item("主页", 602, 644, 1238, 1268),
    ]

    result = build_list_result(
        items,
        (681, 1498),
        {
            "min_box_height_ratio": 0.01,
            "bottom_nav_top_ratio": 0.80,
            "bottom_nav_keywords": ["喜欢", "心动", "家族广场", "消息", "我的"],
            "bottom_nav_accept_any_text": True,
        },
    )

    assert [item["name"] for item in result["bottom_nav"]] == ["邂逅", "动态", "广场", "消息", "主页"]

def test_xinyu_list_rules_reject_name_beyond_right_boundary():
    rules = XINYU_LIST_OCR_MODE["rules"]
    items = [
        _item("rightName", 430, 460, 720, 760),
        _item("validName", 120, 260, 820, 860),
        _item("hello", 120, 190, 870, 905),
    ]

    result = build_list_result(items, (560, 1498), rules)

    assert [row["name"] for row in result["message_list"]] == ["validName"]
    assert result["message_list"][0]["content"] == "hello"
