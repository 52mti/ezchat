from __future__ import annotations

from PIL import Image

import vision.ocr_rules as ocr_rules
from vision.ocr_rules import build_list_result


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

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.01})

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

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.01})

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

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.02})

    assert result["message_list"][0]["name"] == "\u65e0\u654c\u963f\u5f3a"
    assert result["message_list"][0]["unread_count"] == 4


def test_list_rules_caps_right_side_99_plus_badge():
    items = [
        _item("\u7528\u6237\u901a\u77e5", 140, 270, 210, 247),
        _item("14:57", 579, 648, 226, 255),
        _item("99+", 598, 640, 276, 299),
        _item("\u60a8\u6536\u5230\u4e00\u6761\u65b0\u7684\u8bed\u97f3\u901a\u8bdd\u9080\u8bf7", 141, 514, 278, 316),
    ]

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.01})

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

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.01})

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

    result = build_list_result(items, (681, 1498), {"min_box_height_ratio": 0.01})

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
