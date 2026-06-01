from __future__ import annotations

import copy

from PIL import Image

import vision.ocr_rules as ocr_rules
from apps.huanyou.ocr_config import DIALOGUE_OCR_MODE as HUANYOU_DIALOGUE_OCR_MODE
from apps.wechat.config import DEFAULT_CONFIG as WECHAT_CONFIG
from apps.wechat.ocr_config import DIALOGUE_OCR_MODE as WECHAT_DIALOGUE_OCR_MODE
from vision.ocr_rules import build_dialogue_result, run_ocr_mode


def _dialogue_rules(**overrides):
    rules = copy.deepcopy(WECHAT_DIALOGUE_OCR_MODE["rules"])
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(rules.get(key), dict):
            rules[key].update(value)
        else:
            rules[key] = value
    return rules


def test_run_ocr_mode_forces_original_preprocess(monkeypatch):
    calls: list[str] = []

    def fake_ocr(image, config):
        calls.append(str(config.get("preprocess")))
        assert image.size == (100, 200)
        assert config.get("rotate_to_portrait") is False
        return [
            {"text": "[未读]张三", "x": 0.2, "y": 0.3, "w": 0.3, "h": 0.1, "confidence": 0.95}
        ]

    def fake_build_list_result(*_args, **_kwargs):
        return {"message_list": [{"unread_count": 1, "bounds": [0.2, 0.3, 0.5, 0.4]}], "bottom_nav": []}

    monkeypatch.setattr(ocr_rules, "ocr", fake_ocr)
    monkeypatch.setattr(ocr_rules, "build_list_result", fake_build_list_result)
    image = Image.new("RGB", (200, 100), "white")
    config = {
        "ocr": {"runtime_enabled": True},
        "ocr_modes": {
            "list": {
                "preprocess": {"mode": "original", "scale": 2},
                "rules": {"min_confidence": 0.1, "min_box_height_ratio": 0.01},
            }
        },
    }

    result = run_ocr_mode(image, config, "list", region=[0.1, 0.2, 0.5, 0.5])

    assert calls
    assert set(calls) == {"original"}
    assert result["ocr_items"][0]["x"] == 0.2
    assert result["ocr_items"][0]["y"] == 0.35
    assert result["result"]["message_list"][0]["unread_count"] == 1


def test_run_ocr_mode_returns_portrait_bounds_for_landscape_input(monkeypatch):
    def fake_ocr(_image, _config):
        return [{"text": "sample", "x": 0.2, "y": 0.2, "w": 0.2, "h": 0.1, "confidence": 0.99}]

    def fake_build_list_result(*_args, **_kwargs):
        return {
            "message_list": [
                {
                    "name": "sample",
                    "favorability": None,
                    "unread_count": 1,
                    "content": "content",
                    "bounds": [0.1, 0.2, 0.4, 0.5],
                }
            ],
            "bottom_nav": [{"name": "me", "bounds": [0.7, 0.8, 0.9, 0.95]}],
        }

    monkeypatch.setattr(ocr_rules, "ocr", fake_ocr)
    monkeypatch.setattr(ocr_rules, "build_list_result", fake_build_list_result)

    result = run_ocr_mode(
        Image.new("RGB", (200, 100), "white"),
        copy.deepcopy(WECHAT_CONFIG),
        "list",
    )

    assert result["result"]["message_list"][0]["bounds"] == [0.1, 0.2, 0.4, 0.5]
    assert result["result"]["bottom_nav"][0]["bounds"] == [0.7, 0.8, 0.9, 0.95]


def test_dialogue_result_returns_excluded_reasons():
    items = [
        {
            "text": "hello",
            "raw_text": "hello",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 110,
            "min_y": 100,
            "max_y": 140,
            "center_x": 65,
            "center_y": 120,
            "width": 90,
            "height": 40,
            "background_saturation": 0.0,
        },
        {
            "text": "blocked_item_message",
            "raw_text": "blocked_item_message",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 180,
            "min_y": 150,
            "max_y": 190,
            "center_x": 100,
            "center_y": 170,
            "width": 160,
            "height": 40,
            "background_saturation": 0.0,
        },
        {
            "text": "blocked_line_message",
            "raw_text": "blocked_line_message",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 190,
            "min_y": 200,
            "max_y": 240,
            "center_x": 105,
            "center_y": 220,
            "width": 170,
            "height": 40,
            "background_saturation": 0.0,
        },
        {
            "text": "cutoff",
            "raw_text": "cutoff",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 90,
            "min_y": 250,
            "max_y": 290,
            "center_x": 55,
            "center_y": 270,
            "width": 70,
            "height": 40,
            "background_saturation": 0.0,
        },
        {
            "text": "after_cutoff",
            "raw_text": "after_cutoff",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 130,
            "min_y": 300,
            "max_y": 340,
            "center_x": 75,
            "center_y": 320,
            "width": 110,
            "height": 40,
            "background_saturation": 0.0,
        },
    ]

    result = build_dialogue_result(
        items,
        (200, 400),
        _dialogue_rules(
            ocr_result_excluded_keywords=["blocked_item"],
            line_excluded_keywords=["blocked_line"],
            line_and_below_keywords=["cutoff"],
            basic_filter={
                "enable_max_x_filter": False,
                "enable_max_width_filter": False,
                "enable_non_cjk_non_emoji_filter": False,
                "enable_right_narrow_non_cjk_filter": False,
            },
        ),
    )

    excluded = result["debug"]["excluded"]
    assert any(
        item["reason"] == "ocr_keyword"
        and item["text"] == "blocked_item_message"
        and item["matched_keyword"] == "blocked_item"
        for item in excluded
    )
    assert any(
        item["reason"] == "line_excluded_keyword"
        and item["text"] == "helloblocked_line_message"
        and item["matched_keyword"] == "blocked_line"
        for item in excluded
    )
    assert any(
        item["reason"] == "line_and_below_keyword"
        and item["text"] == "cutoff"
        and item["matched_keyword"] == "cutoff"
        for item in excluded
    )
    assert any(item["reason"] == "after_line_and_below_cutoff" and item["text"] == "after_cutoff" for item in excluded)


def test_dialogue_rules_filter_time_and_merge_sender_lines():
    items = [
        {
            "text": "12:30",
            "raw_text": "12:30",
            "confidence": 0.99,
            "min_x": 10,
            "max_x": 50,
            "min_y": 30,
            "max_y": 45,
            "center_x": 30,
            "center_y": 37,
            "width": 40,
            "height": 15,
            "background_saturation": 0.0,
        },
        {
            "text": "你好",
            "raw_text": "你好",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 80,
            "min_y": 100,
            "max_y": 130,
            "center_x": 50,
            "center_y": 115,
            "width": 60,
            "height": 30,
            "background_saturation": 0.0,
        },
        {
            "text": "在吗",
            "raw_text": "在吗",
            "confidence": 0.99,
            "min_x": 22,
            "max_x": 82,
            "min_y": 150,
            "max_y": 180,
            "center_x": 52,
            "center_y": 165,
            "width": 60,
            "height": 30,
            "background_saturation": 0.0,
        },
    ]

    result = build_dialogue_result(items, (200, 300), _dialogue_rules(name_gap=80))

    assert result["dialogues"] == [{"sender": "sender1", "content": "你好在吗"}]


def test_dialogue_name_gap_ratio_scales_with_image_height():
    items = [
        {
            "text": "浣犲ソ",
            "raw_text": "浣犲ソ",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 80,
            "min_y": 220,
            "max_y": 250,
            "center_x": 50,
            "center_y": 235,
            "width": 60,
            "height": 30,
            "background_saturation": 0.0,
        },
        {
            "text": "鍦ㄥ悧",
            "raw_text": "鍦ㄥ悧",
            "confidence": 0.99,
            "min_x": 22,
            "max_x": 82,
            "min_y": 299,
            "max_y": 329,
            "center_x": 52,
            "center_y": 314,
            "width": 60,
            "height": 30,
            "background_saturation": 0.0,
        },
    ]
    rules = _dialogue_rules(name_gap_ratio=0.05)

    tall_result = build_dialogue_result(items, (200, 1000), rules)
    short_result = build_dialogue_result(items, (200, 500), rules)

    assert tall_result["dialogues"] == [{"sender": "sender1", "content": "浣犲ソ鍦ㄥ悧"}]
    assert short_result["dialogues"] == [
        {"sender": "sender1", "content": "浣犲ソ"},
        {"sender": "sender1", "content": "鍦ㄥ悧"},
    ]


def test_dialogue_rules_filter_right_narrow_non_cjk_items():
    items = [
        {
            "text": "12",
            "raw_text": "12",
            "confidence": 0.99,
            "min_x": 185,
            "max_x": 198,
            "min_y": 100,
            "max_y": 120,
            "center_x": 191.5,
            "center_y": 110,
            "width": 13,
            "height": 20,
            "background_saturation": 0.0,
        },
        {
            "text": "你好",
            "raw_text": "你好",
            "confidence": 0.99,
            "min_x": 20,
            "max_x": 80,
            "min_y": 130,
            "max_y": 150,
            "center_x": 50,
            "center_y": 140,
            "width": 60,
            "height": 20,
            "background_saturation": 0.0,
        },
    ]

    result = build_dialogue_result(
        items,
        (200, 300),
        _dialogue_rules(
            basic_filter={
                "enable_right_narrow_non_cjk_filter": True,
                "right_narrow_min_x_ratio": 0.88,
                "right_narrow_max_width_ratio": 0.12,
                "enable_non_cjk_non_emoji_filter": False,
            }
        ),
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "你好"}]


def test_huanyou_dialogue_rules_keep_colored_chat_bubbles_only():
    def item(text: str, min_x: int, max_x: int, min_y: int, max_y: int, sat: float) -> dict:
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
            "background_saturation": sat,
        }

    items = [
        item("火龙果", 233, 313, 56, 92, 0.1),
        item("你觉得相处起来什么最重要", 159, 443, 268, 300, 0.61),
        item("收到搭训礼物", 107, 249, 370, 398, 0.22),
        item("丘比特发现你俩今天很有缘，千万别错过", 85, 473, 618, 643, 0.09),
        item("你好", 126, 182, 786, 820, 0.31),
        item("多回复几句，回复率更高哦~", 82, 406, 1038, 1066, 0.09),
    ]

    result = build_dialogue_result(items, (564, 1222), HUANYOU_DIALOGUE_OCR_MODE["rules"])

    assert result["user_name"] == "火龙果"
    assert result["dialogues"] == [
        {"sender": "sender2", "content": "你觉得相处起来什么最重要"},
        {"sender": "sender1", "content": "你好"},
    ]


def test_huanyou_dialogue_rules_do_not_merge_separate_left_bubbles_with_large_gap():
    def item(text: str, min_x: int, max_x: int, min_y: int, max_y: int) -> dict:
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
            "background_saturation": 0.60,
        }

    items = [
        item("你好", 126, 180, 273, 304),
        item("看你也是个值得托付的人，我", 114, 424, 370, 402),
        item("想把爱交给你", 115, 258, 406, 435),
    ]

    result = build_dialogue_result(items, (681, 1222), HUANYOU_DIALOGUE_OCR_MODE["rules"])

    assert result["dialogues"] == [
        {"sender": "sender1", "content": "你好"},
        {"sender": "sender1", "content": "看你也是个值得托付的人，我想把爱交给你"},
    ]
