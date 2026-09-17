from __future__ import annotations

import copy

from PIL import Image

import apps.xinyu.ocr_rules as ocr_rules
import apps.xinyu.ocr_rules as xinyu_ocr_rules
from apps.xinyu.ocr_config import DIALOGUE_OCR_MODE as XINYU_DIALOGUE_OCR_MODE
from apps.xinyu.ocr_defaults import DIALOGUE_OCR_DEFAULT as XINYU_DIALOGUE_OCR_DEFAULT
from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG
from apps.xinyu.ocr_rules import build_dialogue_result, looks_like_time_text, run_ocr_mode


def _dialogue_rules(**overrides):
    rules = copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"])
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




def test_run_ocr_mode_reuses_result_for_same_image_and_config(monkeypatch):
    ocr_rules._OCR_RESULT_CACHE.clear()
    calls = 0

    def fake_ocr(_image, _config):
        nonlocal calls
        calls += 1
        return [{"text": "张三", "x": 0.2, "y": 0.3, "w": 0.3, "h": 0.1, "confidence": 0.95}]

    def fake_build_list_result(*_args, **_kwargs):
        return {"message_list": [{"name": "张三", "bounds": [0.2, 0.3, 0.5, 0.4]}], "bottom_nav": []}

    monkeypatch.setattr(ocr_rules, "ocr", fake_ocr)
    monkeypatch.setattr(ocr_rules, "build_list_result", fake_build_list_result)
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "ocr": {"runtime_enabled": True},
        "ocr_modes": {
            "list": {
                "preprocess": {"mode": "original", "scale": 1},
                "rules": {"min_confidence": 0.1, "min_box_height_ratio": 0.01},
            }
        },
    }

    first = run_ocr_mode(image, config, "list")
    second = run_ocr_mode(image, config, "list")

    # _queue_seconds 是并发槽计时，毫秒级抖动属正常，不参与缓存复用断言
    first.pop("_queue_seconds", None)
    second.pop("_queue_seconds", None)
    assert first == second
    assert calls == 1




def test_run_ocr_mode_cache_is_scoped_by_app_name(monkeypatch):
    ocr_rules._OCR_RESULT_CACHE.clear()
    calls = 0

    def fake_ocr(_image, _config):
        nonlocal calls
        calls += 1
        return [{"text": f"张三{calls}", "x": 0.2, "y": 0.3, "w": 0.3, "h": 0.1, "confidence": 0.95}]

    def fake_build_list_result(items, *_args, **_kwargs):
        return {"message_list": [{"name": items[0]["text"], "bounds": [0.2, 0.3, 0.5, 0.4]}], "bottom_nav": []}

    monkeypatch.setattr(ocr_rules, "ocr", fake_ocr)
    monkeypatch.setattr(ocr_rules, "build_list_result", fake_build_list_result)
    image = Image.new("RGB", (100, 200), "white")
    base_config = {
        "ocr": {"runtime_enabled": True},
        "ocr_modes": {
            "list": {
                "preprocess": {"mode": "original", "scale": 1},
                "rules": {"min_confidence": 0.1, "min_box_height_ratio": 0.01},
            }
        },
    }
    first_config = copy.deepcopy(base_config)
    first_config["_runtime"] = {"device_ip": "192.168.1.108", "app_name": "huanyou"}
    second_config = copy.deepcopy(base_config)
    second_config["_runtime"] = {"device_ip": "192.168.1.108", "app_name": "xinyu"}

    first = run_ocr_mode(image, first_config, "list")
    second = run_ocr_mode(image, second_config, "list")

    assert calls == 2
    assert first["result"]["message_list"][0]["name"] == "张三1"
    assert second["result"]["message_list"][0]["name"] == "张三2"




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
        copy.deepcopy(XINYU_CONFIG),
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
                "enable_edge_spanning_filter": False,
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
        and item["text"] == "blocked_line_message"
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
    assert [dialogue["content"] for dialogue in result["dialogues"]] == ["hello"]




def test_xinyu_line_excluded_keywords_use_stable_substrings():
    rules = copy.deepcopy(XINYU_DIALOGUE_OCR_MODE["rules"])
    rules["basic_filter"].update({
        "enable_edge_spanning_filter": False,
        "enable_max_width_filter": False,
        "enable_non_cjk_non_emoji_filter": False,
        "enable_right_narrow_non_cjk_filter": False,
    })
    items = []
    for index, text in enumerate((
        "亲密度220",
        "家乡:河北石家庄",
        "身高:170CM家乡:河北石家庄更多>",
    )):
        top = 120 + index * 70
        items.append({
            "text": text,
            "raw_text": text,
            "confidence": 0.99,
            "min_x": 30,
            "max_x": 400,
            "min_y": top,
            "max_y": top + 32,
            "center_x": 215,
            "center_y": top + 16,
            "width": 370,
            "height": 32,
            "background_saturation": 0.0,
        })

    result = xinyu_ocr_rules.build_dialogue_result(items, (500, 600), rules)

    assert result["dialogues"] == []
    excluded = result["debug"]["excluded"]
    matched = {
        item.get("matched_keyword")
        for item in excluded
        if item.get("reason") == "line_excluded_keyword"
    }
    assert matched >= {"亲密度", "家乡"}




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

    result = build_dialogue_result(items, (200, 300), _dialogue_rules(name_gap_ratio=80 / 300))

    assert result["dialogues"] == [{"sender": "sender1", "content": "你好在吗"}]




def test_looks_like_time_text_accepts_relative_day_with_clock():
    assert looks_like_time_text("昨天05:08") is True
    assert looks_like_time_text("前天 12:30") is True
    assert looks_like_time_text("半夜12:02") is True
    assert looks_like_time_text("晚上 11:45") is True




def test_dialogue_filters_time_with_period_prefix_before_sender_detection():
    items = [
        {
            "text": "半夜12:02",
            "raw_text": "半夜12:02",
            "confidence": 0.99,
            "min_x": 120,
            "max_x": 280,
            "min_y": 180,
            "max_y": 220,
            "center_x": 200,
            "center_y": 200,
            "width": 160,
            "height": 40,
            "background_saturation": 0.0,
            "background_hue": None,
            "background_value": 0.9,
        },
    ]

    result = build_dialogue_result(items, (500, 800), _dialogue_rules(name_gap=80))

    assert result["dialogues"] == []




def test_xinyu_dialogue_rules_filter_prompt_keywords_before_sender_detection():
    items = [
        {
            "text": "和他聊聊看，别错过缘分哦。",
            "raw_text": "和他聊聊看，别错过缘分哦。",
            "confidence": 0.99,
            "min_x": 120,
            "max_x": 430,
            "min_y": 250,
            "max_y": 290,
            "center_x": 275,
            "center_y": 270,
            "width": 310,
            "height": 40,
            "background_saturation": 0.41,
            "background_hue": 103.0,
            "background_value": 0.48,
            "background_blue_green_delta": 23.0,
            "background_blue_red_ratio": 9.5,
        },
        {
            "text": "消息发出没有回复？试试热聊话题",
            "raw_text": "消息发出没有回复？试试热聊话题",
            "confidence": 0.99,
            "min_x": 155,
            "max_x": 405,
            "min_y": 960,
            "max_y": 1005,
            "center_x": 280,
            "center_y": 982,
            "width": 250,
            "height": 45,
            "background_saturation": 0.41,
            "background_hue": 103.0,
            "background_value": 0.48,
            "background_blue_green_delta": 23.0,
            "background_blue_red_ratio": 9.5,
        },
        {
            "text": "请输入信息",
            "raw_text": "请输入信息",
            "confidence": 0.99,
            "min_x": 60,
            "max_x": 190,
            "min_y": 1110,
            "max_y": 1155,
            "center_x": 125,
            "center_y": 1132,
            "width": 130,
            "height": 45,
            "background_saturation": 0.41,
            "background_hue": 103.0,
            "background_value": 0.48,
            "background_blue_green_delta": 23.0,
            "background_blue_red_ratio": 9.5,
        },
        {
            "text": "你会把喜欢的人设置成聊天置顶吗？",
            "raw_text": "你会把喜欢的人设置成聊天置顶吗？",
            "confidence": 0.99,
            "min_x": 98,
            "max_x": 462,
            "min_y": 360,
            "max_y": 430,
            "center_x": 280,
            "center_y": 395,
            "width": 364,
            "height": 70,
            "background_saturation": 0.42,
            "background_hue": 4.0,
            "background_value": 0.82,
            "background_blue_green_delta": -12.0,
            "background_blue_red_ratio": 0.62,
        },
    ]

    # 用代码层默认值断言：本机/客户机存在 runtime/configs 客户层时会整表
    # 覆盖关键词列表，用 DIALOGUE_OCR_MODE 会随机器不同而漂移。
    result = xinyu_ocr_rules.build_dialogue_result(items, (562, 1254), XINYU_DIALOGUE_OCR_DEFAULT["rules"])

    # 心遇规则中“家乡”是系统资料/推荐话术的稳定片段，命中后整行过滤。
    assert result["dialogues"] == []




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




def test_xinyu_dialogue_user_name_accepts_configured_single_char_name():
    item = {
        "text": "无",
        "raw_text": "无",
        "confidence": 0.99,
        "min_x": 300,
        "max_x": 344,
        "min_y": 70,
        "max_y": 106,
        "center_x": 322,
        "center_y": 88,
        "width": 44,
        "height": 36,
        "background_saturation": 0.1,
    }

    result = xinyu_ocr_rules.build_dialogue_result([item], (681, 1222), XINYU_DIALOGUE_OCR_MODE["rules"])

    assert result["user_name"] == "无"
    assert result["dialogues"] == []




def test_xinyu_dialogue_strong_right_position_overrides_ambiguous_color():
    item = {
        "text": "好好色",
        "raw_text": "好好色",
        "confidence": 0.99,
        "min_x": 400,
        "max_x": 478,
        "min_y": 216,
        "max_y": 248,
        "center_x": 439,
        "center_y": 232,
        "width": 78,
        "height": 32,
        "background_saturation": 0.16,
        "background_hue": 172,
        "background_value": 0.42,
        "background_blue_green_delta": 21,
        "background_blue_red_ratio": 0.52,
    }
    result = xinyu_ocr_rules.build_dialogue_result([item], (681, 1222), XINYU_DIALOGUE_OCR_MODE["rules"])
    assert result["dialogues"] == [{"sender": "sender2", "content": "好好色"}]
