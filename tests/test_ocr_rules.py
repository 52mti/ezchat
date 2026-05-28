from __future__ import annotations

from PIL import Image

import vision.ocr_rules as ocr_rules
from vision.ocr_rules import build_dialogue_result, run_ocr_mode


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
        {
            "ocr": {"runtime_enabled": True},
            "ocr_modes": {"list": {"preprocess": {"rotate_to_portrait": True}}},
        },
        "list",
    )

    assert result["result"]["message_list"][0]["bounds"] == [0.1, 0.2, 0.4, 0.5]
    assert result["result"]["bottom_nav"][0]["bounds"] == [0.7, 0.8, 0.9, 0.95]


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

    result = build_dialogue_result(items, (200, 300), {"name_gap": 80})

    assert result["dialogues"] == [{"sender": "sender1", "content": "你好在吗"}]


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
        {
            "basic_filter": {
                "enable_right_narrow_non_cjk_filter": True,
                "right_narrow_min_x_ratio": 0.88,
                "right_narrow_max_width_ratio": 0.12,
                "enable_non_cjk_non_emoji_filter": False,
            },
        },
    )

    assert result["dialogues"] == [{"sender": "sender1", "content": "你好"}]
