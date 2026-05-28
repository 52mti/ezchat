from __future__ import annotations

from vision.ocr_rules import build_dialogue_result


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
        {
            "name_gap": 40,
            "basic_filter": {
                "enable_non_cjk_non_emoji_filter": False,
            },
        },
    )

    assert result["dialogues"] == [
        {"sender": "sender1", "content": "left message"},
        {"sender": "sender2", "content": "right message"},
    ]
