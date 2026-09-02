from __future__ import annotations

from apps.xinyu.steps import _dialogue_message_region, _update_dynamic_chat_area


def test_dynamic_chat_area_uses_quick_reply_ocr_top_as_bottom_boundary():
    config = {
        "ocr": {"regions": {"chat_area": [0.0, 0.1, 1.0, 0.8]}},
        "_runtime_cache": {},
    }
    payload = {
        "messages": [{"sender": "sender1", "content": "真实消息"}],
        "quick_replies": [{"type": "quick_reply", "content": "今晚聊聊"}],
    }
    result = {
        "ocr_items": [
            {"text": "真实消息", "bounds": [0.2, 0.35, 0.2, 0.04]},
            {"text": "今晚聊聊", "bounds": [0.1, 0.72, 0.2, 0.04]},
        ]
    }

    assert _update_dynamic_chat_area(config, payload, result)
    regions = config["_runtime_cache"]["ocr_regions"]
    assert regions["dialogue_quick_reply_bottom"] == 0.71
    assert regions["chat_area_dynamic"] == [0.15, 0.1, 0.85, 0.61]


def test_dialogue_message_region_reuses_cached_quick_reply_bottom():
    config = {
        "ocr": {"regions": {"chat_area": [0.0, 0.1, 1.0, 0.8]}},
        "dialogue_message_input_box_top_margin": 0.065,
        "_runtime_cache": {
            "template_points": {"input_box": [0.9, 0.9]},
            "ocr_regions": {"dialogue_quick_reply_bottom": 0.71},
        },
    }

    assert _dialogue_message_region("chat", config) == [0.0, 0.1, 1.0, 0.61]
