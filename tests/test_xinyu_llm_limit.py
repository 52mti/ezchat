from __future__ import annotations

from apps.photo_capture import _sync_coordinate_cache
from apps.xinyu.steps import _acquire_llm_vision_slot
from apps.xinyu.steps import _update_dynamic_chat_area
from services import vision_usage


def test_llm_vision_is_limited_to_50_calls_per_perspective(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    config = {"_runtime": {"device_ip": "192.168.1.1", "app_name": "xinyu", "perspective_signature": "a"}}
    for expected in range(1, 51):
        assert _acquire_llm_vision_slot(config) == (True, expected)
    assert _acquire_llm_vision_slot(config) == (False, 50)


def test_llm_vision_limit_resets_when_perspective_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    config = {
        "_runtime": {"device_ip": "192.168.1.1", "app_name": "xinyu", "perspective_signature": "a"},
        "_runtime_cache": {},
    }
    assert _acquire_llm_vision_slot(config) == (True, 1)


def test_dynamic_chat_area_uses_sender_bounds_and_ocr_coordinates():
    config = {
        "ocr.regions": {},
        "ocr": {"regions": {"chat_area": [0.1, 0.2, 0.8, 0.6]}},
        "_runtime_cache": {},
    }
    payload = {"messages": [
        {"sender": "sender1", "content": "你好"},
        {"sender": "sender2", "content": "你喜欢小孩吗"},
    ]}
    result = {"ocr_items": [
        {"text": "你好", "bounds": [0.25, 0.3, 0.1, 0.04]},
        {"text": "你喜欢小孩吗", "bounds": [0.62, 0.4, 0.2, 0.04]},
    ]}
    assert _update_dynamic_chat_area(config, payload, result)
    assert config["_runtime_cache"]["ocr_regions"]["chat_area_dynamic"] == [0.2, 0.2, 0.67, 0.6]

