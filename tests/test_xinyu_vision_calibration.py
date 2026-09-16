# -*- coding: utf-8 -*-
"""「视觉+OCR识别一次」标定契约。

背景：视觉额度耗尽/规则学习停用后 chat_area_dynamic 永远学不出来，
dialogue_message 一直是全宽兜底。UI 提供一键标定：截图 → 视觉
parse_dialogue → 对话 OCR → 区域学习 → 落盘坐标缓存。手动标定独立
于运行额度：不检查、不消耗视觉识别次数。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import apps.xinyu.steps as steps
from apps.xinyu.config import DEFAULT_CONFIG


def _reply(payload: dict) -> object:
    """镜像 VisionDialogueReply：payload 在属性上（曾因直接当 dict 用而全线失败）。"""
    return SimpleNamespace(payload=payload, provider="openrouter_ensemble", model="fake", raw_text="")


def _base_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["_runtime"] = {"device_ip": "192.168.1.210", "app_name": "xinyu"}
    cfg["_runtime_cache"] = {}
    return cfg


def _fake_payload() -> dict:
    return {
        "messages": [
            {"sender": "sender1", "content": "你好呀"},
            {"sender": "sender2", "content": "很高兴认识你"},
        ]
    }


def _fake_ocr_result() -> dict:
    return {
        "result": {"dialogues": []},
        "ocr_items": [
            {"text": "你好呀", "bounds": [0.15, 0.30, 0.50, 0.05]},
            {"text": "很高兴认识你", "bounds": [0.50, 0.40, 0.45, 0.05]},
        ],
    }


def test_calibrate_dialogue_chat_area_learns_and_persists(monkeypatch):
    cfg = _base_config()
    persisted: list[dict] = []
    acquired: list[dict] = []
    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, config, pre_move=True, label=None: b"chat")
    monkeypatch.setattr(steps, "_is_chat_window_image", lambda image, config: True)
    monkeypatch.setattr(steps, "_acquire_llm_vision_slot", lambda config: acquired.append(config) or (False, steps.LLM_VISION_CALL_LIMIT))
    vision_calls: list[dict] = [{"messages": []}, _fake_payload()]  # 首次空消息，重试成功
    monkeypatch.setattr(
        steps._DIALOGUE_LLM_CLIENT,
        "parse_dialogue",
        lambda image, config: _reply(vision_calls.pop(0)),
    )
    monkeypatch.setattr(steps, "_run_ocr_mode", lambda image, config, mode, region=None: _fake_ocr_result())

    def fake_message_region(image, config):
        region = [0.0, 0.1, 1.0, 0.6]
        steps._cache_set(config, "ocr_regions", "dialogue_message", region)
        return region

    monkeypatch.setattr(steps, "_dialogue_message_region", fake_message_region)
    monkeypatch.setattr(steps, "persist_from_runtime", lambda config, **kwargs: persisted.append(config))

    result = steps.calibrate_dialogue_chat_area(object(), cfg)

    # 额度即使耗尽（acquire 返回 False）也不拦截——手动标定不消耗额度
    assert acquired == []
    assert vision_calls == []  # 空结果触发了一次自动重试
    assert result["ok"] is True
    # sender1 最左 0.15-0.05、sender2 最右 0.95+0.05 → 左右界收窄到 [0.1, 1.0]
    chat_area = result["chat_area_dynamic"]
    assert chat_area is not None
    assert chat_area[0] == pytest.approx(0.1, abs=0.01)
    assert chat_area[0] + chat_area[2] == pytest.approx(1.0, abs=0.01)
    assert result["dialogue_message"] == [0.0, 0.1, 1.0, 0.6]
    assert cfg["_runtime_cache"]["ocr_regions"]["chat_area_dynamic"] == chat_area
    assert len(persisted) >= 1  # 标定结果已落盘坐标缓存（_cache_set 自动持久化）


def test_calibrate_dialogue_chat_area_requires_chat_window(monkeypatch):
    cfg = _base_config()
    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, config, pre_move=True, label=None: b"list")
    monkeypatch.setattr(steps, "_is_chat_window_image", lambda image, config: False)
    # 预检 OCR 先行：模板不中且无消息文本 → 拒绝
    monkeypatch.setattr(steps, "_run_ocr_mode", lambda image, config, mode, region=None: {"result": {}, "ocr_items": []})
    called: list[int] = []
    monkeypatch.setattr(
        steps._DIALOGUE_LLM_CLIENT,
        "parse_dialogue",
        lambda image, config: called.append(1) or _fake_payload(),
    )

    result = steps.calibrate_dialogue_chat_area(object(), cfg)

    assert result["ok"] is False
    assert "聊天界面" in result["reason"]
    assert called == []  # 非聊天界面不发起视觉请求
    assert "chat_area_dynamic" not in cfg["_runtime_cache"].get("ocr_regions", {})


def test_vision_sender_aliases_are_normalized():
    """汇总模型用「对方/我」等别名时不得丢弃消息（视觉返回 0 条事故根因）。"""
    from services.vision_dialogue_client import _normalize_messages

    payload = {
        "messages": [
            {"sender": "对方", "content": "你好"},
            {"sender": "我", "content": "你好呀"},
            {"sender": "left", "content": "在吗"},
            {"sender": "right", "content": "在的"},
            {"sender": "unknown_role", "content": "应当被丢弃"},
        ]
    }

    normalized = _normalize_messages(payload)["messages"]

    assert [item["sender"] for item in normalized] == ["sender1", "sender2", "sender1", "sender2"]


def test_update_dynamic_chat_area_reads_flat_xywh_ocr_items():
    """对话 ocr_items 是平铺 x/y/w/h 形态（无 bounds 键），区域学习必须兼容。

    回归背景：旧代码只认 bounds 键，所有匹配被丢弃，chat_area_dynamic
    从未被学出来——视觉额度耗尽只是次要原因。
    """
    cfg = _base_config()
    payload = {
        "messages": [
            {"sender": "sender1", "content": "气质太好了"},
            {"sender": "sender2", "content": "谢谢你的持续点赞"},
        ]
    }
    ocr_result = {
        "result": {"dialogues": []},
        "ocr_items": [
            {"text": "气质太好了", "x": 0.179, "y": 0.137, "w": 0.439, "h": 0.04},
            {"text": "谢谢你的持续点赞", "x": 0.351, "y": 0.485, "w": 0.443, "h": 0.04},
        ],
    }

    assert steps._update_dynamic_chat_area(cfg, payload, ocr_result) is True

    chat_area = cfg["_runtime_cache"]["ocr_regions"]["chat_area_dynamic"]
    assert chat_area[0] == pytest.approx(0.179 - 0.05)  # sender1 最左 - margin
    assert chat_area[0] + chat_area[2] == pytest.approx(0.351 + 0.443 + 0.05)  # sender2 最右 + margin


def test_calibrate_reuses_provided_image_when_perspective_synced(monkeypatch):
    """UI 预截图后传入 image：透视签名已同步时不得二次截图（省 1~2 秒）。"""
    cfg = _base_config()
    cfg["_runtime"]["perspective_signature"] = "sig-1"
    captured: list[str] = []
    monkeypatch.setattr(
        steps,
        "capture_transform_image",
        lambda robot, config, pre_move=True, label=None: captured.append(label) or b"chat",
    )
    monkeypatch.setattr(steps, "_is_chat_window_image", lambda image, config: True)
    monkeypatch.setattr(
        steps._DIALOGUE_LLM_CLIENT,
        "parse_dialogue",
        lambda image, config: _reply(_fake_payload()),
    )
    monkeypatch.setattr(steps, "_run_ocr_mode", lambda image, config, mode, region=None: _fake_ocr_result())

    def fake_message_region(image, config):
        steps._cache_set(config, "ocr_regions", "dialogue_message", [0.0, 0.1, 1.0, 0.6])
        return [0.0, 0.1, 1.0, 0.6]

    monkeypatch.setattr(steps, "_dialogue_message_region", fake_message_region)
    monkeypatch.setattr(steps, "persist_from_runtime", lambda config, **kwargs: None)

    result = steps.calibrate_dialogue_chat_area(object(), cfg, image=b"pre-captured")

    assert captured == []  # 未二次截图
    assert result["ok"] is True
