"""列表固定条带 OCR 行为测试。

契约（20260919 起 compact 固定 [0, 0.05, 1, 0.30]，与顶部条带同区域）：
条带有行即返回；无行（含横幅把「未回复」Tab/首行压出条带的场景）回退
完整 OCR，首行不漏；旧版自适应缓存（ocr_regions.list_compact）不再生效。
"""

from __future__ import annotations

import pytest

import apps.xinyu.steps as steps


def _config_with_cache(cache: dict) -> dict:
    return {
        "_runtime_cache": cache,
        "enable_compact_list_ocr_region": True,
        "enable_unread_tab_band_cache": True,
    }


def test_run_list_ocr_returns_fixed_band_without_full_fallback(monkeypatch):
    # 缓存里的旧自适应区域应被无视：实际扫描区域恒为固定条带
    cache = {"ocr_regions": {"list_compact": [0.0, 0.06, 1.0, 0.82]}}
    config = _config_with_cache(cache)
    calls: list = []

    def fake_ocr(image, cfg, mode, region=None):
        calls.append(region)
        assert mode == "list"
        if region is not None:
            return {
                "result": {"message_list": [{"name": "微风", "content": "你好", "bounds": [0.1, 0.3, 0.9, 0.4]}]},
                "ocr_items": [
                    {"text": "未回复", "x": 0.05, "y": 0.10, "w": 0.10, "h": 0.03, "confidence": 0.9},
                ],
            }
        return {"result": {"message_list": []}, "ocr_items": []}

    monkeypatch.setattr(steps, "_run_ocr_mode", fake_ocr)
    result = steps._run_list_ocr("img", config)

    assert calls[0] == pytest.approx([0.0, 0.05, 1.0, 0.30])  # 只跑一次固定条带，无全屏回退
    assert len(calls) == 1
    assert result["result"]["message_list"][0]["name"] == "微风"
    item, center = steps._unread_tab_item(result)
    assert item is not None and "未回复" in item.get("name", "")  # 条带结果自带入口位置


def test_compact_falls_back_to_full_when_rows_empty(monkeypatch):
    """条带无行（如 Tab/首行被横幅压到 0.30 以下）→ 回退完整 OCR，首行不漏。"""
    cache = {"ocr_regions": {"list_compact": [0.0, 0.06, 1.0, 0.82]}}
    config = _config_with_cache(cache)
    calls: list = []

    def fake_ocr(image, cfg, mode, region=None):
        calls.append(region)
        if region is not None:
            return {"result": {"message_list": []}, "ocr_items": []}
        return {
            "result": {"message_list": [{"name": "微风", "content": "你好", "bounds": [0.1, 0.3, 0.9, 0.4]}]},
            "ocr_items": [],
            "result_bottom_nav_placeholder": None,
        }

    monkeypatch.setattr(steps, "_run_ocr_mode", fake_ocr)
    result = steps._run_list_ocr("img", config)

    assert calls[0] == pytest.approx([0.0, 0.05, 1.0, 0.30])  # 条带空 → 回退全屏一次
    assert calls[1] is None
    assert result["result"]["message_list"][0]["name"] == "微风"
