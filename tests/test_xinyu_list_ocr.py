"""列表条带小范围 OCR 行为测试。

契约：条带覆盖「未回复」Tab 带，有行即返回（不依赖导航缓存合并，
不再"同帧门槛失败→回退全屏"每轮双 OCR）；Tab 位置永远来自当前帧。
"""

from __future__ import annotations

import apps.xinyu.steps as steps


def _config_with_cache(cache: dict) -> dict:
    return {
        "_runtime_cache": cache,
        "enable_compact_list_ocr_region": True,
        "enable_unread_tab_band_cache": True,
    }


def test_run_list_ocr_returns_compact_without_full_fallback(monkeypatch):
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

    assert calls == [[0.0, 0.06, 1.0, 0.82]]  # 只跑一次条带，无全屏回退
    assert result["result"]["message_list"][0]["name"] == "微风"
    item, center = steps._unread_tab_item(result)
    assert item is not None and "未回复" in item.get("name", "")  # 条带结果自带入口位置


def test_unread_tab_band_extends_compact_top():
    cache = {"ocr_regions": {"list_compact": [0.0, 0.28, 1.0, 0.60]}}
    config = _config_with_cache(cache)

    steps._cache_unread_tab_band(config, [0.0, 0.10, 1.0, 0.15])

    band = cache["ocr_regions"]["unread_tab_band"]
    compact = cache["ocr_regions"]["list_compact"]
    assert compact[1] <= band[1]  # 条带上界已盖住 Tab 带
    assert compact[1] + compact[3] >= 0.85  # 下界不缩水


def test_compact_learn_covers_unread_band_even_above_list_region():
    """Tab 带顶（0.092）可能高于列表区域顶（0.1）：学习不得把它切掉。"""
    cache = {"ocr_regions": {"unread_tab_band": [0.0, 0.092, 1.0, 0.053]}}
    config = _config_with_cache(cache)
    ocr_result = {"result": {"message_list": [{"bounds": [0.1, 0.30, 0.9, 0.40]}]}}

    steps._cache_list_compact_region_from_ocr_result(config, ocr_result)

    compact = cache["ocr_regions"]["list_compact"]
    assert compact[1] <= 0.092


def test_compact_falls_back_to_full_when_rows_empty(monkeypatch):
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

    assert calls == [[0.0, 0.06, 1.0, 0.82], None]  # 条带空 → 回退全屏一次
    assert result["result"]["message_list"][0]["name"] == "微风"
